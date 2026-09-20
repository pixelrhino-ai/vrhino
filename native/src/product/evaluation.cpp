#include "vrhino/product/evaluation.h"
#include "vrhino/product/prepared_execution.h"
#include "vrhino/product/qualification_precision.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/tensor_util.h"
#include "vrhino/error.h"
#include "run_media.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <chrono>
#ifndef _WIN32
#include <nvml.h>
#include <dlfcn.h>
#endif

namespace vrhino::product {
namespace {
Json document(const std::filesystem::path& p) {
    require(std::filesystem::is_regular_file(p)&&std::filesystem::file_size(p)<=65536,"Invalid evaluation options file");
    std::ifstream f(p);require(f.good(),"Cannot read evaluation options");
    return Json::parse(std::string(std::istreambuf_iterator<char>(f),{}),JsonParseLimits{});
}
void save(const std::filesystem::path& p,const Json& j) {
    const auto tmp=p.string()+".tmp";std::ofstream f(tmp);require(f.good(),"Cannot write evaluation evidence");
    f<<j.serialize()<<'\n';f.close();require(f.good(),"Evaluation evidence write failed");std::filesystem::rename(tmp,p);
}
#ifndef _WIN32
class DeviceMonitor {
    void* library_=nullptr;
    nvmlReturn_t (*shutdown_)()=nullptr;
    nvmlReturn_t (*count_)(unsigned int*)=nullptr;
    nvmlReturn_t (*handle_)(unsigned int,nvmlDevice_t*)=nullptr;
    nvmlReturn_t (*memory_)(nvmlDevice_t,nvmlMemory_t*)=nullptr;
    template<class T>T symbol(const char* n){auto p=dlsym(library_,n);require(p!=nullptr,"Missing NVML monitoring symbol");return reinterpret_cast<T>(p);}
public:
    ~DeviceMonitor(){if(shutdown_)shutdown_();if(library_)dlclose(library_);}
    uint64_t used() {
        if(!library_) {
            library_=dlopen("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL);require(library_,"NVML required for bounded GPU evaluation");
            const auto init=symbol<nvmlReturn_t(*)()>("nvmlInit_v2");require(init()==NVML_SUCCESS,"NVML initialization failed");
            shutdown_=symbol<decltype(shutdown_)>("nvmlShutdown");
            count_=symbol<decltype(count_)>("nvmlDeviceGetCount_v2");
            handle_=symbol<decltype(handle_)>("nvmlDeviceGetHandleByIndex_v2");
            memory_=symbol<decltype(memory_)>("nvmlDeviceGetMemoryInfo");
        }
        unsigned int n=0;require(count_(&n)==NVML_SUCCESS&&n>0,"No GPU monitoring devices");uint64_t used=0;
        for(unsigned int i=0;i<n;++i){nvmlDevice_t d{};nvmlMemory_t m{};
            require(handle_(i,&d)==NVML_SUCCESS&&memory_(d,&m)==NVML_SUCCESS,"Cannot monitor GPU memory");used+=m.used;}
        return used;
    }
};
#endif
}
Json evaluate_local_product(const std::filesystem::path& manifest,const std::filesystem::path& resources,
    const std::filesystem::path& request,const std::filesystem::path& options,const std::filesystem::path& output) {
    const Json declaration=document(options);const auto scope=EvaluationScope::parse(declaration);
#ifdef _WIN32
    (void)manifest;(void)resources;(void)request;(void)output;
    throw Error("Bounded Product evaluation requires the Linux process supervisor");
#else
    DeviceMonitor monitor; // NVML is initialized lazily in the parent AFTER fork.
    const auto worker=[&] {
        const auto start=std::chrono::steady_clock::now();auto last=start;
        auto prepared=dry_run_local_product(manifest,resources,request);
        const auto selected=load_qualification_precision(prepared.owner->resources,scope.precision_artifact,"bf16");
        scope.admit(uint64_t(shape_numel(prepared.sampling.latent_shape)),uint64_t(shape_numel(prepared.expected_video_shape)),prepared.sampling.steps);
        if(scope.complete)run_media_detail::check_media_encoder(run_media_detail::product_encoder_path());
        Json::Object evidence{{"schema",Json(std::string("vrhino.product-evaluation-result.v1"))},
            {"status",Json(std::string("RUNNING_UNQUALIFIED"))},{"scope",declaration},
            {"request",prepared.evidence},{"admission",prepared.owner->evidence},
            {"precision_policy",selected.evidence},{"production_numerically_qualified",Json(false)},
            {"reference_numerically_qualified",Json(false)},{"generic_bf16_status",Json(std::string("HOLD"))},
            {"tensor_snapshots_enabled",Json(false)},{"solver_snapshots_enabled",Json(false)}};
        evidence["preparation_seconds"]=Json(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
        save(output/"worker.json",Json(evidence));
        TensorBundle tokens;for(const auto& c:prepared.conditioning){tokens.emplace(c.target+".input_ids",c.input_ids);tokens.emplace(c.target+".attention_mask",c.attention_mask);}
        write_bundle((output/"tokens.vrt").string(),tokens);
        const MemoryBudget budget{size_t(scope.device_bytes),0,size_t(scope.host_bytes),2ULL<<30,1ULL<<30,size_t(scope.weight_cache_bytes)};budget.validate();
        {CudaBackend init;init.synchronize();}
        size_t free=0,total=0;require(cudaMemGetInfo(&free,&total)==cudaSuccess&&free>=scope.device_bytes,"Insufficient device memory for evaluation budget");
        const auto factory=[&]() -> std::unique_ptr<Backend> {
            auto b=std::make_unique<CudaBackend>();b->set_execution_dtype(selected.policy.requested_dtype());
            b->configure_memory_runtime(budget,{true,false,false});b->retain_resource_owners({prepared.owner});
            b->set_vrm_mapped_bytes(prepared.owner->model->file_size()+prepared.owner->conditioning->mapped_bytes());return b;
        };
        PreparedExecutionControl control;control.tensor_trace=false;control.solver_trace=false;
        if(!scope.complete)control.qualification_prefix_steps=scope.max_steps;
        Json::Array steps;last=std::chrono::steady_clock::now();
        control.step_completed=[&](int n,int count,ComponentInstanceID instance,BindingID binding,const Json& memory) {
            steps.emplace_back(Json::Object{{"step",Json(int64_t(n-1))},{"total",Json(int64_t(count))},
                {"instance_id",Json(int64_t(instance.value))},{"binding_id",Json(int64_t(binding.value))},
                {"elapsed_sampling_seconds",Json(std::chrono::duration<double>(std::chrono::steady_clock::now()-last).count())},
                {"resources",memory}});save(output/"steps.json",Json(steps));
        };
        control.phase_completed=[&](ProductExecutionPhase phase,const PreparedProductExecution& r) {
            const auto now=std::chrono::steady_clock::now();const auto seconds=std::chrono::duration<double>(now-last).count();
            if(phase==ProductExecutionPhase::Conditioning) {
                evidence["conditioning_seconds"]=Json(seconds);write_bundle((output/"conditioning.vrt").string(),r.conditioning);
            }else if(phase==ProductExecutionPhase::Sampling) {
                require(r.sampling.trace.empty(),"Unexpected tensor snapshots in bounded evaluation");
                evidence["sampling_seconds"]=Json(seconds);evidence["completed_steps"]=Json(int64_t(r.sampling.completed_steps));
                evidence["program_complete"]=Json(r.sampling.program_complete);evidence["shared_graph_identity"]=Json(r.shared_graph_identity);
                write_bundle((output/"latent.vrt").string(),{{"initial",r.sampling.initial_noise},{"final",r.sampling.final_latent}});
                evidence["rng_seed"]=Json(int64_t(r.sampling.rng_after_initialization.seed));evidence["rng_offset"]=Json(int64_t(r.sampling.rng_after_initialization.offset));
            }else {
                evidence["decoder_seconds"]=Json(seconds);write_bundle((output/"decoded.vrt").string(),{{"video",r.video}});
            }
            evidence["resources"]=Json(r.resources);save(output/"worker.json",Json(evidence));last=std::chrono::steady_clock::now();
        };
        const auto result=execute_prepared_product(prepared,selected.policy,factory,control);
        if(scope.complete) {
            const auto encoder=run_media_detail::product_encoder_path();run_media_detail::check_media_encoder(encoder);
            const auto media=run_media_detail::encode_mp4(result.video,scope.fps,encoder,output/"evaluation.mp4",false,{},scope.video_min,scope.video_max);
            evidence["media_seconds"]=Json(media.seconds);evidence["media_sha256"]=Json(sha256_file(output/"evaluation.mp4"));
        }
        evidence["status"]=Json(std::string("EXECUTED_UNQUALIFIED"));evidence["finite_outputs"]=Json(true);
        save(output/"worker.json",Json(evidence));
    };
    auto result=supervise_evaluation(scope,output,worker,[&]{return monitor.used();}).object();
    result["schema"]=Json(std::string("vrhino.product-evaluation-supervision.v1"));
    result["device_accounting"]=Json(std::string("NVML sum of whole-device used bytes across physical GPUs; includes other processes"));
    result["scope"]=declaration;save(output/"supervision.json",Json(result));return Json(std::move(result));
#endif
}
} // namespace vrhino::product
