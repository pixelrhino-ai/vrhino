// Synthetic generic resource qualification. No checkpoint, model or phase semantics.
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/architecture_binding.h"
#include "vrhino/execution.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <iostream>
#include <set>
#include <utility>

using namespace vrhino;
namespace {
constexpr size_t MiB=1024*1024, weight_bytes=4*MiB;
constexpr int width=1024, slots=4;
enum class Fault { None, Allocation, Upload, Evaluation, Cancellation };
Fault armed=Fault::None;
bool injected=false;
int allocation_failures=0;
std::set<cudaEvent_t> events;
}
extern "C" cudaError_t __real_cudaMalloc(void**,size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** ptr,size_t bytes) {
    if(armed==Fault::Allocation && bytes==weight_bytes) { if(--allocation_failures==0) armed=Fault::None; injected=true; return cudaErrorMemoryAllocation; }
    return __real_cudaMalloc(ptr,bytes);
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream) {
    if(armed==Fault::Upload && bytes==weight_bytes && kind==cudaMemcpyHostToDevice) {
        armed=Fault::None; injected=true; return cudaErrorInvalidValue;
    }
    return __real_cudaMemcpyAsync(dst,src,bytes,kind,stream);
}
extern "C" cudaError_t __real_cudaEventCreate(cudaEvent_t*);
extern "C" cudaError_t __wrap_cudaEventCreate(cudaEvent_t* event) {
    const auto result=__real_cudaEventCreate(event); if(result==cudaSuccess) events.insert(*event); return result;
}
extern "C" cudaError_t __real_cudaEventDestroy(cudaEvent_t);
extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t event) {
    const auto result=__real_cudaEventDestroy(event); if(result==cudaSuccess) events.erase(event); return result;
}
namespace {
struct SyntheticBacking {
    std::vector<std::vector<float>> values;
    std::vector<Tensor> tensors;
    explicit SyntheticBacking(int variant) {
        for(int s=0;s<slots;++s) {
            values.emplace_back(width*width,0.0f);
            for(int i=0;i<width;++i) values.back()[i*width+i]=0.5f+variant*0.125f+s*0.03125f;
            tensors.push_back(Tensor::borrowed(values.back().data(),weight_bytes,{width,width},DType::F32));
        }
    }
    std::vector<uint64_t> hashes() const {
        std::vector<uint64_t> result; for(const auto& t:tensors) result.push_back(hash_host_tensor_content(t)); return result;
    }
};
struct Observation { const void* source; int64_t timestep; bool operator==(const Observation&) const=default; };
struct Selection { size_t step; uint32_t instance,binding; };
class Endpoint final : public Denoiser {
public:
    Endpoint(CudaBackend& b,std::vector<Tensor> p,std::vector<Observation>& seen,std::vector<Selection>& selected,Fault fault)
        : b(b),p(std::move(p)),seen(seen),selected(selected),fault(fault) {}
    std::vector<Tensor> evaluate_step(const Tensor& x,const StepExecutionContext& c) override {
        selected.push_back({c.step_index,c.selected_component.id().value,c.selected_component.binding_id().value});
        return evaluate(x,c.timestep);
    }
    std::vector<Tensor> evaluate(const Tensor& x,const Tensor& t) override {
        seen.push_back({p.at(0).data(),read_scalar_i64(t)});
        if(fault==Fault::Allocation || fault==Fault::Upload) armed=fault;
        if(fault==Fault::Allocation) allocation_failures=2;
        Tensor y=x;
        for(const auto& w:p) {
            y=b.linear(y,w,nullptr,DType::F32,DType::F32);
            // A second use must hit the correct cache entry, including after eviction.
            (void)b.copy_to_device(w,DType::F32);
        }
        if(fault==Fault::Evaluation) throw Error("injected evaluation failure after uploads");
        PreparedTensorKey key;
        key.operation="synthetic.scalar.v1"; key.input_identity=reinterpret_cast<uintptr_t>(p[0].data());
        key.input_content_hash=hash_host_tensor_content(p[0]); key.target_backend=b.name();
        auto handle=prepared_tensors().prepare(key,[&] {
            auto scalar=scalar_f32(p[0].data_as<float>()[0]*0.125f);
            return PreparedTensorMaterialization{{b.copy_to_device(scalar,DType::F32)},1,1,sizeof(float)};
        });
        trace={{"endpoint.timestep",t},{"endpoint.coefficient",scalar_f32(p[0].data_as<float>()[0])}};
        return {y,b.add(y,prepared_tensors().reuse(handle).at(0))};
    }
    std::map<std::string,Tensor> take_trace() override { return std::exchange(trace,{}); }
    CudaBackend& b; std::vector<Tensor> p; std::vector<Observation>& seen; std::vector<Selection>& selected;
    Fault fault; std::map<std::string,Tensor> trace;
};
SamplingProgram program() {
    SamplingProgram p; p.latent_shape={1,width}; p.seed=37; p.steps=9;
    p.guidance_mode=GuidanceMode::CFG; p.guidance_coefficients={-1.0f,2.0f};
    std::vector<FlowScheduleTransition> transitions;
    for(int i=0;i<p.steps;++i) {
        p.model_timesteps.push_back(scalar_i64(90-i*10));
        const float from=float(p.steps-i)/p.steps,to=float(p.steps-i-1)/p.steps;
        transitions.push_back({p.model_timesteps.back(),from,to});
    }
    p.contract.emplace(PredictionContract{PredictionSemantic::Flow},
        SolverContract{SolverSemantic::MultistepPredictorCorrector,2},ScheduleContract::flow_sigma(std::move(transitions)));
    return p;
}
struct MemoryObservation {
    const void* source; DType dtype; size_t bytes;
    bool operator==(const MemoryObservation&) const=default;
};
struct Snapshot {
    std::map<std::string,Tensor> trace;
    std::vector<Observation> seen;
    std::vector<MemoryObservation> accesses;
    MemoryRuntimeStats stats;
    PreparedTensorCacheStats prepared;
    RngState rng;
    size_t peak_allocation=0;
};
Snapshot run(const std::shared_ptr<SyntheticBacking>& a,const std::shared_ptr<SyntheticBacking>& b,
             bool limited,bool legacy=false,bool single=false,Fault fault=Fault::None,bool memory_enabled=true) {
    Snapshot out;
    CudaBackend backend; backend.set_execution_dtype(DType::F32);
    // Generic contexts promote admitted owners automatically. Legacy endpoints
    // register their opaque backing explicitly before any cache use.
    if(legacy) backend.retain_resource_owners({a,b});
    ResourceEstimate estimate;
    estimate.resident_cache=(limited?8:64)*MiB;
    estimate.activations=4*MiB; estimate.workspace=8*MiB;
    estimate.upload_temporary=4*MiB; estimate.sampling_state=MiB;
    estimate.prepared=MiB; estimate.source_backing=32*MiB; estimate.host_staging=8*MiB;
    backend.set_resource_admission({estimate,{estimate.device_peak(),estimate.host_peak()}});
    if(memory_enabled) {
        MemoryRuntimeOptions options; options.enabled=true; options.prefetch=false; options.host_staging=true;
        // Same device/workspace envelope; only the generic cache cap differs.
        backend.configure_memory_runtime({66*MiB,8*MiB,128*MiB,MiB,MiB,limited?8*MiB:0},options);
        backend.set_vrm_mapped_bytes(2*slots*weight_bytes);
    }
    backend.begin_memory_trace();
    auto p=program();
    std::vector<Selection> selected;
    std::vector<ArchitectureParameterSlot> slot_declarations;
    for(int s=0;s<slots;++s) slot_declarations.push_back({"parameter."+std::to_string(s),{width,width}});
    auto declaration=std::make_shared<const ArchitectureBindingDeclaration>(slot_declarations);
    auto admit=[&](const std::shared_ptr<SyntheticBacking>& source) {
        std::map<std::string,const Tensor*> refs;
        for(int s=0;s<slots;++s) refs.emplace(slot_declarations[s].role,&source->tensors[s]);
        return AdmittedArchitectureBinding::admit(declaration,refs,BorrowedBindingLifetime::ExplicitOwners,{source});
    };
    auto ba=admit(a),bb=admit(b);
    ComponentInterface interface{{p.latent_shape,DType::F32},{{},DType::I64},
        {{p.latent_shape,DType::F32},{p.latent_shape,DType::F32}},GuidanceMode::CFG,PredictionSemantic::Flow,0};
    auto graph=std::make_shared<const ComponentGraphDefinition>(interface,
        std::vector<ExecutionTensorContract>(slots,{{width,width},DType::F32}),
        [&](const std::vector<Tensor>& tensors) { return std::make_unique<Endpoint>(backend,tensors,out.seen,selected,fault); });
    std::vector<ComponentInstance> instances;
    instances.push_back(ComponentInstance::bind({0},{10},graph,ba->parameters_for(*declaration),{ba}));
    if(!single) instances.push_back(ComponentInstance::bind({1},{20},graph,bb->parameters_for(*declaration),{bb}));
    ExecutionContext context(std::move(instances));
    require(single || context.at({0}).graph()==context.at({1}).graph(),"Graph was not reused");
    std::vector<ComponentInstanceID> ids;
    for(int i=0;i<p.steps;++i) ids.push_back({static_cast<uint32_t>(single?0:i%2)});
    SamplingRuntime runtime(backend);
    if(fault==Fault::Cancellation) runtime.set_cancellation_requested([&]{ return out.seen.size()==1; });
    bool failed=false;
    try {
        Endpoint old(backend,ba->parameters_for(*declaration),out.seen,selected,Fault::None);
        auto result=legacy?runtime.run(old,p):runtime.run(context,ExecutionProgram::per_step(ids),p);
        out.trace=std::move(result.trace); out.prepared=result.prepared_tensors; out.rng=result.rng_after_initialization;
        if(!legacy) for(size_t i=0;i<selected.size();++i)
            require(selected[i].step==i && selected[i].instance==ids[i].value && selected[i].binding==(ids[i].value?20u:10u),"Wrong admitted selection");
        require(out.seen.size()==static_cast<size_t>(p.steps),"Step count changed");
        for(int i=0;i<p.steps;++i) require(out.seen[i].source==(ids[i].value?b:a)->tensors[0].data() &&
            out.seen[i].timestep==90-i*10,"Stale source/timestep");
    } catch(const Error& error) {
        failed=true;
        if(fault==Fault::None) throw;
        const std::string message=error.what();
        const char* expected=fault==Fault::Allocation?"CUDA allocation failed:":
            fault==Fault::Upload?"CUDA error:":fault==Fault::Cancellation?"sampling cancelled":
            "injected evaluation failure after uploads";
        require(message.starts_with(expected),"Unexpected failure masked injection: "+message);
        require(out.seen.size()==1,"Failure fell back to another binding");
        require((fault!=Fault::Allocation && fault!=Fault::Upload) || injected,"Fault injection missed upload/allocation");
        // SamplingRuntime must have drained ordinary submitted work before return.
        require(cudaStreamQuery(nullptr)==cudaSuccess,"Sampling failure left work outstanding");
        require(a.use_count()>1 && b.use_count()>1,"Source owners released during failure");
    }
    require(failed==(fault!=Fault::None),"Failure injection did not stop execution");
    backend.synchronize();
    for (const auto& access:backend.end_memory_trace())
        out.accesses.push_back({access.tensor.data(),access.target_dtype,access.backend_bytes});
    out.stats=backend.memory_runtime_stats();
    out.peak_allocation=backend.peak_device_bytes();
    return out;
}
void exact(const Snapshot& a,const Snapshot& b,bool memory=false) {
    require(a.trace.size()==b.trace.size() && a.seen==b.seen,"Execution trace mismatch");
    for(const auto& [key,t]:a.trace) {
        const auto& u=b.trace.at(key);
        require(t.device().is_host() && u.device().is_host() && t.shape()==u.shape() && t.dtype()==u.dtype() &&
            t.bytes()==u.bytes() && std::memcmp(t.data(),u.data(),t.bytes())==0,"Output/prediction/guidance/solver drift: "+key);
    }
    require(a.rng.seed==b.rng.seed && a.rng.offset==b.rng.offset && a.rng.algorithm_id==b.rng.algorithm_id,"RNG drift");
    require(a.accesses.size()==b.accesses.size(),"Memory trace length drift");
    for(size_t i=0;i<a.accesses.size();++i) {
        const auto& x=a.accesses[i]; const auto& y=b.accesses[i];
        require(x==y,"Memory identity/order drift");
    }
    if(memory) {
        require(a.stats.cache_hits==b.stats.cache_hits && a.stats.cache_misses==b.stats.cache_misses &&
            a.stats.evictions==b.stats.evictions && a.stats.upload_bytes==b.stats.upload_bytes &&
            a.stats.accounting.peak_device_resident_weight_bytes==b.stats.accounting.peak_device_resident_weight_bytes &&
            a.peak_allocation==b.peak_allocation,"Legacy resource policy drift");
        require(a.prepared.prepare_hits==b.prepared.prepare_hits && a.prepared.prepare_misses==b.prepared.prepare_misses &&
            a.prepared.reuses==b.prepared.reuses && a.prepared.resident_bytes==b.prepared.resident_bytes,"Prepared cache drift");
    }
}
}
int main() {
    try {
        auto a=std::make_shared<SyntheticBacking>(0),b=std::make_shared<SyntheticBacking>(1);
        const auto ah=a->hashes(),bh=b->hashes();
        std::set<const void*> pointers;
        for(const auto* s:{a.get(),b.get()}) for(const auto& t:s->tensors) require(pointers.insert(t.data()).second,"Aliased source storage");
        const auto high=run(a,b,false),limited=run(a,b,true);
        exact(high,limited);
        require(limited.stats.evictions>0 && limited.stats.upload_bytes>2*slots*weight_bytes,"Eviction/re-upload did not occur");
        require(limited.stats.accounting.peak_device_resident_weight_bytes<=8*MiB,"Resident weight budget exceeded");
        require(limited.prepared.prepare_misses==2 && limited.prepared.prepare_hits==7,"Prepared identity crossed instances");
        require(ah==a->hashes() && bh==b->hashes(),"Source backing overwritten");
        std::cout<<"multi_binding_device=PASS steps=9 source_bytes="<<2*slots*weight_bytes
            <<" weight_capacity="<<8*MiB<<" evictions="<<limited.stats.evictions<<" upload_bytes="<<limited.stats.upload_bytes
            <<" peak_cached_weights="<<limited.stats.accounting.peak_device_resident_weight_bytes
            <<" backend_peak_device_bytes="<<limited.peak_allocation<<" exact_trace_entries="<<limited.trace.size()<<'\n';
        for(bool enabled:{false,true}) for(bool cap:{false,true}) {
            auto old=run(a,b,cap,true,true,Fault::None,enabled),now=run(a,b,cap,false,true,Fault::None,enabled);
            exact(old,now,true);
        }
        std::cout<<"legacy_resource_compatibility=PASS cases=4\n";
        bool gap=false;
        for(auto fault:{Fault::Cancellation,Fault::Evaluation,Fault::Allocation,Fault::Upload}) {
            require(events.empty(),"Unexpected baseline event leak"); injected=false;
            (void)run(a,b,true,false,false,fault);
            const auto leaked=events.size(); gap|=leaked!=0;
            std::cout<<"failure_case="<<static_cast<int>(fault)<<" no_fallback=PASS outstanding_events_after_backend_destruction="<<leaked<<'\n';
            // Fresh-session correctness is checked BEFORE harness cleanup; a
            // leak must not be hidden by test recovery before this assertion.
            const auto orphaned=events;
            exact(high,run(a,b,true));
            require(events==orphaned,"Fresh session leaked additional events");
            // Clean only after production destruction and fresh-session probe.
            // This is not a runtime recovery strategy.
            for(auto event:orphaned) require(__wrap_cudaEventDestroy(event)==cudaSuccess,"Test event cleanup failed");
        }
        require(events.empty(),"Events remain after fresh execution session");
        std::weak_ptr<SyntheticBacking> wa=a,wb=b; a.reset(); b.reset();
        require(wa.expired() && wb.expired(),"Source owners escaped session");
        std::cout<<"session_owner_lifetime=PASS new_session_after_each_failure=PASS\n";
        std::cout<<"resource_cleanup_qualification="<<(gap?"GAP":"PASS")<<'\n';
        return gap?2:0;
    } catch(const std::exception& e) { std::cerr<<"resource qualification: "<<e.what()<<'\n'; return 1; }
}
