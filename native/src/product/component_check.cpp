#include "vrhino/product/component_preparation.h"
#include "vrhino/product/prepared_execution.h"
#include "vrhino/product/qualification_precision.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/architecture_binding.h"
#include "vrhino/tensor_util.h"
#include "run_media.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <set>
#include <chrono>
#include <cmath>

namespace vrhino::product {
namespace {
Json read_document(const std::filesystem::path& p) {
    require(std::filesystem::file_size(p)<=65536,"Oversized component-check options");
    std::ifstream f(p); require(f.good(),"Cannot read component-check options");
    return Json::parse(std::string(std::istreambuf_iterator<char>(f),{}),JsonParseLimits{});
}
void save(const std::filesystem::path& p,const Json& j) {
    std::ofstream f(p);require(f.good(),"Cannot write component evidence");f<<j.serialize()<<'\n';
    f.close();require(f.good(),"Component evidence write failed");
}
Json stats(CudaBackend& backend) {
    const auto m=backend.memory_runtime_stats();
    return Json(Json::Object{{"peak_device_bytes",Json(int64_t(backend.peak_device_bytes()))},
        {"weight_upload_bytes",Json(int64_t(backend.weight_upload_bytes()))},
        {"cache_resident_bytes",Json(int64_t(backend.weight_cache_resident_bytes()))},
        {"cache_hits",Json(int64_t(m.cache_hits))},{"cache_misses",Json(int64_t(m.cache_misses))},
        {"source_owner_count",Json(int64_t(backend.resource_owner_count()))},
        {"total_device_memory_bound_guaranteed",Json(false)}});
}
Json device_memory() {
    size_t free=0,total=0;require(cudaMemGetInfo(&free,&total)==cudaSuccess,"Cannot read device memory");
    return Json(Json::Object{{"free_bytes",Json(int64_t(free))},{"total_bytes",Json(int64_t(total))}});
}

}
Json check_local_components(const std::filesystem::path& manifest,
    const std::filesystem::path& resources,const std::filesystem::path& request_file,
    const std::filesystem::path& options_file,const std::filesystem::path& output) {
    const auto options=read_document(options_file);
    require(options.object().size()==8 && options.at("schema").string()=="vrhino.component-check.v1",
            "Invalid component-check declaration");
    // Qualification execution mode, not a model-selected precision override.
    require(options.at("precision").string()=="fp32","Component check v1 requires explicit FP32 qualification mode");
    std::set<std::string> actions;
    for(const auto& a:options.at("actions").array()) {
        const auto action=a.string();require(action=="conditioning" || action=="decoder","Unknown component check");
        require(actions.insert(action).second,"Duplicate component check");
    }
    require(!actions.empty(),"Empty component check");
    const auto device_budget=options.at("device_budget_bytes").integer();
    const auto host_budget=options.at("host_budget_bytes").integer();
    require(device_budget>3LL*1024*1024*1024 && host_budget>0,"Invalid component resource budget");
    const auto fps=options.at("fps").integer();require(fps>0 && fps<=120,"Invalid media fps");
    const auto& range=options.at("media_range").array();require(range.size()==2,"Expected declared media range");
    const float video_min=range[0].number(),video_max=range[1].number();
    require(std::isfinite(video_min) && std::isfinite(video_max) && video_min<video_max,"Invalid media range");
    const auto latent_path=options.at("latent_bundle").string();
    require(!actions.contains("decoder") || !latent_path.empty(),"Decoder requires explicit synthetic latent fixture");
    require(!std::filesystem::exists(output),"Component evidence directory already exists");
    auto prepared=dry_run_local_product(manifest,resources,request_file);
    require(shape_numel(prepared.expected_video_shape)<=1048576,"Decoder qualification output exceeds bounded scope");
    const auto policy=PrecisionPolicy::fp32();
    const MemoryBudget budget{size_t(device_budget),0,size_t(host_budget),2ULL<<30,1ULL<<30};budget.validate();
    // Initialize the device context before lifecycle observations.
    {CudaBackend initialize;initialize.synchronize();}
    require(device_memory().at("free_bytes").integer()>=device_budget,"Insufficient free device memory for declared budget");
    const auto configure=[&](CudaBackend& b) {
        b.set_execution_dtype(policy.requested_dtype());
        b.configure_memory_runtime(budget,MemoryRuntimeOptions{true,false,false});
        b.retain_resource_owners({prepared.owner});
        b.set_vrm_mapped_bytes(prepared.owner->model->file_size()+prepared.owner->conditioning->mapped_bytes());
    };
    require(std::filesystem::create_directories(output),"Cannot create component evidence directory");
    Json::Object evidence{{"schema",Json(std::string("vrhino.component-check-result.v1"))},
        {"admission",prepared.owner->evidence},{"request",prepared.evidence},
        {"production_numerically_qualified",Json(false)},{"generic_bf16_status",Json(std::string("HOLD"))},
        {"denoiser_evaluations",Json(int64_t(0))},{"sampling_steps_executed",Json(int64_t(0))},
        {"precision",Json(std::string("fp32"))},{"device_before",device_memory()}};
    bool passed=true;
    const auto phase=[&](const std::string& name,auto&& operation) {
        const auto start=std::chrono::steady_clock::now();Json::Object result;
        try {result=operation().object();result.emplace("status",Json(std::string("PASS")));}
        catch(const std::exception& error){passed=false;result={{"status",Json(std::string("HOLD"))},{"error",Json(std::string(error.what()))}};}
        result.emplace("seconds",Json(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()));
        result.emplace("device_after_teardown",device_memory());evidence.emplace(name,Json(std::move(result)));
        save(output/"result.json",Json(evidence));
    };
    if(actions.contains("conditioning")) phase("conditioning",[&] {
        TensorBundle inputs;Json encoded_stats;
        {
            CudaBackend encoder;configure(encoder);
            inputs=execute_text_conditioning(encoder,prepared);
            encoded_stats=stats(encoder);
        }
        // Original input IDs/masks are retained as reference-only replay evidence.
        TensorBundle tokens;
        for(const auto& c:prepared.conditioning) {
            tokens.emplace(c.target+".input_ids",c.input_ids);
            tokens.emplace(c.target+".attention_mask",c.attention_mask);
        }
        write_bundle((output/"tokens.vrt").string(),tokens);
        write_bundle((output/"conditioning.vrt").string(),inputs);
        CudaBackend admission;configure(admission);
        auto admitted=admit_conditioned_sampling(admission,policy,prepared,inputs);
        return Json(Json::Object{{"encoder",encoded_stats},{"sampling_admission",admitted},
            {"admission_resources",stats(admission)},{"finite",Json(true)},
            {"reference_numerical_qualified",Json(false)}});
    });
    if(actions.contains("decoder")) phase("decoder",[&] {
        const auto latent_bundle=read_bundle(latent_path);
        require(latent_bundle.size()==1 && latent_bundle.contains("latent"),"Expected isolated latent fixture");
        const auto& latent=latent_bundle.at("latent");
        const auto& declared=prepared.runtime_inputs.at("latent_shape");
        std::vector<int64_t> shape(declared.data_as<int64_t>(),declared.data_as<int64_t>()+declared.numel());
        validate_architecture_tensor(latent,shape,DType::F32);require_finite_component_output(latent);
        Tensor video;Json decoded_stats;
        {
            CudaBackend decoder;configure(decoder);
            // Existing architecture/component topology adapter; no product model dispatch.
            auto device=prepared.owner->architecture->decode(decoder,policy,latent,prepared.runtime_inputs);
            video=device.device().is_host()?device:decoder.copy_to_host(device);decoder.synchronize();
            require_finite_component_output(video);
            require(video.shape()==prepared.expected_video_shape,"Decoder output shape mismatch");
            decoded_stats=stats(decoder);
        }
        write_bundle((output/"decoded.vrt").string(),{{"video",video}});
        return Json(Json::Object{{"resources",decoded_stats},{"finite",Json(true)},
            {"synthetic_latent",Json(true)},{"latent_sha256",Json(sha256_file(latent_path))},
            {"reference_numerical_qualified",Json(false)}});
    });
    if(actions.contains("decoder") && std::filesystem::exists(output/"decoded.vrt")) phase("media",[&] {
        auto video=read_bundle((output/"decoded.vrt").string()).at("video");
        const auto encoder=run_media_detail::product_encoder_path();
        run_media_detail::check_media_encoder(encoder);
        const auto result=run_media_detail::encode_mp4(video,fps,encoder,output/"synthetic-decoder-check.mp4",false,{},video_min,video_max);
        return Json(Json::Object{{"bytes",Json(int64_t(result.bytes))},
            {"synthetic_component_test_only",Json(true)}});
    });
    evidence.emplace("status",Json(std::string(passed?"PASS":"HOLD")));
    evidence.emplace("device_after",device_memory());
    save(output/"result.json",Json(evidence));return Json(std::move(evidence));
}
static Json check_sampling(const std::filesystem::path& manifest,
    const std::filesystem::path& resources,const std::filesystem::path& request_file,
    const std::filesystem::path& options_file,const std::filesystem::path& output,bool complete) {
    const auto options=read_document(options_file);
    const auto schema=options.at("schema").string();
    const bool declared_precision=complete ? schema=="vrhino.sampling-video-check.v2" :
        schema=="vrhino.sampling-step-check.v4";
    require(declared_precision ? options.object().size()==(complete ? 8U : 6U) :
        complete ? (options.object().size()==7 && schema=="vrhino.sampling-video-check.v1") :
            (options.object().size()==5 && (schema=="vrhino.sampling-step-check.v1" ||
            schema=="vrhino.sampling-step-check.v2" || schema=="vrhino.sampling-step-check.v3")),
            "Invalid sampling-step-check declaration");
    const auto precision=options.at("precision").string();
    if(declared_precision) {
        require(precision=="bf16","Declared qualification policy requires BF16 mode");
        require(!options.at("precision_policy_artifact").string().empty(),"Missing qualification precision artifact ID");
    } else require(precision=="fp32","Sampling step check requires explicit FP32 qualification mode");
    auto step_limit=options.at(complete ? "max_steps" : "completed_step_limit").integer();
    if(complete) require(step_limit>=2 && step_limit<=64,"Complete sampling check permits a bound of two to 64 steps");
    else if(schema=="vrhino.sampling-step-check.v1")
        require(step_limit==1,"Sampling step check permits exactly one transition");
    else if(schema=="vrhino.sampling-step-check.v2")
        require(step_limit>=1 && step_limit<=3,"Sampling prefix check permits one to three transitions");
    else require(step_limit>=1 && step_limit<=32,"Sampling transition check permits one to 32 transitions");
    const auto device_budget=options.at("device_budget_bytes").integer();
    const auto host_budget=options.at("host_budget_bytes").integer();
    require(device_budget>3LL*1024*1024*1024 && host_budget>0,"Invalid sampling resource budget");
    int64_t fps=0;float video_min=0,video_max=0;std::filesystem::path media_encoder;
    if(complete) {
        fps=options.at("fps").integer();require(fps>0 && fps<=120,"Invalid media fps");
        const auto& range=options.at("media_range").array();require(range.size()==2,"Expected declared media range");
        video_min=range[0].number();video_max=range[1].number();
        require(std::isfinite(video_min) && std::isfinite(video_max) && video_min<video_max,"Invalid media range");
    }
    require(!std::filesystem::exists(output),"Sampling evidence directory already exists");
    auto prepared=dry_run_local_product(manifest,resources,request_file);
    const auto selected_precision=declared_precision ? load_qualification_precision(
        prepared.owner->resources,options.at("precision_policy_artifact").string(),precision) :
        QualificationPrecision{PrecisionPolicy::fp32(),Json(Json::Object{})};
    if(complete) {
        require(prepared.sampling.steps<=step_limit,"Declared schedule exceeds complete sampling bound");
        step_limit=prepared.sampling.steps; // Bound never rewrites the admitted schedule.
        require(shape_numel(prepared.expected_video_shape)<=1048576,"Decoder qualification output exceeds bounded scope");
        media_encoder=run_media_detail::product_encoder_path();
        run_media_detail::check_media_encoder(media_encoder);
    } else require(step_limit<prepared.sampling.steps,"Sampling qualification must remain a strict program prefix");
    require(shape_numel(prepared.sampling.latent_shape)<=1024,"Sampling qualification input exceeds bounded scope");
    const auto& policy=selected_precision.policy;
    const MemoryBudget budget{size_t(device_budget),0,size_t(host_budget),2ULL<<30,1ULL<<30};budget.validate();
    {CudaBackend initialize;initialize.synchronize();}
    require(device_memory().at("free_bytes").integer()>=device_budget,"Insufficient free device memory for declared budget");
    const auto configure=[&](CudaBackend& b) {
        b.set_execution_dtype(policy.requested_dtype());
        b.configure_memory_runtime(budget,MemoryRuntimeOptions{true,false,false});
        b.retain_resource_owners({prepared.owner});
        b.set_vrm_mapped_bytes(prepared.owner->model->file_size()+prepared.owner->conditioning->mapped_bytes());
    };
    require(std::filesystem::create_directories(output),"Cannot create sampling evidence directory");
    Json::Object evidence{{"schema",Json(std::string(declared_precision ?
            (complete ? "vrhino.sampling-video-check-result.v2" : "vrhino.sampling-step-check-result.v4") :
            complete ? "vrhino.sampling-video-check-result.v1" : schema=="vrhino.sampling-step-check.v1" ?
            "vrhino.sampling-step-check-result.v1" : schema=="vrhino.sampling-step-check.v2" ?
            "vrhino.sampling-step-check-result.v2" : "vrhino.sampling-step-check-result.v3"))},
        {"admission",prepared.owner->evidence},{"request",prepared.evidence},
        {"production_numerically_qualified",Json(false)},{"generic_bf16_status",Json(std::string("HOLD"))},
        {"reference_numerical_qualified",Json(false)},{"precision",Json(precision)},
        {"original_declared_steps",Json(int64_t(prepared.sampling.steps))},
        {"program_complete",Json(false)},{"sampling_steps_executed",Json(int64_t(0))},
        {"decode_executed",Json(false)},{"device_before",device_memory()}};
    if(declared_precision) evidence.emplace("precision_policy",selected_precision.evidence);
    const auto start=std::chrono::steady_clock::now();
    try {
        TensorBundle tokens;
        for(const auto& c:prepared.conditioning) {
            tokens.emplace(c.target+".input_ids",c.input_ids);
            tokens.emplace(c.target+".attention_mask",c.attention_mask);
        }
        write_bundle((output/"tokens.vrt").string(),tokens);
        const bool observe_residency=complete || declared_precision || schema=="vrhino.sampling-step-check.v3";
        Json::Array step_resources;
        PreparedExecutionControl control;
        if(!complete) control.qualification_prefix_steps=static_cast<int>(step_limit);
        control.solver_trace=true;
        control.step_completed=[&](int completed,int,ComponentInstanceID instance,BindingID binding,const Json& resources) {
            evidence["sampling_steps_executed"]=Json(int64_t(completed));
            if(observe_residency) {
                step_resources.emplace_back(Json::Object{{"step",Json(int64_t(completed-1))},
                    {"instance_id",Json(int64_t(instance.value))},{"binding_id",Json(int64_t(binding.value))},
                    {"resources",resources},{"device",device_memory()}});
                save(output/"step-resources.json",Json(step_resources));
            }
        };
        control.phase_completed=[&](ProductExecutionPhase phase,const PreparedProductExecution& result) {
            if(phase==ProductExecutionPhase::Conditioning) {
                write_bundle((output/"conditioning.vrt").string(),result.conditioning);
                evidence.emplace("conditioning_resources",result.resources.at("conditioning"));
                evidence.emplace("device_after_conditioning_teardown",device_memory());
            } else if(phase==ProductExecutionPhase::Sampling) {
                evidence.emplace("sampling_admission",result.sampling_admission);
                if(observe_residency) evidence.emplace("shared_graph_identity",Json(result.shared_graph_identity));
                write_bundle((output/"sampling.vrt").string(),result.sampling.trace);
                if(complete) {
                    write_bundle((output/"latent.vrt").string(),{{"latent",result.sampling.final_latent}});
                    evidence["program_complete"]=Json(true);
                    evidence.emplace("device_after_sampling_teardown",device_memory());
                }
                evidence.emplace("sampling_resources",result.resources.at("sampling"));
                Json::Object calls;
                for(const auto& [name,count]:result.primitive_calls) calls.emplace(name,Json(int64_t(count)));
                evidence.emplace("primitive_calls",Json(std::move(calls)));
                evidence.emplace("rng_seed",Json(int64_t(result.sampling.rng_after_initialization.seed)));
                evidence.emplace("rng_offset",Json(int64_t(result.sampling.rng_after_initialization.offset)));
                evidence.emplace("sampling_seconds",Json(result.sampling.elapsed_seconds));
                evidence.emplace("finite",Json(true));
            } else {
                evidence.emplace("decoder_resources",result.resources.at("decoder"));
                evidence["decode_executed"]=Json(true);
                evidence.emplace("device_after_decoder_teardown",device_memory());
                write_bundle((output/"decoded.vrt").string(),{{"video",result.video}});
            }
            save(output/"result.json",Json(evidence));
        };
        const auto result=execute_prepared_product(prepared,policy,[&]() -> std::unique_ptr<Backend> {
            auto backend=std::make_unique<CudaBackend>();configure(*backend);return backend;
        },control);
        if(complete) {
            const auto media=run_media_detail::encode_mp4(result.video,fps,media_encoder,output/"qualification.mp4",false,{},video_min,video_max);
            evidence.emplace("media",Json(Json::Object{{"bytes",Json(int64_t(media.bytes))},
                {"fps",Json(fps)},{"sha256",Json(sha256_file(output/"qualification.mp4"))},
                {"qualification_only",Json(true)}}));
        }
        evidence.emplace("status",Json(std::string("PASS")));
    } catch(const std::exception& error) {
        evidence.emplace("status",Json(std::string("HOLD")));
        evidence.emplace("error",Json(std::string(error.what())));
    }
    evidence.emplace("seconds",Json(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()));
    evidence.emplace("device_after_teardown",device_memory());
    save(output/"result.json",Json(evidence));return Json(std::move(evidence));
}
Json check_local_sampling_step(const std::filesystem::path& manifest,
    const std::filesystem::path& resources,const std::filesystem::path& request,
    const std::filesystem::path& options,const std::filesystem::path& output) {
    return check_sampling(manifest,resources,request,options,output,false);
}
Json check_local_sampling_video(const std::filesystem::path& manifest,
    const std::filesystem::path& resources,const std::filesystem::path& request,
    const std::filesystem::path& options,const std::filesystem::path& output) {
    return check_sampling(manifest,resources,request,options,output,true);
}
} // namespace vrhino::product
