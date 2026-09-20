// Research harness compiles the actual production private block implementation.
// No test implementation of attention, normalization, modulation or FFN.
#define make_wan_architecture make_wan_numerical_probe_architecture
#define bound_denoiser bound_numerical_probe_denoiser
#define realize realize_numerical_probe_graph
#define ExecutionDefinition NumericalProbeExecutionDefinition
#include "../src/architectures/wan.cpp"
#undef ExecutionDefinition
#undef realize
#undef bound_denoiser
#undef make_wan_architecture
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/package_declaration.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include "vrhino/product/program_declaration.h"
using namespace vrhino;
int main(int argc,char** argv) {try {
    std::string policy_path;
    if(argc>=3 && std::string(argv[argc-2])=="--policy") {
        policy_path=argv[argc-1];argc-=2;
    }
    require(argc==5 || argc==6 || (argc==7 && std::string(argv[5])=="chain"),"usage: wan-numerical-probe MODEL.vrm BINDING INPUT.bundle OUTPUT.bundle [MODE [CONTRACT.json]]");
    if(argc==6 && std::string(argv[5])=="attention") {
        require(policy_path.empty(),"Attention replay mode has a fixed F32 contract");
        auto input=read_bundle(argv[3]);CudaBackend device;device.set_execution_dtype(DType::F32);Backend& backend=device;
        TensorBundle output;
        for(const auto& prefix:{std::string("reference"),std::string("native")}) {
            auto q=backend.copy_to_device(input.at(prefix+".q"),DType::F32);
            auto k=backend.copy_to_device(input.at(prefix+".k"),DType::F32);
            auto v=backend.copy_to_device(input.at(prefix+".v"),DType::F32);
            auto y=backend.attention(q,k,v,nullptr,false,1.0f/std::sqrt(float(q.dim(-1))));
            backend.synchronize();output[prefix+".attention"]=backend.copy_to_host(y);
        }
        backend.synchronize();write_bundle(argv[4],output);return 0;
    }
    const auto model=std::make_shared<VrmModel>(argv[1],false);
    auto package=PackageDeclaration::parse(model->graph());
    package.validate_tensor_references(*model);
    const auto& binding=package.bindings().at(static_cast<uint32_t>(std::stoul(argv[2])));
    auto definition=wan_family::lower(CanonicalArchitectureDeclaration::parse(package.graphs().at(binding.graph)));
    const auto& c=definition->config;
    std::map<std::string,const Tensor*> slots;
    for(const auto& [slot,name]:binding.parameters)slots.emplace(slot,&model->tensor(name));
    WeightMap weights(slots);
    auto policy=PrecisionPolicy::fp32();
    if(!policy_path.empty()) {
        std::ifstream file(policy_path);require(bool(file),"Missing precision contract");
        const auto document=Json::parse(std::string(std::istreambuf_iterator<char>(file),{}));
        policy=PrecisionPolicy::from_json(document);
        for(const auto& name:document.at("research_switches").at("blocked_environment").array())
            require(std::getenv(name.string().c_str())==nullptr,"Blocked precision research environment: "+name.string());
    }
    CudaBackend device; device.set_execution_dtype(policy.requested_dtype()); Backend& backend=device;
    backend.retain_resource_owners({model});
    auto cpu=read_bundle(argv[3]);TensorBundle input,trace;
    for(const auto& [name,t]:cpu) {
        input[name]=backend.copy_to_device(t,t.dtype());
        trace["input."+name]=input.at(name);
    }
    // Research-only controlled replay: hold each operator input fixed to
    // separate local arithmetic error from error propagated through a block.
    if(argc==6 && std::string(argv[5])=="ffn-local") {
        TensorBundle output;
        for(const auto& prefix:{std::string("reference"),std::string("native")}) {
            const auto name=prefix+".";
            output[name+"projection_in"]=backend.linear(input.at(name+"input"),
                weights.at("blocks.0.ffn.0.weight"),&weights.at("blocks.0.ffn.0.bias"));
            output[name+"activation"]=backend.activation(input.at(name+"preactivation"),Activation::GeluTanh);
            output[name+"projection_out"]=backend.linear(input.at(name+"activated"),
                weights.at("blocks.0.ffn.2.weight"),&weights.at("blocks.0.ffn.2.bias"));
        }
        backend.synchronize();TensorBundle host;
        for(const auto& [name,t]:output)host[name]=backend.copy_to_host(t);
        backend.synchronize();write_bundle(argv[4],host);return 0;
    }
    if(argc==6 && std::string(argv[5])=="boundaries") {
        // Test-only preparation/head replay through existing production ops.
        // Inputs are captured independently; anchors must match the full run.
        TensorBundle observed;
        auto time=backend.sinusoidal_embedding(backend.cast(input.at("timestep"),DType::F32),c.frequency,true,0.0,true);
        observed["preparation.sinusoidal"]=time;
        time=operation_linear(backend,policy,PrecisionOperation::Modulation,PrecisionSemantic::TemporaryCompute,time,weights.at("time_embedding.0.weight"),&weights.at("time_embedding.0.bias"));
        observed["preparation.time_linear0"]=time;
        time=backend.activation(time,Activation::Silu);observed["preparation.time_activation"]=time;
        time=operation_linear(backend,policy,PrecisionOperation::Modulation,PrecisionSemantic::TemporaryCompute,time,weights.at("time_embedding.2.weight"),&weights.at("time_embedding.2.bias"));
        observed["preparation.time"]=time;
        auto mod=operation_linear(backend,policy,PrecisionOperation::Modulation,PrecisionSemantic::TemporaryCompute,backend.activation(time,Activation::Silu),weights.at("time_projection.1.weight"),&weights.at("time_projection.1.bias"));
        observed["preparation.modulation"]=backend.reshape(mod,{1,6,c.dim});
        for(int branch=0;branch<2;++branch) {
            auto raw=branch?input.at("raw_context"):backend.mul(input.at("raw_context"),scalar_f32(0));
            auto context=backend.pad(raw,{0,0,0,c.text_length-raw.dim(1)},0.0f);
            const auto prefix="branch."+std::to_string(branch)+".context.";
            context=backend.linear(context,weights.at("text_embedding.0.weight"),&weights.at("text_embedding.0.bias"));observed[prefix+"linear0"]=context;
            context=backend.activation(context,Activation::GeluTanh);observed[prefix+"activation"]=context;
            observed[prefix+"output"]=backend.linear(context,weights.at("text_embedding.2.weight"),&weights.at("text_embedding.2.bias"));
        }
        const auto dtype=policy.operation_contract(PrecisionOperation::Modulation,PrecisionSemantic::TemporaryCompute).temporary_dtype;
        for(const auto& prefix:{std::string("native.0"),std::string("native.1"),std::string("reference.0"),std::string("reference.1")}) {
            auto hidden=input.at(prefix+".hidden");
            auto external=backend.reshape(backend.cast(input.at(prefix+".time"),dtype),{1,1,c.dim});
            auto parts=backend.split(backend.add(backend.cast(weights.at("head.modulation"),dtype),external),{1,1},1);
            auto norm=backend.layer_norm(hidden,nullptr,nullptr,c.epsilon);observed[prefix+".norm"]=norm;
            auto m=modulate(backend,policy,norm,parts[0],parts[1]);observed[prefix+".modulated"]=m;
            observed[prefix+".output"]=producer_linear(backend,policy,PrecisionSemantic::DenoiserOutput,m,weights.at("head.head.weight"),&weights.at("head.head.bias"));
        }
        backend.synchronize();TensorBundle host;
        for(const auto& [name,t]:observed)host[name]=backend.copy_to_host(t);
        write_bundle(argv[4],host);return 0;
    }
    if(argc>=6) {
        const auto state_dtype=effective_sampling_state_dtype(policy);
        input["latent"]=backend.cast(input.at("latent"),state_dtype);
        const bool sequence=std::string(argv[5])=="sequence";
        const bool sampling=std::string(argv[5])=="sample";
        const bool chain=std::string(argv[5])=="chain";
        const auto step=(sequence || sampling || chain)?size_t(0):static_cast<size_t>(std::stoul(argv[5]));
        auto positive=cpu.at("raw_context"), negative=Tensor::host(positive.shape(),DType::F32);
        std::memset(negative.data(),0,negative.bytes());
        auto lower=[&](const Json& config){auto d=wan_family::lower(CanonicalArchitectureDeclaration::parse(config));
            auto e=wan_family::realize_numerical_probe_graph(d,backend,policy,cpu.at("latent").shape(),positive,negative,true,-1,-1);
            return LoweredPackageGraph{d->parameters,e.graph};};
        auto admitted=package.admit(model,lower); auto context=admitted.create_context();
        auto programs=product::admit_program_declaration(model->metadata().at("programs"),package);
        programs.sampling.latent_shape=cpu.at("latent").shape();
        auto plan=programs.execution.admit(context,programs.sampling,state_dtype);
        if(chain) {
            std::ifstream file(argv[6]); require(bool(file),"Missing short-chain contract");
            const auto spec=Json::parse(std::string(std::istreambuf_iterator<char>(file),{}));
            auto program=programs.sampling;
            program.steps=spec.at("steps").integer(); program.seed=spec.at("seed").integer();
            require(program.steps>=3 && program.steps<=8,"Short-chain length required");
            std::vector<FlowScheduleTransition> transitions;
            std::vector<GuidanceParameters> guidance;
            std::vector<ComponentInstanceID> ids;
            for(const auto& row:spec.at("transitions").array()) {
                transitions.push_back({scalar_i64(row.at("model_timestep").integer()),
                    float(row.at("sigma").number()),float(row.at("next_sigma").number())});
                guidance.push_back(GuidanceParameters::cfg(row.at("guidance_scale").number()));
            }
            for(const auto& id:spec.at("selection").array())ids.push_back({uint32_t(id.integer())});
            program.contract.emplace(PredictionContract{PredictionSemantic::Flow},
                SolverContract{SolverSemantic::MultistepPredictorCorrector,2},ScheduleContract::flow_sigma(std::move(transitions)));
            program.guidance_schedule=GuidanceSchedule::per_step(std::move(guidance));
            const auto execution=ExecutionProgram::per_step(std::move(ids));
            const auto selected=execution.admit(context,program,state_dtype);
            for(const auto* instance:selected)require(instance->graph()==selected[0]->graph(),"Graph must be shared");
            SamplingRuntime runtime(backend,policy); runtime.set_solver_trace_enabled(true);
            int observed=0;
            runtime.set_step_observer([&](int complete,int total) {
                require(complete==++observed && total==program.steps,"Non-continuous step observer");
                std::cout<<"Completed step="<<complete<<" instance="<<selected[complete-1]->id().value<<std::endl;
                if(!policy_path.empty())std::cout<<"Resource step="<<complete
                    <<" resident_bytes="<<backend.weight_cache_resident_bytes()
                    <<" cache_capacity_bytes="<<backend.weight_cache_capacity_bytes()
                    <<" weight_upload_bytes="<<backend.weight_upload_bytes()
                    <<" cache_hits="<<backend.weight_cache_hits()<<" cache_misses="<<backend.weight_cache_misses()
                    <<" backend_peak_bytes="<<backend.peak_device_bytes()<<std::endl;
            });
            auto result=runtime.run_with_initial_state(context,execution,program,
                policy_path.empty()?cpu.at("latent"):input.at("latent"));
            require(observed==program.steps,"Incomplete chain");
            auto host=std::move(result.trace);
            for(const auto& [name,count]:runtime.primitives().calls())host["calls."+name]=scalar_i64(count);
            host["calls.rng_normal"]=scalar_i64(runtime.primitives().calls().count("rng_normal")?runtime.primitives().calls().at("rng_normal"):0);
            host["rng_after_initialization.seed"]=scalar_i64(result.rng_after_initialization.seed);
            host["rng_after_initialization.offset"]=scalar_i64(result.rng_after_initialization.offset);
            host["shared_graph"]=scalar_i64(1);
            backend.synchronize();write_bundle(argv[4],host);return 0;
        }
        if(sequence) {
            size_t pivot=1;
            while(pivot<plan.size() && plan[pivot]->id()==plan[0]->id())++pivot;
            require(pivot<plan.size(),"Sequence probe requires two selectable instances");
            const auto& first=*plan[0];const auto& second=*plan[pivot];
            require(first.graph()==second.graph(),"Sequence graph must be shared");
            require(first.binding_id().value!=second.binding_id().value,"Distinct binding identities required");
            require(first.parameters().size()==second.parameters().size(),"Parameter count mismatch");
            for(size_t i=0;i<first.parameters().size();++i)
                require(first.parameters()[i].data()!=second.parameters()[i].data(),"Parameter backing must be distinct");
            // Three actual admitted selections, then a controlled same-timestep
            // comparison using a separately admitted generic uniform program.
            auto uniform=ExecutionProgram::uniform(second.id()).admit(context,programs.sampling,state_dtype);
            TensorBundle host;
            const std::vector<size_t> steps{0,pivot,0,0};
            for(size_t index=0;index<steps.size();++index) {
                const auto s=steps[index];const auto& instance=*(index==3?uniform[s]:plan[s]);
                const auto& t=programs.sampling.model_timestep_at(static_cast<int>(s));
                auto p=instance.denoiser().evaluate_step(input.at("latent"),StepExecutionContext{s,t,instance});
                auto captured=instance.denoiser().take_trace();
                for(size_t b=0;b<p.size();++b)captured["branch."+std::to_string(b)+".prediction"]=p[b];
                backend.synchronize();const auto prefix="evaluation."+std::to_string(index)+".";
                for(const auto& [name,value]:captured)host[prefix+name]=value.device().is_host()?value:backend.copy_to_host(value);
                host[prefix+"instance_id"]=scalar_i64(instance.id().value);
                host[prefix+"binding_id"]=scalar_i64(instance.binding_id().value);
                host[prefix+"timestep"]=t;
                host[prefix+"cache_resident_bytes"]=scalar_i64(backend.weight_cache_resident_bytes());
                host[prefix+"cache_capacity_bytes"]=scalar_i64(backend.weight_cache_capacity_bytes());
                host[prefix+"weight_upload_bytes"]=scalar_i64(backend.weight_upload_bytes());
                std::cout<<"Sequence evaluation="<<index<<" instance="<<instance.id().value<<" timestep="<<*t.data_as<int64_t>()<<std::endl;
            }
            host["shared_graph"]=scalar_i64(1);host["distinct_parameter_backing"]=scalar_i64(1);
            backend.synchronize();write_bundle(argv[4],host);return 0;
        }
        if(sampling) {
            // The complete runtime requires at least two declared steps. Keep
            // the admitted schedule intact and probe its first transition via
            // the same production numerical primitives, without a full run.
            const auto& program=programs.sampling;
            const auto& transition=program.contract->schedule.flow_at(0);
            const auto& guidance=program.guidance_schedule->at(0);
            require(guidance.mode==GuidanceMode::CFG,"CFG probe contract required");
            const auto& selected=*plan[0];
            auto predictions=selected.denoiser().evaluate_step(input.at("latent"),
                StepExecutionContext{0,transition.model_timestep,selected});
            validate_component_predictions(selected,predictions);
            SamplingPrimitives primitives(backend);
            auto guided=primitives.cfg_combine(predictions[0],predictions[1],guidance.scale);
            auto x0=primitives.flow_to_x0(input.at("latent"),guided,transition.sigma);
            std::vector<float> sigmas;
            for(int s=0;s<program.steps;++s)sigmas.push_back(program.contract->schedule.flow_at(s).sigma);
            sigmas.push_back(program.contract->schedule.flow_at(program.steps-1).next_sigma);
            std::vector<Tensor> history{x0};
            const int order=std::min(std::min(program.contract->solver.maximum_order,program.steps),1);
            auto next=primitives.multistep_predictor(input.at("latent"),history,x0,sigmas,0,order);
            primitives.state_advance();
            auto captured=selected.denoiser().take_trace();
            captured["branch.0.prediction"]=predictions[0];captured["branch.1.prediction"]=predictions[1];
            captured["guided_prediction"]=guided;captured["x0"]=x0;captured["next_latent"]=next;
            backend.synchronize();TensorBundle host;
            for(const auto& [name,t]:captured)host[name]=t.device().is_host()?t:backend.copy_to_host(t);
            host["instance_id"]=scalar_i64(selected.id().value);
            host["timestep"]=transition.model_timestep;
            host["sigma"]=scalar_f32(transition.sigma);host["next_sigma"]=scalar_f32(transition.next_sigma);
            host["guidance_scale"]=scalar_f32(guidance.scale);
            host["rng_calls"]=scalar_i64(primitives.calls().count("rng_normal"));
            host["state_advance_calls"]=scalar_i64(primitives.calls().at("state_advance"));
            host["solver_history_entries"]=scalar_i64(history.size());host["solver_order"]=scalar_i64(order);
            backend.synchronize();write_bundle(argv[4],host);return 0;
        }
        const auto& selected=*plan.at(step);
        require(selected.binding_id().value==std::stoul(argv[2]),"Resolved selection mismatch");
        const auto timestep=programs.sampling.model_timestep_at(step);
        auto predictions=selected.denoiser().evaluate_step(input.at("latent"),StepExecutionContext{step,timestep,selected});
        trace=selected.denoiser().take_trace();
        for(size_t i=0;i<predictions.size();++i)trace["branch."+std::to_string(i)+".prediction"]=predictions[i];
        backend.synchronize(); TensorBundle host;
        for(const auto& [name,t]:trace)host[name]=t.device().is_host()?t:backend.copy_to_host(t);
        backend.synchronize();write_bundle(argv[4],host);
        std::cout<<"Resolved single evaluation step="<<step<<" instance="<<selected.id().value<<" timestep="<<*timestep.data_as<int64_t>()<<" checkpoints="<<host.size()<<"\n";
        if(!policy_path.empty())std::cout<<"Resource resident_bytes="<<backend.weight_cache_resident_bytes()
            <<" cache_capacity_bytes="<<backend.weight_cache_capacity_bytes()
            <<" weight_upload_bytes="<<backend.weight_upload_bytes()
            <<" cache_hits="<<backend.weight_cache_hits()<<" cache_misses="<<backend.weight_cache_misses()
            <<" backend_peak_bytes="<<backend.peak_device_bytes()<<std::endl;
        return 0;
    }
    const auto& video=input.at("latent");const auto& raw_context=input.at("raw_context");
    auto time=backend.sinusoidal_embedding(backend.cast(input.at("timestep"),DType::F32),c.frequency,true,0.0,true);
    trace["preparation.sinusoidal"]=time;
    trace["preparation.patch"]=backend.conv3d(video,weights.at("patch_embedding.weight"),&weights.at("patch_embedding.bias"),{c.patch[0],c.patch[1],c.patch[2]},{0,0,0});
    time=operation_linear(backend,policy,PrecisionOperation::Modulation,PrecisionSemantic::TemporaryCompute,time,weights.at("time_embedding.0.weight"),&weights.at("time_embedding.0.bias"));
    time=backend.activation(time,Activation::Silu);
    time=operation_linear(backend,policy,PrecisionOperation::Modulation,PrecisionSemantic::TemporaryCompute,time,weights.at("time_embedding.2.weight"),&weights.at("time_embedding.2.bias"));
    trace["preparation.time"]=time;
    auto mod=operation_linear(backend,policy,PrecisionOperation::Modulation,PrecisionSemantic::TemporaryCompute,backend.activation(time,Activation::Silu),weights.at("time_projection.1.weight"),&weights.at("time_projection.1.bias"));
    trace["preparation.modulation"]=backend.reshape(mod,{1,6,c.dim});
    auto context=backend.pad(raw_context,{0,0,0,c.text_length-raw_context.dim(1)},0.0f);
    trace["preparation.padded_context"]=context;
    context=backend.linear(context,weights.at("text_embedding.0.weight"),&weights.at("text_embedding.0.bias"));
    context=backend.activation(context,Activation::GeluTanh);
    trace["preparation.context"]=backend.linear(context,weights.at("text_embedding.2.weight"),&weights.at("text_embedding.2.bias"));
    std::vector<std::vector<float>> coords;
    for(int f=0;f<2;++f)for(int w=0;w<2;++w)coords.push_back({float(f),0,float(w)});
    auto [cos,sin]=standard_rope(backend,policy,coords,definition->rope_axes,c.rope_theta,true);
    // Match the production denoiser's existing temporary boundary. The old
    // F32-only harness happened to receive this dtype without an explicit cast.
    const auto rope_temporary=policy.operation_contract(PrecisionOperation::Rope,
        PrecisionSemantic::TemporaryCompute).temporary_dtype;
    std::cout<<"RoPE source="<<dtype_name(cos.dtype())<<" temporary="<<dtype_name(rope_temporary)<<std::endl;
    cos=backend.cast(cos,rope_temporary);sin=backend.cast(sin,rope_temporary);
    block(backend,policy,weights.prefix("blocks.0."),input.at("input"),input.at("modulation_input"),input.at("context"),cos,sin,&trace,"block",c);
    backend.synchronize();TensorBundle host;
    for(const auto& [name,t]:trace)host[name]=t.device().is_host()?t:backend.copy_to_host(t);
    backend.synchronize();write_bundle(argv[4],host);
    std::cout<<"Native execution="<<dtype_name(policy.requested_dtype())<<" checkpoints="<<host.size()<<" binding="<<argv[2]<<"\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
