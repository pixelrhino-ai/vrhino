#include "vrhino/product/prepared_execution.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/tensor_util.h"
#include "step_execution_test_support.h"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <bit>

using namespace vrhino;
namespace p=vrhino::product;
namespace fs=std::filesystem;
struct Counts {int evaluated=0,decoded=0;bool decode_failure=false,nonfinite=false,wrong_shape=false;};
struct Endpoint : Denoiser {
    Backend& backend;Tensor parameter;Counts& counts;
    Endpoint(Backend& b,Tensor w,Counts& c):backend(b),parameter(std::move(w)),counts(c){}
    std::vector<Tensor> evaluate(const Tensor& latent,const Tensor&)override{
        ++counts.evaluated;auto u=backend.add(backend.mul(latent,scalar_f32(.25f)),parameter);
        return {u,backend.add(u,scalar_f32(.125f))};
    }
};
struct Adapter : Architecture {
    SamplingProgram program;Counts& counts;
    Adapter(SamplingProgram p,Counts& c):program(std::move(p)),counts(c){}
    std::unique_ptr<Denoiser> create_denoiser(Backend&,const PrecisionPolicy&,const TensorBundle&)override{
        throw Error("No singleton fallback in catalog test");
    }
    SamplingProgram create_program(const TensorBundle&)override{return program;}
    ExecutionSetup create_execution_setup(Backend& backend,const PrecisionPolicy& policy,const TensorBundle& input)override{
        require(input.contains("positive")&&input.contains("negative"),"Missing conditioning");
        const auto dtype=policy.persistent_state_dtype(PrecisionSemantic::SamplingState);
        const auto parameter=[&](float value) {
            if(dtype==DType::F32)return scalar_f32(value);
            require(dtype==DType::BF16,"Unsupported test policy");
            auto tensor=Tensor::host({},dtype);
            tensor.data_as<uint16_t>()[0]=uint16_t(std::bit_cast<uint32_t>(value)>>16);
            return tensor; // Both fixture constants are exactly representable.
        };
        ComponentInterface interface{{program.latent_shape,dtype},{{},DType::I64},
            {{program.latent_shape,dtype},{program.latent_shape,dtype}},GuidanceMode::CFG,PredictionSemantic::Flow,0};
        auto graph=std::make_shared<const ComponentGraphDefinition>(interface,
            std::vector<ExecutionTensorContract>{{{},dtype}},[&backend,this](const std::vector<Tensor>& parameters){
                return std::make_unique<Endpoint>(backend,parameters.at(0),counts);
            });
        std::vector<ComponentInstance> instances;
        instances.push_back(ComponentInstance::bind({7},{70},graph,{parameter(.375f)}));
        instances.push_back(ComponentInstance::bind({9},{90},graph,{parameter(-.25f)}));
        return {ExecutionContext(std::move(instances)),ExecutionProgram::per_step({{7},{9},{7}})};
    }
    Tensor decode(Backend& backend,const PrecisionPolicy&,const Tensor& latent,const TensorBundle&)override{
        ++counts.decoded;if(counts.decode_failure)throw Error("Injected decode failure");
        auto value=backend.add(latent,scalar_f32(counts.nonfinite?std::numeric_limits<float>::quiet_NaN():.125f));
        return counts.wrong_shape?value:backend.reshape(value,{1,1,1,1,2});
    }
};
size_t free_device(){size_t free=0,total=0;require(cudaMemGetInfo(&free,&total)==cudaSuccess,"Device memory query");return free;}
int main(int argc,char**argv){try{
    require(argc==2 || argc==3,"usage: prepared-product-tests OUTPUT [POLICY]");fs::path dir=argv[1];fs::create_directories(dir);
    std::string header=R"({"embedding":{"dtype":"F32","shape":[3,4],"data_offsets":[0,48]},"norm":{"dtype":"F32","shape":[4],"data_offsets":[48,64]}})";while(header.size()%8)header+=' ';
    const std::vector<float> values{1,2,3,4,-1,2,-3,4,4,3,2,1,1,1,1,1};
    const auto file=dir/"conditioning.safetensors";
    {std::ofstream f(file,std::ios::binary);uint64_t length=header.size();f.write(reinterpret_cast<const char*>(&length),8);f<<header;f.write(reinterpret_cast<const char*>(values.data()),64);}
    Counts counts;p::PreparedTextProductRequest request;request.owner=std::make_shared<p::AdmittedLocalProduct>();
    request.owner->conditioning=std::make_unique<p::SafeTensorAsset>(p::SafeTensorAsset::single(file));
    request.sampling=step_test::program(1);request.sampling.guidance_schedule=GuidanceSchedule::per_step({GuidanceParameters::cfg(4),GuidanceParameters::cfg(3),GuidanceParameters::cfg(2)});
    request.execution=ExecutionProgram::per_step({{7},{9},{7}});request.runtime_inputs={{"seed",scalar_i64(11)}};
    request.expected_video_shape={1,1,1,1,2};request.owner->architecture=std::make_unique<Adapter>(request.sampling,counts);
    const auto graph=Json::parse(R"({"schema_version":1,"kind":"pre_norm_transformer","embedding":{"weight":"embedding"},"blocks":[],"final_norm":{"kind":"rms_norm","weight":"norm","eps":0.000001},"output_trim_to_mask":true})");
    request.conditioning.push_back({"positive",graph,host_i64({1,3},{0,1,2}),host_bool({1,3},{1,1,0}),{1,2,4}});
    request.conditioning.push_back({"negative",graph,host_i64({1,3},{1,0,2}),host_bool({1,3},{1,0,0}),{1,1,4}});
    auto policy=PrecisionPolicy::fp32();
    if(argc==3) {std::ifstream f(argv[2]);require(f.good(),"Cannot read test policy");
        policy=PrecisionPolicy::from_json(Json::parse(std::string(std::istreambuf_iterator<char>(f),{})));}
    const auto dtype=policy.requested_dtype();int factories=0;
    p::ProductBackendFactory factory=[&]{++factories;auto b=std::make_unique<CudaBackend>();b->set_execution_dtype(dtype);
        b->configure_memory_runtime({256ULL<<20,0,1ULL<<30,16ULL<<20,8ULL<<20},{true,false,false});return b;};
    p::PreparedExecutionControl control;control.solver_trace=true;std::vector<uint32_t> selected;int phases=0;
    control.step_completed=[&](int completed,int total,ComponentInstanceID instance,BindingID binding,const Json& resources){
        require(total==3&&completed==int(selected.size()+1)&&binding.value==instance.value*10,"Selection trace");
        require(resources.at("source_owner_count").integer()==1,"Missing phase source lease");selected.push_back(instance.value);
    };
    control.phase_completed=[&](p::ProductExecutionPhase phase,const p::PreparedProductExecution& result){
        require(int(phase)==phases++,"Phase order");
        for(const auto& [name,t]:result.conditioning)require(t.device().is_host(),"Conditioning outlived device storage");
        if(phase!=p::ProductExecutionPhase::Conditioning)require(result.sampling.final_latent.device().is_host(),"Latent not on host");
    };
    auto actual=p::execute_prepared_product(request,policy,factory,control);
    require(factories==3&&phases==3&&counts.evaluated==3&&counts.decoded==1&&selected==std::vector<uint32_t>({7,9,7}),"Continuous Product chain");
    require(actual.shared_graph_identity&&actual.sampling.program_complete&&actual.resources.size()==3,"Completion evidence");
    {
        auto backend=factory();auto setup=request.owner->architecture->create_execution_setup(*backend,policy,actual.conditioning);
        SamplingRuntime direct(*backend,policy);direct.set_solver_trace_enabled(true);auto expected=direct.run(setup.context,request.execution,request.sampling);
        for(const auto& [key,value]:expected.trace)step_test::exact(value,actual.sampling.trace.at(key));
        step_test::exact(backend->copy_to_host(expected.final_latent),actual.sampling.final_latent);
        require(actual.primitive_calls==direct.primitives().calls(),"Product changed solver primitives");
        auto video=request.owner->architecture->decode(*backend,policy,expected.final_latent,request.runtime_inputs);
        step_test::exact(backend->copy_to_host(video),actual.video);
    }
    // Turning off observation must preserve every numerical boundary, RNG and
    // solver primitive call. Both F32 and declared BF16 run this same test.
    control={};control.tensor_trace=false;
    const auto untraced=p::execute_prepared_product(request,policy,factory,control);
    require(untraced.sampling.trace.empty(),"Disabled trace retained snapshots");
    step_test::exact(untraced.sampling.initial_noise,actual.sampling.initial_noise);
    step_test::exact(untraced.sampling.final_latent,actual.sampling.final_latent);
    step_test::exact(untraced.video,actual.video);
    require(untraced.primitive_calls==actual.primitive_calls&&
        untraced.sampling.rng_after_initialization.offset==actual.sampling.rng_after_initialization.offset,
        "Observation toggle changed execution");
    const auto free_before=free_device();int rejected=0;
    const auto reject=[&](auto f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"Expected rejection");
        require(free_device()==free_before,"Device allocations survived failed phase");++rejected;};
    control={};control.qualification_prefix_steps=2;factories=0;counts.evaluated=counts.decoded=0;
    auto prefix=p::execute_prepared_product(request,policy,factory,control);
    require(factories==2&&counts.evaluated==2&&counts.decoded==0&&!prefix.video.defined()&&!prefix.sampling.program_complete,"Prefix decoded/reset program");
    step_test::exact(prefix.sampling.final_latent,actual.sampling.trace.at("step.1.latent"));
    control.qualification_prefix_steps=3;reject([&]{p::execute_prepared_product(request,policy,factory,control);});
    control={};control.cancellation_requested=[] {return true;};factories=0;
    reject([&]{p::execute_prepared_product(request,policy,factory,control);});require(factories==0,"Cancelled Product allocated backend");
    bool cancel=false;control.cancellation_requested=[&]{return cancel;};control.step_completed=[&](int,int,ComponentInstanceID,BindingID,const Json&){cancel=true;};counts.evaluated=counts.decoded=0;
    bool cancelled_status=false;
    reject([&]{try{p::execute_prepared_product(request,policy,factory,control);}catch(const p::ModelPackageError& e){cancelled_status=e.code()==p::ModelPackageErrorCode::Cancelled;throw;}});
    require(cancelled_status,"Wrong cancellation status");
    require(counts.evaluated==1&&counts.decoded==0,"Cancellation executed extra steps");control={};
    request.execution=ExecutionProgram::per_step({{7},{99},{7}});counts.evaluated=0;
    reject([&]{p::execute_prepared_product(request,policy,factory,control);});require(counts.evaluated==0,"Unknown instance reached denoiser");request.execution=ExecutionProgram::per_step({{7},{9},{7}});
    counts.decode_failure=true;reject([&]{p::execute_prepared_product(request,policy,factory,control);});counts.decode_failure=false;
    counts.nonfinite=true;reject([&]{p::execute_prepared_product(request,policy,factory,control);});counts.nonfinite=false;
    counts.wrong_shape=true;reject([&]{p::execute_prepared_product(request,policy,factory,control);});counts.wrong_shape=false;
    control.phase_completed=[](auto,const auto&){throw Error("Injected observation failure");};reject([&]{p::execute_prepared_product(request,policy,factory,control);});control={};
    reject([&]{p::execute_prepared_product(request,policy,[]{return std::unique_ptr<Backend>{};});});
    reject([&]{p::execute_prepared_product(request,policy,[&]{auto b=std::make_unique<CudaBackend>();b->set_execution_dtype(dtype==DType::F32?DType::BF16:DType::F32);return b;});});
    std::weak_ptr<p::AdmittedLocalProduct> owner=request.owner;request.owner.reset();require(owner.expired(),"Result retained source owner/device session");
    p::require_finite_component_output(actual.video);p::require_finite_component_output(actual.sampling.final_latent);
    std::cout<<"PASS generic prepared Product execution; direct-runtime byte equality; A/B/A; step CFG; prefix; host ownership; cancellation; failure teardown; negatives="<<rejected<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
