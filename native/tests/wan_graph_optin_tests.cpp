// Compile the actual private production implementation, without exporting a Native API.
#define make_wan_architecture make_wan_optin_test_architecture
#include "../src/architectures/wan.cpp"
#undef make_wan_architecture
#include "neural_graph_test_backend.h"
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#ifdef VRHINO_TEST_CUDA
#include "vrhino/backend/cuda_backend.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <sys/resource.h>
// Test-only CUDA fault injection; successful calls always reach the real CUDA API.
static int fail_error_check = 0;
static size_t observed_syncs = 0;
extern "C" cudaError_t __real_cudaGetLastError();
extern "C" cudaError_t __wrap_cudaGetLastError() {
    auto result=__real_cudaGetLastError();
    return fail_error_check>0 && --fail_error_check==0 ? cudaErrorInvalidValue : result;
}
extern "C" cudaError_t __real_cudaDeviceSynchronize();
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    ++observed_syncs;return __real_cudaDeviceSynchronize();
}
#endif
using namespace vrhino;
namespace wan = wan_internal;
namespace {
class BindingBackend : public neural_graph::test::TinyBackend {
public:
    Tensor copy_to_device(const Tensor& t, DType d) override {
        require(t.dtype()==d,"host binding dtype"); return t;
    }
    Tensor cast(const Tensor& t, DType d) override {
        require(t.dtype()==d,"host cast dtype"); return t;
    }
    Tensor slice(const Tensor& t,int64_t axis,int64_t start,int64_t stop) override {
        begin("slice"); auto s=t.shape();s.at(axis)=stop-start;return owned(s);
    }
};
void negatives() {
    BindingBackend b; const auto policy=PrecisionPolicy::fp32();
    auto graph=wan::self_attention_graph(1,8,DType::F32,DType::F32);
    const auto& d=graph.description(); TensorBundle params;
    std::map<std::string,const Tensor*> slots;
    for(size_t i=0;i<std::size(wan::parameters);++i) {
        params[wan::parameters[i]]=Tensor::host(d.values[d.parameters[i]].shape,DType::F32);
        slots[wan::parameters[i]]=&params.at(wan::parameters[i]);
    }
    auto hidden=Tensor::host({1,8,1536},DType::F32), modulation=Tensor::host({1,6,1536},DType::F32);
    auto rope=Tensor::host({1,8,1,128},DType::F32), context=Tensor::host({1,512,1536},DType::F32);
    auto invoke=[&]() {
        return block(b,policy,WeightMap(slots),hidden,modulation,context,rope,rope,nullptr,"");
    };
    auto reject=[&](const char* name,const std::function<void()>& fn) {
        b.calls.clear();b.produced.clear();bool caught=false;
        try {fn();}catch(const Error&){caught=true;}
        require(caught&&b.calls.empty(),std::string("negative dispatch: ")+name);
        std::cout<<"NEGATIVE\t"<<name<<"\tPASS\n";
    };
    hidden=Tensor::host(hidden.shape(),DType::BF16);
    reject("wrong_hidden_dtype",[&]{invoke();});hidden=Tensor::host(hidden.shape(),DType::F32);
    params[wan::parameters[1]]=Tensor::host({1536,1536},DType::BF16);
    reject("wrong_parameter_dtype",[&]{invoke();});params[wan::parameters[1]]=Tensor::host({1536,1536},DType::F32);
    slots.erase(wan::parameters[1]);reject("missing_parameter",[&]{invoke();});slots[wan::parameters[1]]=&params.at(wan::parameters[1]);
    modulation=Tensor::host({1,5,1536},DType::F32);reject("wrong_modulation_shape",[&]{invoke();});
    modulation=Tensor::host({1,6,1536},DType::F32);
    rope=Tensor::host(rope.shape(),DType::BF16);reject("wrong_rope_dtype",[&]{invoke();});rope=Tensor::host(rope.shape(),DType::F32);
    b.dtype=DType::BF16;reject("wrong_execution_context",[&]{invoke();});b.dtype=DType::F32;
    // Fail after combined modulation was queued. TinyBackend verifies owners during drain.
    b.fail_call=2;bool caught=false;
    try {invoke();}catch(const neural_graph::GraphError&){caught=true;}
    require(caught&&b.syncs==1&&b.calls==std::vector<std::string>({"add","slice"}),"Graph failure drain");
    for(const auto& weak:b.produced)require(weak.expired(),"failed Graph output leaked");
    std::cout<<"NEGATIVE\tgraph_failure_drains_backend\tPASS\n";
}
#ifdef VRHINO_TEST_CUDA
using Clock=std::chrono::steady_clock;
Tensor host(Backend& b,const Tensor& t){return t.device().is_host()?t:b.copy_to_host(t);}
TensorBundle capture(Backend& b,const TensorBundle& t) {
    TensorBundle r;for(const auto& [n,x]:t)r[n]=host(b,x);
    return r;
}
void equal(const Tensor& a,const Tensor& e,const std::string& name) {
    require(a.shape()==e.shape()&&a.dtype()==e.dtype()&&a.bytes()==e.bytes(),"metadata: "+name);
    require(std::memcmp(a.data(),e.data(),a.bytes())==0,"NUMERICAL_REGRESSION: "+name);
    for(int64_t i=0;i<a.numel();++i) {
        float v;
        if(a.dtype()==DType::F32)v=a.data_as<float>()[i];
        else {require(a.dtype()==DType::BF16,"capture dtype");uint32_t bits=uint32_t(a.data_as<uint16_t>()[i])<<16;std::memcpy(&v,&bits,4);}
        require(std::isfinite(v),"nonfinite: "+name);
    }
}
void equal_bundle(const TensorBundle& a,const TensorBundle& e,const std::string& label,std::ostream* out=nullptr) {
    require(a.size()==e.size(),"trace set mismatch");
    for(const auto& [n,t]:e) {equal(a.at(n),t,label+"/"+n);if(out)*out<<label<<'\t'<<n<<'\t'<<dtype_name(t.dtype())<<"\tBITWISE\tPASS\n";}
}
size_t used(){size_t f=0,t=0;require(cudaMemGetInfo(&f,&t)==cudaSuccess,"memory query");return t-f;}
Tensor replay(Backend& b,const PrecisionPolicy& p,const WeightMap& w,const TensorBundle& x,TensorBundle* trace=nullptr) {
    return block(b,p,w,x.at("input"),x.at("modulation_input"),x.at("context"),x.at("cosine"),x.at("sine"),trace,"replay");
}
void qualify(Backend& b,const PrecisionPolicy& policy,const WeightMap& weights,const TensorBundle& boundary,
             const TensorBundle& reference,const std::string& label,const std::string& dir,int repeats) {
    auto x=boundary;auto immutable=capture(b,x);
    auto graph=wan::self_attention_graph(1,8,b.execution_dtype(),x.at("input").dtype());
    auto misses=b.weight_cache_misses(),hits=b.weight_cache_hits();auto p=wan::bind_parameters(b,graph,weights);
    auto again=wan::bind_parameters(b,graph,weights);std::vector<Tensor> original;
    std::ofstream bindings(dir+"/parameter-bindings.tsv",std::ios::app);
    for(size_t i=0;i<p.size();++i){require(p[i].data()==again[i].data(),"cache pointer drift");original.push_back(host(b,p[i]));
        bindings<<label<<'\t'<<wan::parameters[i]<<"\tF32\t"<<dtype_name(p[i].dtype())<<'\t'<<weights.at(wan::parameters[i]).data()<<'\t'<<p[i].data()<<"\tWARM_REUSED\n";}
    require(b.weight_cache_misses()==misses&&b.weight_cache_hits()-hits==22,"production cache not reused");
    // Both hidden contracts bind exactly the same eleven device parameters.
    auto alternate=wan::self_attention_graph(1,8,b.execution_dtype(),
        x.at("input").dtype()==DType::BF16?DType::F32:DType::BF16);
    auto alternate_bound=wan::bind_parameters(b,alternate,weights);
    for(size_t i=0;i<p.size();++i)require(p[i].data()==alternate_bound[i].data(),"hidden variant cache collision");
    const auto resident=b.weight_cache_resident_bytes();TensorBundle first;double seconds=0;
    size_t before=used(),peak=before;
    for(int r=0;r<repeats;++r) {
        TensorBundle trace;
        auto start=Clock::now();auto output=replay(b,policy,weights,x,&trace);b.synchronize();
        seconds+=std::chrono::duration<double>(Clock::now()-start).count();peak=std::max(peak,used());
        auto h=capture(b,trace);equal_bundle(h,reference,label+"/frozen");
        if(r==0)first=h;else equal_bundle(h,first,label+"/determinism");
        equal(host(b,output),h.at("replay.output"),label+"/output_owner");
    }
    std::ofstream results(dir+"/checkpoint-results.tsv",std::ios::app);equal_bundle(first,reference,label,&results);
    // Trace-free execution consumes both owning Graph outputs after evaluator destruction.
    {
        b.enable_profiling(true);auto before_counts=b.profile_stats();
        equal(host(b,replay(b,policy,weights,x)),reference.at("replay.output"),label+"/untraced");
        auto after_counts=b.profile_stats();b.enable_profiling(false);
        std::ofstream ownership(dir+"/operation-counts.tsv",std::ios::app);
        for(const auto& [name,expected]:std::map<std::string,uint64_t>{{"elementwise.add",8},{"elementwise.mul",4},{"slice",6}}) {
            auto count=after_counts[name].calls-before_counts[name].calls;
            require(count==expected,"duplicate modulation or slice computation: "+name);
            ownership<<label<<'\t'<<"NeuralGraph"<<'\t'<<name<<'\t'<<count<<"\tPASS\n";
        }
    }
    equal_bundle(capture(b,x),immutable,label+"/boundary_immutable");
    for(size_t i=0;i<p.size();++i)equal(host(b,p[i]),original[i],label+"/weight_immutable");
    require(b.weight_cache_misses()==misses&&b.weight_cache_resident_bytes()==resident,"reupload or duplicate weights");
    std::ofstream continuation(dir+"/block-continuation.tsv",std::ios::app);
    continuation<<label<<"\tpre_cross_attention_norm/cross_residual/ffn_input/output\tBITWISE\tPASS\n";
    struct rusage rss{};getrusage(RUSAGE_SELF,&rss);std::ofstream resource(dir+"/resource-usage.tsv",std::ios::app);
    resource<<std::setprecision(17)<<label<<'\t'<<before<<'\t'<<peak<<'\t'<<used()<<'\t'<<b.peak_device_bytes()<<'\t'<<rss.ru_maxrss*1024L<<'\t'<<seconds/repeats<<'\t'<<repeats<<'\n';
    // Invalid Graph execution fails closed, then the same production block recovers.
    auto bad=x;bad["cosine"]=b.cast(x.at("cosine"),DType::BF16);bool caught=false;
    try{replay(b,policy,weights,bad);}catch(const Error&){caught=true;}
    require(caught,"invalid Graph silently fell back");b.synchronize();
    equal(host(b,replay(b,policy,weights,x)),reference.at("replay.output"),label+"/recovery");
    auto sync_before=observed_syncs;fail_error_check=2;caught=false;
    try{replay(b,policy,weights,x);}catch(const neural_graph::GraphError&){caught=true;}
    require(caught&&fail_error_check==0&&observed_syncs==sync_before+1,"integrated CUDA Graph failure did not drain");
    equal(host(b,replay(b,policy,weights,x)),reference.at("replay.output"),label+"/CUDA_failure_recovery");
    std::ofstream negatives(dir+"/cuda-negative-controls.tsv",std::ios::app);
    negatives<<label<<"\tinvalid_Graph_no_fallback/CUDA_queued_failure_drain/production_recovery\tPASS\n";
    {
        CudaBackend cold_backend;cold_backend.set_execution_dtype(b.execution_dtype());TensorBundle cold_x;
        for(const auto& [name,t]:immutable)cold_x[name]=cold_backend.copy_to_device(t,t.dtype());
        require(cold_backend.weight_cache_misses()==0,"cold integrated cache starts empty");
        equal(host(cold_backend,replay(cold_backend,policy,weights,cold_x)),
              reference.at("replay.output"),label+"/cold_integrated");
        auto cold_misses=cold_backend.weight_cache_misses(),cold_hits=cold_backend.weight_cache_hits();
        auto cold_bound=wan::bind_parameters(cold_backend,graph,weights);
        require(cold_backend.weight_cache_misses()==cold_misses&&cold_backend.weight_cache_hits()-cold_hits==11,
                "cold integrated Graph did not populate generic cache");
        std::ofstream cache(dir+"/integrated-cache.tsv",std::ios::app);
        cache<<label<<'\t'<<cold_misses<<"\t11\t11\tPASS\n";
        for(size_t i=0;i<cold_bound.size();++i)bindings<<label<<'\t'<<wan::parameters[i]<<"\tF32\t"<<dtype_name(cold_bound[i].dtype())<<'\t'<<weights.at(wan::parameters[i]).data()<<'\t'<<cold_bound[i].data()<<"\tCOLD_INTEGRATED_THEN_WARM_HIT\n";
    }
    std::cout<<"INTEGRATED\t"<<label<<"\t"<<repeats<<"\tPASS\n"<<std::flush;
}
void run(int argc,char** argv) {
    require(argc==7,"usage: TEST MODEL INPUT POLICY FROZEN EVIDENCE MODE(all/bf16/f32)");
    const std::string frozen=argv[4],dir=argv[5],mode=argv[6];VrmModel model(argv[1]);
    auto input=read_bundle(argv[2]);input["audit_trace"]=scalar_i64(1);
    std::ifstream pf(argv[3]);std::stringstream ps;ps<<pf.rdbuf();auto policy=PrecisionPolicy::from_json(Json::parse(ps.str()));
    CudaBackend b;b.set_execution_dtype(DType::BF16);WeightMap weights(model.bindings(model.graph().at("architecture_graph")));
    // Cold bindings precede production; only the existing generic cache materializes weights.
    for(int index:{0,15,29}) {
        auto w=weights.prefix("blocks."+std::to_string(index)+".");auto g=wan::self_attention_graph(1,8,DType::BF16,index?DType::F32:DType::BF16);
        auto m=b.weight_cache_misses(),bytes=b.weight_cache_resident_bytes();auto bound=wan::bind_parameters(b,g,w);
        size_t expected=0;for(const auto& t:bound)expected+=t.bytes();
        require(b.weight_cache_misses()-m==11&&b.weight_cache_resident_bytes()-bytes==expected,"cold binding copies");
        std::ofstream bindings(dir+"/parameter-bindings.tsv",std::ios::app);
        for(size_t i=0;i<bound.size();++i)bindings<<"block"<<index<<'\t'<<wan::parameters[i]<<"\tF32\t"<<dtype_name(bound[i].dtype())<<'\t'<<w.at(wan::parameters[i]).data()<<'\t'<<bound[i].data()<<"\tCOLD_MISS\n";
        std::cout<<"COLD\t"<<index<<"\t11\t"<<expected<<"\tPASS\n";
    }
    for(int step:{0,42,49}) {
        if(mode!="all"&&step!=0)continue;
        auto state=read_bundle(frozen+"/sampling-step"+std::to_string(step)+".bundle");
        auto latent=b.copy_to_device(state.at("latent"),policy.persistent_state_dtype(PrecisionSemantic::SamplingState));
        const auto& timestep=state.at("timestep");const auto time=read_scalar_i64(timestep);
        for(int index:{0,15,29}) {
            if((mode=="bf16"&&index!=0)||(mode=="f32"&&index!=15))continue;
            input["audit_trace_block"]=scalar_i64(index);auto architecture=create_architecture(model);
            auto denoiser=architecture->create_denoiser(b,policy,input);
            auto output=denoiser->evaluate(latent,timestep);auto traced=denoiser->take_trace();auto actual=capture(b,traced);
            std::string label="block"+std::to_string(index)+"-t"+std::to_string(time);
            auto expected=read_bundle(frozen+"/"+label+".bundle");
            for(size_t branch=0;branch<output.size();++branch)actual["prediction."+std::to_string(branch)]=host(b,output[branch]);
            std::ofstream entry(dir+"/architecture-entry-results.tsv",std::ios::app);equal_bundle(actual,expected,label,&entry);
            TensorBundle boundary,reference;auto prefix="branch.1.block."+std::to_string(index)+".";
            for(const char* name:{"input","modulation_input","context","cosine","sine"})boundary[name]=traced.at(prefix+name);
            for(const auto& [name,t]:expected)if(name.starts_with(prefix))reference["replay."+name.substr(prefix.size())]=t;
            require(boundary.at("input").dtype()==(index?DType::F32:DType::BF16),"production hidden dtype drift");
            write_bundle(dir+"/"+label+"-boundary.bundle",capture(b,boundary));
            std::ofstream provenance(dir+"/boundary-provenance.tsv",std::ios::app);
            provenance<<label<<'\t'<<step<<'\t'<<time<<"\tWanArchitecture.create_denoiser/WanDenoiser.evaluate/positive/full_denoiser/block\t"<<index<<'\t'<<dtype_name(boundary.at("input").dtype())<<"\tS8\t"<<frozen<<"/sampling-step"<<step<<".bundle\n";
            qualify(b,policy,weights.prefix("blocks."+std::to_string(index)+"."),boundary,reference,label,dir,mode=="all"?10:1);
        }
        if(mode=="all"&&step==0) {
            // Normal trace-free two-branch production invocation covers all adjacent Graph blocks.
            input["audit_trace"]=scalar_i64(0);auto architecture=create_architecture(model);
            auto denoiser=architecture->create_denoiser(b,policy,input);
            auto expected=read_bundle(frozen+"/block0-t999.bundle");
            for(int repeat=0;repeat<2;++repeat) {
                auto actual=denoiser->evaluate(latent,timestep);
                for(size_t i=0;i<actual.size();++i)equal(host(b,actual[i]),expected.at("prediction."+std::to_string(i)),"adjacent/CFG/state");
            }
            std::ofstream pair(dir+"/multiblock-results.tsv",std::ios::app);pair<<"0->29\t999\tboth_CFG_branches/complete_denoiser/repeated\tBITWISE\tPASS\n";
            input["audit_trace"]=scalar_i64(1);
        }
    }
}
#endif
}
int main(int argc,char** argv) {
    try {negatives();
#ifdef VRHINO_TEST_CUDA
        run(argc,argv);
#else
        (void)argc;(void)argv;
#endif
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
