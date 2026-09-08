#include "../src/architectures/wan_self_attention_graph.h"
#include "neural_graph_test_backend.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#ifdef VRHINO_TEST_CUDA
#include "wan_self_attention_reference.h"
#include "vrhino/backend/cuda_backend.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <sys/resource.h>
#endif
using namespace vrhino;
namespace ng = neural_graph;
namespace wan = wan_internal;
namespace {
void negatives() {
    ng::test::TinyBackend b;
    auto g=wan::self_attention_graph(1,4,DType::F32,DType::F32,true);
    const auto& d=g.description();
    std::vector<Tensor> x,p;
    for(auto id:d.inputs) x.push_back(Tensor::host(d.values[id].shape,d.values[id].dtype));
    for(auto id:d.parameters) p.push_back(Tensor::host(d.values[id].shape,d.values[id].dtype));
    auto reject=[&](const char* name,const std::function<void()>& run) {
        bool caught=false; try {run();} catch(const Error&) {caught=true;}
        require(caught && b.calls.empty(),std::string("negative failed: ")+name);
        std::cout<<"NEGATIVE\t"<<name<<"\tPASS\n";
    };
    auto xx=x;xx[0]=Tensor::host(x[0].shape(),DType::BF16);
    reject("wrong_hidden_dtype",[&]{ng::evaluate(b,g,xx,p);});
    auto pp=p;pp[1]=Tensor::host(p[1].shape(),DType::BF16);
    reject("wrong_parameter_dtype",[&]{ng::evaluate(b,g,x,pp);});
    pp=p;pp[1]=Tensor::host({1536},DType::F32);
    reject("wrong_parameter_shape",[&]{ng::evaluate(b,g,x,pp);});
    pp=p;pp.pop_back();reject("missing_parameter",[&]{ng::evaluate(b,g,x,pp);});
    xx=x;xx[2]=Tensor::host(x[2].shape(),DType::BF16);
    reject("wrong_rope_frequency_dtype",[&]{ng::evaluate(b,g,xx,p);});
    auto bad=d;std::get<ng::LayerNorm>(bad.nodes[4].op).weight=11;
    reject("wrong_layernorm_affine",[&]{ng::admit(bad);});
    bad=d;bad.values[bad.outputs.back()].shape={1,4,1535};
    reject("wrong_output_shape",[&]{ng::admit(bad);});
    b.dtype=DType::F16;reject("unsupported_context",[&]{ng::evaluate(b,g,x,p);});b.dtype=DType::F32;
    auto mixed=wan::self_attention_graph(1,4,DType::BF16,DType::BF16,true).description();
    for(auto& n:mixed.nodes) if(auto* c=std::get_if<ng::Cast>(&n.op)) {
        c->destination=mixed.values[c->x].dtype;break;
    }
    reject("wrong_cast_destination",[&]{ng::admit(mixed);});
    mixed=wan::self_attention_graph(1,4,DType::BF16,DType::BF16,true).description();
    for(auto& n:mixed.nodes) if(auto* c=std::get_if<ng::Cast>(&n.op)) {c->x=99999;break;}
    reject("wrong_cast_source",[&]{ng::admit(mixed);});
    mixed=wan::self_attention_graph(1,4,DType::BF16,DType::BF16,true).description();
    for(auto& n:mixed.nodes) if(std::holds_alternative<ng::Attention>(n.op)) {
        mixed.values[n.result].dtype=DType::F32;break;
    }
    reject("wrong_attention_dtype_tuple",[&]{ng::admit(mixed);});
    std::map<std::string,const Tensor*> slots;
    for(size_t i=0;i<p.size();++i)slots[wan::parameters[i]]=&p[i];
    pp=p;pp[1]=Tensor::host(p[1].shape(),DType::BF16);slots[wan::parameters[1]]=&pp[1];
    reject("wrong_canonical_parameter_dtype",[&]{wan::bind_parameters(b,g,WeightMap(slots));});
    slots.erase(wan::parameters[1]);
    reject("missing_canonical_parameter",[&]{wan::bind_parameters(b,g,WeightMap(slots));});
    for(auto execution:{DType::F32,DType::BF16})for(auto hidden:{DType::F32,DType::BF16}) {
        if(execution==DType::F32&&hidden==DType::BF16) continue;
        auto graph=wan::self_attention_graph(1,4,execution,hidden);
        size_t casts=0;for(const auto& n:graph.description().nodes)casts+=std::holds_alternative<ng::Cast>(n.op);
        const size_t expected=execution==DType::F32?0:hidden==DType::F32?4:6;
        require(casts==expected&&graph.description().nodes.size()==23+expected,"graph design drift");
        std::cout<<"DESIGN\t"<<dtype_name(execution)<<'\t'<<dtype_name(hidden)<<'\t'<<23+expected<<'\t'<<casts<<"\tPASS\n";
        for(const auto& n:graph.description().nodes) {
            const auto& spec=graph.description().values[n.result];
            std::cout<<"NODE\t"<<dtype_name(execution)<<'\t'<<dtype_name(hidden)<<'\t'<<n.result<<'\t'
                <<n.op.index()<<'\t'<<dtype_name(spec.dtype)<<'\t';
            for(auto extent:spec.shape)std::cout<<extent<<',';
            if(const auto* c=std::get_if<ng::Cast>(&n.op))std::cout<<'\t'<<c->x<<'\t'
                <<dtype_name(graph.description().values[c->x].dtype);
            std::cout<<'\n';
        }
    }
}
#ifdef VRHINO_TEST_CUDA
using Clock=std::chrono::steady_clock;
std::string shape(const Tensor& t) {std::ostringstream s;s<<'[';for(auto n:t.shape())s<<n<<',';return s.str()+']';}
Tensor host(Backend& b,const Tensor& t) {return t.device().is_host()?t:b.copy_to_host(t);}
TensorBundle capture(Backend& b,const TensorBundle& x) {
    TensorBundle h;for(const auto& [n,t]:x)h[n]=host(b,t);return h;
}
double number(const Tensor& t,int64_t i) {
    if(t.dtype()==DType::F32)return t.data_as<float>()[i];
    require(t.dtype()==DType::BF16,"numeric capture dtype");
    uint32_t bits=uint32_t(t.data_as<uint16_t>()[i])<<16;float f;std::memcpy(&f,&bits,4);return f;
}
void compare(const Tensor& a,const Tensor& e,const std::string& label,const std::string& name,std::ostream* out) {
    require(a.shape()==e.shape()&&a.dtype()==e.dtype(),"checkpoint metadata mismatch: "+name);
    double maximum=0,sum=0,relative=0,dot=0,aa=0,ee=0;size_t nonfinite=0,exceed=0;
    for(int64_t i=0;i<a.numel();++i) {
        double av=number(a,i),ev=number(e,i),err=std::abs(av-ev);
        maximum=std::max(maximum,err);sum+=err;
        relative=std::max(relative,ev==0?(err==0?0:INFINITY):err/std::abs(ev));
        dot+=av*ev;aa+=av*av;ee+=ev*ev;
        nonfinite+=!std::isfinite(av)||!std::isfinite(ev);exceed+=err>2e-5+2e-5*std::abs(ev);
    }
    bool exact=a.bytes()==e.bytes()&&std::memcmp(a.data(),e.data(),a.bytes())==0;
    if(out)*out<<label<<'\t'<<name<<'\t'<<shape(a)<<'\t'<<dtype_name(a.dtype())<<'\t'<<maximum<<'\t'
        <<sum/a.numel()<<'\t'<<relative<<'\t'<<(aa*ee==0?(exact?1:0):dot/std::sqrt(aa*ee))<<'\t'
        <<nonfinite<<'\t'<<exact<<'\t'<<(nonfinite==0&&exceed==0&&exact?"PASS":"FAIL")<<std::endl;
    require(nonfinite==0&&exceed==0&&exact,"frozen numerical gate: "+label+" / "+name);
}
void qualify(Backend& b,const PrecisionPolicy& policy,const WeightMap& weights,const TensorBundle& x,
             const std::string& label,const std::string& dir,int repeats) {
    auto graph=wan::self_attention_graph(1,x.at("hidden").dim(1),b.execution_dtype(),x.at("hidden").dtype(),true);
    auto misses=b.weight_cache_misses(),hits=b.weight_cache_hits(),resident=b.weight_cache_resident_bytes();
    auto p=wan::bind_parameters(b,graph,weights);
    auto cold=b.weight_cache_misses()-misses;
    auto again=wan::bind_parameters(b,graph,weights);
    for(size_t i=0;i<p.size();++i)require(p[i].data()==again[i].data(),"parameter cache pointer drift");
    std::vector<Tensor> parameter_before;for(const auto& t:p)parameter_before.push_back(host(b,t));
    std::ofstream bindings(dir+"/parameter-bindings.tsv",std::ios::app);
    for(size_t i=0;i<p.size();++i)bindings<<label<<'\t'<<wan::parameters[i]<<'\t'<<shape(p[i])
        <<"\tF32\t"<<dtype_name(p[i].dtype())<<'\t'<<weights.at(wan::parameters[i]).data()<<'\t'<<p[i].data()<<"\tREUSED\n";
    std::cout<<"CACHE\t"<<label<<'\t'<<cold<<'\t'<<b.weight_cache_hits()-hits<<'\t'
        <<b.weight_cache_resident_bytes()<<'\t'<<b.weight_cache_resident_bytes()-resident<<std::endl;
    write_bundle(dir+"/"+label+"-boundary.bundle",capture(b,x));
    std::ofstream results(dir+"/checkpoint-results.tsv",std::ios::app);results<<std::setprecision(17);
    TensorBundle ref_first,graph_first;
    double rt=0,gt=0;size_t free=0,total=0;cudaMemGetInfo(&free,&total);size_t before=total-free,peak=before;
    for(int repeat=0;repeat<repeats;++repeat) {
        auto start=Clock::now();auto ref=wan_test::reference(b,policy,weights,x);b.synchronize();
        rt+=std::chrono::duration<double>(Clock::now()-start).count();
        // Additional diagnostic only, outside timed unchanged imperative slice.
        ref["gated"]=b.mul(b.cast(ref.at("self_attention.output"),DType::F32),ref.at("gate"));
        start=Clock::now();auto warm_misses=b.weight_cache_misses();
        auto values=wan::self_attention(b,policy,weights,x.at("hidden"),x.at("modulation"),x.at("cosine"),x.at("sine"),true);
        gt+=std::chrono::duration<double>(Clock::now()-start).count();
        require(b.weight_cache_misses()==warm_misses,"warm Graph parameter reupload");
        cudaMemGetInfo(&free,&total);peak=std::max(peak,total-free);
        TensorBundle actual;for(size_t i=0;i<values.size();++i)actual[wan::checkpoints[i]]=host(b,values[i]);
        auto expected=capture(b,ref);
        for(const auto* name:wan::checkpoints) {
            compare(actual.at(name),expected.at(name),label,name,repeat==0?&results:nullptr);
            if(repeat) {
                compare(actual.at(name),graph_first.at(name),label,"graph_repeat",nullptr);
                compare(expected.at(name),ref_first.at(name),label,"imperative_repeat",nullptr);
            }
        }
        if(!repeat){ref_first=expected;graph_first=actual;}
    }
    b.enable_profiling(true);
    auto normal=wan::self_attention(b,policy,weights,x.at("hidden"),x.at("modulation"),x.at("cosine"),x.at("sine"));
    std::map<std::string,uint64_t> counts;
    for(const auto& [name,stat]:b.profile_stats())counts[name.substr(0,name.find('|'))]+=stat.calls;
    b.enable_profiling(false);
    require(counts["elementwise.add"]==4&&counts["elementwise.mul"]==2&&counts["slice"]==3,
            "Graph modulation ownership dispatch count");
    for(size_t i=0;i<p.size();++i)compare(host(b,p[i]),parameter_before[i],label,"immutable_parameter",nullptr);
    std::cout<<"OWNERSHIP\t"<<label<<"\t4_add_2_mul_3_slice\timmutable_parameters\tzero_warm_misses\tPASS\n";
    compare(host(b,normal[0]),ref_first.at("first_residual"),label,"normal_first_residual",&results);
    compare(host(b,normal[1]),ref_first.at("combined"),label,"normal_combined",&results);
    // Verify the complete production Graph block against the isolated imperative reference.
    TensorBundle original;
    (void)block(b,policy,weights,x.at("hidden"),x.at("modulation"),x.at("context"),x.at("cosine"),x.at("sine"),&original,"original");
    compare(host(b,original.at("original.first_residual")),ref_first.at("first_residual"),label,"unchanged_block_reference",&results);
    write_bundle(dir+"/"+label+"-imperative.bundle",ref_first);
    write_bundle(dir+"/"+label+"-graph.bundle",graph_first);
    cudaMemGetInfo(&free,&total);rusage usage{};getrusage(RUSAGE_SELF,&usage);
    peak=std::max(peak,total-free);
    std::ofstream resources(dir+"/resource-usage.tsv",std::ios::app);
    resources<<label<<'\t'<<before<<'\t'<<peak<<'\t'<<total-free<<'\t'<<b.peak_device_bytes()<<'\t'
        <<usage.ru_maxrss*1024L<<'\t'<<rt/repeats<<'\t'<<gt/repeats<<'\t'<<repeats<<'\n';
    std::cout<<"CASE\t"<<label<<'\t'<<dtype_name(x.at("hidden").dtype())<<'\t'<<repeats<<"\tPASS\n"<<std::flush;
}
struct CaptureDenoiser final:Denoiser {
    Backend& b;Denoiser& original;int index=0;std::map<int,TensorBundle> states;
    CaptureDenoiser(Backend& b,Denoiser& d):b(b),original(d){}
    std::vector<Tensor> evaluate(const Tensor& x,const Tensor& t) override {
        if(index==0||index==42||index==49)states[index]={{"latent",host(b,x)},{"timestep",host(b,t)}};
        ++index;return original.evaluate(x,t);
    }
};
void run(int argc,char** argv) {
    require(argc==6,"usage: TEST MODEL INPUT POLICY EVIDENCE MODE(all/f32/bf16)");
    const std::string dir=argv[4],mode=argv[5];
    VrmModel model(argv[1]);require(model.architecture_id()=="wan","real Wan identity");
    auto input=read_bundle(argv[2]);input["sampling_steps"]=scalar_i64(50);
    input["latent_shape"]=host_i64({5},{1,16,2,4,4}); // Existing bounded production contract, S=8, all RoPE axes.
    std::ifstream pf(argv[3]);std::stringstream ps;ps<<pf.rdbuf();auto mixed=PrecisionPolicy::from_json(Json::parse(ps.str()));
    WeightMap weights(model.bindings(model.graph().at("architecture_graph")));
    if(mode=="cache") {
        CudaBackend b;b.set_execution_dtype(DType::BF16);
        const auto& source=weights.at("blocks.0.self_attn.q.bias");
        auto f=b.copy_to_device(source,DType::F32),h=b.copy_to_device(source,DType::BF16);
        require(f.data()!=h.data()&&b.copy_to_device(source,DType::F32).data()==f.data()&&
                b.copy_to_device(source,DType::BF16).data()==h.data(),"stale dtype cache collision");
        compare(host(b,f),source,"cache","exact_F32_binding",nullptr);
        compare(host(b,h),host(b,b.cast(f,DType::BF16)),"cache","exact_BF16_binding",nullptr);
        require(b.weight_cache_misses()==2&&b.weight_cache_hits()==2&&
                b.weight_cache_resident_bytes()==static_cast<size_t>(source.numel())*6,"dtype cache accounting");
        std::cout<<"DTYPE_CACHE\t2_misses\t2_hits\t"<<b.weight_cache_resident_bytes()
            <<"\tF32_pointer="<<f.data()<<"\tBF16_pointer="<<h.data()<<"\tPASS\n";
        return;
    }
    for(auto dtype:{DType::F32,DType::BF16}) {
        if((mode=="f32"&&dtype!=DType::F32)||(mode=="bf16"&&dtype!=DType::BF16))continue;
        CudaBackend b;b.set_execution_dtype(dtype);auto policy=dtype==DType::F32?PrecisionPolicy::fp32():mixed;
        auto architecture=make_wan_test_architecture(model);auto program=architecture->create_program(input);
        RngState rng{program.seed,0,"pytorch_compat.v1"};
        auto latent=b.rng_normal(rng,program.latent_shape,policy.persistent_state_dtype(PrecisionSemantic::SamplingState));
        auto context=b.copy_to_device(input.at("positive"),policy.boundary_dtype(PrecisionSemantic::Conditioning));
        auto initial=wan_test::producer(b,policy,weights,latent,program.model_timestep_at(0),context);
        const std::string prefix=dtype==DType::F32?"f32":"bf16";
        qualify(b,policy,weights.prefix("blocks.0."),initial,prefix+"-early-block0",dir,10);
        if(mode!="all")continue;
        if(dtype==DType::F32)continue;
        // Local middle/late checkpoints gate the broader real sampling capture.
        auto state=initial;
        for(int i=0;i<29;++i) {
            state["hidden"]=block(b,policy,weights.prefix("blocks."+std::to_string(i)+"."),state.at("hidden"),
                state.at("modulation"),state.at("context"),state.at("cosine"),state.at("sine"));
            if(i+1==15||i+1==29)qualify(b,policy,weights.prefix("blocks."+std::to_string(i+1)+"."),state,
                prefix+"-early-block"+std::to_string(i+1),dir,10);
        }
        auto production=architecture->create_denoiser(b,policy,input);CaptureDenoiser recorder(b,*production);
        SamplingRuntime sampling(b,policy);auto sampled=sampling.run(recorder,program);
        require(recorder.index==50,"production sampling call count");
        std::ofstream timesteps(dir+"/timestep-manifest.tsv",std::ios::app);
        for(const auto& [step,boundary]:recorder.states) {
            timesteps<<step<<'\t'<<read_scalar_i64(boundary.at("timestep"))<<"\tunchanged 50-step SamplingRuntime CFG B1 branches\n";
            write_bundle(dir+"/sampling-step"+std::to_string(step)+".bundle",boundary);
            if(step==0)continue;
            auto x=wan_test::producer(b,policy,weights,b.copy_to_device(boundary.at("latent"),latent.dtype()),boundary.at("timestep"),context);
            const std::string level=step==42?"middle":"late";
            for(int i=0;i<30;++i) {
                if(i==0||i==15||i==29)qualify(b,policy,weights.prefix("blocks."+std::to_string(i)+"."),x,
                    prefix+"-"+level+"-block"+std::to_string(i),dir,10);
                if(i<29)x["hidden"]=block(b,policy,weights.prefix("blocks."+std::to_string(i)+"."),x.at("hidden"),
                    x.at("modulation"),x.at("context"),x.at("cosine"),x.at("sine"));
            }
        }
    }
}
#endif
}
int main(int argc,char** argv) {
    try {std::cout<<std::setprecision(17);negatives();
#ifdef VRHINO_TEST_CUDA
        run(argc,argv);
#else
        (void)argc;(void)argv;
#endif
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<std::endl;return 1;}
}
