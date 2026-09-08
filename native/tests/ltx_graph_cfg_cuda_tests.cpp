#include "ltx_self_attention_graph.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/loader.h"
#include "vrhino/tensor_util.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <sys/resource.h>

namespace {
std::string frozen_dir;
using namespace vrhino;
namespace ng = neural_graph;
namespace ltx = ltx_internal;
using Clock = std::chrono::steady_clock;
size_t peak_used = 0, graph_calls = 0;
std::ostream* batch_manifest = nullptr;
std::ofstream resources;
std::string shape(const Tensor& t) {
    std::ostringstream s; s << '[';
    for (size_t i=0;i<t.shape().size();++i) s << (i ? "," : "") << t.shape()[i];
    return s.str()+']';
}
Tensor capture(Backend& b,const Tensor& t) {
    cudaPointerAttributes a{};
    require(t.dtype()==DType::F32 && t.logical_dtype()==DType::F32 && !t.is_quantized() &&
            t.device()==DeviceId::accelerator() &&
            cudaPointerGetAttributes(&a,t.data())==cudaSuccess && a.type==cudaMemoryTypeDevice,
            "capture requires real CUDA F32");
    return b.copy_to_host(t);
}
void compare(const Tensor& actual,const Tensor& expected,std::ostream* report,
             const std::string& label,const std::string& checkpoint,const std::string& path) {
    require(actual.shape()==expected.shape() && actual.dtype()==DType::F32 &&
            expected.dtype()==DType::F32,"comparison metadata");
    double max_abs=0,sum=0,relative=0,dot=0,aa=0,ee=0,gate_ratio=0;
    size_t nonfinite=0,exceedances=0;
    for (int64_t i=0;i<actual.numel();++i) {
        double a=actual.data_as<float>()[i],e=expected.data_as<float>()[i],err=std::abs(a-e);
        nonfinite+=!std::isfinite(a) || !std::isfinite(e);
        max_abs=std::max(max_abs,err); sum+=err;
        relative=std::max(relative,e==0 ? (err==0 ? 0 : INFINITY) : err/std::abs(e));
        gate_ratio=std::max(gate_ratio,err/(2e-5+2e-5*std::abs(e)));
        exceedances+=err>2e-5+2e-5*std::abs(e);
        dot+=a*e; aa+=a*a; ee+=e*e;
    }
    const bool bitwise=actual.bytes()==expected.bytes() &&
        std::memcmp(actual.data(),expected.data(),actual.bytes())==0;
    const double cosine=aa*ee==0 ? (bitwise ? 1 : 0) : dot/std::sqrt(aa*ee);
    if (report) *report << label << '\t' << checkpoint << '\t' << path << '\t'
        << shape(actual) << "\tF32\t" << max_abs << '\t' << sum/actual.numel() << '\t'
        << relative << '\t' << cosine << '\t' << nonfinite << '\t' << gate_ratio << '\t'
        << exceedances << '\t' << (nonfinite==0 && exceedances==0 ? "PASS" : "FAIL")
        << '\t' << (bitwise ? "PASS" : "FAIL") << '\n';
    require(nonfinite==0 && exceedances==0 && bitwise,"frozen Class 2 qualification failed");
}
TensorBundle capture_bundle(Backend& b,const TensorBundle& source) {
    TensorBundle host;
    for(const auto& [name,t]:source)
        host[name]=t.device().is_host() ? t : b.copy_to_host(t);
    return host;
}
void exact_bundle(Backend& b,const TensorBundle& actual,const TensorBundle& expected,
                  std::ostream* report,const std::string& label,const std::string& path) {
    require(actual.size()==expected.size(),"trace key count changed");
    for(const auto& [name,t]:actual) {
        const Tensor h=t.device().is_host() ? t : b.copy_to_host(t);
        const Tensor& e=expected.at(name);
        if(h.dtype()==DType::F32) compare(h,e,report,label,name,path);
        else require(h.shape()==e.shape() && h.dtype()==e.dtype() && h.bytes()==e.bytes() &&
                     std::memcmp(h.data(),e.data(),h.bytes())==0,"discrete binding changed");
    }
}
Tensor half(const Tensor& t,int index) {
    require(t.device().is_host() && t.ndim()>0 && t.dim(0)==2,"half requires host B2");
    auto shape=t.shape(); shape[0]=1;
    auto out=Tensor::host(shape,t.dtype());
    std::memcpy(out.data(),static_cast<const uint8_t*>(t.data())+index*out.bytes(),out.bytes());
    return out;
}
void resource(const std::string& label,const std::string& phase,Backend* b=nullptr) {
    size_t free=0,total=0; require(cudaMemGetInfo(&free,&total)==cudaSuccess,"CUDA resource query");
    peak_used=std::max(peak_used,total-free);
    rusage usage{}; require(getrusage(RUSAGE_SELF,&usage)==0,"RSS query");
    uint64_t rss=0; std::ifstream status("/proc/self/status"); std::string line;
    while(std::getline(status,line)) if(line.rfind("VmRSS:",0)==0) {
        std::istringstream value(line.substr(6)); value>>rss;
    }
    resources<<label<<'\t'<<phase<<'\t'<<total-free<<'\t'<<peak_used<<'\t'
        <<(b?b->peak_device_bytes():0)<<'\t'<<rss*1024<<'\t'<<usage.ru_maxrss*1024<<'\n';
    resources.flush();
}
using Counts=std::map<std::string,uint64_t>;
Counts counts(Backend& b) {
    Counts c;
    for(const auto& [name,stat]:b.profile_stats()) c[name.substr(0,name.find('|'))]+=stat.calls;
    return c;
}
struct Run { TensorBundle trace; double seconds; uint64_t hits,misses; size_t calls; };
Run forward(Backend& b,Denoiser& d,const Tensor& latent,const Tensor& sigma,
            const std::string& label,bool graph,bool profile,std::ostream& cache) {
    const auto hits=b.weight_cache_hits(),misses=b.weight_cache_misses();
    Counts before;
    if(profile) { before=counts(b); b.enable_profiling(true); }
    graph_calls=0;
    auto start=Clock::now();
    auto out=d.evaluate(latent,sigma); b.synchronize();
    const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
    const auto executed=graph_calls;
    require(executed==(graph?28u:0u),"normal denoiser did not execute requested mode");
    require(out.size()==2 && out[0].shape()==std::vector<int64_t>({1,4,128}) &&
            out[1].shape()==out[0].shape(),"normal CFG return contract");
    auto first=reinterpret_cast<uintptr_t>(out[0].data()),second=reinterpret_cast<uintptr_t>(out[1].data());
    require(first+out[0].bytes()<=second || second+out[1].bytes()<=first,"CFG output buffers overlap");
    auto trace=capture_bundle(b,d.take_trace());
    trace["prediction.0"]=capture(b,out[0]); trace["prediction.1"]=capture(b,out[1]);
    if(profile) {
        auto after=counts(b);
        for(const auto& [name,want]:Counts{{"elementwise.add",227},{"slice",172}}) {
            const auto count=after[name]-before[name];
            std::cout<<"OWNERSHIP\t"<<label<<'\t'<<name<<'\t'<<count<<'\n';
            require(count==want,"duplicate/missing CFG preparation");
        }
        b.enable_profiling(false);
    }
    const auto hit_delta=b.weight_cache_hits()-hits,miss_delta=b.weight_cache_misses()-misses;
    cache<<label<<'\t'<<executed<<'\t'<<hit_delta<<'\t'<<miss_delta<<'\t'
         <<b.weight_cache_resident_bytes()<<'\t'<<seconds<<'\n';
    resource(label,"after_forward",&b);
    return {std::move(trace),seconds,hit_delta,miss_delta,executed};
}
void normal_contract(const TensorBundle& trace,const Tensor& sigma,std::ostream* results,const std::string& label) {
    const std::map<std::string,std::vector<int64_t>> expected={
        {"denoiser.latent",{2,4,128}},{"denoiser.sigma",{2,1}},
        {"denoiser.cosine",{2,4,2048}},{"denoiser.sine",{2,4,2048}},
        {"denoiser.text",{2,2,4096}},{"conditioning.modulation",{2,1,12288}},
        {"conditioning.embedded",{2,1,2048}},{"output_projection",{2,4,128}}};
    for(const auto& [name,shape]:expected)
        require(trace.at(name).shape()==shape,"normal orchestration batch shape: "+name);
    compare(trace.at("denoiser.scalar_sigma"),sigma,results,label,"caller_scalar","identity");
    compare(half(trace.at("denoiser.sigma"),0),sigma,results,label,"expanded_sigma_value","identity");
    for(const auto* name:{"denoiser.latent","denoiser.sigma","denoiser.cosine","denoiser.sine",
                          "conditioning.modulation","conditioning.embedded"})
        compare(half(trace.at(name),0),half(trace.at(name),1),results,label,name,"intentional_shared_CFG_value");
    for(int block:{0,14,27}) {
        const auto prefix="block."+std::to_string(block)+".";
        require(trace.at(prefix+"modulation_table").shape()==std::vector<int64_t>({2,1,6,2048}),"B2 combined table shape");
        for(const auto* suffix:{"modulation_table","shift","scale","gate","ffn_shift","ffn_scale","ffn_gate"})
            compare(half(trace.at(prefix+suffix),0),half(trace.at(prefix+suffix),1),results,label,prefix+suffix,"shared_sigma_modulation");
    }
}
TensorBundle run_case(Backend& b,const VrmModel& model,const TensorBundle& input,
    const Tensor& latent,const Tensor& sigma,const std::string& label,int repeats,
    const std::string& dir,std::ostream& results,std::ostream& cases,std::ostream& blocks,std::ostream& cache,
    std::ostream& batch) {
    const auto policy=PrecisionPolicy::fp32();
    // Normal production factory; reference bundles were frozen before migration.
    auto architecture=create_architecture(model);
    auto graph=architecture->create_denoiser(b,policy,input);
    auto latent_before=capture(b,latent);
    resource(label,"before",&b);
    if(repeats==10)
        (void)forward(b,*graph,latent,sigma,label+".warm.production",true,false,cache);
    const auto ref_first=read_bundle(frozen_dir+"/ltx-cfg/"+label+"-imperative.bundle");
    TensorBundle graph_first;
    double gtime=0;
    for(int repeat=0;repeat<repeats;++repeat) {
        const auto tag=label+"."+std::to_string(repeat);
        if(repeats==10&&repeat==0) batch_manifest=&batch;
        auto actual=forward(b,*graph,latent,sigma,tag+".production",true,repeats==10&&repeat==0,cache);
        batch_manifest=nullptr;
        if(repeats==10) require(actual.misses==0,"warm full forward rematerialized model parameters");
        gtime+=actual.seconds;
        exact_bundle(b,actual.trace,ref_first,repeat==0?&results:nullptr,label,"frozen_reference");
        normal_contract(actual.trace,sigma,repeat==0?&results:nullptr,label);
        compare(capture(b,latent),latent_before,nullptr,label,"latent","immutable");
        if(repeat==0) graph_first=actual.trace;
        else exact_bundle(b,actual.trace,graph_first,nullptr,label,"production_repeat");
    }
    if(repeats==10) {
        graph_calls=0; bool rejected=false;
        try { (void)graph->evaluate(latent,host_f32({2,1},{.75f,.5f})); }
        catch(const Error& e) { rejected=std::string(e.what()).find("Expected CPU f32 scalar")!=std::string::npos; }
        require(rejected && graph_calls==0,"unsupported per-batch sigma was accepted");
        auto recovered=forward(b,*graph,latent,sigma,label+".recovery",true,false,cache);
        exact_bundle(b,recovered.trace,graph_first,&results,label,"scalar_recovery");
        std::cout<<"BATCHED_SIGMA\tUNSUPPORTED_REJECTED\n";
    }
    write_bundle(dir+"/"+label+"-graph.bundle",graph_first);
    for(int block:{0,14,27}) {
        size_t n=0;const auto prefix="block."+std::to_string(block)+".";
        for(const auto& [name,t]:ref_first) { (void)t; n+=name.rfind(prefix,0)==0; }
        require(n==43,"required real block checkpoints missing");
        blocks<<label<<'\t'<<block<<'\t'<<n<<"\tPASS\tPASS\n";
    }
    cases<<label<<"\t2\t4\t"<<sigma.data_as<float>()[0]<<'\t'<<repeats<<'\t'<<gtime<<"\tPASS\tPASS\n";
    resource(label,"after",&b);
    std::cout<<"CASE_PASS\t"<<label<<"\trepeats="<<repeats<<"\ttrace_tensors="<<ref_first.size()<<std::endl;
    return graph_first;
}
void independence(Backend& b,const TensorBundle& control,const TensorBundle& original,
                  int actual_half,int original_half,const std::string& label,std::ostream& results) {
    size_t checked=0;
    for(const auto& [name,t]:original) if(t.ndim()>0 && t.dim(0)==2 && t.dtype()==DType::F32) {
        compare(half(control.at(name),actual_half),half(t,original_half),&results,label,name,"half_independence");
        ++checked;
    }
    const auto a="prediction."+std::to_string(actual_half),e="prediction."+std::to_string(original_half);
    compare(control.at(a),original.at(e),&results,label,"returned_prediction","half_independence");
    (void)b;
    std::cout<<"INDEPENDENCE_PASS\t"<<label<<"\tcheckpoints="<<checked+1<<'\n';
}
void require_changed(const TensorBundle& control,const TensorBundle& original,int index) {
    for(const auto* name:{"conditioning.context","block.14.input","output_projection"}) {
        auto a=half(control.at(name),index),e=half(original.at(name),index);
        require(std::memcmp(a.data(),e.data(),a.bytes())!=0,"conditioning perturbation was vacuous");
    }
}
void baseline_b1(Backend& b,const VrmModel& model,const TensorBundle& prepared,
                 std::ostream& cases,std::ostream& results) {
    (void)prepared;
    WeightMap w(model.bindings(model.graph().at("architecture_graph")));
    const auto input=read_bundle(frozen_dir+"/ltx-optin/forward-input-S3-M1.bundle");
    const auto ref=read_bundle(frozen_dir+"/ltx-optin/forward-output-S3-M1.bundle");
    graph_calls=0;auto start=Clock::now();
    auto actual=ltx::denoiser_forward(b,PrecisionPolicy::fp32(),w,input.at("latent"),
        input.at("cosine"),input.at("sine"),input.at("text"),input.at("mask"),input.at("sigma"));
    b.synchronize();double seconds=std::chrono::duration<double>(Clock::now()-start).count();
    require(graph_calls==28,"B1 production Graph route");
    compare(capture(b,actual),ref.at("imperative"),&results,"A_B1","final_output","frozen_reference");
    cases<<"A_B1\t1\t3\t"<<input.at("sigma").data_as<float>()[0]<<"\t1\t"<<seconds<<"\tPASS\tPASS\n";

}
} // namespace
// ELF observation only; no replacement implementation and no Runtime changes.
std::vector<vrhino::Tensor> real_graph_evaluate(vrhino::Backend&,const vrhino::neural_graph::Graph&,
    const std::vector<vrhino::Tensor>&,const std::vector<vrhino::Tensor>&)
    asm("__real__ZN6vrhino12neural_graph8evaluateERNS_7BackendERKNS0_5GraphERKSt6vectorINS_6TensorESaIS7_EESB_");
std::vector<vrhino::Tensor> observed_graph_evaluate(vrhino::Backend& b,const vrhino::neural_graph::Graph& g,
    const std::vector<vrhino::Tensor>& x,const std::vector<vrhino::Tensor>& p)
    asm("__wrap__ZN6vrhino12neural_graph8evaluateERNS_7BackendERKNS0_5GraphERKSt6vectorINS_6TensorESaIS7_EESB_");
std::vector<vrhino::Tensor> observed_graph_evaluate(vrhino::Backend& b,const vrhino::neural_graph::Graph& g,
    const std::vector<vrhino::Tensor>& x,const std::vector<vrhino::Tensor>& p) {
    const auto block=graph_calls++;
    vrhino::require(g.description().nodes.size()==28 && p.size()==11,"unexpected integrated Graph contract");
    if(batch_manifest) for(size_t id=0;id<g.description().values.size();++id) {
        const auto& spec=g.description().values[id];
        *batch_manifest<<block<<'\t'<<id<<'\t'<<(id<4?"input":id<15?"parameter":id==15?"constant":"node")<<'\t';
        for(auto dim:spec.shape) *batch_manifest<<dim<<',';
        *batch_manifest<<'\t'<<vrhino::dtype_name(spec.dtype)<<'\n';
    }
    return real_graph_evaluate(b,g,x,p);
}
int main(int argc,char** argv) {
    try {
        require(argc==5,"usage: ltx-graph-cfg-cuda-tests MODEL INPUT EVIDENCE_DIRECTORY FROZEN_QUALIFICATION_DIRECTORY");
        frozen_dir=argv[4];
        std::string dir=argv[3];std::cout<<std::setprecision(17);
        resources.open(dir+"/resource-usage.tsv");require(resources.good(),"resource output unavailable");
        resources<<"case\tphase\tcuda_used_bytes\tobserved_peak_bytes\tbackend_peak_bytes\thost_rss_bytes\thost_peak_rss_bytes\n";
        resource("process","before_backend");
        VrmModel model(argv[1]); auto fixture=read_bundle(argv[2]);
        require(model.architecture_id()=="ltx_v0_9_1","fixture identity");
        fixture["audit_trace"]=scalar_i64(1);
        const auto positive=fixture.at("positive"),negative=fixture.at("negative");
        require(positive.shape()==negative.shape() && std::memcmp(positive.data(),negative.data(),positive.bytes())!=0,"distinct CFG fixture required");
        std::ofstream results(dir+"/checkpoint-results.tsv"),cases(dir+"/case-results.tsv"),blocks(dir+"/block-position-results.tsv"),cache(dir+"/cache-results.tsv"),batch(dir+"/batch-contract.tsv");
        require(results.good() && cases.good() && blocks.good() && cache.good() && batch.good(),"evidence streams unavailable");
        results<<std::setprecision(17)<<"case\tcheckpoint\tcomparison\tshape\tdtype\tmax_abs\tmean_abs\tmax_relative\tcosine\tnonfinite\tmax_gate_ratio\texceedances\tdirect_criterion\tbitwise\n";
        cases<<std::setprecision(17)<<"case\tB\tS\tsigma\trepeats\tproduction_seconds\tdirect_criterion\tbitwise\n";
        blocks<<"case\tblock\tcheckpoints\tdirect_criterion\tbitwise\n";
        cache<<std::setprecision(17)<<"run\tgraph_calls\tparameter_cache_hits\tparameter_cache_misses\tresident_weight_bytes\tseconds\n";
        batch<<"block\tvalue_id\tkind\tshape\tdtype\n";
        {
            CudaBackend b;b.set_execution_dtype(DType::F32);
            const auto architecture=create_architecture(model);const auto program=architecture->create_program(fixture);
            RngState rng{program.seed,0,"pytorch_compat.v1"};auto latent=b.rng_normal(rng,program.latent_shape,DType::F32);
            require(latent.shape()==std::vector<int64_t>({1,4,128}),"existing canary shape required");
            const auto sigma=program.model_timesteps.at(0),sigma2=program.model_timesteps.at(1);
            require(sigma.shape()==std::vector<int64_t>({1,1}) && read_scalar_f32(sigma)!=read_scalar_f32(sigma2),"normal scalar schedule required");
            write_bundle(dir+"/caller-input.bundle",{{"latent",capture(b,latent)},{"sigma",sigma},{"sigma2",sigma2},{"positive",positive},{"negative",negative}});
            TensorBundle pp=fixture;pp["negative"]=positive;
            auto same=run_case(b,model,pp,latent,sigma,"B_PP",1,dir,results,cases,blocks,cache,batch);
            baseline_b1(b,model,same,cases,results);
            auto distinct=run_case(b,model,fixture,latent,sigma,"C_NP",10,dir,results,cases,blocks,cache,batch);
            auto second=run_case(b,model,fixture,latent,sigma2,"D_NP_SECOND_SIGMA",1,dir,results,cases,blocks,cache,batch);
            for(const auto* name:{"conditioning.modulation","block.14.modulation_table","output_projection"})
                require(std::memcmp(second.at(name).data(),distinct.at(name).data(),second.at(name).bytes())!=0,
                        "second scalar sigma did not reach the denoiser state");
            TensorBundle nn=fixture;nn["positive"]=negative;
            auto same_negative=run_case(b,model,nn,latent,sigma,"CONTROL_NN",1,dir,results,cases,blocks,cache,batch);
            TensorBundle swapped=fixture;swapped["positive"]=negative;swapped["negative"]=positive;
            auto reverse=run_case(b,model,swapped,latent,sigma,"CONTROL_PN",1,dir,results,cases,blocks,cache,batch);
            independence(b,same,distinct,1,1,"negative_changed_positive_held",results);require_changed(same,distinct,0);
            independence(b,same_negative,distinct,0,0,"positive_changed_negative_held",results);require_changed(same_negative,distinct,1);
            independence(b,reverse,distinct,0,1,"swap_half0",results);
            independence(b,reverse,distinct,1,0,"swap_half1",results);
            independence(b,same,same,0,1,"identical_conditioning_halves",results);
            resource("process","before_backend_destroy",&b);
        }
        resource("process","after_backend_destroy");
        std::cout<<"QUALIFICATION_PASS\tNORMAL_CFG_B2_SCALAR\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n';return 1; }
}
