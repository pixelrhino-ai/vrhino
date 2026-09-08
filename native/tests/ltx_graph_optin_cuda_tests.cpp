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

namespace {
using namespace vrhino;
namespace ng = neural_graph;
namespace ltx = ltx_internal;
using Clock = std::chrono::steady_clock;
size_t peak_used = 0;
std::string frozen_dir;
std::string shape(const Tensor& t) {
    std::ostringstream s; s << '[';
    for (size_t i=0;i<t.shape().size();++i) s << (i ? "," : "") << t.shape()[i];
    return s.str()+']';
}
void resource(Backend& b) {
    size_t free=0,total=0;
    require(cudaMemGetInfo(&free,&total)==cudaSuccess,"CUDA memory query");
    peak_used=std::max(peak_used,total-free);
    std::cout << "MEMORY\t" << total-free << '\t' << peak_used << '\t' << b.peak_device_bytes() << '\n';
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
             int s,int m,int block,const std::string& checkpoint,const std::string& path) {
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
    if (report) *report << s << '\t' << m << '\t' << block << '\t' << checkpoint << '\t' << path << '\t'
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
                  std::ostream* report,int s,int m,int block,const std::string& path) {
    require(actual.size()==expected.size(),"trace key count changed");
    for(const auto& [name,t]:actual) {
        const Tensor h=t.device().is_host() ? t : b.copy_to_host(t);
        const Tensor& e=expected.at(name);
        if(h.dtype()==DType::F32) compare(h,e,report,s,m,block,name,path);
        else require(h.shape()==e.shape() && h.dtype()==e.dtype() && h.bytes()==e.bytes() &&
                     std::memcmp(h.data(),e.data(),h.bytes())==0,"discrete binding changed");
    }
}
using Counts=std::map<std::string,uint64_t>;
Counts counts(Backend& b) {
    Counts c;
    for(const auto& [name,stat]:b.profile_stats()) c[name.substr(0,name.find('|'))]+=stat.calls;
    return c;
}
void ownership_counts(Backend& b,const Counts& before,int block,const char* mode) {
    const auto after=counts(b);
    for(const auto& [name,want]:std::map<std::string,uint64_t>{{"elementwise.add",8},{"slice",6}}) {
        const auto old=before.find(name),now=after.find(name);
        uint64_t count=(now==after.end()?0:now->second)-(old==before.end()?0:old->second);
        std::cout<<"OWNERSHIP\t"<<block<<'\t'<<mode<<'\t'<<name<<'\t'<<count<<'\n';
        require(count==want,"duplicate or missing block modulation computation");
    }
    b.enable_profiling(false);
}
Tensor block_run(Backend& b,const WeightMap& w,const TensorBundle& boundary,
                 TensorBundle* trace=nullptr) {
    return ltx::transformer_block_forward(b,PrecisionPolicy::fp32(),w,
        boundary.at("input"),boundary.at("context"),boundary.at("timestep"),
        boundary.at("cosine"),boundary.at("sine"),boundary.at("mask"),trace);
}
void qualify_block(Backend& b,const WeightMap& all_weights,const TensorBundle& boundary,
                   int block,int s,int m,const std::string& dir,std::ostream& results,
                   std::ostream& positions,std::ostream& manifest) {
    const auto weights=all_weights.prefix("transformer_blocks."+std::to_string(block)+".");
    const std::string tag="S"+std::to_string(s)+"-M"+std::to_string(m)+"-block"+std::to_string(block);
    const auto frozen=capture_bundle(b,boundary);
    write_bundle(dir+"/boundary-"+tag+".bundle",frozen);
    for(const auto& [name,t]:frozen)
        manifest<<s<<'\t'<<m<<'\t'<<block<<'\t'<<name<<'\t'<<shape(t)<<'\t'<<dtype_name(t.dtype())
                <<"\tactual_native_full_denoiser_pre_block\tboundary-"<<tag<<".bundle\n";
    const auto ref_first=read_bundle(frozen_dir+"/imperative-"+tag+".bundle");
    TensorBundle graph_first;
    double graph_time=0;
    for(int repeat=0;repeat<10;++repeat) {
        TensorBundle graph;
        const auto misses=b.weight_cache_misses();
        Counts before;
        if(repeat==0) { before=counts(b); b.enable_profiling(true); }
        auto start=Clock::now();
        auto actual=block_run(b,weights,boundary,&graph);
        b.synchronize(); graph_time+=std::chrono::duration<double>(Clock::now()-start).count();
        if(repeat==0) ownership_counts(b,before,block,"production_graph");
        require(b.weight_cache_misses()==misses,"replay unnecessarily rematerialized model weights");
        auto graph_host=capture_bundle(b,graph);
        exact_bundle(b,graph_host,ref_first,repeat==0?&results:nullptr,s,m,block,"frozen_reference");
        compare(capture(b,actual),ref_first.at("block.0.output"),repeat==0?&results:nullptr,s,m,block,"block_return","frozen_reference");
        if(repeat==0) graph_first=graph_host;
        else exact_bundle(b,graph_host,graph_first,nullptr,s,m,block,"production_repeat");
        exact_bundle(b,boundary,frozen,repeat==0?&results:nullptr,s,m,block,"boundary_immutable");
    }
    // No mode argument: exercise normal non-diagnostic production output/state.
    auto final=block_run(b,weights,boundary);
    compare(capture(b,final),ref_first.at("block.0.output"),&results,s,m,block,"default_no_trace","frozen_reference");
    write_bundle(dir+"/graph-"+tag+".bundle",graph_first);
    positions<<s<<'\t'<<m<<'\t'<<block<<"\t10\tPASS\tPASS\t"<<ref_first.size()<<'\t'<<graph_time<<'\n';
    std::cout<<"BLOCK_PASS\t"<<tag<<"\tcheckpoints="<<ref_first.size()<<"\trepeats=10\n"<<std::flush;
}
void invalid_contexts(Backend& b) {
    for(int kind=0;kind<2;++kind) {
        const auto policy=kind==0 ? PrecisionPolicy::unqualified_default(DType::BF16) : PrecisionPolicy::fp32();
        if(kind==1) b.set_execution_dtype(DType::BF16);
        bool rejected=false;
        try { (void)ltx::transformer_block_forward(b,policy,{}, {},{},{},{},{},{}); }
        catch(const Error& e) {
            rejected=std::string(e.what()).find("LTX NeuralGraph requires")!=std::string::npos;
        }
        b.set_execution_dtype(DType::F32);
        require(rejected,"invalid context must reject before dereferencing inputs/weights");
    }
    std::cout<<"INVALID_PRECISION_CONTEXT\t2\tPASS\n";
}
void qualify_shape(Backend& b,const VrmModel& model,const TensorBundle& original,int s,int m,
                   const std::string& dir,std::ostream& results,std::ostream& positions,std::ostream& manifest) {
    const auto policy=PrecisionPolicy::fp32();
    WeightMap weights(model.bindings(model.graph().at("architecture_graph")));
    TensorBundle input=original;
    std::vector<float> coords(static_cast<size_t>(3*s));
    for(int axis=0;axis<3;++axis) for(int token=0;token<s;++token)
        coords[axis*s+token]=original.at("coordinates").data_as<float>()[axis*4+token];
    input["coordinates"]=host_f32({1,3,s},coords);
    auto architecture=create_architecture(model);
    const auto program=architecture->create_program(input);
    RngState rng{program.seed,0,"pytorch_compat.v1"};
    auto latent=b.rng_normal(rng,{1,s,128},DType::F32);
    auto [cosine,sine]=fractional_rope(b,policy,input.at("coordinates"),{20.0f,2048.0f,2048.0f},2048,10000.0f,true);
    auto text=b.copy_to_device(input.at("positive"),DType::F32);
    const auto mask=host_bool({1,1,text.dim(1)},std::vector<uint8_t>(text.dim(1),1));
    std::vector<float> sigma_values(static_cast<size_t>(m));
    for(int i=0;i<m;++i) sigma_values[i]=program.sigmas[i%program.steps];
    const auto sigma=host_f32({1,m},sigma_values);
    const std::string tag="S"+std::to_string(s)+"-M"+std::to_string(m);
    write_bundle(dir+"/forward-input-"+tag+".bundle",capture_bundle(b,
        {{"latent",latent},{"cosine",cosine},{"sine",sine},{"text",text},{"mask",mask},{"sigma",sigma}}));
    ltx::BlockBoundaries graph{{0,{}},{14,{}},{27,{}}};
    std::cout<<"CAPTURE_BEGIN\t"<<tag<<std::endl;
    auto actual=ltx::denoiser_forward(b,policy,weights,latent,cosine,sine,text,mask,sigma,&graph);
    const auto ref=read_bundle(frozen_dir+"/forward-output-"+tag+".bundle");
    compare(capture(b,actual),ref.at("imperative"),&results,s,m,-1,"denoiser_output","frozen_reference");
    write_bundle(dir+"/forward-output-"+tag+".bundle",{{"graph",capture(b,actual)}});
    for(int block:{0,14,27}) {
        const auto boundary=read_bundle(frozen_dir+"/boundary-"+tag+"-block"+std::to_string(block)+".bundle");
        exact_bundle(b,graph.at(block),boundary,&results,s,m,block,"frozen_full_forward_boundary");
        qualify_block(b,weights,graph.at(block),block,s,m,dir,results,positions,manifest);
    }
    resource(b);
    std::cout<<"SHAPE_PASS\t"<<tag<<"\tfull_28_block_forward_bitwise\n"<<std::flush;
}
}
int main(int argc,char** argv) {
    try {
        require(argc==5,"usage: ltx-graph-optin-cuda-tests MODEL INPUT EVIDENCE_DIRECTORY FROZEN_QUALIFICATION_DIRECTORY");
        frozen_dir=std::string(argv[4])+"/ltx-optin";
        const std::string dir=argv[3];
        std::cout<<std::setprecision(17);
        VrmModel model(argv[1]);
        require(model.architecture_id()=="ltx_v0_9_1","LTX artifact identity");
        auto input=read_bundle(argv[2]);
        require(input.at("coordinates").shape()==std::vector<int64_t>({1,3,4}),"existing four-token fixture required");
        CudaBackend b; b.set_execution_dtype(DType::F32);
        std::cout<<"BACKEND\t"<<b.name()<<std::endl;
        invalid_contexts(b);
        std::ofstream results(dir+"/checkpoint-results.tsv"),positions(dir+"/block-position-results.tsv"),manifest(dir+"/real-boundary-manifest.tsv");
        require(results.good() && positions.good() && manifest.good(),"evidence streams unavailable");
        results<<std::setprecision(17)<<"S\tM\tblock\tcheckpoint\tcomparison\tshape\tdtype\tmax_abs\tmean_abs\tmax_relative\tcosine\tnonfinite\tmax_gate_ratio\texceedances\tdirect_criterion\tbitwise\n";
        positions<<std::setprecision(17)<<"S\tM\tblock\tproduction_repeats\tfrozen_reference\tdeterminism\tcheckpoints\tproduction_seconds\n";
        manifest<<"S\tM\tblock\ttensor\tshape\tdtype\tprovenance\tartifact\n";
        qualify_shape(b,model,input,3,1,dir,results,positions,manifest);
        qualify_shape(b,model,input,3,3,dir,results,positions,manifest);
        qualify_shape(b,model,input,4,4,dir,results,positions,manifest);
        std::cout<<"QUALIFICATION_PASS\t9_cases\t90_production_graph_replays_frozen_reference\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
