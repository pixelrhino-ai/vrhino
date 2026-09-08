#include "ltx_self_attention_graph.h"
#include "ltx_self_attention_reference.h"
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
             int s,int m,const std::string& checkpoint,const std::string& path) {
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
    if (report) *report << s << '\t' << m << '\t' << checkpoint << '\t' << path << '\t'
        << shape(actual) << "\tF32\t" << max_abs << '\t' << sum/actual.numel() << '\t'
        << relative << '\t' << cosine << '\t' << nonfinite << '\t' << gate_ratio << '\t'
        << exceedances << '\t' << (nonfinite==0 && exceedances==0 ? "PASS" : "FAIL")
        << '\t' << (bitwise ? "PASS" : "FAIL") << '\n';
    require(nonfinite==0 && exceedances==0 && bitwise,"frozen Class 2 qualification failed");
}
std::vector<Tensor> fixture(Backend& b,int s,int m,TensorBundle& host) {
    const std::vector<std::vector<int64_t>> shapes={{1,s,2048},{1,m,12288},{1,s,2048},{1,s,2048}};
    std::vector<Tensor> x;
    for (int j=0;j<4;++j) {
        std::vector<float> data(static_cast<size_t>(shape_numel(shapes[j])));
        for (size_t i=0;i<data.size();++i) {
            if (j<2) data[i]=(int((i*17+j*11+3)%101)-50)/128.0f;
            else {
                const double angle=double((i/2)%97+1)/128.0;
                data[i]=float(j==2 ? std::cos(angle) : std::sin(angle));
            }
        }
        auto t=host_f32(shapes[j],data);
        host["input."+std::to_string(j)]=t;
        x.push_back(b.copy_to_device(t,DType::F32));
    }
    return x;
}
void manifest(const ng::Graph& g,int s,int m,std::ostream& out) {
    static const char* names[]={"unsupported","Add","Mul","Reshape","Linear","RMSNorm",
        "Activation","Permute","Slice","Concat","Gather","RoPE","Attention"};
    for (const auto& n:g.description().nodes) {
        out << s << '\t' << m << '\t' << n.result << '\t' << names[n.op.index()] << '\t';
        std::visit([&](const auto& op) {
            using T=std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T,ng::Add> || std::is_same_v<T,ng::Mul>) out<<op.a<<','<<op.b;
            else if constexpr (std::is_same_v<T,ng::Linear>) out<<op.x<<','<<op.weight<<','<<*op.bias;
            else if constexpr (std::is_same_v<T,ng::RmsNorm>) {
                out<<op.x; if(op.weight) out<<','<<*op.weight;
                out<<";axis="<<op.axis<<";epsilon="<<op.epsilon;
            } else if constexpr (std::is_same_v<T,ng::Rope>) out<<op.x<<','<<op.cosine<<','<<op.sine;
            else if constexpr (std::is_same_v<T,ng::Attention>) out<<op.q<<','<<op.k<<','<<op.v<<";scale="<<op.scale;
            else if constexpr (std::is_same_v<T,ng::Slice>) out<<op.x<<";axis="<<op.axis<<";start="<<op.start<<";stop="<<op.stop;
            else if constexpr (std::is_same_v<T,ng::Reshape>) out<<op.x;
        },n.op);
        out<<'\t'; for(auto dim:g.description().values[n.result].shape) out<<dim<<',';
        out<<"\tF32\n";
    }
}
void negatives(Backend& b,const ng::Graph& g,const std::vector<Tensor>& x,const std::vector<Tensor>& p) {
    auto reject=[&](const std::vector<Tensor>& xx,const std::vector<Tensor>& pp,ng::Code code) {
        bool rejected=false;
        try { (void)ng::evaluate(b,g,xx,pp); }
        catch(const ng::GraphError& e) { rejected=e.phase==ng::Phase::Bind && e.code==code; }
        require(rejected,"bad binding was not rejected before dispatch");
    };
    auto xx=x; xx.pop_back(); reject(xx,p,ng::Code::InvalidInputBinding);
    auto pp=p; pp.pop_back(); reject(x,pp,ng::Code::InvalidParameterBinding);
    xx=x; xx[0]=host_f32({1},{0}); reject(xx,p,ng::Code::InvalidInputBinding);
    xx=x; xx[0]=b.cast(x[0],DType::BF16); reject(xx,p,ng::Code::InvalidInputBinding);
    pp=p; std::swap(pp[0],pp[1]); reject(x,pp,ng::Code::InvalidParameterBinding);
    bool rejected=false;
    try { (void)ltx::self_attention_graph(1,3,2); } catch(const Error&) { rejected=true; }
    require(rejected,"illegal modulation shape accepted");
    std::cout<<"NEGATIVE\tPASS\t6_cases\n";
}
void qualify(Backend& b,const std::vector<Tensor>& p,int s,int m,const std::string& dir,
             std::ostream& results,std::ostream& graph_manifest) {
    TensorBundle source;
    auto x=fixture(b,s,m,source);
    write_bundle(dir+"/input-S"+std::to_string(s)+"-M"+std::to_string(m)+".bundle",source);
    std::map<std::string,const Tensor*> binding;
    for(size_t i=0;i<p.size();++i) binding[ltx::self_attention_parameters[i]]=&p[i];
    WeightMap weights(binding);
    auto graph=ltx::self_attention_graph(1,s,m,true);
    require(graph.description().nodes.size()==28,"frozen node count");
    manifest(graph,s,m,graph_manifest);
    negatives(b,graph,x,p);
    TensorBundle reference_first,graph_first;
    double ref_seconds=0,graph_seconds=0;
    for(int repeat=0;repeat<10;++repeat) {
        TensorBundle ref;
        auto start=Clock::now();
        auto result=ltx_test_reference::self_attention_reference(b,weights,x[0],x[1],x[2],x[3],&ref);
        b.synchronize();
        ref_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
        resource(b);
        start=Clock::now();
        auto out=ng::evaluate(b,graph,x,p);
        graph_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
        resource(b);
        require(out.size()==std::size(ltx::self_attention_checkpoints),"checkpoint arity");
        for(size_t i=0;i<out.size();++i) {
            const auto name=ltx::self_attention_checkpoints[i];
            auto actual=capture(b,out[i]),expected=capture(b,ref.at(name));
            compare(actual,expected,repeat==0 ? &results : nullptr,s,m,name,"cross_path");
            if(repeat==0) { reference_first[name]=expected; graph_first[name]=actual; }
            else {
                compare(expected,reference_first.at(name),nullptr,s,m,name,"imperative_repeat");
                compare(actual,graph_first.at(name),nullptr,s,m,name,"graph_repeat");
            }
        }
        for(size_t i=0;i<x.size();++i)
            compare(capture(b,x[i]),source.at("input."+std::to_string(i)),repeat==0 ? &results : nullptr,
                    s,m,"input."+std::to_string(i),"input_identity");
        compare(capture(b,result),reference_first.at("block.0.first_residual"),nullptr,s,m,"return","reference_return");
    }
    auto final=ng::evaluate(b,ltx::self_attention_graph(1,s,m),x,p);
    compare(capture(b,final[0]),reference_first.at("block.0.first_residual"),&results,s,m,"final_only","cross_path");
    write_bundle(dir+"/imperative-S"+std::to_string(s)+"-M"+std::to_string(m)+".bundle",reference_first);
    write_bundle(dir+"/graph-S"+std::to_string(s)+"-M"+std::to_string(m)+".bundle",graph_first);
    std::cout<<"PASS\tS="<<s<<"\tM="<<m<<"\trepeats=10\timperative_bitwise=PASS\tgraph_bitwise=PASS\tcross_path_bitwise=PASS\n"
        <<"TIME\t"<<s<<'\t'<<m<<'\t'<<ref_seconds<<'\t'<<graph_seconds<<"\t10\n";
}
}
int main(int argc,char** argv) {
    try {
        require(argc==3,"usage: ltx-graph-first-cuda-tests MODEL EVIDENCE_DIRECTORY");
        std::cout<<std::setprecision(17);
        const std::string dir=argv[2];
        VrmModel model(argv[1]);
        require(model.architecture_id()=="ltx_v0_9_1","fixture architecture identity");
        CudaBackend b; b.set_execution_dtype(DType::F32);
        std::cout<<"BACKEND\t"<<b.name()<<"\n";
        std::vector<Tensor> parameters;
        TensorBundle materialized;
        std::ofstream bindings(dir+"/tensor-bindings.tsv");
        bindings<<"value_id\tslot\tcanonical_name\tstored_dtype\tshape\texecution_dtype\n";
        size_t elements=0;
        for(const auto* name:ltx::self_attention_parameters) {
            const auto key="denoiser.transformer_blocks.0."+std::string(name);
            const auto& source=model.tensor(key);
            auto t=b.copy_to_device(source,DType::F32);
            bindings<<parameters.size()+4<<'\t'<<parameters.size()<<'\t'<<key<<'\t'
                <<dtype_name(source.dtype())<<'\t'<<shape(t)<<"\tF32\n";
            elements+=t.numel(); materialized[name]=capture(b,t); parameters.push_back(t);
        }
        write_bundle(dir+"/parameters-f32.bundle",materialized);
        std::cout<<"PARAMETERS\t"<<parameters.size()<<'\t'<<elements<<'\n';
        std::ofstream results(dir+"/checkpoint-results.tsv"),manifest_file(dir+"/graph-manifest.tsv");
        results<<std::setprecision(17)<<"S\tM\tcheckpoint\tcomparison\tshape\tdtype\tmax_abs\tmean_abs\tmax_relative\tcosine\tnonfinite\tmax_gate_ratio\texceedances\tdirect_criterion\tbitwise\n";
        manifest_file<<std::setprecision(17)<<"S\tM\tvalue_id\tprimitive\toperands_and_attributes\tshape\tdtype\n";
        qualify(b,parameters,3,1,dir,results,manifest_file);
        qualify(b,parameters,3,3,dir,results,manifest_file);
        qualify(b,parameters,4,4,dir,results,manifest_file);
        for(size_t i=0;i<parameters.size();++i)
            compare(capture(b,parameters[i]),materialized.at(ltx::self_attention_parameters[i]),nullptr,0,0,"parameter","immutable");
        resource(b);
        std::cout<<"IMMUTABLE_PARAMETERS\tPASS\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
