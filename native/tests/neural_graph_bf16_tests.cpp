#include "neural_graph.h"
#include "neural_graph_test_backend.h"
#include "vrhino/tensor_util.h"
#ifdef VRHINO_TEST_CUDA
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/loader.h"
#include "vrhino/bundle.h"
#endif
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
using namespace vrhino;
namespace ng = vrhino::neural_graph;
namespace {
constexpr DType f = DType::F32, b = DType::BF16;
void check(bool ok, const std::string& why) { if (!ok) throw Error(why); }
ng::Description single(std::vector<ng::TensorSpec> specs, ng::Primitive op) {
    ng::Description d; d.schema=ng::schema_v1; d.execution_dtype=b; d.values=std::move(specs);
    const auto out=static_cast<ng::ValueId>(d.values.size()-1);
    for(ng::ValueId i=0;i<out;++i)d.inputs.push_back(i);
    d.nodes={{out,std::move(op)}};d.outputs={out};return d;
}
struct Case { std::string name; ng::Description graph; };
std::vector<Case> cases() {
    std::vector<Case> result;
    auto add=[&](std::string name,std::vector<ng::TensorSpec> s,ng::Primitive op) {
        result.push_back({std::move(name),single(std::move(s),std::move(op))});
    };
    add("Cast_F32_BF16",{{{2,4},f},{{2,4},b}},ng::Cast{0,b});
    add("Cast_BF16_F32",{{{2,4},b},{{2,4},f}},ng::Cast{0,f});
    auto round=single({{{2,4},f},{{2,4},b}},ng::Cast{0,b});
    round.values.push_back({{2,4},f});round.nodes.push_back({2,ng::Cast{1,f}});round.outputs={1,2};
    result.push_back({"Cast_roundtrip",round});
    add("Add",{{{2,4},b},{{4},b},{{2,4},b}},ng::Add{0,1});
    add("Mul",{{{2,4},b},{{4},b},{{2,4},b}},ng::Mul{0,1});
    add("Reshape",{{{2,4},b},{{4,2},b}},ng::Reshape{0,{4,2}});
    add("GeluTanh",{{{2,4},b},{{2,4},b}},ng::Activate{0,Activation::GeluTanh});
    add("SiLU",{{{2,4},b},{{2,4},b}},ng::Activate{0,Activation::Silu});
    add("Permute",{{{2,4},b},{{4,2},b}},ng::Permute{0,{1,0}});
    add("Slice",{{{2,4},b},{{2,2},b}},ng::Slice{0,1,1,3});
    add("Concat",{{{2,4},b},{{2,2},b},{{2,6},b}},ng::Concat{{0,1},1});
    // Explicit F32 islands inside a BF16 context must retain F32 too.
    const size_t preserving_end=result.size();
    for(size_t i=3;i<preserving_end;++i) {
        auto copy=result[i];copy.name+="_F32_island";
        for(auto& spec:copy.graph.values)spec.dtype=f;
        result.push_back(std::move(copy));
    }
    for(auto table:{f,b}) for(auto index:{DType::I32,DType::I64})
        add("Gather_"+dtype_name(table)+"_"+dtype_name(index),
            {{{4,3},table},{{3},index},{{3,3},b}},ng::Gather{0,1});
    // Every admitted storage/compute/output tuple, including biasless consumers.
    for(auto x:{f,b})for(auto w:{f,b})for(auto bias:{f,b})for(auto compute:{f,b})for(auto out:{f,b}) {
        if(compute==f && out==b)continue;
        std::string name="Linear_"+dtype_name(x)+"_"+dtype_name(w)+"_"+dtype_name(bias)+"_"+dtype_name(compute)+"_"+dtype_name(out);
        add(name,{{{2,4},x},{{6,4},w},{{6},bias},{{2,6},out}},ng::Linear{0,1,2,compute});
    }
    add("Linear_biasless",{{{2,4},b},{{6,4},b},{{2,6},b}},ng::Linear{0,1,{},b});
    for(auto x:{f,b})for(auto w:{f,b})for(auto out:{f,b})for(int axis:{0,1}) {
        if(out!=x && !(x==b && out==f && axis==1))continue;
        add("RMSNorm_"+dtype_name(x)+"_"+dtype_name(w)+"_"+dtype_name(out)+"_"+std::to_string(axis),
            {{{2,4},x},{{axis==0?2:4},w},{{2,4},out}},ng::RmsNorm{0,1,axis,1e-6f});
    }
    for(auto x:{f,b})for(int axis:{0,1})
        add("RMSNorm_unweighted_"+dtype_name(x)+"_"+std::to_string(axis),
            {{{2,4},x},{{2,4},x}},ng::RmsNorm{0,{},axis,1e-6f});
    for(auto freq:{f,b})
        add("RoPE_F32_"+dtype_name(freq),{{{1,3,2,4},f},{{1,3,1,4},freq},{{1,3,1,4},freq},{{1,3,2,4},f}},ng::Rope{0,1,2});
    for(auto q:{f,b})for(auto k:{f,b})for(auto v:{f,b})
        add("Attention_"+dtype_name(q)+"_"+dtype_name(k)+"_"+dtype_name(v),
            {{{1,3,2,4},q},{{1,2,2,4},k},{{1,2,2,4},v},{{1,3,2,4},b}},ng::Attention{0,1,2,.5f});
    for(auto type:{f,b}) {
        ng::Description d;d.schema=ng::schema_v1;d.execution_dtype=b;d.values={{{1},type}};
        d.constants={{0,-0.0f}};d.outputs={0};result.push_back({"Constant_"+dtype_name(type),d});
    }
    return result;
}
void reject(const ng::Description& d,const std::string& name) {
    bool rejected=false;try{(void)ng::admit(d);}catch(const ng::GraphError& e){rejected=e.phase==ng::Phase::Admit;}
    check(rejected,"Accepted invalid dtype: "+name);std::cout<<"NEG\t"<<name<<"\tAdmit\tPASS\n";
}
void admission() {
    for(const auto& c:cases()) {
        (void)ng::admit(c.graph);std::cout<<"HOST_ADMIT\t"<<c.name<<"\tPASS\n";
        if(!c.graph.nodes.empty()) {
            auto bad=c.graph;auto id=bad.nodes.back().result;
            bad.values[id].dtype=bad.values[id].dtype==f?b:f;
            if(std::holds_alternative<ng::Linear>(bad.nodes.back().op) ||
               std::holds_alternative<ng::RmsNorm>(bad.nodes.back().op))bad.values[id].dtype=DType::I32;
            reject(bad,"wrong_output_"+c.name);
        }
    }
    reject(single({{{2},f},{{2},b},{{2},f}},ng::Add{0,1}),"mixed_Add");
    reject(single({{{2},f},{{2},b},{{2},f}},ng::Mul{0,1}),"mixed_Mul");
    reject(single({{{2},f},{{2},b},{{4},f}},ng::Concat{{0,1},0}),"mixed_Concat");
    for(auto type:{DType::I32,DType::I64}) {
        reject(single({{{2},type},{{2},b},{{2},b}},ng::Add{0,1}),"integer_arithmetic_"+dtype_name(type));
        reject(single({{{2},f},{{2},type}},ng::Cast{0,type}),"Cast_integer_"+dtype_name(type));
    }
    for(auto type:{DType::F16,DType::I8,static_cast<DType>(255)})
        reject(single({{{2},f},{{2},type}},ng::Cast{0,type}),"Cast_unsupported_"+std::to_string(static_cast<int>(type)));
    for(auto type:{f,b})reject(single({{{2},type},{{2},type}},ng::Cast{0,type}),"Cast_identity_"+dtype_name(type));
    reject(single({{{4,3},b},{{2},b},{{2,3},b}},ng::Gather{0,1}),"BF16_selector");
    reject(single({{{1,2,1,4},b},{{1,2,1,4},b},{{1,2,1,4},b},{{1,2,1,4},b}},ng::Rope{0,1,2}),"RoPE_BF16_data");
    reject(single({{{1,2,1,4},f},{{1,2,1,4},f},{{1,2,1,4},b},{{1,2,1,4},f}},ng::Rope{0,1,2}),"RoPE_mixed_frequency");
    reject(single({{{1,2,1,4},b},{{1,2,1,4},b},{{1,2,1,4},DType::I32},{{1,2,1,4},b}},ng::Attention{0,1,2,.5f}),"Attention_integer");
    reject(single({{{2,4},b},{{2,4},f}},ng::RmsNorm{0,{},1,1e-6f}),"RMSNorm_widen_without_weight");
    reject(single({{{2,4},b},{{2},b},{{2,4},f}},ng::RmsNorm{0,1,0,1e-6f}),"RMSNorm_widen_nonfinal");
    reject(single({{{2,4},f},{{4},f},{{2,4},b}},ng::RmsNorm{0,1,1,1e-6f}),"RMSNorm_narrow");
    reject(single({{{2,4},b},{{6,4},b},{{2,6},b}},ng::Linear{0,1,{},f}),"Linear_F32_compute_BF16_output");
    reject(single({{{2,4},b},{{6,4},b},{{2,6},b}},ng::Linear{0,1,{},DType::F16}),"Linear_implicit_compute");
    auto bad=cases()[0].graph;bad.execution_dtype=f;reject(bad,"F32_context_BF16_value");
    bad.execution_dtype=DType::F16;reject(bad,"unsupported_context");
    for(auto id:{ng::ValueId{99},ng::ValueId{1}}) {
        bad=cases()[0].graph;bad.nodes[0].op=ng::Cast{id,b};reject(bad,"Cast_dependency_"+std::to_string(id));
    }
    ng::Description c;c.schema=ng::schema_v1;c.execution_dtype=b;c.values={{{1},b}};c.constants={{0,1.001f}};c.outputs={0};
    reject(c,"constant_dtype_mismatch_nonrepresentable");
    c.constants[0].scalar=std::numeric_limits<float>::infinity();reject(c,"constant_nonfinite");
}
// Host control double exercises lifetime/failure boundaries; it is not a CPU numerical Backend.
class BindingBackend final : public ng::test::TinyBackend {
public:
    int devices=0;bool support=true,throw_upload=false,bad_upload=false;
    int uploads=0;std::vector<std::weak_ptr<Storage>> uploads_alive;
    BindingBackend(){dtype=b;}
    bool supports(DType) const override{return support;}
    int device_count() const override{return devices;}
    Tensor copy_to_device(const Tensor& t,DType type) override {
        ++uploads;check(t.dtype()==type,"upload changed dtype");
        if(throw_upload && uploads==2)throw Error("injected upload failure");
        auto result=owned(t.shape(),bad_upload?f:type);
        std::memcpy(result.data(),t.data(),t.bytes());
        produced.back().lock()->device=DeviceId::accelerator();produced.back().lock()->domain=MemoryDomain::DeviceLocal;
        return result;
    }
};
void binding() {
    ng::Description d;d.schema=ng::schema_v1;d.execution_dtype=b;
    d.values={{{2},b},{{2},b}};d.inputs={0};d.parameters={1};d.outputs={0,1};
    auto graph=ng::admit(d);Tensor x=Tensor::host({2},b),w=Tensor::host({2},b);
    auto rejects=[&](BindingBackend& backend,const std::vector<Tensor>& in,const std::vector<Tensor>& params,const std::string& name,ng::Code code) {
        bool failed=false;try{(void)ng::evaluate(backend,graph,in,params);}catch(const ng::GraphError& e){failed=e.phase==ng::Phase::Bind&&e.code==code;}
        check(failed,"binding accepted: "+name);check(backend.calls.empty(),"binding dispatched arithmetic");
        std::cout<<"NEG\t"<<name<<"\tBind\tPASS\n";
    };
    BindingBackend backend;
    rejects(backend,{Tensor::host({2},f)},{w},"binding_dtype_mismatch",ng::Code::InvalidInputBinding);
    rejects(backend,{x},{Tensor::host({2},f)},"parameter_dtype_mismatch",ng::Code::InvalidParameterBinding);
    backend.dtype=f;rejects(backend,{x},{w},"context_mismatch",ng::Code::InvalidContext);backend.dtype=b;
    backend.support=false;rejects(backend,{x},{w},"BF16_unsupported",ng::Code::InvalidContext);backend.support=true;
    auto outputs=ng::evaluate(backend,graph,{x},{w});check(outputs[0].data()==x.data(),"host alias");
    backend.devices=1;backend.syncs=0;
    rejects(backend,{x},{Tensor::host({2},f)},"all_bindings_before_upload",ng::Code::InvalidParameterBinding);
    check(backend.uploads==0 && backend.syncs==0,"invalid binding submitted");
    backend.throw_upload=true;
    rejects(backend,{x},{w},"upload_failure_drain",ng::Code::BackendFailure);
    check(backend.syncs==1,"upload failure did not drain");
    BindingBackend bad;bad.devices=1;bad.bad_upload=true;
    rejects(bad,{x},{w},"upload_dtype_mismatch",ng::Code::InvalidContext);
    check(bad.syncs==1,"bad upload did not drain");
    BindingBackend constant;
    for(float value:{0.f,-0.f,1.f,-2.f}) {
        ng::Description c;c.schema=ng::schema_v1;c.execution_dtype=b;c.values={{{1},b}};c.constants={{0,value}};c.outputs={0};
        auto out=ng::evaluate(constant,ng::admit(c),{},{});uint32_t bits;std::memcpy(&bits,&value,4);
        check(out[0].data_as<uint16_t>()[0]==bits>>16,"constant payload changed");
    }
    std::cout<<"HOST_BIND\tmetadata_context_upload_drain_constant_bits\tPASS\n";
}
#ifdef VRHINO_TEST_CUDA
Tensor fixture(Backend& backend,const ng::TensorSpec& spec,int slot) {
    Tensor t=Tensor::host(spec.shape,spec.dtype);
    if(spec.dtype==DType::I32 || spec.dtype==DType::I64) {
        for(int64_t i=0;i<t.numel();++i) {
            const auto value=i==1?0:3;
            if(spec.dtype==DType::I32)t.data_as<int32_t>()[i]=value;else t.data_as<int64_t>()[i]=value;
        }
        return t;
    }
    std::vector<float> values(t.numel());
    for(size_t i=0;i<values.size();++i)values[i]=(int((i*17+slot*11+3)%41)-20)/32.f;
    Tensor source=host_f32(spec.shape,values);
    return spec.dtype==f?source:backend.copy_to_host(backend.cast(source,b));
}
Tensor direct(Backend& backend,const ng::Primitive& op,const std::vector<Tensor>& v,DType out) {
    switch(op.index()) {
    case 1:{auto p=std::get<ng::Add>(op);return backend.add(v[p.a],v[p.b]);}
    case 2:{auto p=std::get<ng::Mul>(op);return backend.mul(v[p.a],v[p.b]);}
    case 3:{auto p=std::get<ng::Reshape>(op);return backend.reshape(v[p.x],p.shape);}
    case 4:{auto p=std::get<ng::Linear>(op);return backend.linear(v[p.x],v[p.weight],p.bias?&v[*p.bias]:nullptr,p.compute_dtype,out);}
    case 5:{auto p=std::get<ng::RmsNorm>(op);return backend.rms_norm(v[p.x],p.weight?&v[*p.weight]:nullptr,p.epsilon,p.axis,out);}
    case 6:{auto p=std::get<ng::Activate>(op);return backend.activation(v[p.x],p.kind);}
    case 7:{auto p=std::get<ng::Permute>(op);return backend.permute(v[p.x],p.axes);}
    case 8:{auto p=std::get<ng::Slice>(op);return backend.slice(v[p.x],p.axis,p.start,p.stop);}
    case 9:{auto p=std::get<ng::Concat>(op);std::vector<Tensor> parts;for(auto id:p.tensors)parts.push_back(v[id]);return backend.concat(parts,p.axis);}
    case 10:{auto p=std::get<ng::Gather>(op);return backend.indexed_gather(v[p.table],v[p.indices]);}
    case 11:{auto p=std::get<ng::Rope>(op);return backend.rope_nd(v[p.x],v[p.cosine],v[p.sine]);}
    case 12:{auto p=std::get<ng::Attention>(op);return backend.attention(v[p.q],v[p.k],v[p.v],nullptr,false,p.scale,nullptr,nullptr);}
    case 13:{auto p=std::get<ng::Cast>(op);return backend.cast(v[p.x],p.destination);}
    default:throw Error("unsupported test reference");
    }
}
void exact(Backend& backend,const Tensor& x,const Tensor& ref,const std::string& label) {
    check(x.dtype()==ref.dtype() && x.shape()==ref.shape(),label+" metadata");
    auto host=[&](const Tensor& t){return t.device().type==DeviceType::CPU?t:backend.copy_to_host(t);};
    Tensor a=host(x),r=host(ref);
    check(a.bytes()==r.bytes() && std::memcmp(a.data(),r.data(),a.bytes())==0,label+" bytes");
    for(int64_t i=0;i<a.numel();++i) {
        float v;
        if(a.dtype()==b){uint32_t bits=uint32_t(a.data_as<uint16_t>()[i])<<16;std::memcpy(&v,&bits,4);}
        else v=a.data_as<float>()[i];
        check(std::isfinite(v),label+" nonfinite");
    }
}
void numerical() {
    CudaBackend backend;backend.set_execution_dtype(b);
    for(const auto& c:cases()) {
        const auto& d=c.graph;const auto graph=ng::admit(d);
        std::vector<Tensor> host_inputs,device_inputs,values(d.values.size());
        for(auto id:d.inputs) {
            Tensor h=fixture(backend,d.values[id],id);host_inputs.push_back(h);
            Tensor dev=(h.dtype()==f||h.dtype()==b)?backend.copy_to_device(h,h.dtype()):h;
            values[id]=dev;device_inputs.push_back(dev);
        }
        for(auto constant:d.constants) {
            Tensor t=host_f32({1},{constant.scalar});values[constant.value]=backend.copy_to_device(t,d.values[constant.value].dtype);
        }
        for(const auto& node:d.nodes)values[node.result]=direct(backend,node.op,values,d.values[node.result].dtype);
        backend.synchronize();
        for(bool host_binding:{true,false}) {
            for(int repeat=0;repeat<20;++repeat) {
                auto outputs=ng::evaluate(backend,graph,host_binding?host_inputs:device_inputs,{});
                for(size_t i=0;i<outputs.size();++i)exact(backend,outputs[i],values[d.outputs[i]],c.name);
            }
            std::cout<<"SYNTH\t"<<c.name<<"\t"<<(host_binding?"HostBindings/CUDA":"DeviceBindings/CUDA")<<"\t";
            for(auto dim:d.values[d.outputs.back()].shape)std::cout<<dim<<',';
            std::cout<<'\t'<<dtype_name(d.values[d.outputs.back()].dtype)<<"\tfinite\tbitwise\t20/20\tPASS\n";
        }
    }
    // Exact ties, signed zero and subnormals use the existing conversion implementation.
    std::vector<float> ties;for(uint32_t bits:{0u,0x80000000u,0x3f808000u,0x3f818000u,0xbf808000u,0xbf818000u,0x00010000u}) {
        float value;std::memcpy(&value,&bits,4);ties.push_back(value);
    }
    auto d=single({{{7},f},{{7},b}},ng::Cast{0,b});auto x=host_f32({7},ties);
    auto output=ng::evaluate(backend,ng::admit(d),{x},{});
    exact(backend,output[0],backend.cast(x,b),"Cast ties");
    std::cout<<"SYNTH\tCast_ties_signed_zero_subnormal\tHostBindings/CUDA\t7\tbf16\tfinite\tbitwise\tPASS\n";
}
void mochi(const char* path,const char* output_dir) {
    CudaBackend backend;backend.set_execution_dtype(b);VrmModel model(path);
    const std::string prefix="denoiser.transformer_blocks.0.ff";
    Tensor w0=model.tensor(prefix+".net.0.proj.weight"),w1=model.tensor(prefix+".net.2.weight");
    check(w0.dtype()==b && w1.dtype()==b,"Mochi real BF16 weights required");
    check(w0.shape()==std::vector<int64_t>({16384,3072}) && w1.shape()==std::vector<int64_t>({3072,8192}),"Mochi weight shape");
    for(int tokens:{1,3}) {
        ng::Description d;d.schema=ng::schema_v1;d.execution_dtype=b;
        d.values={{{1,tokens,3072},b},{{16384,3072},b},{{3072,8192},b},{{1,tokens,16384},b},
                  {{1,tokens,8192},b},{{1,tokens,8192},b},{{1,tokens,8192},b},{{1,tokens,8192},b},{{1,tokens,3072},b}};
        d.inputs={0};d.parameters={1,2};d.nodes={{3,ng::Linear{0,1,{},b}},
            {4,ng::Slice{3,2,0,8192}},{5,ng::Slice{3,2,8192,16384}},{6,ng::Activate{5,Activation::Silu}},
            {7,ng::Mul{4,6}},{8,ng::Linear{7,2,{},b}}};d.outputs={3,4,5,6,7,8};
        auto graph=ng::admit(d);Tensor x=backend.copy_to_device(fixture(backend,d.values[0],0),b);
        // Literal existing mochi.cpp swiglu algorithm: default BF16 consumers and split.
        Tensor projected=backend.linear(x,w0);
        auto parts=backend.split(projected,{8192,8192},-1);
        Tensor silu=backend.activation(parts[1],Activation::Silu);
        Tensor hidden=backend.mul(parts[0],silu);Tensor final=backend.linear(hidden,w1);
        std::vector<Tensor> refs={projected,parts[0],parts[1],silu,hidden,final};backend.synchronize();
        Tensor p0=backend.copy_to_device(w0,b),p1=backend.copy_to_device(w1,b);
        for(int repeat=0;repeat<20;++repeat) {
            auto actual=ng::evaluate(backend,graph,{x},{p0,p1});
            for(size_t i=0;i<actual.size();++i)exact(backend,actual[i],refs[i],"Mochi checkpoint "+std::to_string(i));
        }
        auto host_bound=ng::evaluate(backend,graph,{backend.copy_to_host(x)},{w0,w1});
        std::map<std::string,Tensor> saved;
        saved["input"]=backend.copy_to_host(x);
        for(size_t i=0;i<refs.size();++i) {
            exact(backend,host_bound[i],refs[i],"Mochi host bindings");
            saved["checkpoint_"+std::to_string(i)]=backend.copy_to_host(refs[i]);
            std::cout<<"MOCHI\t"<<tokens<<'\t'<<i<<'\t';for(auto dim:refs[i].shape())std::cout<<dim<<',';
            std::cout<<"\tbf16\tfinite\tbitwise\t20/20\tPASS\n";
        }
        write_bundle(std::string(output_dir)+"/mochi-"+std::to_string(tokens)+".bundle",saved);
    }
}
#endif
}
int main(int argc,char** argv) {
    try {
#ifdef VRHINO_TEST_CUDA
        if(argc==4 && std::string(argv[1])=="mochi"){mochi(argv[2],argv[3]);return 0;}
#else
        (void)argc;(void)argv;
#endif
        admission();binding();
#ifdef VRHINO_TEST_CUDA
        numerical();
#endif
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
