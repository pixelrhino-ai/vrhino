#include "neural_graph.h"
#include "neural_graph_test_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/tensor_util.h"
#ifdef VRHINO_TEST_CUDA
#include "vrhino/backend/cuda_backend.h"
#endif
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
using namespace vrhino;
namespace ng = vrhino::neural_graph;
namespace {
constexpr DType f32 = DType::F32, bf16 = DType::BF16;
void check(bool ok, const std::string& why) { if (!ok) throw Error(why); }
ng::Description description(std::vector<int64_t> shape, DType x, DType param,
                            bool affine, DType context, float epsilon) {
    ng::Description d; d.schema = ng::schema_v1; d.execution_dtype = context;
    d.values = {{shape, x}}; d.inputs = {0};
    if (affine) {
        d.values.push_back({{shape.back()}, param}); d.values.push_back({{shape.back()}, param});
        d.parameters = {1, 2};
    }
    auto out = static_cast<ng::ValueId>(d.values.size()); d.values.push_back({shape, x});
    d.nodes = {{out, ng::LayerNorm{0, affine ? std::optional<ng::ValueId>{1} : std::nullopt,
                                  affine ? std::optional<ng::ValueId>{2} : std::nullopt, epsilon}}};
    d.outputs = {out}; return d;
}
void admission() {
    auto base = description({2, 3, 17}, f32, f32, true, bf16, 1e-6f);
    auto reject = [&](const char* name, const std::function<void(ng::Description&)>& mutate, ng::Code code) {
        auto d = base; mutate(d); bool caught = false;
        try { (void)ng::admit(d); } catch (const ng::GraphError& e) {
            caught = e.phase == ng::Phase::Admit && e.code == code;
        }
        check(caught, std::string("admission negative failed: ") + name);
        std::cout << "NEG\t" << name << "\tAdmit\tPASS\n";
    };
    using C = ng::Code;
    reject("weight_shape", [](auto& d){d.values[1].shape={18};}, C::InvalidShape);
    reject("bias_shape", [](auto& d){d.values[2].shape={18};}, C::InvalidShape);
    reject("weight_rank", [](auto& d){d.values[1].shape={1,17};}, C::InvalidShape);
    reject("bias_rank", [](auto& d){d.values[2].shape={17,1};}, C::InvalidShape);
    reject("weight_only", [](auto& d){std::get<ng::LayerNorm>(d.nodes[0].op).bias.reset();}, C::InvalidOperation);
    reject("bias_only", [](auto& d){std::get<ng::LayerNorm>(d.nodes[0].op).weight.reset();}, C::InvalidOperation);
    for (auto type : {DType::F16, DType::I32, DType::I64, DType::U8, DType::I8, DType::Bool, static_cast<DType>(255)}) {
        for (size_t slot : {size_t{0},size_t{1},size_t{2},size_t{3}}) {
            auto name = "unsupported_dtype_" + std::to_string(slot) + "_" + std::to_string(static_cast<int>(type));
            reject(name.c_str(), [=](auto& d){d.values[slot].dtype=type;}, C::InvalidDType);
        }
    }
    reject("mixed_weight_dtype", [](auto& d){d.values[1].dtype=bf16;}, C::InvalidDType);
    reject("mixed_bias_dtype", [](auto& d){d.values[2].dtype=bf16;}, C::InvalidDType);
    reject("invalid_f32_tuple", [](auto& d){d.values[3].dtype=bf16;}, C::InvalidDType);
    reject("invalid_bf16_tuple", [](auto& d){d.values[0].dtype=bf16;}, C::InvalidDType);
    reject("rank_zero", [](auto& d){d.values[0].shape={};}, C::InvalidShape);
    reject("rank_nine", [](auto& d){d.values[0].shape=std::vector<int64_t>(9,1);}, C::InvalidShape);
    reject("zero_dimension", [](auto& d){d.values[0].shape={0,17};}, C::InvalidShape);
    reject("negative_dimension", [](auto& d){d.values[0].shape={-1,17};}, C::InvalidShape);
    reject("element_overflow", [](auto& d){d.values[0].shape={INT32_MAX,2};}, C::InvalidShape);
    reject("wrong_normalized_dimension", [](auto& d){d.values[1].shape=d.values[2].shape={3};}, C::InvalidShape);
    reject("wrong_output_shape", [](auto& d){d.values[3].shape={6,17};}, C::InvalidShape);
    for (float eps : {0.f,-0.f,-1e-6f,std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}) {
        auto name="epsilon_"+std::to_string(eps);
        reject(name.c_str(), [=](auto& d){std::get<ng::LayerNorm>(d.nodes[0].op).epsilon=eps;}, C::InvalidOperation);
    }
    reject("unsupported_context", [](auto& d){d.execution_dtype=DType::F16;}, C::InvalidDType);
    reject("f32_context_bf16_value", [](auto& d){d.execution_dtype=f32;d.values[1].dtype=d.values[2].dtype=bf16;}, C::InvalidDType);
    for (ng::ValueId bad : {ng::ValueId{99},ng::ValueId{3}}) for (int slot : {0,1,2}) {
        auto name="dependency_"+std::to_string(slot)+"_"+std::to_string(bad);
        reject(name.c_str(), [=](auto& d){auto& op=std::get<ng::LayerNorm>(d.nodes[0].op);
            if(slot==0)op.x=bad;else if(slot==1)op.weight=bad;else op.bias=bad;}, C::InvalidDependency);
    }
}
// Records dispatch/owners only. This is deliberately not a CPU numerical implementation.
class RecordingBackend : public ng::test::TinyBackend {
public:
    bool support = true; const Tensor *expected_x=nullptr, *expected_w=nullptr, *expected_b=nullptr;
    float epsilon=1e-6f;
    bool supports(DType) const override { return support; }
    Tensor layer_norm(const Tensor& x, const Tensor* w, const Tensor* b, float eps) override {
        begin("layer_norm"); check(eps==epsilon,"epsilon forwarding");
        check((w!=nullptr)==(expected_w!=nullptr) && (b!=nullptr)==(expected_b!=nullptr),"affine forwarding");
        check(x.data()==expected_x->data(),"input identity");
        if(w)check(w->data()==expected_w->data() && b->data()==expected_b->data() &&
                   w->dtype()==expected_w->dtype() && b->dtype()==expected_b->dtype(),"parameter identity/dtype");
        Tensor out=owned(x.shape(),x.dtype());std::memset(out.data(),0,out.bytes());return finish(out);
    }
};
void binding() {
    using C=ng::Code;auto d=description({2,17},bf16,bf16,true,bf16,1e-6f);auto g=ng::admit(d);
    Tensor x=Tensor::host({2,17},bf16),w=Tensor::host({17},bf16),b=Tensor::host({17},bf16);
    auto reject=[&](const char* name,std::vector<Tensor> xs,std::vector<Tensor> ps,DType context,bool support,C code){
        RecordingBackend backend;backend.dtype=context;backend.support=support;bool caught=false;
        try{(void)ng::evaluate(backend,g,xs,ps);}catch(const ng::GraphError& e){caught=e.phase==ng::Phase::Bind&&e.code==code;}
        check(caught&&backend.calls.empty()&&backend.syncs==0,std::string("binding negative failed: ")+name);
        std::cout<<"NEG\t"<<name<<"\tBind\tPASS\n";
    };
    reject("input_binding_dtype",{Tensor::host({2,17},f32)},{w,b},bf16,true,C::InvalidInputBinding);
    reject("weight_binding_dtype",{x},{Tensor::host({17},f32),b},bf16,true,C::InvalidParameterBinding);
    reject("bias_binding_dtype",{x},{w,Tensor::host({17},f32)},bf16,true,C::InvalidParameterBinding);
    reject("context_dtype",{x},{w,b},f32,true,C::InvalidContext);
    reject("unsupported_bf16_backend",{x},{w,b},bf16,false,C::InvalidContext);
    reject("missing_affine_binding",{x},{w},bf16,true,C::InvalidParameterBinding);
    reject("extra_affine_binding",{x},{w,b,w},bf16,true,C::InvalidParameterBinding);
    reject("missing_input",{},{w,b},bf16,true,C::InvalidInputBinding);
    reject("extra_input",{x,x},{w,b},bf16,true,C::InvalidInputBinding);
    reject("binding_shape",{Tensor::host({1,34},bf16)},{w,b},bf16,true,C::InvalidInputBinding);
    reject("undefined_binding",{Tensor{}},{w,b},bf16,true,C::InvalidInputBinding);
    reject("null_binding",{Tensor::borrowed(nullptr,x.bytes(),x.shape(),bf16)},{w,b},bf16,true,C::InvalidInputBinding);
    reject("binding_byte_extent",{Tensor::borrowed(x.data(),x.bytes()+2,x.shape(),bf16)},{w,b},bf16,true,C::InvalidInputBinding);
    auto info=std::make_shared<QuantizationInfo>();info->type=QuantType::INT8Symmetric;info->scales=Tensor::host({1},f32);
    Tensor quant=Tensor::host({2,17},DType::U8);quant.set_quantization(info);
    reject("quantized_binding",{quant},{w,b},bf16,true,C::InvalidInputBinding);
    for(auto type:{f32,bf16}) for(bool affine:{false,true}) {
        auto graph=ng::admit(description({2,17},type,type,affine,bf16,.25f));
        Tensor a=Tensor::host({2,17},type),u=Tensor::host({17},type),v=Tensor::host({17},type);
        RecordingBackend backend;backend.dtype=bf16;backend.epsilon=.25f;backend.expected_x=&a;
        backend.expected_w=affine?&u:nullptr;backend.expected_b=affine?&v:nullptr;
        auto out=ng::evaluate(backend,graph,{a},affine?std::vector<Tensor>{u,v}:std::vector<Tensor>{});
        check(out[0].dtype()==type && backend.calls==std::vector<std::string>{"layer_norm"}&&backend.syncs==1,"single dispatch/no cast");
    }
    for(auto fault:{RecordingBackend::Fault::Shape,RecordingBackend::Fault::Dtype,RecordingBackend::Fault::Undefined}) {
        RecordingBackend backend;backend.dtype=bf16;backend.expected_x=&x;backend.expected_w=&w;backend.expected_b=&b;backend.fault=fault;
        bool caught=false;try{(void)ng::evaluate(backend,g,{x},{w,b});}catch(const ng::GraphError& e){caught=e.code==C::InvalidNodeOutput;}
        check(caught&&backend.syncs==1,"invalid result must drain");
        std::cout<<"FAULT\toutput_"<<static_cast<int>(fault)<<"\tRun/drain\tPASS\n";
    }
    RecordingBackend backend;backend.dtype=bf16;backend.fail_call=1;bool caught=false;
    try{(void)ng::evaluate(backend,g,{x},{w,b});}catch(const ng::GraphError& e){caught=e.code==C::BackendFailure;}
    check(caught&&backend.syncs==1,"throwing dispatch drain");
    std::cout<<"HOST\tidentity_dtype_no_cast_owners_failure_drain\tPASS\n";
}
#ifdef VRHINO_TEST_CUDA
Tensor host(Backend& b,const Tensor& t){return t.device()==Device::CPU?t:b.copy_to_host(t);}
std::vector<long double> numbers(Backend& b,const Tensor& t){
    Tensor h=host(b,t);std::vector<long double> v(h.numel());
    for(int64_t i=0;i<h.numel();++i){float value;
        if(h.dtype()==f32)value=h.data_as<float>()[i];
        else{check(h.dtype()==bf16,"reference dtype");uint32_t bits=uint32_t(h.data_as<uint16_t>()[i])<<16;std::memcpy(&value,&bits,4);}
        check(std::isfinite(value),"nonfinite capture");v[i]=value;
    }return v;
}
void exact(Backend& b,const Tensor& a,const Tensor& ref){
    auto x=host(b,a),y=host(b,ref);check(x.shape()==y.shape()&&x.dtype()==y.dtype()&&x.bytes()==y.bytes(),"parity metadata");
    check(std::memcmp(x.data(),y.data(),x.bytes())==0,"same-route bitwise mismatch");(void)numbers(b,x);
}
// Independent centered two-pass population variance in long double. No Native second-moment staging.
void independent(Backend& b,const Tensor& x,const std::vector<Tensor>& params,float eps,const Tensor& actual){
    auto values=numbers(b,x),a=numbers(b,actual);std::vector<long double>w,bias;
    if(!params.empty()){w=numbers(b,params[0]);bias=numbers(b,params[1]);}
    const int64_t width=x.dim(-1);
    for(int64_t row=0;row<x.numel()/width;++row){
        long double mean=0,variance=0;for(int64_t c=0;c<width;++c)mean+=values[row*width+c];mean/=width;
        for(int64_t c=0;c<width;++c){auto delta=values[row*width+c]-mean;variance+=delta*delta;}
        auto inverse=1/std::sqrt(variance/width+static_cast<long double>(eps));
        for(int64_t c=0;c<width;++c){auto ref=(values[row*width+c]-mean)*inverse;if(!w.empty())ref=ref*w[c]+bias[c];
            check(std::isfinite(ref)&&std::abs(a[row*width+c]-ref)<=2e-5L+2e-5L*std::abs(ref),"independent F32 fixed gate");}
    }
}
Tensor fixture(Backend& b,const std::vector<int64_t>& shape,DType dtype,int slot,int pattern=0){
    std::vector<float> v(shape_numel(shape));for(size_t i=0;i<v.size();++i){
        v[i]=(int((i*17+slot*11+3)%41)-20)/32.f;
        if(slot)v[i]+=(slot==1?1.f:.125f)+float(i%3)/1024.f;
        if(!slot&&pattern==1)v[i]=.5f;
        if(!slot&&pattern==2)v[i]=0;
        if(!slot&&pattern==3)v[i]=(i%2?-.5f:.5f);
        if(!slot&&pattern==4)v[i]=1024.f+(i%2?-.5f:.5f);
    }auto h=host_f32(shape,v);return dtype==f32?h:b.copy_to_host(b.cast(h,dtype));
}
void qualify(CudaBackend& b,const std::string& name,const Tensor& x,const std::vector<Tensor>& params,
             float eps,const std::string& savedir="",const std::vector<Tensor>& imperative_params={}){
    bool affine=!params.empty();auto d=description(x.shape(),x.dtype(),affine?params[0].dtype():f32,affine,b.execution_dtype(),eps);
    auto graph=ng::admit(d);Tensor dx=b.copy_to_device(x,x.dtype());std::vector<Tensor> device;
    for(const auto& p:params)device.push_back(b.copy_to_device(p,p.dtype()));
    const auto& direct_params=imperative_params.empty()?device:imperative_params;
    Tensor ref=b.layer_norm(dx,affine?&direct_params[0]:nullptr,affine?&direct_params[1]:nullptr,eps);b.synchronize();
    if(x.dtype()==f32)independent(b,x,params,eps,ref);
    for(int placement=0;placement<3;++placement){
        for(int repeat=0;repeat<20;++repeat){auto out=ng::evaluate(b,graph,{placement==0?x:dx},placement==1?device:params);
            exact(b,out[0],ref);if(x.dtype()==f32&&repeat==0)independent(b,x,params,eps,out[0]);}
    }
    // Verify external parameter payloads after execution, including Backend conversion/cache use.
    for(size_t i=0;i<params.size();++i)exact(b,device[i],params[i]);
    if(!savedir.empty()){
        auto out=ng::evaluate(b,graph,{dx},device);TensorBundle data{{"input",host(b,x)},{"imperative",host(b,ref)},
            {"checkpoint",host(b,out[0])},{"final",host(b,out[0])}};
        if(affine){data["weight"]=host(b,params[0]);data["bias"]=host(b,params[1]);}
        write_bundle(savedir+"/"+name+".bundle",data);
    }
    std::cout<<"RESULT\t"<<name<<'\t'<<dtype_name(b.execution_dtype())<<'\t'<<dtype_name(x.dtype())<<'\t'
        <<(affine?dtype_name(params[0].dtype()):"none")<<'\t';for(auto dim:x.shape())std::cout<<dim<<',';
    std::cout<<'\t'<<eps<<"\tHost/Device/Mixed\tbitwise\tfinite\t20/20\t"
        <<(x.dtype()==f32?"independent_fixed_gate":"same_route_only")<<"\tPASS\n";
}
void synthetic(DType type){
    CudaBackend b;std::vector<std::vector<int64_t>> shapes={{17},{2,3},{1,2,4},{3,128},{3,1536},{1,3,3072},{1,3,30,64},{3,1},{3,256},{3,257}};
    for(int rank=2;rank<=8;++rank){std::vector<int64_t>s(rank,1);s[rank-2]=3;s.back()=17;shapes.push_back(s);}
    int id=0;for(auto context:{f32,bf16}){
        if(type==bf16&&context==f32)continue;
        b.set_execution_dtype(context);
        for(auto param:{f32,bf16})for(bool affine:{false,true}){
            if((!affine&&param==bf16)||(context==f32&&param==bf16))continue;
            for(const auto& shape:shapes)for(float eps:{1e-6f,1e-5f}){
                auto x=fixture(b,shape,type,0);std::vector<Tensor>ps;
                if(affine)ps={fixture(b,{shape.back()},param,1),fixture(b,{shape.back()},param,2)};
                qualify(b,"synthetic_"+std::to_string(id++),x,ps,eps);
            }
        }
    }
    b.set_execution_dtype(type);for(int pattern=1;pattern<=4;++pattern)
        qualify(b,"pattern_"+std::to_string(pattern),fixture(b,{3,256},type,0,pattern),{},.25f);
    if(type==f32){
        auto x=fixture(b,{2,3},f32,0);auto y=host(b,b.layer_norm(x,nullptr,nullptr,1e-6f));
        for(float corruption:{1.f,std::numeric_limits<float>::infinity()}){
            auto damaged=host_f32(y.shape(),std::vector<float>(y.data_as<float>(),y.data_as<float>()+y.numel()));
            damaged.data_as<float>()[0]+=corruption;bool caught=false;
            try{independent(b,x,{},1e-6f,damaged);}catch(const Error&){caught=true;}
            check(caught,"qualification must reject corrupt/nonfinite capture");
            std::cout<<"CONTROL\tcorrupt_capture_"<<corruption<<"\tqualification\tPASS\n";
        }
    }
}
void real(const std::string& family,const std::string& path,const std::string& outdir){
    auto data=read_bundle(path);CudaBackend b;
    if(family=="hunyuan"){
        b.set_execution_dtype(bf16);for(const char* norm:{"norm1","norm2"}){
            auto w=data.at(std::string(norm)+".weight"),bias=data.at(std::string(norm)+".bias");
            check(w.dtype()==bf16&&bias.dtype()==bf16&&w.shape()==std::vector<int64_t>{3072}&&bias.shape()==w.shape(),"Hunyuan real params");
            for(int tokens:{1,3})for(auto type:{f32,bf16})
                qualify(b,std::string(norm)+"_"+dtype_name(type)+"_"+std::to_string(tokens),fixture(b,{1,tokens,3072},type,0),{w,bias},1e-6f,outdir);
        }
        qualify(b,"hunyuan_no_affine",fixture(b,{1,3,3072},bf16,0),{},1e-6f,outdir);
    }else{
        check(family=="cogvideox","unknown real family");b.set_execution_dtype(f32);
        for(const char* norm:{"norm_q","norm_k"}){
            auto raww=data.at(std::string(norm)+".weight"),rawb=data.at(std::string(norm)+".bias");
            check(raww.dtype()==DType::F16&&rawb.dtype()==DType::F16&&raww.shape()==std::vector<int64_t>{64}&&rawb.shape()==raww.shape(),"CogVideoX real params");
            auto w=b.copy_to_host(b.copy_to_device(raww,f32)),bias=b.copy_to_host(b.copy_to_device(rawb,f32));
            exact(b,w,data.at(std::string(norm)+".weight.f32"));exact(b,bias,data.at(std::string(norm)+".bias.f32"));
            for(auto shape:{std::vector<int64_t>{1,3,30,64},std::vector<int64_t>{2,230,30,64}})
                qualify(b,std::string(norm)+"_"+std::to_string(shape[1]),fixture(b,shape,f32,0),{w,bias},1e-6f,outdir,{raww,rawb});
        }
        for(auto shape:{std::vector<int64_t>{1,3,30,64},std::vector<int64_t>{2,230,30,64}}){
            ng::Description d;d.schema=ng::schema_v1;
            d.values={{shape,f32},{shape,f32},{{64},f32},{{64},f32},{{64},f32},{{64},f32},{shape,f32},{shape,f32}};
            d.inputs={0,1};d.parameters={2,3,4,5};
            d.nodes={{6,ng::LayerNorm{0,2,3,1e-6f}},{7,ng::LayerNorm{1,4,5,1e-6f}}};d.outputs={6,7};
            auto graph=ng::admit(d);auto q=fixture(b,shape,f32,0),k=fixture(b,shape,f32,0,3);
            std::vector<Tensor> params={data.at("norm_q.weight.f32"),data.at("norm_q.bias.f32"),
                data.at("norm_k.weight.f32"),data.at("norm_k.bias.f32")};
            std::vector<Tensor> refs={b.layer_norm(q,&data.at("norm_q.weight"),&data.at("norm_q.bias"),1e-6f),
                b.layer_norm(k,&data.at("norm_k.weight"),&data.at("norm_k.bias"),1e-6f)};
            b.synchronize();std::vector<Tensor> actual;
            for(int repeat=0;repeat<20;++repeat){actual=ng::evaluate(b,graph,{q,k},params);
                for(size_t i=0;i<2;++i)exact(b,actual[i],refs[i]);}
            independent(b,q,{params[0],params[1]},1e-6f,actual[0]);independent(b,k,{params[2],params[3]},1e-6f,actual[1]);
            write_bundle(outdir+"/joint_"+std::to_string(shape[1])+".bundle",{{"q.input",q},{"k.input",k},
                {"q.imperative",host(b,refs[0])},{"k.imperative",host(b,refs[1])},
                {"q.final",host(b,actual[0])},{"k.final",host(b,actual[1])}});
            std::cout<<"JOINT\t"<<shape[1]<<"\tq/k\tF32\tbitwise\tfinite\t20/20\tPASS\n";
        }
    }
}
#endif
}
int main(int argc,char**argv){try{
    std::cout.precision(9);
#ifdef VRHINO_TEST_CUDA
    if(argc==2&&std::string(argv[1])=="f32"){synthetic(f32);return 0;}
    if(argc==2&&std::string(argv[1])=="bf16"){synthetic(bf16);return 0;}
    if(argc==4){real(argv[1],argv[2],argv[3]);return 0;}
#else
    (void)argc;(void)argv;
#endif
    admission();binding();return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
