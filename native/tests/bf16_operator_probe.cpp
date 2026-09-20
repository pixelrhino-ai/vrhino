// Same-input generic operator replay. No graph family or model identity.
#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include <fstream>
#include <iostream>
using namespace vrhino;
int main(int argc,char** argv) {try {
    require(argc==4,"usage: bf16-operator-probe INPUT.bundle OUTPUT.bundle POLICY.json");
    std::ifstream stream(argv[3]);require(bool(stream),"Policy missing");
    const auto policy=PrecisionPolicy::from_json(Json::parse(std::string(std::istreambuf_iterator<char>(stream),{})));
    CudaBackend device;device.set_execution_dtype(policy.requested_dtype());Backend& b=device;
    auto source=read_bundle(argv[1]);TensorBundle operands,output;
    for(const auto& [n,t]:source)if(t.dtype()==DType::F32 || t.dtype()==DType::BF16)operands[n]=b.copy_to_device(t,t.dtype());
    for(const auto& [n,t]:source) {
        if(n.size()<5 || n.substr(n.size()-5)!=".kind")continue;
        require(t.dtype()==DType::I64 && t.numel()==1,"Kind contract");
        const auto p=n.substr(0,n.size()-5);
        auto get=[&](const char* k)->const Tensor& {return operands.at(p+"."+k);};
        Tensor result;
        switch(*t.data_as<int64_t>()) {
        case 0: result=b.layer_norm(get("x"),&get("w"),&get("b"),1e-6f);break;
        case 1: result=operation_linear(b,policy,PrecisionOperation::Linear,PrecisionSemantic::TemporaryCompute,get("x"),get("w"),&get("b"));break;
        case 2: result=operation_attention(b,policy,get("q"),get("k"),get("v"));break;
        case 3: {
            auto h=b.linear(get("x"),get("w"),&get("b"));
            if(source.count(p+".trace"))output[p+".linear0"]=b.copy_to_host(h);
            h=b.activation(h,Activation::GeluTanh);
            if(source.count(p+".trace"))output[p+".activation"]=b.copy_to_host(h);
            result=b.linear(h,get("w2"),&get("b2"));break;}
        case 4: result=gated_residual(b,policy,get("x"),get("branch"),get("gate"));break;
        case 5: result=attention_norm(b,policy,get("x"),&get("w"),1e-6f);break;
        case 6: result=b.activation(get("x"),Activation::GeluTanh);break;
        case 7: result=b.softmax(get("x"),-1);break;
        case 8: result=operation_rope(b,policy,get("x"),get("cos"),get("sin"));break;
        default:throw Error("Unknown operator kind");
        }
        b.synchronize();output[p]=b.copy_to_host(result);
    }
    require(!output.empty(),"Empty operator set");b.synchronize();write_bundle(argv[2],output);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
