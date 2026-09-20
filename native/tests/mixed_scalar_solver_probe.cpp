// Research-only generic scalar/solver qualification. No model, selector or policy override.
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/sampling.h"
#include "vrhino/tensor_util.h"
#include <iostream>
#include <fstream>
#include "vrhino/json.h"
using namespace vrhino;
int main(int argc,char** argv) { try {
    require(argc==3 || argc==4,"usage: mixed-scalar-solver-probe INPUT OUTPUT [POLICY]");
    auto policy=PrecisionPolicy::unqualified_default(DType::BF16);
    if(argc==4) {std::ifstream f(argv[3]);require(bool(f),"Policy missing");policy=PrecisionPolicy::from_json(Json::parse(std::string(std::istreambuf_iterator<char>(f),{})));}
    CudaBackend b; b.set_execution_dtype(DType::BF16); SamplingPrimitives s(b,policy);
    const auto input=read_bundle(argv[1]); TensorBundle output;
    // Configure observation before uploads create pending CUDA events.
    for(const auto& [key,value]:input) if(key.ends_with(".kind") && read_scalar_i64(value)==2) {
        b.enable_profiling(true); break;
    }
    for(const auto& [key,kind]:input) {
        if(!key.ends_with(".kind")) continue;
        const auto p=key.substr(0,key.size()-5);
        auto get=[&](const std::string& n)->const Tensor&{return input.at(p+"."+n);};
        auto gpu=[&](const std::string& n){const auto& t=get(n);return b.copy_to_device(t,t.dtype());};
        auto save=[&](const std::string& n,const Tensor& t){b.synchronize();output[p+"."+n]=b.copy_to_host(t);};
        const Tensor x=gpu("x"),y=gpu("y");
        if(read_scalar_i64(kind)==0) {
            const Tensor host=get("scalar"), device=b.copy_to_device(host,DType::F32);
            for(const auto& [name,scalar]:std::vector<std::pair<std::string,Tensor>>{{"host",host},{"device",device}}) {
                save(name+".mul",b.mul(x,scalar));save(name+".mul_reverse",b.mul(scalar,x));
                save(name+".add",b.add(x,scalar));save(name+".add_reverse",b.add(scalar,x));
                save(name+".div",b.div(x,scalar));
                if(argc==4) for(const auto& [opname,op]:std::vector<std::pair<std::string,ScalarBinaryOperation>>{{"mul",ScalarBinaryOperation::Multiply},{"add",ScalarBinaryOperation::Add},{"div",ScalarBinaryOperation::Divide}})
                    save("semantic."+name+"."+opname,precision_scalar_binary(b,policy,PrecisionScalarRole::SolverCoefficient,op,x,scalar));
            }
            // Counterfactual realized with existing primitives, no production change:
            // preserve scalar F32, do each elementary operation F32, round its result.
            Tensor xf=b.cast(x,DType::F32);
            save("preserve.mul",b.cast(b.mul(xf,device),x.dtype()));
            save("preserve.add",b.cast(b.add(xf,device),x.dtype()));
            save("preserve.div",b.cast(b.div(xf,device),x.dtype()));
        } else if(read_scalar_i64(kind)==3) {
            save("host_only",b.mul(get("x"),get("scalar")));
            if(argc==4) save("semantic",precision_scalar_binary(b,policy,PrecisionScalarRole::TimestepScale,ScalarBinaryOperation::Multiply,get("x"),get("scalar")));
            save("explicit_f32",b.mul(b.copy_to_device(get("x"),DType::F32),get("scalar")));
        } else if(read_scalar_i64(kind)==2) {
            auto bias=gpu("bias");const DType compute=x.dtype();
            save("linear",b.linear(x,y,&bias,compute,compute));
            save("linear_no_bias",b.linear(x,y,nullptr,compute,compute));
            auto rounded_bias=b.cast(bias,compute);
            save("accumulator",b.linear(x,y,&rounded_bias,compute,DType::F32));
            save("no_bias",b.linear(x,y,nullptr,compute,DType::F32));
        } else if(read_scalar_i64(kind)==4) {
            // Matched accumulator replay: isolate conversion from reduction.
            save("cast",b.cast(x,DType::BF16));
        } else if(read_scalar_i64(kind)==5) {
            // Generic F32 modulation/broadcast replay on identical operands.
            auto combined=b.add(y,gpu("external"));save("combined",combined);
            auto shift=b.slice(combined,1,0,1),scale=b.slice(combined,1,1,2);
            auto scale_one=b.add(scalar_f32(1.0f),scale);save("scale_one",scale_one);
            auto scaled=b.mul(x,scale_one);save("scaled",scaled);
            save("modulated",b.add(scaled,shift));
        } else {
            const float c=read_scalar_f32(get("scalar"));
            const Tensor z=gpu("z");
            save("flow",s.flow_to_x0(x,y,c));save("v",s.v_to_x0(x,y,c));
            save("epsilon",s.epsilon_to_x0(x,y,c));
            save("affine",s.affine_first_order(x,y,c,1.0f-c));
            save("cfg",s.cfg_combine(x,y,3.1415927f));
            save("linear",s.linear_combine({x,y,z},{c,1.0f-c,-0.1234567f}));
            save("euler",s.euler_update(x,y,scalar_f32(-0.012076616287231445f),false));
            const std::vector<float> sigmas={0.8778772950172424f,0.865800678730011f,0.8522725701332092f,0.8370144367218018f};
            save("predictor1",s.multistep_predictor(x,{y},z,sigmas,0,1));
            save("predictor2",s.multistep_predictor(x,{z,y},z,sigmas,1,2));
            save("corrector1",s.multistep_corrector(x,x,{y},z,sigmas,1,1,x));
            save("corrector2",s.multistep_corrector(x,x,{z,y},z,sigmas,2,2,x));
        }
    }
    require(!output.empty(),"Empty corpus");
    if(b.profiling_enabled()) for(const auto& entry:b.profile_stats())
        std::cout<<"profile "<<entry.first<<'\n';
    TensorBundle part;size_t part_index=0;
    for(const auto& [name,tensor]:output) {
        part.emplace(name,tensor);
        if(part.size()==500) {write_bundle(std::string(argv[2])+"."+std::to_string(part_index++),part);part.clear();}
    }
    if(!part.empty())write_bundle(std::string(argv[2])+"."+std::to_string(part_index),part);
    std::cout<<"outputs="<<output.size()<<"\n"; return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
