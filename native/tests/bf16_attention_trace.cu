// Test-only stage observer. No changes to production kernels or dispatch.
#include "../src/backend/cuda/cuda_backend.cu"
#include "bf16_attention_observer.generated.cuh"
#include <iostream>
using namespace vrhino;
int main(int argc,char** argv){try {
    require(argc==3,"usage: bf16-attention-trace INPUT OUTPUT");
    auto input=read_bundle(argv[1]);TensorBundle results;
    CudaBackend device;device.set_execution_dtype(DType::BF16);Backend& b=device;
    for(const auto& [name,kind]:input) {
        if(name.size()<5 || name.substr(name.size()-5)!=".kind")continue;
        require(kind.dtype()==DType::I64 && *kind.data_as<int64_t>()==2,"attention-only fixture");
        const auto prefix=name.substr(0,name.size()-5);
        auto q=b.copy_to_device(input.at(prefix+".q"),DType::BF16);
        auto k=b.copy_to_device(input.at(prefix+".k"),DType::BF16);
        auto v=b.copy_to_device(input.at(prefix+".v"),DType::BF16);
        require(q.ndim()==4 && q.dim(0)==1 && q.dim(1)==1 && q.dim(2)==1 && q.dim(3)<=128 &&
                k.shape()==v.shape() && k.dim(0)==1 && k.dim(2)==1 && k.dim(3)==q.dim(3) && k.dim(1)<=4096,
                "bounded single-row observer contract");
        const int width=q.dim(3),keys=k.dim(1);const float scale=1.0f/std::sqrt(float(width));
        std::vector<Tensor> stages;
        for(int i=0;i<6;++i)stages.push_back(b.allocate_device({1,keys},DType::F32));
        for(int i=0;i<2;++i)stages.push_back(b.allocate_device(q.shape(),DType::F32));
        auto traced=b.allocate_device(q.shape(),DType::BF16);
        attention_observer_kernel<<<1,256,cuda_attention_config::workspace_bytes(width)>>>(
            q.data_as<__nv_bfloat16>(),k.data_as<__nv_bfloat16>(),v.data_as<__nv_bfloat16>(),
            nullptr,static_cast<const __nv_bfloat16*>(nullptr),traced.data_as<__nv_bfloat16>(),
            1,1,keys,1,width,1,false,scale,Meta{},Meta{},
            stages[0].data_as<float>(),stages[1].data_as<float>(),stages[2].data_as<float>(),
            stages[3].data_as<float>(),stages[4].data_as<float>(),stages[5].data_as<float>(),
            stages[6].data_as<float>(),stages[7].data_as<float>());
        CUDA_CHECK(cudaGetLastError());b.synchronize();
        auto actual=b.attention(q,k,v);b.synchronize();
        auto nh=b.copy_to_host(actual),th=b.copy_to_host(traced);
        const char* names[]={"qk","score","current","previous","denominator","maximum","accumulation","pre_round"};
        for(int i=0;i<8;++i)results[prefix+"."+names[i]]=b.copy_to_host(stages[i]);
        results[prefix+".native"]=nh;results[prefix+".observed"]=th;
        results[prefix+".q"]=b.copy_to_host(q);results[prefix+".k"]=b.copy_to_host(k);
        results[prefix+".v"]=b.copy_to_host(v);
        results[prefix+".scale"]=scalar_f32(scale);
        write_bundle(argv[2],results);
        require(nh.bytes()==th.bytes() && std::memcmp(nh.data(),th.data(),nh.bytes())==0,
                "observer differs from production; reject attribution");
        auto rounded=b.copy_to_host(b.cast(stages[7],DType::BF16));
        require(std::memcmp(nh.data(),rounded.data(),nh.bytes())==0,"pre-round cast differs");
        std::cout<<prefix<<" production_observer_exact=PASS pre_round_cast_exact=PASS\n";
    }
    require(!results.empty(),"Empty observer fixture");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
