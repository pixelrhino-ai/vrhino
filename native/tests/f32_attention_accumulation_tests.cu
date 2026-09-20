// Research-only stage capture: compile the actual private CUDA kernels.
// No production observer API or model identity enters the Backend.
#include "../src/backend/cuda/cuda_backend.cu"
#include <iostream>
using namespace vrhino;
struct BlasLease {
    cublasHandle_t handle{};
    BlasLease(){CUBLAS_CHECK(cublasCreate(&handle));CUBLAS_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));}
    ~BlasLease(){if(handle)cublasDestroy(handle);}
};
int main(int argc,char** argv){try {
    require(argc==3,"usage: f32-attention-accumulation-tests INPUT.bundle OUTPUT.bundle");
    auto input=read_bundle(argv[1]);TensorBundle results;
    CudaBackend device; device.set_execution_dtype(DType::F32);Backend& backend=device;BlasLease blas;
    // Scratch is mutable allocation, not an uploaded immutable constant (the
    // Backend may intern identical small constants).
    auto zero=[&](std::vector<int64_t> shape){auto t=backend.allocate_device(shape,DType::F32);CUDA_CHECK(cudaMemsetAsync(t.data(),0,t.bytes()));return t;};
    auto q=backend.copy_to_device(input.at("q"),DType::F32),k=backend.copy_to_device(input.at("k"),DType::F32),v=backend.copy_to_device(input.at("v"),DType::F32);
    int b=q.dim(0),qs=q.dim(1),hs=q.dim(2),d=q.dim(3),ks=k.dim(1),rows=b*qs*hs;
    require(q.dtype()==DType::F32 && k.shape()==v.shape() && k.dim(0)==b && k.dim(2)==hs && k.dim(3)==d,"F32 BSHD contract");
    require(int64_t(rows)*ks<=4*1024*1024,"bounded research capture only");
    float scale=input.count("scale")?*input.at("scale").data_as<float>():1.0f/std::sqrt(float(d));
    bool causal=input.count("causal") && *input.at("causal").data_as<int64_t>()!=0;
    Tensor mask,bias;const uint8_t* mp=nullptr;const float* bp=nullptr;Meta mm{},bm{};
    if(input.count("mask")){mask=backend.copy_to_device(input.at("mask"),DType::Bool);mp=mask.data_as<uint8_t>();mm=meta(mask.shape());}
    if(input.count("bias")){bias=backend.copy_to_device(input.at("bias"),DType::F32);bp=bias.data_as<float>();bm=meta(bias.shape());}
    int tile=cuda_attention_config::kOrderedCandidateKeyTile;
    if(const char* text=std::getenv("VRHINO_CUDA_ATTENTION_KEY_TILE"))tile=std::stoi(text);
    require(tile>=8 && tile<=256 && tile%8==0,"key tile contract");
    auto maximum=zero({rows}),denom=zero({rows}),acc=zero(q.shape()),dc=zero({rows}),ac=zero(q.shape());
    auto scores=zero({b,hs,qs,tile}),previous=zero(scores.shape());
    attention_ordered_initialize_kernel<<<blocks(acc.numel()),kThreads>>>(maximum.data_as<float>(),denom.data_as<float>(),acc.data_as<float>(),rows,d);
    auto all_scores=Tensor::host({b,hs,qs,ks},DType::F32),all_current=Tensor::host(all_scores.shape(),DType::F32),all_previous=Tensor::host(all_scores.shape(),DType::F32);
    float alpha=1,beta=0;
    auto capture=[&](const Tensor& src,Tensor& dst,int base,int count){auto h=backend.copy_to_host(src);for(int row=0;row<rows;++row)std::memcpy(dst.data_as<float>()+row*ks+base,h.data_as<float>()+row*count,count*sizeof(float));};
    for(int base=0;base<ks;base+=tile){int count=std::min(tile,ks-base);
        for(int batch=0;batch<b;++batch){
            CUBLAS_CHECK(cublasSgemmStridedBatched(blas.handle,CUBLAS_OP_T,CUBLAS_OP_N,count,qs,d,&alpha,
                k.data_as<float>()+(int64_t(batch)*ks+base)*hs*d,hs*d,d,
                q.data_as<float>()+int64_t(batch)*qs*hs*d,hs*d,d,&beta,
                scores.data_as<float>()+int64_t(batch)*hs*qs*count,count,int64_t(qs)*count,hs));
        }
        capture(scores,all_scores,base,count);
        attention_ordered_state_kernel<<<blocks(rows),kThreads>>>(scores.data_as<float>(),previous.data_as<float>(),maximum.data_as<float>(),denom.data_as<float>(),mp,bp,b,qs,ks,hs,base,count,scale,causal,mm,bm,dc.data_as<float>());
        capture(scores,all_current,base,count);capture(previous,all_previous,base,count);
        if(d%4==0)attention_ordered_pv_float4_kernel<<<blocks(acc.numel()/4),kThreads>>>(scores.data_as<float>(),previous.data_as<float>(),v.data_as<float>(),acc.data_as<float>(),b,qs,ks,hs,d,base,count,ac.data_as<float>());
        else attention_ordered_pv_kernel<<<blocks(acc.numel()),kThreads>>>(scores.data_as<float>(),previous.data_as<float>(),v.data_as<float>(),acc.data_as<float>(),b,qs,ks,hs,d,base,count,ac.data_as<float>());
        CUDA_CHECK(cudaGetLastError());
    }
    results["qk"]=all_scores;results["current"]=all_current;results["previous"]=all_previous;
    results["denominator"]=backend.copy_to_host(denom);results["maximum"]=backend.copy_to_host(maximum);results["accumulation"]=backend.copy_to_host(acc);
    auto traced=zero(q.shape());
    attention_ordered_finalize_kernel<<<blocks(acc.numel()),kThreads>>>(acc.data_as<float>(),denom.data_as<float>(),traced.data_as<float>(),rows,d);
    results["traced"]=backend.copy_to_host(traced);
    auto actual=backend.attention(q,k,v,mp?&mask:nullptr,causal,scale,bp?&bias:nullptr);
    results["output"]=backend.copy_to_host(actual);
    write_bundle(argv[2],results); // Keep failed capture evidence as well.
    require(results.at("output").bytes()==results.at("traced").bytes() && std::memcmp(results.at("output").data(),results.at("traced").data(),results.at("output").bytes())==0,"captured kernels must reproduce production output bitwise");
    if(input.count("projection")){
        auto flat=backend.reshape(actual,{b,qs,hs*d});
        results["projection"]=backend.copy_to_host(backend.linear(flat,input.at("projection"),nullptr));
    }
    backend.synchronize();write_bundle(argv[2],results);
    std::cout<<"PASS production/stage replay bitwise; B="<<b<<" Q="<<qs<<" K="<<ks<<" H="<<hs<<" D="<<d<<" tile="<<tile<<"\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
