#include "vrhino/backend/cuda_backend.h"
#include "vrhino/execution.h"
#include "vrhino/tensor_util.h"
#include "vrhino/error.h"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <cstdlib>
#include <iostream>
#include <set>
#include <new>
#include <sys/wait.h>
#include <unistd.h>
using namespace vrhino;
namespace probe {
enum class Fault { None,Allocation,Event,EventFlags,Record,Upload,Staging,Stream,Publication };
Fault fault=Fault::None;
int calls=0,at=1,remaining=1;
// GNU/ELF qualification target: derive the container allocation size from the
// exact resource-entry layout, excluding Tensor vector metadata allocations.
struct ObservedWeightEntry {Tensor value;size_t bytes=0;uint64_t last_use=0;bool prefetched=false;cudaEvent_t ready{};};
using ObservedWeightKey=std::pair<const void*,DType>;
constexpr size_t weight_node_bytes=sizeof(std::_Rb_tree_node<std::pair<const ObservedWeightKey,ObservedWeightEntry>>);
size_t failing_new_size=0;
bool injected=false,persistent_sync_failure=false;
std::set<void*> device,pinned,events,streams,pools,blas,blaslt,dnn;
size_t peak_events=0,peak_device=0,peak_owned=0;
int double_free=0;
bool hit(Fault kind) {
    if(fault!=kind) return false;
    if(++calls<at) return false;
    injected=true;
    if(--remaining==0) fault=Fault::None;
    return true;
}
void arm(Fault kind,int ordinal=1,int repetitions=1) {
    fault=kind;calls=0;at=ordinal;remaining=repetitions;injected=false;failing_new_size=0;
}
void remove(std::set<void*>& set,void* p) {if(p && !set.erase(p)) ++double_free;}
size_t count(){return device.size()+pinned.size()+events.size()+streams.size()+pools.size()+blas.size()+blaslt.size()+dnn.size();}
void add(std::set<void*>& set,void* pointer){set.insert(pointer);peak_owned=std::max(peak_owned,count());peak_events=std::max(peak_events,events.size());peak_device=std::max(peak_device,device.size());}
void baseline(const char* label){require(count()==0 && double_free==0,std::string("Owned resources remain/double free: ")+label);}
}
// Keep the paired malloc/free fault-injection replacements opaque to callers.
// Inlining delete into Release callers makes GCC diagnose a spurious
// mismatched-new-delete despite this replacement new allocating with malloc.
[[gnu::noinline]] void* operator new(std::size_t n) {
    if(n==probe::failing_new_size && n>0) {probe::failing_new_size=0;probe::injected=true;throw std::bad_alloc();}
    if(void* p=std::malloc(n?n:1))return p;
    throw std::bad_alloc();
}
[[gnu::noinline]] void operator delete(void* p) noexcept {std::free(p);}
[[gnu::noinline]] void operator delete(void* p,std::size_t) noexcept {std::free(p);}
#define REAL(ret,name,args) extern "C" ret __real_##name args
REAL(cudaError_t,cudaMalloc,(void**,size_t));
extern "C" cudaError_t __wrap_cudaMalloc(void** p,size_t n){if(probe::hit(probe::Fault::Allocation))return cudaErrorMemoryAllocation;auto r=__real_cudaMalloc(p,n);if(r==cudaSuccess){probe::add(probe::device,*p);probe::peak_device=std::max(probe::peak_device,probe::device.size());}return r;}
REAL(cudaError_t,cudaFree,(void*));
extern "C" cudaError_t __wrap_cudaFree(void* p){auto r=__real_cudaFree(p);if(r==cudaSuccess)probe::remove(probe::device,p);return r;}
REAL(cudaError_t,cudaMallocFromPoolAsync,(void**,size_t,cudaMemPool_t,cudaStream_t));
extern "C" cudaError_t __wrap_cudaMallocFromPoolAsync(void** p,size_t n,cudaMemPool_t pool,cudaStream_t stream){if(probe::hit(probe::Fault::Allocation))return cudaErrorMemoryAllocation;auto r=__real_cudaMallocFromPoolAsync(p,n,pool,stream);if(r==cudaSuccess)probe::add(probe::device,*p);return r;}
REAL(cudaError_t,cudaFreeAsync,(void*,cudaStream_t));
extern "C" cudaError_t __wrap_cudaFreeAsync(void* p,cudaStream_t stream){auto r=__real_cudaFreeAsync(p,stream);if(r==cudaSuccess)probe::remove(probe::device,p);return r;}
REAL(cudaError_t,cudaHostAlloc,(void**,size_t,unsigned));
extern "C" cudaError_t __wrap_cudaHostAlloc(void** p,size_t n,unsigned flags){if(probe::hit(probe::Fault::Staging))return cudaErrorMemoryAllocation;auto r=__real_cudaHostAlloc(p,n,flags);if(r==cudaSuccess)probe::add(probe::pinned,*p);return r;}
REAL(cudaError_t,cudaFreeHost,(void*));
extern "C" cudaError_t __wrap_cudaFreeHost(void* p){auto r=__real_cudaFreeHost(p);if(r==cudaSuccess)probe::remove(probe::pinned,p);return r;}
REAL(cudaError_t,cudaEventCreate,(cudaEvent_t*));
extern "C" cudaError_t __wrap_cudaEventCreate(cudaEvent_t* p){if(probe::hit(probe::Fault::Event))return cudaErrorMemoryAllocation;auto r=__real_cudaEventCreate(p);if(r==cudaSuccess){probe::add(probe::events,*p);probe::peak_events=std::max(probe::peak_events,probe::events.size());}return r;}
REAL(cudaError_t,cudaEventCreateWithFlags,(cudaEvent_t*,unsigned));
extern "C" cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* p,unsigned flags){if(probe::hit(probe::Fault::EventFlags))return cudaErrorMemoryAllocation;auto r=__real_cudaEventCreateWithFlags(p,flags);if(r==cudaSuccess)probe::add(probe::events,*p);return r;}
REAL(cudaError_t,cudaEventDestroy,(cudaEvent_t));
extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t p){auto r=__real_cudaEventDestroy(p);if(r==cudaSuccess)probe::remove(probe::events,p);return r;}
REAL(cudaError_t,cudaEventRecord,(cudaEvent_t,cudaStream_t));
extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t p,cudaStream_t s){if(probe::hit(probe::Fault::Record))return cudaErrorInvalidValue;return __real_cudaEventRecord(p,s);}
REAL(cudaError_t,cudaMemcpyAsync,(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t));
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* d,const void* s,size_t n,cudaMemcpyKind k,cudaStream_t stream){if(probe::hit(probe::Fault::Upload))return cudaErrorInvalidValue;return __real_cudaMemcpyAsync(d,s,n,k,stream);}
REAL(cudaError_t,cudaStreamWaitEvent,(cudaStream_t,cudaEvent_t,unsigned));
extern "C" cudaError_t __wrap_cudaStreamWaitEvent(cudaStream_t s,cudaEvent_t e,unsigned f){auto r=__real_cudaStreamWaitEvent(s,e,f);if(probe::hit(probe::Fault::Publication)){probe::injected=false;probe::failing_new_size=probe::weight_node_bytes;}return r;}
REAL(cudaError_t,cudaStreamCreateWithFlags,(cudaStream_t*,unsigned));
extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* p,unsigned f){if(probe::hit(probe::Fault::Stream))return cudaErrorMemoryAllocation;auto r=__real_cudaStreamCreateWithFlags(p,f);if(r==cudaSuccess)probe::add(probe::streams,*p);return r;}
REAL(cudaError_t,cudaStreamDestroy,(cudaStream_t));
extern "C" cudaError_t __wrap_cudaStreamDestroy(cudaStream_t p){auto r=__real_cudaStreamDestroy(p);if(r==cudaSuccess)probe::remove(probe::streams,p);return r;}
REAL(cudaError_t,cudaMemPoolCreate,(cudaMemPool_t*,const cudaMemPoolProps*));
extern "C" cudaError_t __wrap_cudaMemPoolCreate(cudaMemPool_t* p,const cudaMemPoolProps* props){auto r=__real_cudaMemPoolCreate(p,props);if(r==cudaSuccess)probe::add(probe::pools,*p);return r;}
REAL(cudaError_t,cudaMemPoolDestroy,(cudaMemPool_t));
extern "C" cudaError_t __wrap_cudaMemPoolDestroy(cudaMemPool_t p){auto r=__real_cudaMemPoolDestroy(p);if(r==cudaSuccess)probe::remove(probe::pools,p);return r;}
REAL(cublasStatus_t,cublasCreate_v2,(cublasHandle_t*));
extern "C" cublasStatus_t __wrap_cublasCreate_v2(cublasHandle_t* p){auto r=__real_cublasCreate_v2(p);if(r==CUBLAS_STATUS_SUCCESS)probe::add(probe::blas,*p);return r;}
REAL(cublasStatus_t,cublasDestroy_v2,(cublasHandle_t));
extern "C" cublasStatus_t __wrap_cublasDestroy_v2(cublasHandle_t p){auto r=__real_cublasDestroy_v2(p);if(r==CUBLAS_STATUS_SUCCESS)probe::remove(probe::blas,p);return r;}
REAL(cublasStatus_t,cublasLtCreate,(cublasLtHandle_t*));
extern "C" cublasStatus_t __wrap_cublasLtCreate(cublasLtHandle_t* p){auto r=__real_cublasLtCreate(p);if(r==CUBLAS_STATUS_SUCCESS)probe::add(probe::blaslt,*p);return r;}
REAL(cublasStatus_t,cublasLtDestroy,(cublasLtHandle_t));
extern "C" cublasStatus_t __wrap_cublasLtDestroy(cublasLtHandle_t p){auto r=__real_cublasLtDestroy(p);if(r==CUBLAS_STATUS_SUCCESS)probe::remove(probe::blaslt,p);return r;}
REAL(cudnnStatus_t,cudnnCreate,(cudnnHandle_t*));
extern "C" cudnnStatus_t __wrap_cudnnCreate(cudnnHandle_t* p){auto r=__real_cudnnCreate(p);if(r==CUDNN_STATUS_SUCCESS)probe::add(probe::dnn,*p);return r;}
REAL(cudnnStatus_t,cudnnDestroy,(cudnnHandle_t));
extern "C" cudnnStatus_t __wrap_cudnnDestroy(cudnnHandle_t p){auto r=__real_cudnnDestroy(p);if(r==CUDNN_STATUS_SUCCESS)probe::remove(probe::dnn,p);return r;}
REAL(cudaError_t,cudaDeviceSynchronize,());
extern "C" cudaError_t __wrap_cudaDeviceSynchronize(){if(probe::persistent_sync_failure)return cudaErrorUnknown;return __real_cudaDeviceSynchronize();}
namespace {
constexpr size_t MiB=1024*1024;
struct Backing {std::vector<float> a=std::vector<float>(256*256,0.5f);Tensor tensor=Tensor::borrowed(a.data(),a.size()*4,{256,256},DType::F32);};
void setup(CudaBackend& b,bool staging=true){MemoryRuntimeOptions o;o.enabled=true;o.prefetch=false;o.host_staging=staging;b.configure_memory_runtime({4*MiB,MiB,16*MiB,MiB,MiB},o);}
Tensor quantized(const std::shared_ptr<Backing>& source){auto t=Tensor::borrowed(source->a.data(),256*256,{256,256},DType::U8);auto q=std::make_shared<QuantizationInfo>();q->type=QuantType::INT8Symmetric;q->logical_dtype=DType::BF16;q->compute_dtype=DType::BF16;q->accumulation_dtype=DType::F32;q->axis=1;q->group_size=64;q->granularity="per_group";q->symmetric=true;q->zero_point_mode="none";q->packing_layout="byte_twos_complement";q->scales=host_f32({256,4},std::vector<float>(1024,0.01f));t.set_quantization(q);return t;}
void case_upload(const char* label,probe::Fault fault,int ordinal=1,bool quant=false){
    probe::baseline(label);std::weak_ptr<Backing> weak;
    {
        CudaBackend b;setup(b);auto source=std::make_shared<Backing>();weak=source;
        auto tensor=quant?quantized(source):source->tensor;b.retain_resource_owners({source});source.reset();
        probe::arm(fault,ordinal,fault==probe::Fault::Allocation?2:1);
        bool threw=false;try{(void)b.copy_to_device(tensor,quant?DType::BF16:DType::F32);}catch(const Error&){threw=true;}
        require(threw && probe::injected,"Fault not exercised");require(!weak.expired(),"Borrowed source freed on failure");
        require(b.resource_session_terminal(),"Failed upload session reusable");
        bool refused=false;try{(void)b.allocate_device({1},DType::F32);}catch(const Error&){refused=true;}require(refused,"Terminal session dispatched device work");
        require(b.weight_cache_resident_bytes()==0,"Partially initialized weight published");
    }
    require(weak.expired(),"Session lease escaped teardown");probe::baseline(label);std::cout<<"PASS "<<label<<" owned_after=0\n";
}
void publication(){
    probe::baseline("cache publication");
    {CudaBackend b;setup(b,false);auto a=std::make_shared<Backing>(),c=std::make_shared<Backing>();b.retain_resource_owners({a,c});
     (void)b.copy_to_device(a->tensor,DType::F32);b.synchronize();
     const auto resident=b.weight_cache_resident_bytes(),copies=b.memory_runtime_stats().upload_copies;
     // Warm transfer-vector capacity; fail the actual weight-cache tree node
     // allocation after H2D. Tensor metadata/error-string allocations succeed.
     // Counters require completed transfer publication before the throw.
     probe::arm(probe::Fault::Publication);bool threw=false;
     try{(void)b.copy_to_device(c->tensor,DType::F32);}catch(const std::bad_alloc&){threw=true;}
     require(threw && probe::injected && b.memory_runtime_stats().upload_copies==copies+1,"Cache insertion injection missed publication boundary");
     require(b.weight_cache_resident_bytes()==resident && b.resource_session_terminal(),"Failed cache publication corrupted residency");}
    probe::baseline("cache publication");std::cout<<"PASS cache_insertion_bad_alloc owned_after=0\n";
}
void leases(){
    probe::baseline("leases");std::weak_ptr<Backing> weak;
    {
        CudaBackend b;setup(b);Tensor tensor;
        {auto owner=std::make_shared<Backing>();weak=owner;tensor=owner->tensor;b.retain_resource_owners({owner});}
        require(!weak.expired(),"Registered source released");
        auto first=b.copy_to_host(b.copy_to_device(tensor,DType::F32));
        auto hit=b.copy_to_host(b.copy_to_device(tensor,DType::F32));
        require(hash_host_tensor_content(first)==hash_host_tensor_content(hit) && b.weight_cache_hits()>0,"Leased address cache mismatch");
        tensor={};require(!weak.expired(),"Lease tied only to Tensor handle");
    }
    require(weak.expired(),"Backend did not release source lease");probe::baseline("leases");std::cout<<"PASS explicit_source_lease owned_after=0\n";
}
class LeaseEndpoint final : public Denoiser {
public:
    LeaseEndpoint(CudaBackend& backend,Tensor parameter,bool fail=false,int* calls=nullptr):backend(backend),parameter(std::move(parameter)),fail(fail),calls(calls){}
    std::vector<Tensor> evaluate(const Tensor&,const Tensor&) override {
        if(calls) ++*calls;
        auto value=backend.copy_to_device(parameter,DType::F32);
        if(fail) throw Error("synthetic evaluation failure after upload");
        return {value,value};
    }
    CudaBackend& backend; Tensor parameter;bool fail;int* calls;
};
void admitted_leases(){
    probe::baseline("admitted leases");
    std::weak_ptr<Backing> weak_a,weak_b;
    {
        CudaBackend b;setup(b);Tensor saved;
        {
            auto a=std::make_shared<Backing>(),other=std::make_shared<Backing>();
            weak_a=a;weak_b=other;saved=a->tensor;
            ComponentInterface interface{{{256,256},DType::F32},{{},DType::I64},
                {{{256,256},DType::F32},{{256,256},DType::F32}},GuidanceMode::CFG,std::nullopt,0};
            auto graph=std::make_shared<const ComponentGraphDefinition>(interface,
                std::vector<ExecutionTensorContract>{{{256,256},DType::F32}},
                [&](const std::vector<Tensor>& p){return std::make_unique<LeaseEndpoint>(b,p.at(0));});
            std::vector<ComponentInstance> instances;
            instances.push_back(ComponentInstance::bind({0},{10},graph,{a->tensor},{a}));
            instances.push_back(ComponentInstance::bind({1},{20},graph,{other->tensor},{other}));
            ExecutionContext context(std::move(instances));a.reset();other.reset();
            SamplingProgram p;p.latent_shape={256,256};p.steps=3;p.guidance_mode=GuidanceMode::CFG;
            p.guidance_coefficients={0,1};p.model_timesteps={scalar_i64(3),scalar_i64(2),scalar_i64(1)};
            p.update_deltas={-0.1f,-0.1f,-0.1f};
            (void)SamplingRuntime(b).run(context,ExecutionProgram::per_step({{0},{1},{0}}),p);
        }
        require(!weak_a.expired() && !weak_b.expired() && b.resource_owner_count()==2,
                "Execution context destruction released Backend cache sources");
        auto value=b.copy_to_host(b.copy_to_device(saved,DType::F32));
        require(value.data_as<float>()[0]==0.5f,"Promoted lease cache became stale");
    }
    require(weak_a.expired() && weak_b.expired(),"Promoted source leases leaked");
    probe::baseline("admitted leases");std::cout<<"PASS automatic_A_B_A_session_leases owned_after=0\n";
}
void runtime_failure(bool cancel){
    probe::baseline("runtime failure");std::weak_ptr<Backing> weak_a,weak_b;
    {CudaBackend b;setup(b);int evaluations=0;Tensor saved;
     {
      auto a=std::make_shared<Backing>(),other=std::make_shared<Backing>();weak_a=a;weak_b=other;saved=a->tensor;
      ComponentInterface interface{{{256,256},DType::F32},{{},DType::I64},
          {{{256,256},DType::F32},{{256,256},DType::F32}},GuidanceMode::CFG,std::nullopt,0};
      auto graph=std::make_shared<const ComponentGraphDefinition>(interface,
          std::vector<ExecutionTensorContract>{{{256,256},DType::F32}},
          [&](const std::vector<Tensor>& p){return std::make_unique<LeaseEndpoint>(b,p.at(0),!cancel,&evaluations);});
      std::vector<ComponentInstance> instances;
      instances.push_back(ComponentInstance::bind({0},{10},graph,{a->tensor},{a}));
      instances.push_back(ComponentInstance::bind({1},{20},graph,{other->tensor},{other}));
      ExecutionContext context(std::move(instances));a.reset();other.reset();
      SamplingProgram p;p.latent_shape={256,256};p.steps=3;p.guidance_mode=GuidanceMode::CFG;
      p.guidance_coefficients={0,1};p.model_timesteps={scalar_i64(3),scalar_i64(2),scalar_i64(1)};p.update_deltas={-0.1f,-0.1f,-0.1f};
      SamplingRuntime runtime(b);if(cancel)runtime.set_cancellation_requested([&]{return evaluations>0;});
      bool failed=false;try{runtime.run(context,ExecutionProgram::per_step({{0},{1},{0}}),p);}catch(const Error&){failed=true;}
      require(failed && evaluations==1 && cudaStreamQuery(nullptr)==cudaSuccess,"Runtime failure did not drain / fell back");
     }
     require(!weak_a.expired() && !weak_b.expired(),"Failure lost source lease");
     require(!b.resource_session_terminal(),"Successfully drained host cancellation/evaluation error needlessly retired session");
     auto value=b.copy_to_host(b.copy_to_device(saved,DType::F32));require(value.data_as<float>()[0]==0.5f,"Drained session stale cache");}
    require(weak_a.expired() && weak_b.expired(),"Failed runtime source lease leaked");probe::baseline("runtime failure");
    std::cout<<"PASS "<<(cancel?"cancellation_partial_preparation":"execution_error_after_upload")<<" owned_after=0 no_fallback=yes\n";
}
void budget_pressure(){
    probe::baseline("budget admission");
    {CudaBackend b;setup(b);const auto before=probe::count();ResourceEstimate e;e.resident_cache=2*MiB;
     e.activations=MiB;e.workspace=MiB;e.upload_temporary=MiB;e.sampling_state=MiB;e.component_allocations=MiB;
     e.prepared=MiB;e.source_backing=8*MiB;e.host_staging=MiB;
     bool rejected=false;try{b.set_resource_admission({e,{e.device_peak()-1,e.host_peak()}});}catch(const Error&){rejected=true;}
     require(rejected && probe::count()==before && b.weight_upload_bytes()==0 && !b.resource_session_terminal(),
             "Insufficient estimate budget acquired device resources or poisoned session");
     b.set_resource_admission({e,{e.device_peak(),e.host_peak()}});b.admit_execution_resources();}
    probe::baseline("budget admission");std::cout<<"PASS insufficient_budget_reject_before_upload owned_after=0\n";
}
void owned_upload_lifetime(){
    probe::baseline("owned upload");
    {CudaBackend b;setup(b,false);std::weak_ptr<const void> weak;
     {auto source=host_f32({128},std::vector<float>(128,1.0f));weak=source.storage_lease();
      (void)b.copy_to_device(source,DType::F32);}
     require(!weak.expired(),"Asynchronous owned upload released its host storage early");
     b.synchronize();require(weak.expired(),"Completed upload kept transient source storage");}
    probe::baseline("owned upload");std::cout<<"PASS owned_upload_storage_lifetime owned_after=0\n";
}
void auxiliary(){
    probe::baseline("constructor");probe::arm(probe::Fault::Stream);bool caught=false;try{CudaBackend b;}catch(const Error&){caught=true;}require(caught && probe::injected,"Constructor injection missing");probe::baseline("constructor");std::cout<<"PASS partial_constructor owned_after=0\n";
    {CudaBackend b;probe::arm(probe::Fault::EventFlags);caught=false;try{b.create_fence();}catch(const Error&){caught=true;}require(caught && probe::injected,"Fence injection missing");}probe::baseline("fence");std::cout<<"PASS fence_creation owned_after=0\n";
    {CudaBackend b;b.enable_profiling(true);probe::arm(probe::Fault::Event,2);caught=false;try{(void)b.add(scalar_f32(1),scalar_f32(2));}catch(const Error&){caught=true;}require(caught && probe::injected,"Profile injection missing");}probe::baseline("profile");std::cout<<"PASS profile_partial_creation owned_after=0\n";
    {CudaBackend b;b.enable_profiling(true);b.profile_region_begin("generic.region");probe::arm(probe::Fault::Event);caught=false;try{b.profile_region_end();}catch(const Error&){caught=true;}require(caught && probe::injected,"Region injection missing");}probe::baseline("region");std::cout<<"PASS profile_region_failure owned_after=0\n";
    {CudaBackend b;auto dst=b.allocate_device({64},DType::F32);auto src=host_f32({64},std::vector<float>(64,1));probe::arm(probe::Fault::Upload);caught=false;try{b.copy(src,dst,Backend::CopyMode::Asynchronous);}catch(const Error&){caught=true;}require(caught && probe::injected,"Async copy injection missing");}probe::baseline("copy");std::cout<<"PASS public_copy_failure owned_after=0\n";
    {CudaBackend b;TransferFence fence;{auto dst=b.allocate_device({64},DType::F32);auto src=host_f32({64},std::vector<float>(64,1));fence=b.copy(src,dst,Backend::CopyMode::Asynchronous);}require(!probe::device.empty(),"Async destination not retained");b.destroy_fence(fence);}probe::baseline("copy lifetime");std::cout<<"PASS public_copy_buffer_lifetime owned_after=0\n";
    {CudaBackend b;setup(b);PreparedTensorCache cache;PreparedTensorKey key;key.operation="synthetic.prepared";key.target_backend=b.name();caught=false;try{cache.prepare(key,[&]()->PreparedTensorMaterialization{auto t=b.copy_to_device(scalar_f32(3),DType::F32);throw Error("prepared builder failure");});}catch(const Error&){caught=true;}require(caught && cache.stats().resident_entries==0,"Failed prepared resource published");}probe::baseline("prepared");std::cout<<"PASS prepared_builder_failure owned_after=0\n";
}
std::weak_ptr<Backing> terminal_owner;
[[noreturn]] void terminal_child(){
    std::set_terminate([]{_exit(!terminal_owner.expired() && probe::count()>0?77:78);});
    {CudaBackend b;setup(b);auto owner=std::make_shared<Backing>();terminal_owner=owner;b.retain_resource_owners({owner});(void)b.copy_to_device(owner->tensor,DType::F32);owner.reset();probe::persistent_sync_failure=true;
     bool caught=false;try{b.synchronize();}catch(const Error&){caught=true;}require(caught && b.resource_session_terminal(),"Completion failure did not retire session");}
    _exit(79);
}
}
int main(int argc,char** argv){
    if(argc==2)terminal_child();
    try {
        case_upload("first_allocation",probe::Fault::Allocation);
        case_upload("middle_allocation",probe::Fault::Allocation,2,true);
        case_upload("first_event",probe::Fault::Event);
        case_upload("second_event",probe::Fault::Event,2);
        case_upload("staging_completion_event",probe::Fault::Event,3);
        case_upload("staging_allocation",probe::Fault::Staging);
        case_upload("upload",probe::Fault::Upload);
        case_upload("second_upload",probe::Fault::Upload,2,true);
        case_upload("record_after_submission",probe::Fault::Record,2);
        publication();leases();admitted_leases();runtime_failure(false);runtime_failure(true);budget_pressure();owned_upload_lifetime();auxiliary();
        const auto pid=fork();require(pid>=0,"fork failed");if(pid==0){execl(argv[0],argv[0],"--terminal-child",nullptr);_exit(80);}int status=0;require(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==77,"Unconfirmed completion did not fail closed retaining owners");
        probe::baseline("final");std::cout<<"PASS unconfirmed_completion_fail_closed\nCUDA_RESOURCE_TRANSACTION=PASS owned_before=0 peak_events="<<probe::peak_events<<" peak_allocations="<<probe::peak_device<<" peak_owned="<<probe::peak_owned<<" owned_after=0 double_free=0\n";
    }catch(const std::exception& e){std::cerr<<"resource transaction qualification: "<<e.what()<<" live="<<probe::count()<<" double_free="<<probe::double_free<<'\n';return 1;}
}
