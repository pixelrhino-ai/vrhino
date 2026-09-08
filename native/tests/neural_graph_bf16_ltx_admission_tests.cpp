#include "../src/architectures/ltx_self_attention_graph.h"
#include "neural_graph_test_backend.h"
#include "vrhino/loader.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <type_traits>
#ifdef VRHINO_LTX_BINDING_CUDA
#include "vrhino/backend/cuda_backend.h"
#endif
using namespace vrhino;
namespace ng=vrhino::neural_graph;
namespace {
class BindOnly final : public ng::test::TinyBackend {
public:
    BindOnly(){dtype=DType::BF16;}
    bool supports(DType) const override{return true;}
};
// Frozen admission-only candidate from 34ced2f; independent topology oracle.
ng::Description frozen_candidate(DType hidden,int64_t batch,int64_t tokens,int64_t modulation) {
    auto d=ltx_internal::self_attention_graph(batch,tokens,modulation,true).description();
    d.execution_dtype=DType::BF16;d.values[0].dtype=hidden;
    for(auto id:{2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,29,30,31,38,39,40,41})d.values[id].dtype=DType::BF16;
    d.values[25].dtype=hidden;
    auto original=d.nodes;d.nodes.clear();
    auto cast=[&](ng::ValueId source,DType destination,ng::ValueId consumer) {
        auto id=static_cast<ng::ValueId>(d.values.size());
        d.values.push_back({d.values[source].shape,destination});d.nodes.push_back({id,ng::Cast{source,destination}});
        std::cout<<"CAST\t"<<dtype_name(hidden)<<'\t'<<source<<'\t'<<id<<'\t'<<consumer<<'\t'<<dtype_name(destination)<<'\n';
        return id;
    };
    for(auto node:original) {
        if(auto* op=std::get_if<ng::Linear>(&node.op))op->compute_dtype=DType::BF16;
        if(node.result==18)std::get<ng::Add>(node.op).b=cast(17,DType::BF16,18);
        if(node.result==26)std::get<ng::Add>(node.op).b=cast(22,DType::F32,26);
        if(node.result==27 && hidden==DType::BF16)std::get<ng::Mul>(node.op).a=cast(25,DType::F32,27);
        if(node.result==28)std::get<ng::Add>(node.op).b=cast(20,DType::F32,28);
        if(node.result==42) {
            std::get<ng::Mul>(node.op).a=cast(41,DType::F32,42);
            std::get<ng::Mul>(node.op).b=cast(24,DType::F32,42);
        }
        if(node.result==43 && hidden==DType::BF16)std::get<ng::Add>(node.op).a=cast(0,DType::F32,43);
        d.nodes.push_back(std::move(node));
    }
    auto graph=ng::admit(d);const size_t count=hidden==DType::F32?33:35;
    require(graph.description().nodes.size()==count,"expanded node count");
    return graph.description();
}
std::string contract_text(const ng::Description& d) {
    std::ostringstream out;
    auto list=[&](const auto& values){for(auto v:values)out<<v<<',';out<<';';};
    out<<d.schema<<';'<<static_cast<int>(d.execution_dtype)<<';';
    for(const auto& spec:d.values){list(spec.shape);out<<static_cast<int>(spec.dtype)<<';';}
    list(d.inputs);list(d.parameters);list(d.outputs);
    for(const auto& c:d.constants)out<<c.value<<':'<<std::hexfloat<<c.scalar<<';';
    for(const auto& n:d.nodes) {
        out<<n.result<<':'<<n.op.index()<<':';
        std::visit([&](const auto& op){
            using T=std::decay_t<decltype(op)>;
            if constexpr(std::is_same_v<T,ng::Add> || std::is_same_v<T,ng::Mul>)out<<op.a<<','<<op.b;
            else if constexpr(std::is_same_v<T,ng::Reshape>){out<<op.x<<':';list(op.shape);}
            else if constexpr(std::is_same_v<T,ng::Slice>)out<<op.x<<','<<op.axis<<','<<op.start<<','<<op.stop;
            else if constexpr(std::is_same_v<T,ng::Linear>)out<<op.x<<','<<op.weight<<','<<op.bias.value_or(999)<<','<<static_cast<int>(op.compute_dtype);
            else if constexpr(std::is_same_v<T,ng::RmsNorm>)out<<op.x<<','<<op.weight.value_or(999)<<','<<op.axis<<','<<std::hexfloat<<op.epsilon;
            else if constexpr(std::is_same_v<T,ng::Rope>)out<<op.x<<','<<op.cosine<<','<<op.sine;
            else if constexpr(std::is_same_v<T,ng::Attention>)out<<op.q<<','<<op.k<<','<<op.v<<','<<std::hexfloat<<op.scale;
            else if constexpr(std::is_same_v<T,ng::Cast>)out<<op.x<<','<<static_cast<int>(op.destination);
            else throw Error("Unexpected LTX primitive in topology oracle");
        },n.op);
        out<<';';
    }
    return out.str();
}
template<class F> void negative(const std::string& name,F fn,ng::Phase phase,ng::Code code) {
    bool rejected=false;
    try{fn();}catch(const ng::GraphError& e){rejected=e.phase==phase && e.code==code;}
    require(rejected,"Negative did not fail closed: "+name);
    std::cout<<"NEG\t"<<name<<"\tPASS\n";
}
class DispatchProbe final : public ng::test::TinyBackend {
public:
    const void* expected=nullptr;
    DispatchProbe(){dtype=DType::BF16;}
    bool supports(DType t) const override{return t==DType::F32 || t==DType::BF16;}
    Tensor reshape(const Tensor& t,const std::vector<int64_t>&) override {
        require(t.data()==expected && t.dtype()==DType::BF16,"Production binding identity changed");
        begin("bound_first_reshape");
        throw Error("intentional structural dispatch stop");
    }
};
void qualify(VrmModel& model,DType hidden,int64_t batch,int64_t tokens,int64_t modulation) {
    const auto contract=ltx_internal::self_attention_contract(DType::BF16,hidden);
    const auto graph=ltx_internal::self_attention_graph(batch,tokens,modulation,true,false,contract);
    const auto& d=graph.description();
    require(contract_text(d)==contract_text(frozen_candidate(hidden,batch,tokens,modulation)),"Admitted topology changed");
    const size_t count=hidden==DType::F32?33:35;
    auto state=ltx_internal::self_attention_graph(batch,tokens,modulation,false,true,contract).description();
    require(state.nodes.size()==count && state.outputs==std::vector<ng::ValueId>({43,18}) &&
            state.values[43].dtype==DType::F32 && state.values[18].dtype==DType::BF16,"Block state output contract");
    auto final=ltx_internal::self_attention_graph(batch,tokens,modulation,false,false,contract).description();
    require(final.outputs==std::vector<ng::ValueId>{43},"Final output contract");
    // Evaluate only external slots and the actual constant, without arithmetic.
    ng::Description bindings;bindings.schema=ng::schema_v1;bindings.execution_dtype=DType::BF16;
    std::vector<Tensor> inputs,parameters;
    for(auto id:d.inputs) {
        bindings.inputs.push_back(bindings.values.size());bindings.values.push_back(d.values[id]);
        inputs.push_back(Tensor::host(d.values[id].shape,d.values[id].dtype));
    }
    for(size_t slot=0;slot<d.parameters.size();++slot) {
        auto id=d.parameters[slot];bindings.parameters.push_back(bindings.values.size());bindings.values.push_back(d.values[id]);
        parameters.push_back(model.tensor("denoiser.transformer_blocks.0."+std::string(ltx_internal::self_attention_parameters[slot])));
    }
    const auto constant=static_cast<ng::ValueId>(bindings.values.size());
    bindings.values.push_back(d.values[15]);bindings.constants={{constant,d.constants.at(0).scalar}};
    for(ng::ValueId i=0;i<bindings.values.size();++i)bindings.outputs.push_back(i);
    BindOnly backend;auto outputs=ng::evaluate(backend,ng::admit(bindings),inputs,parameters);
    require(outputs.size()==bindings.values.size() && backend.calls.empty(),"binding projection dispatched computation");
    require(outputs.back().dtype()==DType::F32 && outputs.back().data_as<float>()[0]==1.0f,"Constant binding");
    for(size_t i=0;i<parameters.size();++i)
        require(outputs[inputs.size()+i].data()==parameters[i].data(),"Parameter binding copied storage");
    auto bad_inputs=inputs;bad_inputs[0]=Tensor::host(d.values[0].shape,hidden==DType::F32?DType::BF16:DType::F32);
    negative("hidden_mismatch_"+dtype_name(hidden),[&]{ng::evaluate(backend,graph,bad_inputs,parameters);},ng::Phase::Bind,ng::Code::InvalidInputBinding);
    auto bad_parameters=parameters;bad_parameters[1]=Tensor::host(d.values[5].shape,DType::F32);
    negative("BF16_parameter_mismatch",[&]{ng::evaluate(backend,graph,inputs,bad_parameters);},ng::Phase::Bind,ng::Code::InvalidParameterBinding);
    auto bad=d;bad.nodes.erase(std::find_if(bad.nodes.begin(),bad.nodes.end(),[](const auto& n){return n.result==44;}));
    negative("missing_cast_value",[&]{ng::admit(bad);},ng::Phase::Admit,ng::Code::InvalidDependency);
    backend.dtype=DType::F32;
    negative("wrong_context",[&]{ng::evaluate(backend,graph,inputs,parameters);},ng::Phase::Bind,ng::Code::InvalidContext);
    backend.dtype=DType::BF16;
    bad=d;bad.values[0].dtype=DType::F16;
    negative("unsupported_dtype",[&]{ng::admit(bad);},ng::Phase::Admit,ng::Code::InvalidDType);
    require(backend.calls.empty(),"Negative control dispatched arithmetic");
    // Actual production block route: bindings validated, first node dispatched,
    // then intentionally stopped by a test double. No numerical smoke/forward.
    std::map<std::string,const Tensor*> slots;
    for(size_t i=0;i<parameters.size();++i)slots[ltx_internal::self_attention_parameters[i]]=&parameters[i];
    DispatchProbe probe;probe.expected=parameters[0].data();
    negative("production_dispatch_probe_"+dtype_name(hidden),[&]{
        ltx_internal::transformer_block_forward(probe,PrecisionPolicy::unqualified_default(DType::BF16),WeightMap(slots),
            inputs[0],{},inputs[1],inputs[2],inputs[3],{});
    },ng::Phase::Run,ng::Code::BackendFailure);
    require(probe.calls==std::vector<std::string>{"bound_first_reshape"},"Private production Graph did not dispatch exactly once");
    DispatchProbe rejected_probe;rejected_probe.expected=parameters[0].data();slots[ltx_internal::self_attention_parameters[1]]=&bad_parameters[1];
    negative("production_wrong_BF16_parameter",[&]{
        ltx_internal::transformer_block_forward(rejected_probe,PrecisionPolicy::unqualified_default(DType::BF16),WeightMap(slots),
            inputs[0],{},inputs[1],inputs[2],inputs[3],{});
    },ng::Phase::Bind,ng::Code::InvalidParameterBinding);
    require(rejected_probe.calls.empty(),"Production mismatch dispatched");
    for(size_t id=0;id<d.values.size();++id) {
        std::cout<<"FLOW\t"<<dtype_name(hidden)<<'\t'<<id<<'\t'<<dtype_name(d.values[id].dtype)<<'\t';
        for(auto dim:d.values[id].shape)std::cout<<dim<<',';
        std::cout<<'\n';
    }
    std::cout<<"LTX_ADMISSION\t"<<dtype_name(hidden)<<'\t'<<batch<<'\t'<<tokens<<'\t'<<modulation<<'\t'<<count<<'\t'<<count-28<<"\tPASS\n";
}
}
void legacy_and_selection() {
    const auto graph=ltx_internal::self_attention_graph(1,1,1);
    const auto& d=graph.description();require(d.nodes.size()==28 && d.execution_dtype==DType::F32,"Legacy F32 contract");
    std::vector<Tensor> inputs,parameters;
    for(auto id:d.inputs)inputs.push_back(Tensor::host(d.values[id].shape,DType::F32));
    for(auto id:d.parameters)parameters.push_back(Tensor::host(d.values[id].shape,DType::F32));
    auto projection=d;projection.nodes.clear();projection.values.resize(16);projection.outputs=d.parameters;
    BindOnly backend;backend.dtype=DType::F32;
    auto outputs=ng::evaluate(backend,ng::admit(projection),inputs,parameters);
    for(size_t i=0;i<parameters.size();++i)require(outputs[i].data()==parameters[i].data(),"F32 binding copied parameter");
    parameters[1]=Tensor::host(d.values[5].shape,DType::BF16);
    negative("F32_parameter_mismatch",[&]{ng::evaluate(backend,graph,inputs,parameters);},ng::Phase::Bind,ng::Code::InvalidParameterBinding);
    require(backend.calls.empty(),"F32 negative dispatched");
    for(auto execution:{DType::F32,DType::BF16,DType::F16})for(auto hidden:{DType::F32,DType::BF16,DType::F16}) {
        const bool valid=(execution==DType::F32 && hidden==DType::F32) ||
                         (execution==DType::BF16 && (hidden==DType::F32 || hidden==DType::BF16));
        bool accepted=false;try{(void)ltx_internal::self_attention_contract(execution,hidden);accepted=true;}catch(const Error&){}
        require(accepted==valid,"Dtype selector did not fail closed");
        std::cout<<"SELECTOR\t"<<dtype_name(execution)<<'\t'<<dtype_name(hidden)<<"\tPASS\n";
    }
    bool rejected=false;try{ltx_internal::self_attention_graph(1,1,1,false,false,static_cast<ltx_internal::SelfAttentionContract>(99));}catch(const Error&){rejected=true;}
    require(rejected,"Invalid contract enum accepted");
    std::cout<<"LEGACY_F32_BINDING_AND_SELECTION\tPASS\n";
}
#ifdef VRHINO_LTX_BINDING_CUDA
void cache_bindings(VrmModel& model) {
    CudaBackend backend;backend.set_execution_dtype(DType::BF16);
    size_t expected_bytes=0;
    std::vector<Tensor> resident;
    for(auto hidden:{DType::F32,DType::BF16}) {
        auto d=ltx_internal::self_attention_graph(2,1,1,false,true,
            ltx_internal::self_attention_contract(DType::BF16,hidden)).description();
        // Real CUDA placement/cache test only: no Graph arithmetic executes.
        d.nodes.clear();d.values.resize(16);d.outputs=d.parameters;d.outputs.push_back(15);
        const auto graph=ng::admit(d);
        std::vector<Tensor> inputs,parameters;
        for(auto id:d.inputs)inputs.push_back(Tensor::host(d.values[id].shape,d.values[id].dtype));
        for(const char* name:ltx_internal::self_attention_parameters)
            parameters.push_back(model.tensor("denoiser.transformer_blocks.0."+std::string(name)));
        if(!expected_bytes)for(const auto& t:parameters)expected_bytes+=t.bytes();
        for(int repeat=0;repeat<2;++repeat) {
            const auto misses=backend.weight_cache_misses(),hits=backend.weight_cache_hits();
            auto outputs=ng::evaluate(backend,graph,inputs,parameters);
            const bool cold=resident.empty();
            require(backend.weight_cache_misses()-misses==(cold?parameters.size():0) &&
                    backend.weight_cache_hits()-hits==(cold?0:parameters.size()),"Unexpected binding cache access");
            require(backend.weight_cache_resident_bytes()==expected_bytes,"Duplicate/shadow parameter materialization");
            for(size_t i=0;i<parameters.size();++i) {
                require(outputs[i].dtype()==DType::BF16 && outputs[i].shape()==parameters[i].shape(),"CUDA binding metadata");
                if(!cold)require(outputs[i].data()==resident[i].data(),"Parameter cache identity changed");
                require(backend.copy_to_device(parameters[i],DType::BF16).data()==outputs[i].data(),"Graph has private parameter copy");
            }
            if(cold)resident=outputs;
            const auto before=backend.weight_cache_misses();
            std::vector<Tensor> device_parameters(outputs.begin(),outputs.end()-1);
            const auto device_outputs=ng::evaluate(backend,graph,inputs,device_parameters);
            for(size_t i=0;i<parameters.size();++i)require(device_outputs[i].data()==outputs[i].data(),"Device binding copied weights");
            require(backend.weight_cache_misses()==before && backend.weight_cache_resident_bytes()==expected_bytes,"Device binding rematerialized weights");
            std::cout<<"CACHE\t"<<dtype_name(hidden)<<'\t'<<repeat<<'\t'<<(cold?parameters.size():0)
                <<'\t'<<(cold?0:parameters.size())<<'\t'<<expected_bytes<<"\tPASS\n";
        }
    }
}
#endif
int main(int argc,char** argv) {
    try {
        require(argc==2,"usage: neural-graph-bf16-ltx-admission-tests LTX.vrm");VrmModel model(argv[1]);
        legacy_and_selection();
        for(auto hidden:{DType::F32,DType::BF16})for(int batch:{1,2})for(int modulation:{1,3})qualify(model,hidden,batch,3,modulation);
#ifdef VRHINO_LTX_BINDING_CUDA
        cache_bindings(model);
#endif
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
