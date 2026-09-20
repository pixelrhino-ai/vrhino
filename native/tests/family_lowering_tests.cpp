#include "../src/architectures/wan_execution.h"
#include "../src/architectures/wan_self_attention_graph.h"
#include "family_lowering_test_backend.h"
#include "step_execution_test_support.h"
#include "vrhino/runtime.h"
#include <fstream>
#include <iostream>

using namespace vrhino;
namespace wf=vrhino::wan_family;
namespace ft=vrhino::family_test;
namespace st=vrhino::step_test;
namespace {
int groups=0, rejects_count=0;
void pass(const char* name) { ++groups; std::cout<<"PASS "<<name<<'\n'; }
template<class F> void rejects(F action) {
    bool failed=false; try { action(); } catch (const Error&) { failed=true; }
    require(failed,"Invalid family declaration admitted"); ++rejects_count;
}
wf::Config small(int variant=0) {
    wf::Config c; c.dim=variant?48:12; c.heads=variant?4:2; c.layers=variant?3:2; c.ffn=variant?28:20;
    c.frequency=6; c.text_dim=8; c.text_length=4; c.input_channels=c.output_channels=2;
    c.patch=variant?std::array<int,3>{2,1,2}:std::array<int,3>{1,2,1};
    c.rope_theta=variant?8000:10000; c.epsilon=variant?1e-5f:1e-6f; return c;
}
Tensor filled(const std::vector<int64_t>& shape,float base=0) {
    auto t=Tensor::host(shape,DType::F32);
    for (int64_t i=0;i<t.numel();++i) t.data_as<float>()[i]=base+0.013f*std::sin(float(i%23));
    return t;
}
TensorBundle weights(const wf::Definition& d,float delta=0) {
    TensorBundle result; int i=0;
    for (const auto& slot:d.parameters->slots()) {
        float base=slot.role.find("norm")!=std::string::npos && slot.role.ends_with("weight")?1.0f:0.0f;
        result.emplace(slot.role,filled(slot.shape,base+delta+0.0001f*(i++%7)));
    } return result;
}
std::map<std::string,const Tensor*> references(const TensorBundle& tensors) {
    std::map<std::string,const Tensor*> result; for (const auto& [key,t]:tensors) result.emplace(key,&t); return result;
}
auto admit(const std::shared_ptr<const wf::Definition>& d,const TensorBundle& w) {
    return AdmittedArchitectureBinding::admit(d->parameters,references(w),BorrowedBindingLifetime::ExplicitOwners);
}
void register_sources(ft::Backend& b,const TensorBundle& w) { for (const auto& [key,t]:w) { (void)key; b.cacheable.insert(t.data()); } }
SamplingProgram program() { auto p=st::program(1); p.latent_shape={1,2,2,4,2}; return p; }

void lowering() {
    std::ifstream file(VRHINO_FAMILY_SPEC); require(file.good(),"Missing family spec");
    const auto json=Json::parse(std::string(std::istreambuf_iterator<char>(file),{}));
    auto d=wf::lower(canonical_architecture_from_vrm_descriptor(json.at("architecture_graph")));
    require(d->config==wf::Config{} && d->head_dim==128 && d->rope_axes==std::vector<int>({44,42,42}),"Legacy config lowering drift");
    const auto& roles=json.at("architecture_graph").at("runtime_tensor_bindings").object();
    require(d->parameters->slots().size()==roles.size() && roles.size()==825,"Legacy parameter set drift");
    for (const auto& slot:d->parameters->slots()) require(roles.contains(slot.role),"Legacy role missing");
    // Declaration only: never allocate these large shapes.
    auto large=d->config; large.dim=5120; large.heads=40; large.layers=40; large.ffn=13824;
    auto large_d=wf::lower(large); require(large_d->parameters->slots().size()==1095 && large_d->rope_axes==d->rope_axes,"Large metadata lowering");
    for (int variant=0;variant<2;++variant) {
        auto current=wf::lower(small(variant));
        auto g=wan_internal::self_attention_graph(1,4,DType::F32,DType::F32,true,current->config);
        require(g.description().values[0].shape==std::vector<int64_t>({1,4,current->config.dim}),"Graph ignores hidden size");
        bool checked=false;
        for (const auto& n:g.description().nodes) if (const auto* a=std::get_if<neural_graph::Attention>(&n.op)) {
            require(g.description().values[n.result].shape==std::vector<int64_t>({1,4,current->config.heads,current->head_dim}) &&
                    a->scale==1/std::sqrt(float(current->head_dim)),"Graph ignores heads"); checked=true;
        }
        require(checked,"Missing attention declaration");
    }
    auto valid=small();
    for (int bad=0;bad<9;++bad) rejects([&] {
        auto c=valid;
        if(bad==0)c.heads=0;
        if(bad==1)c.dim=13;
        if(bad==2)c.layers=0;
        if(bad==3)c.ffn=-1;
        if(bad==4)c.patch[0]=0;
        if(bad==5)c.frequency=5;
        if(bad==6)c.output_channels=3;
        if(bad==7)c.rope_theta=INFINITY;
        if(bad==8)c.epsilon=0;
        wf::lower(c);
    });
    for (const auto& key:{"qk_norm","cross_attn_norm"}) rejects([&] {
        auto object=json.at("architecture_graph").at("config").object(); object[key]=Json(false); auto descriptor=json.at("architecture_graph").object(); descriptor["config"]=Json(object);
        wf::lower(canonical_architecture_from_vrm_descriptor(Json(descriptor)));
    });
    rejects([&] { auto object=json.at("architecture_graph").at("config").object(); object["unknown"]=Json(int64_t(1)); auto descriptor=json.at("architecture_graph").object(); descriptor["config"]=Json(object);
        wf::lower(canonical_architecture_from_vrm_descriptor(Json(descriptor))); });
    pass("config_schema_legacy_large_metadata_and_two_small_graphs");
}

void admission() {
    auto d=wf::lower(small()); auto w=weights(*d); auto refs=references(w); auto accepted=admit(d,w);
    require(accepted->parameters_for(*d->parameters).size()==w.size(),"Binding not execution-ready");
    for (int bad=0;bad<7;++bad) rejects([&] {
        auto inputs=refs; Tensor altered;
        const auto& original=*inputs.begin()->second;
        if(bad==0)inputs.erase(inputs.begin());
        if(bad==1)inputs.emplace("extra",&original);
        if(bad==2) { altered=Tensor::host(original.shape(),DType::BF16); inputs.begin()->second=&altered; }
        if(bad==3) { altered=filled({1}); inputs.begin()->second=&altered; }
        if(bad==4) { altered=Tensor::borrowed(original.data(),sizeof(float),original.shape(),original.dtype()); inputs.begin()->second=&altered; }
        if(bad==5) inputs.begin()->second=nullptr;
        if(bad==6) { auto storage=std::make_shared<Storage>(); storage->data=original.data(); storage->bytes=original.bytes();
            storage->device=DeviceId::accelerator(0); storage->domain=MemoryDomain::DeviceLocal;
            altered=Tensor(storage,0,original.shape(),original.dtype()); inputs.begin()->second=&altered; }
        AdmittedArchitectureBinding::admit(d->parameters,inputs,BorrowedBindingLifetime::CallerRetained);
    });
    rejects([&] { auto c=d->config; c.epsilon*=2; auto other=wf::lower(c); accepted->parameters_for(*other->parameters); });
    rejects([&] { auto slots=d->parameters->slots(); slots[0].layout=static_cast<ParameterLayout>(99); ArchitectureBindingDeclaration invalid(slots); });
    const auto positive=filled({1,2,8}),negative=filled({1,3,8});
    for (int bad=0;bad<3;++bad) rejects([&] {
        auto shape=program().latent_shape; Tensor p=positive;
        if(bad==0)shape[3]=3;
        if(bad==1)shape[1]=3;
        if(bad==2)p=filled({1,5,8});
        wf::validate_request(*d,shape,p,negative);
    });
    ft::Backend backend; auto policy=PrecisionPolicy::fp32();
    auto e=wf::realize(d,backend,policy,program().latent_shape,positive,negative);
    rejects([&] { e.bind({0},{0},admit(wf::lower(small()),w)); });
    require(backend.calls.empty() && backend.address_cache.empty(),"Lowering/admission executed Backend");
    pass("admission_slots_shape_dtype_layout_extent_semantics_before_execution");
}

void qualification() {
    for (int variant=0;variant<2;++variant) {
        auto d=wf::lower(small(variant)); auto a=weights(*d),b=weights(*d,0.007f);
        auto A=admit(d,a),B=admit(d,b); const auto p=program();
        const auto positive=filled({1,2,8},0.1f),negative=filled({1,3,8},-0.1f);
        auto policy=PrecisionPolicy::fp32(); ft::Backend backend,old_backend;
        register_sources(backend,a); register_sources(backend,b); register_sources(old_backend,a);
        auto e=wf::realize(d,backend,policy,p.latent_shape,positive,negative,true,-1,0);
        std::vector<ComponentInstance> items; items.push_back(e.bind({0},{10},A)); items.push_back(e.bind({1},{20},B));
        ExecutionContext context(std::move(items));
        require(context.at({0}).graph()==context.at({1}).graph(),"Graph not reused");
        std::map<std::string,uint64_t> before;
        for (const auto& [key,t]:a) before[key]=hash_host_tensor_content(t);
        SamplingRuntime runtime(backend); std::vector<size_t> misses;
        runtime.set_step_observer([&](int,int) { misses.push_back(backend.misses); });
        auto result=runtime.run(context,ExecutionProgram::per_step({{0},{1},{0}}),p);
        const void* pa=a.at("patch_embedding.weight").data(); const void* pb=b.at("patch_embedding.weight").data();
        require(pa!=pb && backend.patch_weights==std::vector<const void*>({pa,pa,pb,pb,pa,pa}),"Wrong binding/branch order");
        require(misses[0]>0 && misses[1]==2*misses[0] && misses[2]==misses[1] && backend.hits>0,"Address cache collision/reuse failure");
        for (const auto& [key,t]:a) require(hash_host_tensor_content(t)==before.at(key),"Weight overwrite");
        require(backend.rng_calls==1 && result.rng_after_initialization.offset==static_cast<uint64_t>(shape_numel(p.latent_shape)),"RNG restarted");
        require(runtime.primitives().calls().at("state_advance")==3 && runtime.primitives().calls().at("multistep_corrector")==2,"Solver history reset");
        // Re-evaluate each actual input with an independent bound endpoint.
        for (int i=0;i<3;++i) {
            ft::Backend oracle; auto endpoint=wf::bound_denoiser(d,i==1?B:A,oracle,policy,p.latent_shape,positive,negative);
            const auto prefix="step."+std::to_string(i)+".";
            auto predictions=endpoint->evaluate(result.trace.at(prefix+"input_latent"),p.model_timestep_at(i));
            st::exact(predictions[0],result.trace.at(prefix+"prediction.0"));
            st::exact(predictions[1],result.trace.at(prefix+"prediction.1"));
        }
        // Legacy Denoiser API versus typed singleton: complete numerical traces
        // and primitive ordering, with either RNG or explicit initial state.
        for (bool initial:{false,true}) {
            ft::Backend legacy_backend,generic_backend;
            auto legacy=wf::bound_denoiser(d,A,legacy_backend,policy,p.latent_shape,positive,negative,true,-1,0);
            auto generic=wf::realize(d,generic_backend,policy,p.latent_shape,positive,negative,true,-1,0);
            std::vector<ComponentInstance> single; single.push_back(generic.bind({0},{0},A)); ExecutionContext single_context(std::move(single));
            SamplingRuntime first(legacy_backend),second(generic_backend); const auto state=filled(p.latent_shape);
            auto x=initial?first.run_with_initial_state(*legacy,p,state):first.run(*legacy,p);
            auto y=initial?second.run_with_initial_state(single_context,ExecutionProgram::uniform({0}),p,state):second.run(single_context,ExecutionProgram::uniform({0}),p);
            st::exact(x,y); require(legacy_backend.calls==generic_backend.calls && first.primitives().calls()==second.primitives().calls(),"Legacy execution order changed");
        }
        // Global step trace must fire on A's second invocation at global step 2.
        ft::Backend traced; auto te=wf::realize(d,traced,policy,p.latent_shape,positive,negative,true,2,0);
        std::vector<ComponentInstance> ti; ti.push_back(te.bind({0},{0},A)); ti.push_back(te.bind({1},{1},B));
        auto tr=SamplingRuntime(traced).run(ExecutionContext(std::move(ti)),ExecutionProgram::per_step({{0},{1},{0}}),p);
        require(!tr.trace.contains("step.0.branch.0.input_projection") && tr.trace.contains("step.2.branch.0.input_projection"),"Global trace step lost");
    }
    pass("real_family_A_B_A_two_configs_cache_solver_rng_trace_and_legacy_exact_4_cases");
}

void lifetime() {
    auto d=wf::lower(small()); auto owner=std::make_shared<TensorBundle>(weights(*d)); std::weak_ptr<TensorBundle> weak=owner;
    TensorBundle borrowed;
    for (const auto& [key,t]:*owner) borrowed.emplace(key,Tensor::borrowed(t.data(),t.bytes(),t.shape(),t.dtype()));
    rejects([&] { AdmittedArchitectureBinding::admit(d->parameters,references(borrowed),BorrowedBindingLifetime::ExplicitOwners); });
    auto binding=AdmittedArchitectureBinding::admit(d->parameters,references(borrowed),BorrowedBindingLifetime::ExplicitOwners,{owner});
    const auto positive=filled({1,2,8}),negative=filled({1,2,8}); auto policy=PrecisionPolicy::fp32();
    // Execution admission promotes source ownership into Backend session.
    {
        ft::Backend backend; register_sources(backend,*owner);
        auto e=wf::realize(d,backend,policy,program().latent_shape,positive,negative);
        std::vector<ComponentInstance> instances; instances.push_back(e.bind({0},{0},binding));
        owner.reset(); binding.reset(); borrowed.clear();
        { ExecutionContext context(std::move(instances)); SamplingRuntime(backend).run(context,ExecutionProgram::uniform({0}),program()); }
        require(!weak.expired() && backend.misses>0,"Source owner expired while address cache live");
    }
    require(weak.expired(),"Binding owners leaked");
    pass("borrowed_tensor_objects_backing_owner_and_backend_session_lifetime");
}

class SyntheticArchitecture final : public Architecture {
public:
    SyntheticArchitecture(std::shared_ptr<const wf::Definition> d,
                          std::shared_ptr<const AdmittedArchitectureBinding> a,
                          std::shared_ptr<const AdmittedArchitectureBinding> b)
        : definition(std::move(d)), A(std::move(a)), B(std::move(b)) {}
    std::unique_ptr<Denoiser> create_denoiser(vrhino::Backend& backend,const PrecisionPolicy& policy,const TensorBundle&) override {
        return wf::bound_denoiser(definition,A,backend,policy,program().latent_shape,positive,negative);
    }
    ExecutionSetup create_execution_setup(vrhino::Backend& backend,const PrecisionPolicy& policy,const TensorBundle&) override {
        auto e=wf::realize(definition,backend,policy,program().latent_shape,positive,negative);
        std::vector<ComponentInstance> instances; instances.push_back(e.bind({0},{0},A)); instances.push_back(e.bind({1},{1},B));
        return {ExecutionContext(std::move(instances)),ExecutionProgram::per_step({{0},{1},{0}})};
    }
    SamplingProgram create_program(const TensorBundle&) override { return program(); }
    Tensor decode(vrhino::Backend&,const PrecisionPolicy&,const Tensor& latent,const TensorBundle&) override { return latent; }
    std::shared_ptr<const wf::Definition> definition;
    std::shared_ptr<const AdmittedArchitectureBinding> A,B;
    Tensor positive=filled({1,2,8}),negative=filled({1,2,8});
};

void shared_runtime() {
    auto d=wf::lower(small()); auto wa=weights(*d),wb=weights(*d,0.007f); auto A=admit(d,wa),B=admit(d,wb);
    SyntheticArchitecture architecture(d,A,B); ft::Backend backend;
    const auto result=NativeRuntime(backend).execute(architecture,{});
    require(result.outputs.at("video").shape()==program().latent_shape && backend.rng_calls==1,"Shared Runtime integration failed");
    const void* pa=wa.at("patch_embedding.weight").data(); const void* pb=wb.at("patch_embedding.weight").data();
    require(backend.patch_weights==std::vector<const void*>({pa,pa,pb,pb,pa,pa}),"Shared Runtime misrouted instances");
    ft::Backend legacy_backend; auto policy=PrecisionPolicy::fp32();
    auto legacy=wf::bound_denoiser(d,A,legacy_backend,policy,program().latent_shape,architecture.positive,architecture.negative,true,2,0);
    SamplingRuntime sampling(legacy_backend);
    const auto first=sampling.run(*legacy,program()),second=sampling.run(*legacy,program());
    require(first.trace.contains("step.2.branch.0.input_projection") && !second.trace.contains("step.2.branch.0.input_projection"),
            "Legacy reused endpoint counter reset");
    pass("shared_NativeRuntime_real_family_and_legacy_repeated_run_counter");
}
}  // namespace
int main() {
    try { lowering(); admission(); qualification(); lifetime(); shared_runtime();
        std::cout<<"FAMILY_LOWERING=PASS groups="<<groups<<" rejected="<<rejects_count<<'\n';
    } catch (const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
