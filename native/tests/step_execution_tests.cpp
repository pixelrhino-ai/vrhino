#include <iostream>
#include <limits>

#include "step_execution_test_support.h"
// Windows SDK headers define this object-like macro. The public execution
// header and its accessor must compile without undefining a caller's macro.
#define interface struct
#include "vrhino/execution.h"
interface WindowsInterfaceMacroPreserved { int value; };
using ComponentInterfaceAccessor = decltype(&vrhino::ComponentGraphDefinition::component_interface);
#undef interface
#include "vrhino/runtime.h"

using namespace vrhino;
namespace st = vrhino::step_test;

namespace {
int cases = 0;
void pass(const char* name) { ++cases; std::cout << "PASS " << name << '\n'; }
struct Observation { size_t step; uint32_t instance, binding; };

class ContextDenoiser final : public st::LegacyDenoiser {
public:
    ContextDenoiser(st::Backend& backend, Tensor parameter, std::vector<Observation>& seen)
        : LegacyDenoiser(backend, std::move(parameter)), seen(seen) {}
    std::vector<Tensor> evaluate_step(const Tensor& latent, const StepExecutionContext& context) override {
        seen.push_back({context.step_index, context.selected_component.id().value,
                        context.selected_component.binding_id().value});
        auto result = evaluate(latent, context.timestep);
        trace.emplace("endpoint.global_step", scalar_i64(context.step_index));
        trace.emplace("endpoint.instance", scalar_i64(context.selected_component.id().value));
        return result;
    }
    std::vector<Observation>& seen;
};

ComponentInterface interface_for(const SamplingProgram& p) {
    return {{p.latent_shape, DType::F32}, {{},DType::I64},
            {{p.latent_shape,DType::F32},{p.latent_shape,DType::F32}},
            p.guidance_mode, p.contract ? std::optional(p.contract->prediction.semantic) : std::nullopt, 0};
}

std::shared_ptr<const ComponentGraphDefinition> graph_for(st::Backend& backend,
        const SamplingProgram& p, std::vector<Observation>* seen = nullptr) {
    return std::make_shared<const ComponentGraphDefinition>(interface_for(p),
        std::vector<ExecutionTensorContract>{{{},DType::F32}},
        [&backend, seen](const std::vector<Tensor>& parameters) -> std::unique_ptr<Denoiser> {
            if (seen) return std::make_unique<ContextDenoiser>(backend, parameters.at(0), *seen);
            return std::make_unique<st::LegacyDenoiser>(backend, parameters.at(0));
        });
}

ExecutionContext catalog(std::shared_ptr<const ComponentGraphDefinition> graph, bool two = true) {
    std::vector<ComponentInstance> instances;
    instances.push_back(ComponentInstance::bind({0},{10},graph,{scalar_f32(0.375f)}));
    if (two) instances.push_back(ComponentInstance::bind({1},{20},graph,{scalar_f32(-0.25f)}));
    return ExecutionContext(std::move(instances));
}

void bounded_prefix() {
    for (int kind = 0; kind < 4; ++kind) for (int limit : {1,2,3}) {
        st::Backend full_backend, prefix_backend;
        auto p = st::program(kind);
        p.guidance_schedule=GuidanceSchedule::per_step({GuidanceParameters::cfg(4),
            GuidanceParameters::cfg(3),GuidanceParameters::cfg(2)});
        auto full_context=catalog(graph_for(full_backend,p));
        auto prefix_context=catalog(graph_for(prefix_backend,p));
        auto execution=ExecutionProgram::per_step({{0},{1},{0}});
        SamplingRuntime full(full_backend), prefix(prefix_backend);
        full.set_solver_trace_enabled(true); prefix.set_solver_trace_enabled(true);
        int observed=0;
        prefix.set_step_observer([&](int completed,int total) {
            require(total==p.steps && completed==++observed,"Prefix changed observer schedule");
        });
        const auto expected=full.run(full_context,execution,p);
        const auto actual=prefix.run_prefix(prefix_context,execution,p,limit);
        require(actual.completed_steps==limit && observed==limit &&
            actual.program_complete==(limit==p.steps),"Prefix completion state");
        st::exact(actual.initial_noise,expected.initial_noise);
        st::exact(actual.final_latent,expected.trace.at("step."+std::to_string(limit-1)+".latent"));
        for(const auto& [name,tensor]:actual.trace)
            if(name!="final_latent") st::exact(tensor,expected.trace.at(name));
        require(!actual.trace.contains("step."+std::to_string(limit)+".input_latent"),"Extra prefix step");
        require(prefix_backend.rng_calls==1 && actual.rng_after_initialization.offset==expected.rng_after_initialization.offset,
            "Prefix RNG changed");
        require(prefix.primitives().calls().at("state_advance")==size_t(limit),"Prefix transition count");
    }
    for(int mode=0;mode<5;++mode) {
        st::Backend backend; auto p=st::program(1);
        auto context=catalog(graph_for(backend,p));
        auto execution=mode==3 ? ExecutionProgram::per_step({{0},{1},{99}}) : ExecutionProgram::per_step({{0},{1},{0}});
        if(mode==4) p.guidance_schedule=GuidanceSchedule::per_step({GuidanceParameters::cfg(4)});
        bool rejected=false;
        try { (void)SamplingRuntime(backend).run_prefix(context,execution,p,mode==0?0:mode==1?-1:mode==2?4:1); }
        catch(const Error&) { rejected=true; }
        require(rejected && backend.rng_calls==0,"Prefix bypassed full admission or bound check");
    }
    pass("bounded_prefix_exact_12_cases_and_5_fail_closed_cases");
}

void compatibility() {
    for (int kind = 0; kind < 4; ++kind) for (bool initial : {false,true})
        for (auto mode : {GuidanceMode::CFG,GuidanceMode::Linear}) for (int schedule = 0; schedule < 3; ++schedule) {
            st::Backend before_backend, after_backend;
            st::LegacyDenoiser before(before_backend, scalar_f32(0.375f));
            auto p = st::program(kind);
            p.guidance_mode = mode;
            if (mode == GuidanceMode::Linear) p.guidance_coefficients = {0.25f,0.75f};
            SamplingRuntime old_runtime(before_backend), new_runtime(after_backend);
            const auto state = host_f32({1,2}, {0.2f,-0.3f});
            const auto a = initial ? old_runtime.run_with_initial_state(before,p,state) : old_runtime.run(before,p);
            auto context = catalog(graph_for(after_backend,p), false);
            if (schedule) {
                auto guidance = mode == GuidanceMode::CFG ? GuidanceParameters::cfg(2.0f)
                                                          : GuidanceParameters::linear({0.25f,0.75f});
                p.guidance_schedule = schedule == 1 ? GuidanceSchedule::constant(guidance)
                                                    : GuidanceSchedule::per_step({guidance,guidance,guidance});
                // Explicit guidance is authoritative, including when legacy fields are empty.
                p.guidance_coefficients.clear();
            }
            const auto execution = ExecutionProgram::uniform({0});
            const auto b = initial ? new_runtime.run_with_initial_state(context,execution,p,state)
                                   : new_runtime.run(context,execution,p);
            st::exact(a,b);
            require(before_backend.calls == after_backend.calls, "Backend primitive order changed");
            require(old_runtime.primitives().calls() == new_runtime.primitives().calls(), "Sampling operation counts changed");
        }
    pass("legacy_vs_typed_singleton_exact_48_cases");
}

// Single-instance reference supplies the same synthetic numerical outputs
// without any component selection, so routing cannot reset its solver state.
class SequenceReference final : public Denoiser {
public:
    explicit SequenceReference(st::Backend& backend) : backend(backend) {}
    std::vector<Tensor> evaluate(const Tensor& latent, const Tensor&) override {
        // Numerical oracle only: no binding table, address replacement or cache.
        return graph.evaluate(backend, latent, values.at(evaluations++));
    }
    st::Backend& backend;
    st::FakeGraph graph;
    const std::vector<Tensor> values{scalar_f32(0.375f),scalar_f32(-0.25f),scalar_f32(0.375f)};
    size_t evaluations = 0;
};

void multi_binding() {
    for (int kind : {0,1}) {
        st::Backend backend, reference_backend;
        auto p = st::program(kind);
        p.guidance_schedule = GuidanceSchedule::per_step({GuidanceParameters::cfg(2.0f),
            GuidanceParameters::cfg(0.0f),GuidanceParameters::cfg(1.0f)});
        std::vector<Observation> seen;
        auto graph = graph_for(backend,p,&seen);
        auto context = catalog(graph);
        require(context.instances()[0].graph() == context.instances()[1].graph(), "Topology not shared");
        require(context.instances()[0].parameters()[0].data() != context.instances()[1].parameters()[0].data(),
                "Bindings alias mutable storage");
        SamplingRuntime runtime(backend), reference_runtime(reference_backend);
        SequenceReference reference(reference_backend);
        std::vector<int> steps;
        runtime.set_step_observer([&](int step,int total) { require(total == 3,"Step total"); steps.push_back(step); });
        auto result = runtime.run(context,ExecutionProgram::per_step({{0},{1},{0}}),p);
        auto expected = reference_runtime.run(reference,p);
        require(seen.size() == 3 && steps == std::vector<int>({1,2,3}), "Step count");
        for (size_t i = 0; i < 3; ++i) {
            require(seen[i].step == i && seen[i].instance == (i == 1 ? 1U : 0U) &&
                        seen[i].binding == (i == 1 ? 20U : 10U), "Wrong selected binding/global step");
            const auto prefix = "step." + std::to_string(i) + ".";
            for (const char* suffix : {"input_latent","prediction.0","prediction.1","guidance","latent"})
                st::exact(result.trace.at(prefix+suffix),expected.trace.at(prefix+suffix));
            require(read_scalar_i64(result.trace.at(prefix+"endpoint.global_step")) == static_cast<int64_t>(i), "Wrong trace attribution");
            require(read_scalar_i64(result.trace.at(prefix+"endpoint.local_call")) == (i == 2 ? 2 : 1), "Instance call state lost");
        }
        st::exact(result.final_latent,expected.final_latent);
        require(backend.rng_calls == 1 && result.rng_after_initialization.offset == 2, "RNG restarted at binding boundary");
        require(runtime.primitives().calls() == reference_runtime.primitives().calls(), "Solver reset or extra calls");
        require(runtime.primitives().calls().at("state_advance") == 3, "Solver transition count");
        if (kind == 1) require(runtime.primitives().calls().at("multistep_corrector") == 2 &&
                              runtime.primitives().calls().at("multistep_predictor") == 3, "Multistep history reset");
        require(result.prepared_tensors.prepare_misses == 2 && result.prepared_tensors.prepare_hits == 1 &&
                result.prepared_tensors.reuses == 3 && result.prepared_tensors.resident_entries == 2,
                "Per-binding cache contamination or double-counting");
    }
    pass("A_B_A_selection_continuous_solver_rng_trace_cache");
}

void solver_trace_observation() {
    for (bool external : {false,true}) {
        st::Backend a_backend,b_backend;
        auto p=st::program(1);
        p.guidance_schedule=GuidanceSchedule::per_step({GuidanceParameters::cfg(4),
            GuidanceParameters::cfg(3),GuidanceParameters::cfg(2)});
        auto a_context=catalog(graph_for(a_backend,p));
        auto b_context=catalog(graph_for(b_backend,p));
        auto execution=ExecutionProgram::per_step({{0},{1},{0}});
        SamplingRuntime baseline(a_backend), observed(b_backend);
        observed.set_solver_trace_enabled(true);
        auto initial=host_f32({1,2},{0.2f,-0.3f});
        auto a=external?baseline.run_with_initial_state(a_context,execution,p,initial):baseline.run(a_context,execution,p);
        auto b=external?observed.run_with_initial_state(b_context,execution,p,initial):observed.run(b_context,execution,p);
        st::exact(a.final_latent,b.final_latent);
        for(const auto& [key,value]:a.trace) st::exact(value,b.trace.at(key));
        require(a_backend.calls==b_backend.calls,"Trace changed backend numerical operations");
        require(baseline.primitives().calls()==observed.primitives().calls(),"Trace changed solver operations");
        for(int i=0;i<3;++i) {
            const auto prefix="step."+std::to_string(i)+".solver.";
            auto control=[&](const std::string& key){return read_scalar_i64(b.trace.at(prefix+key));};
            require(control("step")==i && control("state_id")==0,"Solver trace step/identity");
            require(control("instance_id")== (i==1?1:0),"Solver trace selection");
            require(control("history_before")==std::min(i,2) && control("history_after")==std::min(i+1,2),"Solver trace history size");
            require(control("warmup_before")==std::min(i,2) && control("warmup_after")==std::min(i+1,2),"Solver trace warmup");
            require(control("order")== (i==1?2:1),"Solver trace order");
            require(control("rng_offset")== (external?0:2),"Solver trace RNG continuity");
            for(int j=0;j<control("history_after");++j) {
                const int origin=i+1-control("history_after")+j;
                st::exact(b.trace.at(prefix+"history_after."+std::to_string(j)),
                    b.trace.at("step."+std::to_string(origin)+".solver.converted"));
            }
            if(i) {
                const auto previous="step."+std::to_string(i-1)+".";
                st::exact(b.trace.at("step."+std::to_string(i)+".input_latent"),b.trace.at(previous+"latent"));
                for(int j=0;j<control("history_before");++j)
                    st::exact(b.trace.at(prefix+"history_before."+std::to_string(j)),b.trace.at(previous+"solver.history_after."+std::to_string(j)));
            }
        }
        // The default path retains exactly its previous trace vocabulary.
        for(const auto& [key,value]:a.trace)require(key.find(".solver.")==std::string::npos,"Trace was not opt-in");
    }
    pass("solver_trace_observation_only_history_rng_warmup_and_legacy_exact");
}

template<class F> void rejects(st::Backend& backend, F action) {
    auto calls = backend.calls.size();
    auto rng = backend.rng_calls;
    bool rejected = false;
    try { action(); } catch (const Error&) { rejected = true; }
    require(rejected && backend.calls.size() == calls && backend.rng_calls == rng,
            "Admission failed to reject before numerical execution");
}

void admission() {
    st::Backend backend;
    auto p = st::program(1);
    auto graph = graph_for(backend,p);
    auto context = catalog(graph);
    SamplingRuntime runtime(backend);
    rejects(backend,[&] { runtime.run(context,ExecutionProgram::per_step({{0},{1}}),p); });
    rejects(backend,[&] { runtime.run(context,ExecutionProgram::per_step({{0},{7},{0}}),p); });
    rejects(backend,[&] { runtime.run(context,ExecutionProgram::uniform({7}),p); });
    rejects(backend,[&] { runtime.run(context,ExecutionProgram::partition({SelectionCoordinate::ModelTimestepScalar,
        SelectionComparison::GreaterOrEqual,int64_t(-1),{0},{7}}),p); });
    rejects(backend,[&] { ComponentInstance::bind({0},{0},graph,{}); });
    rejects(backend,[&] { ComponentInstance::bind({0},{0},graph,{host_f32({1},{1})}); });
    rejects(backend,[&] { ComponentInstance::bind({0},{0},graph,{scalar_i64(1)}); });
    rejects(backend,[&] {
        std::vector<ComponentInstance> items;
        items.push_back(ComponentInstance::bind({0},{0},graph,{scalar_f32(0)}));
        items.push_back(ComponentInstance::bind({0},{1},graph,{scalar_f32(1)}));
        ExecutionContext invalid(std::move(items));
    });
    rejects(backend,[&] {
        std::vector<ComponentInstance> items;
        items.push_back(ComponentInstance::bind({0},{0},graph,{scalar_f32(0)}));
        items.push_back(ComponentInstance::bind({1},{0},graph,{scalar_f32(1)}));
        ExecutionContext invalid(std::move(items));
    });
    for (int mismatch = 0; mismatch < 6; ++mismatch) {
        rejects(backend,[&] {
            auto wrong = interface_for(p);
            if (mismatch == 0) wrong.latent.shape = {2};
            if (mismatch == 1) wrong.latent.dtype = DType::BF16;
            if (mismatch == 2) wrong.timestep.dtype = DType::F32;
            if (mismatch == 3) wrong.predictions[0].shape = {2};
            if (mismatch == 4) wrong.prediction = PredictionSemantic::V;
            if (mismatch == 5) wrong.conditioning_contract = 42;
            auto bad_graph = std::make_shared<ComponentGraphDefinition>(wrong,std::vector<ExecutionTensorContract>{{{},DType::F32}},
                [&](const std::vector<Tensor>& params) { return std::make_unique<st::LegacyDenoiser>(backend,params[0]); });
            std::vector<ComponentInstance> items;
            items.push_back(ComponentInstance::bind({0},{0},graph,{scalar_f32(0)}));
            items.push_back(ComponentInstance::bind({1},{1},bad_graph,{scalar_f32(1)}));
            ExecutionContext invalid(std::move(items));
            runtime.run(invalid,ExecutionProgram::uniform({0}),p); // Even unselected candidates must be compatible.
        });
    }
    for (int bad = 0; bad < 4; ++bad) rejects(backend,[&] {
        auto invalid = p;
        if (bad == 0) invalid.guidance_schedule = GuidanceSchedule::per_step({GuidanceParameters::cfg(1)});
        if (bad == 1) invalid.guidance_schedule = GuidanceSchedule::constant(GuidanceParameters::cfg(std::numeric_limits<float>::quiet_NaN()));
        if (bad == 2) invalid.guidance_schedule = GuidanceSchedule::per_step({GuidanceParameters::cfg(1),GuidanceParameters::linear({1}),GuidanceParameters::cfg(1)});
        if (bad == 3) invalid.guidance_schedule = GuidanceSchedule::constant(GuidanceParameters::linear({1}));
        runtime.run(context,ExecutionProgram::uniform({0}),invalid);
    });
    pass("admission_rejects_19_invalid_declarations_before_rng_and_ops");
}

void partitions() {
    st::Backend backend;
    auto p = st::program();
    auto context = catalog(graph_for(backend,p));
    auto execution = ExecutionProgram::partition({SelectionCoordinate::ModelTimestepScalar,
        SelectionComparison::GreaterOrEqual,int64_t(5),{0},{1}});
    auto plan = execution.admit(context,p,DType::F32);
    require(plan[0]->id().value == 0 && plan[1]->id().value == 0 && plan[2]->id().value == 1,"Inclusive partition boundary");
    p.model_timesteps = {scalar_i64(5),scalar_i64(5),scalar_i64(9)};
    plan = execution.admit(context,p,DType::F32);
    require(plan[0]->id().value == 0 && plan[1]->id().value == 0 && plan[2]->id().value == 0,"Repeated timestep selection");
    constexpr int64_t large = 9007199254740993LL;
    p.model_timesteps = {scalar_i64(large-1),scalar_i64(large),scalar_i64(large+1)};
    plan = ExecutionProgram::partition({SelectionCoordinate::ModelTimestepScalar,
        SelectionComparison::LessThan,large,{0},{1}}).admit(context,p,DType::F32);
    require(plan[0]->id().value == 0 && plan[1]->id().value == 1,"Integer precision lost");
    rejects(backend,[&] { ExecutionProgram::partition({SelectionCoordinate::ModelTimestepScalar,
        SelectionComparison::GreaterOrEqual,5.0f,{0},{1}}).admit(context,p,DType::F32); });
    rejects(backend,[&] { ExecutionProgram::partition({SelectionCoordinate::FlowSigma,
        SelectionComparison::GreaterOrEqual,0.5f,{0},{1}}).admit(context,p,DType::F32); });
    auto flow = st::program(1);
    auto flow_context = catalog(graph_for(backend,flow));
    plan = ExecutionProgram::partition({SelectionCoordinate::FlowSigma,
        SelectionComparison::GreaterOrEqual,0.5f,{0},{1}}).admit(flow_context,flow,DType::F32);
    require(plan[1]->id().value == 0 && plan[2]->id().value == 1,"Flow partition boundary");
    auto borrowed = ExecutionContext::legacy(context.at({0}).denoiser());
    p.model_timesteps[0] = host_i64({2},{5,5});
    rejects(backend,[&] { execution.admit(borrowed,p,DType::F32); });
    p.model_timesteps = {scalar_f32(0.9f),scalar_f32(0.5f),scalar_f32(0.1f)};
    plan = ExecutionProgram::partition({SelectionCoordinate::ModelTimestepScalar,
        SelectionComparison::GreaterOrEqual,0.5f,{0},{0}}).admit(borrowed,p,DType::F32);
    require(plan.size() == 3,"Floating timestep partition");
    pass("typed_partitions_boundaries_repeated_steps_and_exact_i64");
}

class TestArchitecture final : public Architecture {
public:
    explicit TestArchitecture(bool multi) : multi(multi) {}
    std::unique_ptr<Denoiser> create_denoiser(vrhino::Backend& backend, const PrecisionPolicy&, const TensorBundle&) override {
        order.push_back(1);
        return std::make_unique<st::LegacyDenoiser>(dynamic_cast<st::Backend&>(backend),scalar_f32(0.375f));
    }
    SamplingProgram create_program(const TensorBundle&) override { order.push_back(2); return st::program(1); }
    ExecutionSetup create_execution_setup(vrhino::Backend& backend, const PrecisionPolicy& policy, const TensorBundle& input) override {
        if (!multi) return Architecture::create_execution_setup(backend,policy,input);
        order.push_back(1);
        return {catalog(graph_for(dynamic_cast<st::Backend&>(backend),st::program(1))),
                ExecutionProgram::per_step({{0},{1},{0}})};
    }
    Tensor decode(vrhino::Backend&, const PrecisionPolicy&, const Tensor& latent, const TensorBundle&) override {
        order.push_back(3); return latent;
    }
    bool multi;
    std::vector<int> order;
};

void native_runtime_and_lifetime() {
    for (bool multi : {false,true}) {
        st::Backend backend;
        TestArchitecture architecture(multi);
        auto result = NativeRuntime(backend).execute(architecture,{});
        require(architecture.order == std::vector<int>({1,2,3}),"Architecture factory/decode order changed");
        require(backend.rng_calls == 1 && result.outputs.contains("video"),"NativeRuntime setup not integrated");
        require(!result.outputs.contains("step.0.endpoint.instance"),"Default trace keys changed");
    }
    auto p = st::program();
    auto owner = std::make_shared<std::vector<float>>(1,0.25f);
    std::weak_ptr<std::vector<float>> weak = owner;
    {
        st::Backend backend;
        {
        std::vector<ComponentInstance> items;
        Tensor borrowed = Tensor::borrowed(owner->data(),sizeof(float),{},DType::F32);
        items.push_back(ComponentInstance::bind({0},{0},graph_for(backend,p),{borrowed},{owner}));
        owner.reset();
        ExecutionContext context(std::move(items));
        require(!weak.expired(),"Resource owner not retained");
        SamplingRuntime(backend).run(context,ExecutionProgram::uniform({0}),p);
    }
    require(!weak.expired(),"Backend session lost promoted source lease");
    }
    require(weak.expired(),"Execution resources leaked after Backend teardown");
    pass("NativeRuntime_default_and_multi_setup_and_owner_lifetime");
}

class WrongOutput final : public st::LegacyDenoiser {
public:
    WrongOutput(st::Backend& backend, Tensor parameter, int fault)
        : LegacyDenoiser(backend,std::move(parameter)), fault(fault) {}
    std::vector<Tensor> evaluate(const Tensor& latent,const Tensor& t) override {
        auto outputs = LegacyDenoiser::evaluate(latent,t);
        if (fault == 0) outputs.pop_back();
        if (fault == 1) outputs[0] = outputs[0].reshape({2});
        if (fault == 2) outputs[0] = host_i64({1,2},{1,2});
        return outputs;
    }
    int fault;
};

void failures_and_independent_schedules() {
    for (int fault = 0; fault < 3; ++fault) {
        st::Backend backend;
        auto p = st::program(1);
        auto graph = std::make_shared<ComponentGraphDefinition>(interface_for(p),
            std::vector<ExecutionTensorContract>{{{},DType::F32}},
            [&](const std::vector<Tensor>& params) { return std::make_unique<WrongOutput>(backend,params[0],fault); });
        auto context = catalog(graph);
        SamplingRuntime runtime(backend);
        bool failed = false;
        try { runtime.run(context,ExecutionProgram::per_step({{0},{1},{0}}),p); }
        catch (const Error&) { failed = true; }
        require(failed && !runtime.primitives().calls().contains("state_advance") && backend.syncs == 1 &&
                    dynamic_cast<WrongOutput&>(context.at({0}).denoiser()).evaluations == 1 &&
                    dynamic_cast<WrongOutput&>(context.at({1}).denoiser()).evaluations == 0,
                "Bad output reached solver or fallback endpoint");
    }
    {
        st::Backend backend;
        backend.fail_call = 2; backend.fail_sync = true;
        auto p = st::program();
        auto context = catalog(graph_for(backend,p));
        bool primary = false;
        try { SamplingRuntime(backend).run(context,ExecutionProgram::uniform({0}),p); }
        catch (const Error& e) { primary = std::string(e.what()) == "injected primitive failure"; }
        require(primary && backend.syncs == 1,"Primary failure replaced by completion failure");
    }
    {
        st::Backend backend;
        auto p = st::program(1);
        std::vector<Observation> seen;
        auto context = catalog(graph_for(backend,p,&seen));
        SamplingRuntime runtime(backend);
        bool cancel = false;
        runtime.set_cancellation_requested([&] { return cancel; });
        runtime.set_step_observer([&](int,int) { cancel = true; });
        bool failed = false;
        try { runtime.run(context,ExecutionProgram::per_step({{0},{1},{0}}),p); }
        catch (const Error&) { failed = true; }
        require(failed && seen.size() == 1 && runtime.primitives().calls().at("state_advance") == 1 &&
                    backend.rng_calls == 1 && backend.syncs == 1,"Cancellation executed another instance");
    }
    {
        st::Backend backend;
        auto p = st::program();
        auto context = catalog(graph_for(backend,p),false);
        p.guidance_schedule = GuidanceSchedule::per_step({GuidanceParameters::cfg(1),
            GuidanceParameters::cfg(2),GuidanceParameters::cfg(3)});
        auto result = SamplingRuntime(backend).run(context,ExecutionProgram::uniform({0}),p);
        for (int i = 0; i < 3; ++i) {
            auto prefix = "step."+std::to_string(i)+".";
            const auto& u = result.trace.at(prefix+"prediction.0");
            const auto& c = result.trace.at(prefix+"prediction.1");
            const auto& guided = result.trace.at(prefix+"guidance");
            for (int64_t j = 0; j < u.numel(); ++j)
                require(guided.data_as<float>()[j] == u.data_as<float>()[j] +
                    static_cast<float>(i+1)*(c.data_as<float>()[j]-u.data_as<float>()[j]),"Guidance tied to binding");
        }
    }
    {
        st::Backend backend, reference_backend;
        auto p = st::program(1);
        auto context = catalog(graph_for(backend,p));
        auto result = SamplingRuntime(backend).run(context,ExecutionProgram::per_step({{0},{1},{0}}),p);
        SequenceReference reference(reference_backend);
        auto expected = SamplingRuntime(reference_backend).run(reference,p);
        st::exact(result.final_latent,expected.final_latent);
        for (int i = 0; i < 3; ++i) {
            auto name = "step."+std::to_string(i)+".guidance";
            st::exact(result.trace.at(name),expected.trace.at(name));
        }
    }
    {
        st::Backend backend;
        auto p = st::program();
        auto graph = graph_for(backend,p);
        auto parameter = scalar_f32(0.375f);
        std::vector<ComponentInstance> items;
        items.push_back(ComponentInstance::bind({0},{10},graph,{parameter}));
        items.push_back(ComponentInstance::bind({1},{10},graph,{parameter}));
        ExecutionContext context(std::move(items));
        auto result = SamplingRuntime(backend).run(context,ExecutionProgram::per_step({{0},{1},{0}}),p);
        require(result.prepared_tensors.resident_entries == 2,"Instance-private caches merged by binding ID");
    }
    pass("output_contract_errors_cancellation_drain_independent_guidance_and_shared_binding");
}
}  // namespace

int main() {
    try {
        bounded_prefix(); compatibility(); multi_binding(); solver_trace_observation(); admission(); partitions(); native_runtime_and_lifetime();
        failures_and_independent_schedules();
        std::cout << "GENERIC_STEP_EXECUTION=PASS groups=" << cases << '\n';
    } catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
