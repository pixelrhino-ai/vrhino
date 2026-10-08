#include "ltx_schema2_test_support.h"
#include "step_execution_test_support.h"
#include <limits>
#include <sstream>
using namespace vrhino;
namespace s = vrhino::ltx_schema2_test;
namespace st = vrhino::step_test;

ExecutionContext context(st::Backend &backend, const SamplingProgram &p) {
    ComponentInterface io{{p.latent_shape, DType::F32},
                          {{1, 1}, DType::F32},
                          {{p.latent_shape, DType::F32}, {p.latent_shape, DType::F32}},
                          GuidanceMode::CFG,
                          PredictionSemantic::Flow,
                          1};
    auto graph = std::make_shared<ComponentGraphDefinition>(
        io, std::vector<ExecutionTensorContract>{{{1}, DType::F32}},
        [&backend](const auto &params) {
            return std::make_unique<st::LegacyDenoiser>(backend, params.at(0));
        });
    std::vector<ComponentInstance> instances;
    instances.push_back(ComponentInstance::bind({0}, {0}, graph, {host_f32({1}, {0.375f})}));
    return ExecutionContext(std::move(instances));
}
int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: ltx-schema2-contract-tests OUTPUT SPECS");
        s::fs::path dir = argv[1];
        s::fs::create_directories(dir);
        auto marker = s::legacy_marker(dir);
        auto legacy = create_architecture(*marker);
        const auto decl = ltx_internal::token_flow_declaration(DType::BF16);
        const auto graph = s::catalog(decl, {{"marker", Json(std::string("marker"))}});
        const auto package = PackageDeclaration::parse(graph);
        for (auto pair : {std::pair{int64_t(4), 3}, std::pair{int64_t(5280), 40}}) {
            auto reference = s::legacy_schedule(*legacy, pair.first, pair.second);
            const auto j = s::programs(reference);
            s::write(dir / ("programs-" + std::to_string(pair.second) + ".json"), j);
            const auto restored =
                s::read(dir / ("programs-" + std::to_string(pair.second) + ".json"));
            auto admitted = s::p::admit_program_declaration(restored, package);
            for (int i = 0; i < reference.steps; ++i) {
                const auto &t = admitted.sampling.contract->schedule.flow_at(i);
                s::exact(reference.model_timesteps[i], t.model_timestep);
                require(std::bit_cast<uint32_t>(t.sigma) ==
                                std::bit_cast<uint32_t>(reference.sigmas[i]) &&
                            std::bit_cast<uint32_t>(t.sigma - t.next_sigma) ==
                                std::bit_cast<uint32_t>(reference.update_deltas[i]) &&
                            admitted.sampling.guidance_schedule->at(i).scale ==
                                reference.guidance_coefficients[1],
                        "Frozen sigma/delta/CFG drift");
            }
            // Independent deterministic scheduler recurrence on a small state.
            reference.latent_shape = {1, 2};
            auto typed = admitted.sampling;
            typed.latent_shape = reference.latent_shape;
            const auto initial = host_f32({1, 2}, {0.2f, -0.3f});
            st::Backend b1, b2;
            st::LegacyDenoiser endpoint(b1, scalar_f32(0.375f));
            SamplingRuntime old(b1, PrecisionPolicy::fp32()), runtime(b2, PrecisionPolicy::fp32());
            auto ctx = context(b2, typed);
            require(!ctx.is_legacy(), "Legacy context leak");
            auto before = old.run_with_initial_state(endpoint, reference, initial);
            auto after = runtime.run_with_initial_state(ctx, admitted.execution, typed, initial);
            s::exact(before.final_latent, after.final_latent);
            for (int i = 0; i < reference.steps; ++i)
                for (const auto *key : {"latent", "guidance", "prediction.0", "prediction.1"})
                    s::exact(before.trace.at("step." + std::to_string(i) + "." + key),
                             after.trace.at("step." + std::to_string(i) + "." + key));
            require(old.primitives().calls() == runtime.primitives().calls() &&
                        !runtime.primitives().calls().contains("flow_to_x0") &&
                        !runtime.primitives().calls().contains("multistep_predictor"),
                    "Primitive substitution");
            runtime.set_solver_trace_enabled(true);
            auto observed = runtime.run_with_initial_state(ctx, admitted.execution, typed, initial);
            s::exact(after.final_latent, observed.final_latent);
            for (int i = 0; i < typed.steps; ++i)
                require(read_scalar_i64(observed.trace.at("step." + std::to_string(i) +
                                                          ".solver.history_before")) == 0,
                        "Euler accumulated history");
            s::reject("legacy bridge for typed Euler",
                      [&] { SamplingRuntime(b1, PrecisionPolicy::fp32()).run(endpoint, typed); });
            std::cout << "PASS frozen_schedule_bits delta_order CFG stepwise_latent_parity steps="
                      << typed.steps << '\n';
        }
        const auto good = s::programs(s::legacy_schedule(*legacy, 4));
        auto negative = [&](const std::string &name, Json bad) {
            s::reject(name, [&] { s::p::admit_program_declaration(bad, package); });
        };
        s::reject("reader without Euler capability", [&] {
            s::p::admit_program_declaration(
                good, package, {"execution.per_step.v1", "sampling.flow_sigma_cfg.v1"});
        });
        negative("unknown schema",
                 s::set(good, "schema", Json(std::string("vrhino.programs.v99"))));
        negative("workflow", s::set(good, "workflow", Json(Json::Array{})));
        negative("old capabilities",
                 s::set(good, "required_capabilities",
                        Json(Json::Array{Json(std::string("execution.per_step.v1")),
                                         Json(std::string("sampling.flow_sigma_cfg.v1"))})));
        negative("duplicate capability",
                 s::set(good, "required_capabilities",
                        Json(Json::Array{Json(std::string("execution.per_step.v1")),
                                         Json(std::string("execution.per_step.v1")),
                                         Json(std::string("sampling.flow_euler_cfg.v1"))})));
        for (auto pair :
             {std::pair{"solver", "multistep_predictor_corrector"},
              std::pair{"prediction", "epsilon"}, std::pair{"schedule", "alpha_cumprod"},
              std::pair{"branch_order", "conditional_then_unconditional"}})
            negative(pair.first, s::set(good, "sampling",
                                        s::set(good.at("sampling"), pair.first,
                                               Json(std::string(pair.second)))));
        negative("maximum order",
                 s::set(good, "sampling",
                        s::set(good.at("sampling"), "maximum_order", Json(int64_t(2)))));
        for (auto pair : {std::pair{"dtype", "int64"}, std::pair{"source", "model_timestep"}})
            negative(pair.first,
                     s::set(good, "sampling",
                            s::set(good.at("sampling"), "model_timestep",
                                   s::set(good.at("sampling").at("model_timestep"), pair.first,
                                          Json(std::string(pair.second))))));
        negative("time shape", s::set(good, "sampling",
                                      s::set(good.at("sampling"), "model_timestep",
                                             s::set(good.at("sampling").at("model_timestep"),
                                                    "shape", Json(Json::Array{})))));
        negative("floating shape dimension",
                 s::set(good, "sampling",
                        s::set(good.at("sampling"), "model_timestep",
                               s::set(good.at("sampling").at("model_timestep"), "shape",
                                      Json(Json::Array{Json(1.0), Json(1.0)})))));
        for (int mode = 0; mode < 8; ++mode) {
            auto ts = good.at("sampling").at("transitions").array();
            if (mode == 0)
                ts[0] = s::set(ts[0], "sigma", Json(0.9));
            if (mode == 1)
                ts[2] = s::set(ts[2], "next_sigma", Json(0.1));
            if (mode == 2)
                ts[1] = s::set(ts[1], "sigma", Json(0.5));
            if (mode == 3)
                ts[0] = s::set(ts[0], "next_sigma", Json(1.0));
            if (mode == 4)
                ts[0] = s::set(ts[0], "guidance_scale", Json(1e100));
            if (mode == 5)
                ts[0] = s::set(ts[0], "guidance_scale", Json(-1.0));
            if (mode == 6)
                ts[0] = s::set(ts[0], "model_timestep", Json(int64_t(1000)));
            if (mode == 7)
                ts[2] = s::set(ts[2], "next_sigma", Json(-0.0));
            negative(
                "transition " + std::to_string(mode),
                s::set(good, "sampling", s::set(good.at("sampling"), "transitions", Json(ts))));
        }
        for (int n : {1, 1001}) {
            auto ids = Json::Array(n, Json(int64_t(0)));
            auto ts = Json::Array(n, good.at("sampling").at("transitions").array()[0]);
            auto bad =
                s::set(good, "execution", s::set(good.at("execution"), "instance_ids", Json(ids)));
            negative("step bound " + std::to_string(n),
                     s::set(bad, "sampling", s::set(bad.at("sampling"), "transitions", Json(ts))));
        }
        negative("unknown instance",
                 s::set(good, "execution",
                        s::set(good.at("execution"), "instance_ids",
                               Json(Json::Array{Json(int64_t(0)), Json(int64_t(9)),
                                                Json(int64_t(0))}))));
        auto witness =
            s::marker(dir / "program-authority.vrm", graph,
                      Json(Json::Object{{"architecture", Json(std::string("sampling_test"))},
                                        {"programs", good}}),
                      "sampling_test");
        s::write(dir / "programs.json", good);
        const auto sha = s::p::sha256_file(dir / "programs.json");
        auto verified = s::p::admit_verified_program_artifact(
            dir / "programs.json", s::fs::file_size(dir / "programs.json"), sha, *witness);
        s::write(dir / "typed-evidence.json", verified.evidence);
        require(
            verified.evidence.at("steps").array()[0].at("model_timestep").at("dtype").string() ==
                "float32",
            "I64 trace leak");
        s::reject("artifact SHA", [&] {
            s::p::admit_verified_program_artifact(dir / "programs.json",
                                                  s::fs::file_size(dir / "programs.json"),
                                                  std::string(64, '0'), *witness);
        });
        auto changed = s::set(good, "sampling",
                              s::set(good.at("sampling"), "maximum_order", Json(int64_t(2))));
        s::write(dir / "changed.json", changed);
        s::reject("competing authority", [&] {
            s::p::admit_verified_program_artifact(
                dir / "changed.json", s::fs::file_size(dir / "changed.json"),
                s::p::sha256_file(dir / "changed.json"), *witness);
        });
        auto g =
            s::p::admit_token_grid_geometry(s::geometry(), 704, 480, 121, UINT64_MAX, 1ULL << 30);
        require(g.latent_shape == std::vector<int64_t>({1, 5280, 128}) &&
                    read_scalar_i64(g.runtime_inputs.at("decode_seed")) == 99,
                "Geometry or seed wrap");
        auto small = s::p::admit_token_grid_geometry(s::geometry(), 64, 32, 9, 11, 1ULL << 20);
        require(small.latent_shape == std::vector<int64_t>({1, 4, 128}), "Synthetic geometry");
        for (auto dims : {std::array<int64_t, 3>{63, 32, 9},
                          {64, 31, 9},
                          {64, 32, 8},
                          {64, 32, 0},
                          {INT64_MAX, INT64_MAX, INT64_MAX}})
            s::reject("geometry dimensions", [&] {
                s::p::admit_token_grid_geometry(s::geometry(), dims[0], dims[1], dims[2], 11,
                                                1ULL << 30);
            });
        s::reject("geometry tensor budget",
                  [&] { s::p::admit_token_grid_geometry(s::geometry(), 704, 480, 121, 11, 4); });
        auto badg = small;
        badg.runtime_inputs = small.runtime_inputs;
        badg.runtime_inputs["coordinates"] = Tensor::host({1, 3, 4}, DType::F32);
        s::reject("wrong/pre-scaled coordinates",
                  [&] { s::p::validate_token_grid_geometry(badg); });
        badg = small;
        badg.expected_video_shape[2] = 10;
        s::reject("decoder output geometry", [&] { s::p::validate_token_grid_geometry(badg); });
        s::reject("geometry DSL", [&] {
            s::p::admit_token_grid_geometry(s::set(s::geometry(), "workflow", Json(Json::Array{})),
                                            64, 32, 9, 11, 1ULL << 20);
        });
        auto def = ltx_internal::lower_token_flow(decl);
        require(def.parameters->slots().size() == 715, "715 slots missing");
        require(ltx_internal::decoder_slots(DType::BF16)->slots().size() == 297,
                "Decoder slots missing");
        for (const auto &k : {"latent_channels", "hidden_size", "head_count", "block_count",
                              "feed_forward_size", "time_frequency_size"})
            s::reject(k, [&] {
                ltx_internal::lower_token_flow(
                    s::set(decl, "dimensions", s::set(decl.at("dimensions"), k, Json(int64_t(1)))));
            });
        s::reject("architecture op escape", [&] {
            ltx_internal::lower_token_flow(s::set(decl, "operators", Json(Json::Array{})));
        });
        s::reject("floating architecture dimension", [&] {
            ltx_internal::lower_token_flow(s::set(
                decl, "dimensions", s::set(decl.at("dimensions"), "hidden_size", Json(2048.0))));
        });
        // Independent frozen TSV role/shape check, not a mirrored slot generator.
        std::ifstream map(s::fs::path(argv[2]) / "ltx_v0_9_1/tensor-mapping.tsv");
        require(map.good(), "Missing slot oracle");
        std::map<std::string, std::vector<int64_t>> oracle;
        std::string line;
        std::getline(map, line);
        while (std::getline(map, line)) {
            std::vector<std::string> cols;
            std::stringstream stream(line);
            std::string col;
            while (std::getline(stream, col, '\t'))
                cols.push_back(col);
            if (cols.at(7) != "denoiser")
                continue;
            auto shape = Json::parse(cols.at(6));
            std::vector<int64_t> dims;
            for (const auto &v : shape.array())
                dims.push_back(v.integer());
            oracle.emplace(cols.at(0).substr(std::string("model.diffusion_model.").size()), dims);
        }
        require(oracle.size() == 715, "Frozen oracle count");
        for (const auto &slot : def.parameters->slots())
            require(oracle.at(slot.role) == slot.shape && slot.dtype == DType::BF16,
                    "Slot oracle mismatch");
        auto clean = s::p::admit_program_declaration(good, package).sampling;
        clean.latent_shape = {1, 2};
        for (int fault = 0; fault < 6; ++fault) {
            st::Backend b;
            auto ctx = context(b, clean);
            s::reject("runtime timestep tensor contract " + std::to_string(fault), [&] {
                auto p = clean;
                std::vector<FlowScheduleTransition> transitions;
                for (int i = 0; i < p.steps; ++i)
                    transitions.push_back(p.contract->schedule.flow_at(i));
                if (fault == 0)
                    transitions[1].model_timestep = scalar_i64(1);
                if (fault == 1)
                    transitions[1].model_timestep = scalar_f32(transitions[1].sigma);
                if (fault == 2)
                    transitions[1].model_timestep = Tensor::host({1, 1}, DType::BF16);
                if (fault == 3) {
                    auto storage = std::make_shared<Storage>();
                    storage->data = transitions[1].model_timestep.data();
                    storage->bytes = 4;
                    storage->device = DeviceId::accelerator(0);
                    transitions[1].model_timestep = Tensor(storage, 0, {1, 1}, DType::F32);
                }
                if (fault == 4) {
                    auto time = Tensor::host({1, 1}, DType::U8);
                    auto q = std::make_shared<QuantizationInfo>();
                    q->type = QuantType::INT8Symmetric;
                    q->logical_dtype = DType::F32;
                    q->scales = scalar_f32(1);
                    time.set_quantization(q);
                    transitions[1].model_timestep = time;
                }
                if (fault == 5) {
                    auto storage = std::make_shared<Storage>();
                    storage->data = transitions[1].model_timestep.data();
                    storage->bytes = 4;
                    transitions[1].model_timestep = Tensor(storage, 0, {1, 1}, DType::F32);
                    storage->bytes = 0;
                }
                p.contract.emplace(PredictionContract{PredictionSemantic::Flow},
                                   SolverContract{SolverSemantic::FlowEuler, 1},
                                   ScheduleContract::flow_sigma(transitions));
                SamplingRuntime(b, PrecisionPolicy::fp32())
                    .run(ctx, ExecutionProgram::uniform({0}), p);
            });
            require(b.rng_calls == 0 && b.calls.empty(), "Bad timestep tensor dispatched");
        }
        for (int fault = 0; fault < 3; ++fault) {
            st::Backend failing;
            auto ctx = context(failing, clean);
            SamplingRuntime runtime(failing, PrecisionPolicy::fp32());
            if (fault == 0)
                failing.fail_call = 2;
            if (fault == 1) {
                failing.fail_call = 2;
                failing.fail_sync = true;
            }
            if (fault == 2)
                runtime.set_cancellation_requested([] { return true; });
            s::reject("typed failure/cancellation " + std::to_string(fault),
                      [&] { runtime.run(ctx, ExecutionProgram::uniform({0}), clean); });
            require(!runtime.primitives().calls().contains("state_advance") &&
                        failing.rng_calls == (fault == 2 ? 0 : 1) &&
                        failing.syncs == (fault == 2 ? 0 : 1),
                    "Typed failure cleanup/order regression");
        }
        auto typed = s::p::admit_program_declaration(good, package).sampling;
        typed.latent_shape = {1, 2};
        st::Backend backend;
        auto ctx = context(backend, typed);
        typed.contract->schedule.flow_at(1).model_timestep.data_as<float>()[0] = 0.5f;
        s::reject("mutated timestep before RNG", [&] {
            SamplingRuntime(backend, PrecisionPolicy::fp32())
                .run(ctx, ExecutionProgram::uniform({0}), typed);
        });
        require(backend.rng_calls == 0 && backend.calls.empty(), "Rejected program executed");
        std::cout << "LTX_SCHEMA2_CONTRACT=PASS negative_cases=" << s::negatives
                  << " slots=715 frozen_schedule_40=BITWISE geometry=PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
