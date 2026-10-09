#include "../src/architectures/ltx_self_attention.h"
#include "ltx_schema2_test_support.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/runtime.h"
#include <chrono>
#include <cmath>

using namespace vrhino;
namespace s = vrhino::ltx_schema2_test;
namespace {
TensorBundle host(Backend &b, const TensorBundle &values) {
    TensorBundle out;
    for (const auto &[k, t] : values)
        out[k] = t.device().is_host() ? t : b.copy_to_host(t);
    return out;
}
void finite(const Tensor &t) {
    require(t.device().is_host() && t.dtype() == DType::F32, "F32 host audit");
    for (int64_t i = 0; i < t.numel(); ++i)
        require(std::isfinite(t.data_as<float>()[i]), "Nonfinite synthetic value");
}
void configure(CudaBackend &b) {
    b.set_execution_dtype(DType::F32);
    MemoryBudget budget{22ULL << 30, 0, 72ULL << 30, 512ULL << 20, 512ULL << 20};
    budget.weight_cache_budget_bytes = 6ULL << 30;
    b.configure_memory_runtime(budget, {true, false, false});
}
// Independent legacy assembly around the existing full 28-block numerical
// implementation. Used only as the reference, never as Schema2 execution.
class Reference final : public Denoiser {
  public:
    Reference(Backend &b, WeightMap w, const TensorBundle &input) : b_(b), w_(w) {
        const auto policy = PrecisionPolicy::fp32();
        text_ = b.concat({b.copy_to_device(input.at("negative"), DType::F32),
                          b.copy_to_device(input.at("positive"), DType::F32)},
                         0);
        auto m = Tensor::host({2, 1, 128}, DType::Bool);
        std::memcpy(m.data(), input.at("negative_mask").data(), 128);
        std::memcpy(m.data_as<uint8_t>() + 128, input.at("positive_mask").data(), 128);
        mask_ = b.copy_to_device(m, DType::Bool);
        const auto &c = input.at("coordinates");
        const int64_t n = c.dim(2);
        auto transformed = Tensor::host({2, 3, n}, DType::F32);
        const float scale[] = {0.32f, 32.0f, 32.0f};
        for (int batch = 0; batch < 2; ++batch)
            for (int a = 0; a < 3; ++a)
                for (int64_t i = 0; i < n; ++i)
                    transformed.data_as<float>()[(batch * 3 + a) * n + i] =
                        c.data_as<float>()[a * n + i] * scale[a];
        rope_ = fractional_rope(b, policy, transformed, {20.0f, 2048.0f, 2048.0f}, 2048, 10000.0f,
                                true);
    }
    std::vector<Tensor> evaluate(const Tensor &x, const Tensor &time) override {
        const float sigma = read_scalar_f32(time);
        auto pred = ltx_internal::denoiser_forward(b_, PrecisionPolicy::fp32(), w_,
                                                   b_.concat({x, x}, 0), rope_.first, rope_.second,
                                                   text_, mask_, host_f32({2, 1}, {sigma, sigma}));
        return b_.split(pred, {1, 1}, 0);
    }

  private:
    Backend &b_;
    WeightMap w_;
    Tensor text_, mask_;
    std::pair<Tensor, Tensor> rope_;
};
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: ltx-schema2-cuda-tests OUTPUT");
        const auto started = std::chrono::steady_clock::now();
        s::fs::path dir = argv[1];
        s::fs::create_directories(dir);
        auto marker = s::legacy_marker(dir);
        auto legacy = create_architecture(*marker);
        auto geometry = s::p::admit_token_grid_geometry(s::geometry(), 64, 32, 9, 11, 16ULL << 20);
        auto legacy_program = s::legacy_schedule(*legacy, 4);
        auto programs = s::programs(legacy_program);
        programs =
            s::set(programs, "execution",
                   s::set(programs.at("execution"), "instance_ids",
                          Json(Json::Array{Json(int64_t(7)), Json(int64_t(0)), Json(int64_t(7))})));
        s::write(dir / "programs.json", programs);
        const auto definition =
            ltx_internal::lower_token_flow(ltx_internal::token_flow_declaration(DType::BF16));
        s::Source source;
        std::vector<s::p::TensorMapping> maps;
        Json::Object parameters, decoder_parameters;
        for (const auto &slot : definition.parameters->slots()) {
            const auto name = "denoiser." + slot.role;
            s::add(source, maps, name, slot);
            parameters.emplace(slot.role, Json(name));
        }
        const auto decoder_slots = ltx_internal::decoder_slots(DType::BF16);
        for (const auto &slot : decoder_slots->slots()) {
            const auto name = "decoder." + slot.role;
            s::add(source, maps, name, slot);
            decoder_parameters.emplace(slot.role, Json(name));
        }
        auto graph = s::catalog(ltx_internal::token_flow_declaration(DType::BF16), parameters);
        auto instances = graph.at("instances").array();
        instances.push_back(Json(Json::Object{
            {"id", Json(int64_t(7))}, {"graph", Json(int64_t(0))}, {"binding", Json(int64_t(0))}}));
        graph = s::set(graph, "instances", Json(instances));
        auto decoder = Json::parse(
            R"JSON({"schema_version":1,"implementation_id":"dit_flow.ltx_v0_9_1.vae_decoder.v1","latent_contract":{"channels":128,"layout":"BCTHW","sampling_layout":"BLC","spatial_scale":32,"temporal_decode":"1+8*(F-1)"},"runtime_tensor_bindings":{}})JSON");
        decoder = s::set(decoder, "runtime_tensor_bindings", Json(decoder_parameters));
        const auto meta =
            Json(Json::Object{{"architecture", Json(std::string("ltx_v0_9_1"))},
                              {"programs", programs},
                              {"shared_components", Json(Json::Object{{"decoder", decoder}})}});
        const auto path = dir / "synthetic-ltx.vrm";
        std::cout << "BUILD_SYNTHETIC_VRM slots=" << definition.parameters->slots().size()
                  << " decoder_slots=297\n"
                  << std::flush;
        s::fs::remove(path);
        s::p::write_vrm_streaming(path, "dit-flow", "ltx_v0_9_1", meta, graph, source, maps);
        std::cout << "SYNTHETIC_VRM bytes=" << s::fs::file_size(path) << '\n' << std::flush;
        auto model = std::make_shared<VrmModel>(path.string());
        std::weak_ptr<const VrmModel> weak = model;
        auto verified = s::p::admit_verified_program_artifact(
            dir / "programs.json", s::fs::file_size(dir / "programs.json"),
            s::p::sha256_file(dir / "programs.json"), *model);
        s::write(dir / "admitted-program-evidence.json", verified.evidence);
        auto architecture = create_architecture(std::shared_ptr<const VrmModel>(model));
        auto input = geometry.runtime_inputs;
        for (const auto *name : {"positive", "negative"}) {
            auto t = Tensor::host({1, 128, 4096}, DType::F32);
            for (int64_t i = 0; i < t.numel(); ++i)
                t.data_as<float>()[i] =
                    float(int((i + (name[0] == 'p' ? 3 : 7)) % 23) - 11) * 0.015625f;
            input[name] = t;
            auto m = Tensor::host({1, 128}, DType::Bool);
            std::memset(m.data(), 0, 128);
            m.data_as<uint8_t>()[0] = 1;
            m.data_as<uint8_t>()[1] = 1;
            input[std::string(name) + "_mask"] = m;
        }
        input["audit_trace"] = scalar_i64(1);
        const auto typed = architecture->create_program(input);
        validate_flow_euler_program(typed);
        require(typed.contract->solver.semantic == SolverSemantic::FlowEuler &&
                    typed.latent_shape == geometry.latent_shape,
                "Lost typed contract");
        for (int i = 0; i < typed.steps; ++i)
            s::exact(typed.model_timestep_at(i), legacy_program.model_timesteps[i]);
        // Real backing admission failures before any endpoint/device computation.
        std::map<std::string, const Tensor *> weights;
        for (const auto &[role, name] : parameters)
            weights.emplace(role, &model->tensor(name.string()));
        auto wrong = weights;
        wrong.erase(wrong.begin());
        s::reject("715 missing slot", [&] {
            AdmittedArchitectureBinding::admit(definition.parameters, wrong,
                                               BorrowedBindingLifetime::ExplicitOwners, {model});
        });
        wrong = weights;
        wrong["patchify_proj.weight"] = &model->tensor("denoiser.scale_shift_table");
        s::reject("715 slot shape", [&] {
            AdmittedArchitectureBinding::admit(definition.parameters, wrong,
                                               BorrowedBindingLifetime::ExplicitOwners, {model});
        });
        auto f32 = Tensor::host({2048}, DType::F32);
        wrong = weights;
        wrong["transformer_blocks.0.attn1.q_norm.weight"] = &f32;
        s::reject("715 slot dtype", [&] {
            AdmittedArchitectureBinding::admit(definition.parameters, wrong,
                                               BorrowedBindingLifetime::ExplicitOwners, {model});
        });
        s::reject("715 borrowed owner", [&] {
            AdmittedArchitectureBinding::admit(definition.parameters, weights,
                                               BorrowedBindingLifetime::ExplicitOwners);
        });
        auto overridden = input;
        overridden["sampling_steps"] = scalar_i64(1);
        s::reject("schedule override", [&] { architecture->create_program(overridden); });
        auto missing_mask = input;
        missing_mask.erase("positive_mask");
        s::reject("mask fallback",
                  [&] { ltx_internal::validate_token_flow_inputs(missing_mask, true); });
        auto initial = Tensor::host(geometry.latent_shape, DType::F32);
        for (int64_t i = 0; i < initial.numel(); ++i)
            initial.data_as<float>()[i] = float(int(i % 19) - 9) * 0.0625f;
        TensorBundle references[2];
        for (int mode = 0; mode < 2; ++mode) {
            CudaBackend backend;
            configure(backend);
            backend.retain_resource_owners({model});
            Reference reference(backend, WeightMap(weights), input);
            SamplingRuntime runtime(backend, PrecisionPolicy::fp32());
            auto r = mode == 0 ? runtime.run_with_initial_state(reference, legacy_program, initial)
                               : runtime.run(reference, legacy_program);
            backend.synchronize();
            references[mode] = host(backend, r.trace);
            for (const auto &[k, t] : references[mode]) {
                (void)k;
                finite(t);
            }
            s::write(dir / ("reference-primitives-" + std::to_string(mode) + ".json"),
                     Json(Json::Object{
                         {"euler_update",
                          Json(int64_t(runtime.primitives().calls().at("euler_update")))}}));
            std::cout << "LEGACY_REFERENCE_FULL_28_BLOCKS mode=" << mode << " PASS\n" << std::flush;
        }
        TensorBundle first;
        for (int mode = 0; mode < 2; ++mode) {
            CudaBackend backend;
            configure(backend);
            NativeRuntime runtime(backend, PrecisionPolicy::fp32());
            auto r = mode == 0 ? runtime.execute_with_external_initial_state_for_test(
                                     *architecture, input, initial)
                               : runtime.execute(*architecture, input);
            for (int step = 0; step < 3; ++step)
                for (const auto *name : {"prediction.0", "prediction.1", "guidance", "latent"}) {
                    const auto key = "step." + std::to_string(step) + "." + name;
                    s::exact(references[mode].at(key), r.outputs.at(key));
                }
            for (int block = 0; block < 28; ++block)
                require(r.outputs.contains("step.0.block." + std::to_string(block)),
                        "Full topology block not executed");
            for (int i = 0; i < 3; ++i)
                s::exact(r.outputs.at("step." + std::to_string(i) + ".timestep"),
                         legacy_program.model_timesteps[i]);
            require(r.sampling_primitive_calls.at("euler_update") == 3 &&
                        !r.sampling_primitive_calls.contains("flow_to_x0") &&
                        !r.sampling_primitive_calls.contains("multistep_predictor"),
                    "Wrong solver primitives");
            require(r.outputs.at("video").shape() == geometry.expected_video_shape,
                    "Decoder output mismatch");
            finite(r.outputs.at("video"));
            for (int64_t i = 0; i < r.outputs.at("video").numel(); ++i)
                require(r.outputs.at("video").data_as<float>()[i] >= 0 &&
                            r.outputs.at("video").data_as<float>()[i] <= 1,
                        "Video range mismatch");
            if (mode == 0)
                first = r.outputs;
            auto repeat = mode == 0 ? runtime.execute_with_external_initial_state_for_test(
                                          *architecture, input, initial)
                                    : runtime.execute(*architecture, input);
            require(repeat.outputs.size() == r.outputs.size(), "Repeated output key mismatch");
            for (const auto &[name, t] : r.outputs)
                if (!name.starts_with("metric."))
                    s::exact(t, repeat.outputs.at(name));
            std::cout << "TYPED_NATIVE_RUNTIME_FULL_28_BLOCKS mode=" << mode
                      << " stepwise_bitwise=PASS repeat=PASS decoder=PASS peak_device="
                      << backend.peak_device_bytes() << '\n'
                      << std::flush;
            s::reject("managed singleton fallback", [&] {
                architecture->create_denoiser(backend, PrecisionPolicy::fp32(), input);
            });
        }
        // Geometry scaling is visible once in the actual architecture trace.
        const auto &c = first.at("step.0.denoiser.cosine");
        require(c.shape() == std::vector<int64_t>({2, 4, 2048}), "RoPE hidden width mismatch");
        {
            CudaBackend backend;
            configure(backend);
            {
                auto setup =
                    architecture->create_execution_setup(backend, PrecisionPolicy::fp32(), input);
                require(!setup.context.is_legacy() &&
                            setup.context.instances()[0].parameters().size() == 715,
                        "Typed endpoint bridge");
                const auto selected = setup.program.admit(setup.context, typed, DType::F32);
                require(selected.size() == 3 && selected[0]->id().value == 7 &&
                            selected[1]->id().value == 0 && selected[2] == selected[0] &&
                            selected[0]->graph() == selected[1]->graph(),
                        "Frozen instance selection lost/shared topology duplicated");
                require(
                    setup.context.instances()[0].parameters()[0].data() ==
                        model
                            ->tensor(parameters.at(definition.parameters->slots()[0].role).string())
                            .data(),
                    "Lost exact borrowed tensor");
            }
            architecture.reset();
            model.reset();
            require(!weak.expired(), "Backend lost source owner after context teardown");
        }
        require(weak.expired(), "Model mmap owner leaked after Backend teardown");
        require(references[0].at("final_latent").shape() == geometry.latent_shape,
                "Final latent shape");
        std::cout
            << "STAGE1B_LTX_SYNTHETIC=PASS full_topology=28 slots=715 decoder_slots=297 "
               "typed_only=PASS selection=7/0/7 ownership=PASS seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
