#include "ltx_schema2_test_support.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/conditioning.h"
#include "vrhino/product/declared_run.h"
#include "vrhino/product/prepared_execution.h"
#include "vrhino/product/qualification_precision.h"
#include "vrhino/product/vrm_verification.h"
#include <iostream>
namespace s = vrhino::ltx_schema2_test;
namespace p = vrhino::product;
using namespace vrhino;
int main(int argc, char **argv) {
    try {
        require(argc == 5, "usage: reference-compare PACKAGE OLD_VRM OUTPUT PROMPT");
        const s::fs::path package = argv[1], dir = argv[3];
        s::fs::create_directories(dir);
        auto admitted = std::make_shared<p::AdmittedLocalProduct>(
            p::preflight_local_product(package / "vrhino-model.json", package / "local.json"));
        p::RunOptions options;
        options.prompt = argv[4];
        options.seed = 5703;
        options.model_reference = admitted->resources.manifest.identity.reference();
        const auto plan = p::lower_declared_text_run(admitted->resources, options);
        const auto request = p::prepare_text_product_request(admitted, plan.request);
        const auto precision =
            p::load_declared_product_precision(admitted->resources, plan.precision_artifact);
        auto factory = [&]() -> std::unique_ptr<Backend> {
            auto b = std::make_unique<CudaBackend>();
            b->set_execution_dtype(precision.policy.requested_dtype());
            b->configure_memory_runtime(plan.memory, {true, false, false});
            return b;
        };
        p::PreparedExecutionControl control;
        control.qualification_prefix_steps = 2;
        control.tensor_trace = true;
        control.solver_trace = true;
        const auto typed = p::execute_prepared_product(request, precision.policy, factory, control);
        std::shared_ptr<const VrmModel> old(p::load_verified_vrm_component(
            argv[2], {"legacy-reference", "ltx_v0_9_1", 5717174080ULL,
                      "267a95330f48dbe2134220e6116c60cddddf54f7548a661e20b18531fb70fa7d"}));
        auto legacy = create_architecture(*old);
        auto input = typed.conditioning;
        input["sampling_steps"] = scalar_i64(40);
        input["guidance_scale"] = scalar_f32(3);
        const auto program = legacy->create_program(input);
        for (int i = 0; i < 40; ++i) {
            const auto &t = request.sampling.contract->schedule.flow_at(i);
            s::exact(t.model_timestep, program.model_timesteps[i]);
            require(std::bit_cast<uint32_t>(t.sigma) ==
                            std::bit_cast<uint32_t>(program.sigmas[i]) &&
                        std::bit_cast<uint32_t>(t.next_sigma) ==
                            std::bit_cast<uint32_t>(program.sigmas[i + 1]) &&
                        std::bit_cast<uint32_t>(t.sigma - t.next_sigma) ==
                            std::bit_cast<uint32_t>(program.update_deltas[i]),
                    "Frozen Legacy program bits differ");
        }
        // Independent old conditioning declaration execution; no altered weights or Python
        // reference.
        {
            auto backend = factory();
            backend->retain_resource_owners({admitted});
            ConditioningComponentExecutor executor(*backend, request.conditioning_weights());
            for (const auto &c : request.conditioning) {
                auto result = executor.execute(c.graph, c.input_ids, c.attention_mask);
                auto hidden = result.hidden_states.device().is_host()
                                  ? result.hidden_states
                                  : backend->copy_to_host(result.hidden_states);
                backend->synchronize();
                s::exact(hidden, typed.conditioning.at(c.target));
            }
        }
        {
            auto backend = factory();
            backend->retain_resource_owners({old, admitted});
            auto denoiser = legacy->create_denoiser(*backend, precision.policy, input);
            SamplingRuntime runtime(*backend, precision.policy);
            runtime.set_solver_trace_enabled(true);
            auto context = ExecutionContext::legacy(*denoiser);
            auto result = runtime.run_prefix(context, ExecutionProgram::uniform({0}), program, 2);
            backend->synchronize();
            auto host = [&](const Tensor &t) {
                return t.device().is_host() ? t : backend->copy_to_host(t);
            };
            s::exact(host(result.initial_noise), typed.sampling.initial_noise);
            s::exact(host(result.final_latent), typed.sampling.final_latent);
            for (int step = 0; step < 2; ++step)
                for (const auto *name :
                     {"prediction.0", "prediction.1", "guidance", "latent", "timestep"}) {
                    const auto key = "step." + std::to_string(step) + "." +
                                     (std::string(name) == "timestep" ? "solver.timestep" : name);
                    s::exact(host(result.trace.at(key)), typed.sampling.trace.at(key));
                }
        }
        s::write(dir / "comparison.json",
                 Json(Json::Object{
                     {"status", Json(std::string("PASS"))},
                     {"real_weights", Json(true)},
                     {"conditioning_bitwise", Json(true)},
                     {"all_40_schedule_bits", Json(true)},
                     {"prefix_steps", Json(int64_t(2))},
                     {"prefix_predictions_guidance_latents_bitwise", Json(true)},
                     {"full_raw_video_bitwise_proven", Json(false)},
                     {"numerical_qualification_granted", Json(false)},
                     {"generic_bf16_status", Json(std::string("HOLD"))},
                     {"scope", Json(std::string("same frozen Legacy production profile; comparison "
                                                "only, not a BF16 candidate"))}}));
        std::cout << "STAGE1C_REAL_REFERENCE_PREFIX=PASS conditioning=BITWISE schedule40=BITWISE "
                     "steps2_predictions_guidance_latents=BITWISE full_raw_video=UNPROVEN "
                     "qualification=HOLD\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
