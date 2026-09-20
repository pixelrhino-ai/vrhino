#include "vrhino/product/declared_run.h"
#include "vrhino/product/prepared_execution.h"
#include "vrhino/product/qualification_precision.h"
#include "vrhino/backend/cuda_backend.h"
#include "run_media.h"
#include <algorithm>
#include <chrono>

namespace vrhino::product {
RunResult run_declared_text_product(const ResolvedRunnableModel& model,
                                    const RunOptions& options, RunEventSink events) {
    // This guard is intentionally independent of structurally valid declarations.
    // No current schema2 package is numerically admitted by this implementation.
    require_numerical_product_admission(model.manifest);
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    const auto cancelled = [&] {
        if (options.cancellation_requested && options.cancellation_requested())
            throw ModelPackageError(ModelPackageErrorCode::Cancelled, "Product run cancelled");
    };
    try {
        cancelled();
        const auto output = options.output.empty() ? std::filesystem::path("output.mp4") : options.output;
        preflight_output_destination(output, options.overwrite);
        const auto encoder = options.encoder_path.empty() ? default_media_encoder_path() : options.encoder_path;
        preflight_media_encoder(encoder);
        const auto hardware = preflight_runnable_model(model, options.preset);
        if (hardware.status == PreflightStatus::InsufficientVram)
            throw ModelPackageError(ModelPackageErrorCode::InsufficientVram, hardware.message);
        if (hardware.status == PreflightStatus::UnsupportedGpu)
            throw ModelPackageError(ModelPackageErrorCode::UnsupportedGpu, hardware.message);
        if (hardware.status == PreflightStatus::DriverIncompatible)
            throw ModelPackageError(ModelPackageErrorCode::DriverIncompatible, hardware.message);
        if (events) events(RunEvent::stage_changed(RunStage::Loading, "Loading declared Product"));
        auto admitted = std::make_shared<AdmittedLocalProduct>(preflight_resolved_product(model));
        const auto plan = lower_declared_text_run(admitted->resources, options);
        if (plan.memory.device_budget_bytes > hardware.hardware.available_vram_bytes)
            throw ModelPackageError(ModelPackageErrorCode::InsufficientVram,
                                    "Declared device budget exceeds currently available memory");
        const auto request = prepare_text_product_request(admitted, plan.request);
        const auto precision = load_declared_product_precision(admitted->resources, plan.precision_artifact);
        cancelled();
        RunResult result;
        result.identity = model.manifest.identity;
        result.preset = model.manifest.default_preset;
        result.seed = request.sampling.seed;
        result.width = request.expected_video_shape.at(4);
        result.height = request.expected_video_shape.at(3);
        result.frames = request.expected_video_shape.at(2);
        result.fps = plan.fps;
        result.preflight = hardware;
        result.package_validation_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        const auto factory = [&]() -> std::unique_ptr<Backend> {
            auto backend = std::make_unique<CudaBackend>();
            backend->set_execution_dtype(precision.policy.requested_dtype());
            backend->configure_memory_runtime(plan.memory, {true, false, false});
            backend->set_vrm_mapped_bytes(admitted->model->file_size() + admitted->conditioning->mapped_bytes());
            return backend;
        };
        PreparedExecutionControl control;
        control.solver_trace = false;
        control.tensor_trace = false;
        control.cancellation_requested = options.cancellation_requested;
        auto phase_started = Clock::now();
        if (events) events(RunEvent::stage_changed(RunStage::Conditioning, "Encoding prompt"));
        control.phase_completed = [&](ProductExecutionPhase phase, const PreparedProductExecution&) {
            const auto now = Clock::now();
            const double elapsed = std::chrono::duration<double>(now - phase_started).count();
            if (phase == ProductExecutionPhase::Conditioning) {
                result.conditioning_seconds = elapsed;
                if (events) events(RunEvent::stage_changed(RunStage::Sampling, "Sampling"));
            } else if (phase == ProductExecutionPhase::Sampling) {
                result.sampling_seconds = elapsed;
                if (events) events(RunEvent::stage_changed(RunStage::Decoding, "Decoding video"));
            } else result.decode_seconds = elapsed;
            phase_started = now;
        };
        control.step_completed = [&](int step, int total, ComponentInstanceID, BindingID, const Json&) {
            if (events) events(RunEvent::progress(RunStage::Sampling, static_cast<uint64_t>(step),
                static_cast<uint64_t>(total), RunProgressUnit::Step, "Sampling"));
        };
        const auto execution = execute_prepared_product(request, precision.policy, factory, control);
        cancelled();
        if (events) events(RunEvent::stage_changed(RunStage::Encoding, "Encoding MP4"));
        const auto encoded = run_media_detail::encode_mp4(execution.video, plan.fps, encoder, output,
            options.overwrite, options.cancellation_requested, plan.video_minimum, plan.video_maximum);
        result.encoding_seconds = encoded.seconds;
        result.output_bytes = encoded.bytes;
        result.video_minimum = encoded.minimum;
        result.video_maximum = encoded.maximum;
        result.nan_count = encoded.nan_count;
        result.inf_count = encoded.inf_count;
        for (const auto& [name, memory] : execution.resources) {
            (void)name;
            result.peak_device_bytes = std::max(result.peak_device_bytes,
                static_cast<uint64_t>(memory.at("peak_device_bytes").integer()));
        }
        result.total_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        if (events) events(RunEvent::stage_changed(RunStage::Finalizing, "Done"));
        return result;
    } catch (const ModelPackageError&) { throw; }
      catch (const std::exception& error) {
        cancelled();
        const std::string message = error.what();
        const bool oom = message.find("out of memory") != std::string::npos ||
            message.find("Out of memory") != std::string::npos || message.find("cudaErrorMemoryAllocation") != std::string::npos;
        throw ModelPackageError(oom ? ModelPackageErrorCode::OutOfMemory : ModelPackageErrorCode::RuntimeError, message);
    }
}
} // namespace vrhino::product
