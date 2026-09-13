#include "vrhino/product/run.h"
#ifdef _WIN32
#include "vrhino/product/windows_process.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>

#include "vrhino/audio_conditioning.h"
#include "vrhino/audio_encoder.h"
#include "vrhino/autoencoder_kl.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/lip_sync_diffusion_workflow.h"
#include "vrhino/pose_estimator.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/vrm_verification.h"
#include "vrhino/sampling.h"
#include "vrhino/temporal_conditional_unet_2d.h"
#include "vrhino/vision_detector.h"

namespace vrhino::product {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const ModelPackageErrorCode code,
                       const std::string& message) {
    throw ModelPackageError(code, message);
}

double elapsed_seconds(const Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

void check_cancelled(const RunOptions& options, const std::string& stage) {
    if (options.cancellation_requested && options.cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "lip-sync diffusion run cancelled before " + stage);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "required product configuration is missing");
    return {std::istreambuf_iterator<char>(input), {}};
}

const ResolvedArtifact& resolved_artifact(const ResolvedRunnableModel& model,
                                          const std::string& id) {
    const auto found = model.artifacts.find(id);
    if (found == model.artifacts.end())
        fail(ModelPackageErrorCode::ArtifactMissing,
             "required package artifact is missing: " + id);
    return found->second;
}

std::string read_verified_text(const ResolvedRunnableModel& model,
                               const std::string& id) {
    const ResolvedArtifact& artifact = resolved_artifact(model, id);
    if (sha256_file(artifact.path) != artifact.declaration.sha256)
        fail(ModelPackageErrorCode::ChecksumMismatch,
             "required product configuration failed integrity validation: " + id);
    return read_text(artifact.path);
}

const ComponentDeclaration& component_role(const ResolvedRunnableModel& model,
                                            const std::string& role) {
    const auto found = std::find_if(
        model.manifest.components.begin(), model.manifest.components.end(),
        [&](const ComponentDeclaration& value) { return value.role == role; });
    if (found == model.manifest.components.end() || found->artifact_ids.size() != 1)
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync diffusion component contract mismatch: " + role);
    return *found;
}

const ResolvedArtifact& component_artifact(const ResolvedRunnableModel& model,
                                           const std::string& role) {
    return resolved_artifact(model,
        component_role(model, role).artifact_ids.front());
}

ComponentIdentityContract identity_contract(const Json& workflow,
                                            const std::string& role) {
    for (const Json& component : workflow.at("components").array()) {
        if (component.at("role").string() != role) continue;
        return {role, component.at("architecture").string(),
                static_cast<uint64_t>(component.at("bytes").integer()),
                component.at("sha256").string()};
    }
    fail(ModelPackageErrorCode::PackageInvalid,
         "workflow component identity is missing: " + role);
}

VrmComponentIntegrityContract integrity_contract(
        const ComponentIdentityContract& contract) {
    return {contract.semantic_name, contract.architecture,
            contract.bytes, contract.sha256};
}

std::vector<float> tensor_values(CudaBackend& backend, const Tensor& tensor) {
    const Tensor host = tensor.device().is_host()
        ? tensor : backend.copy_to_host(tensor);
    return {host.data_as<float>(), host.data_as<float>() + host.numel()};
}

std::vector<uint8_t> rgb_to_bgr(const RgbFrame& frame) {
    std::vector<uint8_t> output = frame.pixels;
    for (size_t index = 0; index < output.size(); index += 3)
        std::swap(output[index], output[index + 2]);
    return output;
}

Tensor horizontal_flip(const Tensor& input) {
    Tensor output = Tensor::host(input.shape(), DType::F32);
    for (int64_t channel = 0; channel < input.dim(1); ++channel)
        for (int64_t y = 0; y < input.dim(2); ++y)
            for (int64_t x = 0; x < input.dim(3); ++x)
                output.data_as<float>()[
                    (channel * input.dim(2) + y) * input.dim(3) + x] =
                    input.data_as<float>()[
                        (channel * input.dim(2) + y) * input.dim(3) +
                        (input.dim(3) - 1 - x)];
    return output;
}

const std::vector<int64_t>& flip_indices() {
    static const std::vector<int64_t> value = {
        0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,20,21,22,17,18,19,
        39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,49,48,47,
        46,45,44,43,42,41,40,50,51,52,53,58,57,56,55,54,68,67,66,65,
        70,69,62,61,60,59,64,63,77,76,75,74,73,72,71,82,81,80,79,78,
        87,86,85,84,83,90,89,88,112,113,114,115,116,117,118,119,120,
        121,122,123,124,125,126,127,128,129,130,131,132,91,92,93,94,
        95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111};
    return value;
}

Tensor slice_host(const Tensor& source, int64_t begin, int64_t end) {
    if (!source.device().is_host() || source.dtype() != DType::F32 ||
        source.ndim() < 1 || begin < 0 || begin >= end || end > source.dim(0))
        fail(ModelPackageErrorCode::RuntimeError,
             "host tensor slice contract mismatch");
    std::vector<int64_t> shape = source.shape();
    shape[0] = end - begin;
    Tensor output = Tensor::host(shape, DType::F32);
    const size_t row_bytes = source.bytes() / static_cast<size_t>(source.dim(0));
    std::memcpy(output.data(), static_cast<const uint8_t*>(source.data()) +
        static_cast<size_t>(begin) * row_bytes, output.bytes());
    return output;
}

Tensor zeros_like_host(const Tensor& source) {
    Tensor output = Tensor::host(source.shape(), DType::F32);
    std::fill(output.data_as<float>(),
              output.data_as<float>() + output.numel(), 0.0f);
    return output;
}

class TemporalWorkflowDenoiser final : public Denoiser {
public:
    TemporalWorkflowDenoiser(
        CudaBackend& backend,
        TemporalConditionalUNet2DComponentExecutor& executor,
        const Json& graph, const Tensor& assembled, const Tensor& audio)
        : backend_(backend), executor_(executor), graph_(graph) {
        context_ = backend_.slice(assembled, 1, 4, 13);
        Tensor batched = backend_.reshape(
            audio, {1, audio.dim(0), audio.dim(1), audio.dim(2)});
        Tensor unconditional = backend_.copy_to_device(
            zeros_like_host(audio).reshape(
                {1, audio.dim(0), audio.dim(1), audio.dim(2)}),
            DType::F32);
        audio_ = backend_.concat({unconditional, batched}, 0);
    }

    std::vector<Tensor> evaluate(const Tensor& latent,
                                 const Tensor& timestep) override {
        Tensor input = backend_.concat({latent, context_}, 1);
        Tensor pair = backend_.concat({input, input}, 0);
        const TemporalConditionalUNetResult result =
            executor_.execute(graph_, pair, timestep, audio_);
        return backend_.split(result.epsilon, {1, 1}, 0);
    }

private:
    CudaBackend& backend_;
    TemporalConditionalUNet2DComponentExecutor& executor_;
    const Json& graph_;
    Tensor context_;
    Tensor audio_;
};

SamplingProgram sampling_program(const uint64_t seed, const int64_t frames) {
    SamplingProgram result;
    result.latent_shape = {1, 4, frames, 64, 64};
    result.seed = seed;
    result.steps = 20;
    result.guidance_mode = GuidanceMode::CFG;
    result.guidance_coefficients = {0.0f, 1.5f};
    result.contract.emplace(
        PredictionContract{PredictionSemantic::Epsilon},
        SolverContract{SolverSemantic::AffineFirstOrder, 1},
        make_scaled_linear_ddim_schedule(
            1000, 0.00085f, 0.012f, 20, 1, false));
    return result;
}

PrecisionPolicy admitted_precision_policy(
        const ResolvedRunnableModel& model, const Json& execution) {
    const std::string execution_dtype = execution.at("execution_dtype").string();
    if (execution_dtype == "float32") return PrecisionPolicy::fp32();
    if (execution_dtype != "bfloat16")
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync diffusion execution dtype is unsupported");
    const Json* artifact = execution.find("precision_policy_artifact");
    if (artifact == nullptr || !artifact->is_string())
        fail(ModelPackageErrorCode::PackageInvalid,
             "BF16 execution requires a declared generic PrecisionPolicy");
    const PrecisionPolicy policy = PrecisionPolicy::from_json(
        Json::parse(read_verified_text(model, artifact->string())));
    const auto require_dtype = [](const bool condition, const char* message) {
        if (!condition)
            fail(ModelPackageErrorCode::PackageInvalid, message);
    };
    require_dtype(
        policy.requested_dtype() == DType::BF16 &&
        policy.operation_compute_dtype(PrecisionOperation::Linear) == DType::BF16 &&
        policy.operation_compute_dtype(PrecisionOperation::Convolution) == DType::BF16 &&
        policy.operation_compute_dtype(PrecisionOperation::AttentionProjection) == DType::BF16 &&
        policy.operation_compute_dtype(PrecisionOperation::AttentionQK) == DType::BF16 &&
        policy.operation_compute_dtype(PrecisionOperation::AttentionPV) == DType::BF16,
        "BF16 heavy-consumer PrecisionPolicy contract mismatch");
    require_dtype(
        policy.persistent_state_dtype(PrecisionSemantic::ResidualState) == DType::F32 &&
        policy.persistent_state_dtype(PrecisionSemantic::SamplingState) == DType::F32 &&
        policy.boundary_dtype(PrecisionSemantic::DenoiserOutput) == DType::F32 &&
        policy.boundary_dtype(PrecisionSemantic::VaeInput) == DType::F32 &&
        policy.reduction_dtype(PrecisionSemantic::NormalizationStatistic) == DType::F32 &&
        policy.reduction_dtype(PrecisionSemantic::AttentionStatistic) == DType::F32 &&
        policy.operation_compute_dtype(PrecisionOperation::Guidance) == DType::F32 &&
        policy.operation_compute_dtype(PrecisionOperation::Scheduler) == DType::F32 &&
        policy.operation_compute_dtype(PrecisionOperation::Vae) == DType::F32,
        "BF16 heavy-consumer FP32-island contract mismatch");
    return policy;
}

PrecisionPolicy validate_frozen_configs(
        const ResolvedRunnableModel& model,
        const Json& workflow, const Json& execution) {
    if (workflow.at("family").string() !=
            kLipSyncDiffusionWorkflowFamily ||
        execution.at("product_family").string() != "lip_sync" ||
        execution.at("workflow_identity").string() !=
            kLipSyncDiffusionWorkflowFamily)
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync diffusion workflow/execution identity mismatch");
    const PrecisionPolicy precision_policy =
        admitted_precision_policy(model, execution);
    const Json& profile = workflow.at("profile");
    const Json& sampling = profile.at("sampling");
    const Json& rng = profile.at("rng");
    if (profile.at("fps").integer() != 25 ||
        profile.at("audio_sample_rate").integer() != 16000 ||
        profile.at("temporal_chunk").integer() != 16 ||
        profile.at("temporal_overlap").integer() != 0 ||
        sampling.at("steps").integer() != 20 ||
        std::abs(sampling.at("guidance").number() - 1.5) > 1.0e-12 ||
        std::abs(sampling.at("eta").number()) > 1.0e-12 ||
        sampling.at("prediction").string() != "epsilon" ||
        rng.at("algorithm").string() != "pytorch_compat.v1" ||
        rng.at("posterior_semantic_index").string() !=
            "source_frame_identity" ||
        rng.at("initial_noise_semantic_index").string() !=
            "temporal_chunk_start" ||
        !rng.at("repeat_initial_noise_across_chunk_frames").boolean())
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync diffusion frozen workflow profile drift");
    const Json scheduler = Json::parse(
        read_verified_text(model, "scheduler-config"));
    if (scheduler.at("_class_name").string() != "DDIMScheduler" ||
        scheduler.at("num_train_timesteps").integer() != 1000 ||
        scheduler.at("beta_schedule").string() != "scaled_linear" ||
        std::abs(scheduler.at("beta_start").number() - 0.00085) > 1.0e-12 ||
        std::abs(scheduler.at("beta_end").number() - 0.012) > 1.0e-12 ||
        scheduler.at("steps_offset").integer() != 1 ||
        scheduler.at("set_alpha_to_one").boolean() ||
        scheduler.at("clip_sample").boolean())
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync diffusion scheduler configuration drift");
    (void)resolved_artifact(model, "mel-filters");
    (void)resolved_artifact(model, "unet-config");
    (void)resolved_artifact(model, "fixed-mask");
    return precision_policy;
}

}  // namespace

RunResult run_lip_sync_diffusion_product(
        const ResolvedRunnableModel& model, const RunOptions& options,
        RunEventSink progress) {
    const auto process_started = Clock::now();
    if (model.manifest.product.family != "lip_sync" ||
        model.manifest.product.workflow_identity !=
            kLipSyncDiffusionWorkflowFamily ||
        (model.manifest.product.status != "technical_private" &&
         model.manifest.product.status != "public_supported"))
        fail(ModelPackageErrorCode::PackageInvalid,
             "installed package is not a qualified lip-sync diffusion product");
    if (!options.prompt.empty())
        fail(ModelPackageErrorCode::InvalidInput,
             "lip-sync products do not accept --prompt");
    if (options.video.empty())
        fail(ModelPackageErrorCode::InvalidInput,
             "lip-sync product requires --video");
    if (options.audio.empty())
        fail(ModelPackageErrorCode::InvalidInput,
             "lip-sync product requires --audio");
    const ProductInputSchema* product_schema =
        model.manifest.product.input_schema
            ? &*model.manifest.product.input_schema : nullptr;
#ifdef _WIN32
    const std::filesystem::path output = windows_process::wide(resolve_product_output(
        product_schema, windows_process::utf8(options.output)));
#else
    const std::string output = resolve_product_output(
        product_schema, options.output.string());
#endif
    preflight_output_destination(output, options.overwrite);
    fs::path helper = options.encoder_path;
    if (helper.empty()) {
#ifdef _WIN32
        if (const wchar_t* configured = _wgetenv(L"VRHINO_FFMPEG");
            configured != nullptr && *configured != L'\0')
#else
        if (const char* configured = std::getenv("VRHINO_FFMPEG");
            configured != nullptr && *configured != '\0')
#endif
            helper = configured;
        else
            helper = default_media_encoder_path();
    }
    preflight_media_encoder(helper);

    const auto validation_started = Clock::now();
    const Json workflow = Json::parse(read_verified_text(
        model, model.manifest.product.workflow_artifact_id));
    const Json execution = Json::parse(read_verified_text(
        model, model.manifest.product.execution_artifact_id));
    const PrecisionPolicy precision_policy =
        validate_frozen_configs(model, workflow, execution);
    if (product_schema != nullptr)
        validate_product_execution_consistency(
            model.manifest.product.family,
            model.manifest.product.workflow_identity,
            *product_schema,
            *model.manifest.product.frozen_profile,
            execution, &workflow);
    const uint64_t seed = resolve_product_seed(
        product_schema, options.seed,
        static_cast<uint64_t>(execution.at("default_seed").integer()));

    const ResolvedArtifact& audio_artifact = component_artifact(model, "audio_encoder");
    const ResolvedArtifact& vae_artifact = component_artifact(model, "image_autoencoder");
    const ResolvedArtifact& unet_artifact = component_artifact(model, "neural_edit");
    const ResolvedArtifact& detector_artifact = component_artifact(model, "face_detector");
    const ResolvedArtifact& pose_artifact = component_artifact(model, "pose_estimator");
    if (component_role(model, "neural_edit").kind !=
            "temporal_conditional_unet_2d" ||
        component_role(model, "face_detector").kind !=
            "vision_detector_dense_anchors")
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync diffusion component/profile contract mismatch");

    if (progress)
        progress(RunEvent::stage_changed(
            RunStage::Validation, "Validating five Native components"));
    auto load_component = [&](const ResolvedArtifact& artifact,
                              const std::string& role) {
        return load_verified_vrm_component(
            artifact.path, integrity_contract(identity_contract(workflow, role)));
    };
    auto whisper_storage = load_component(audio_artifact, "audio_encoder");
    auto vae_storage = load_component(vae_artifact, "image_autoencoder");
    auto unet_storage = load_component(unet_artifact, "neural_edit");
    auto detector_storage = load_component(detector_artifact, "face_detector");
    auto pose_storage = load_component(pose_artifact, "pose_estimator");
    VrmModel& whisper = *whisper_storage;
    VrmModel& vae = *vae_storage;
    VrmModel& unet = *unet_storage;
    VrmModel& detector_model = *detector_storage;
    VrmModel& pose_model = *pose_storage;

    RunResult result;
    result.identity = model.manifest.identity;
    result.preset = options.preset.empty()
        ? model.manifest.default_preset : options.preset;
    result.seed = seed;
    result.package_validation_seconds = elapsed_seconds(validation_started);
    result.preflight = preflight_runnable_model(model, result.preset);
    if (result.preflight.status == PreflightStatus::UnsupportedGpu)
        fail(ModelPackageErrorCode::UnsupportedGpu, result.preflight.message);
    if (result.preflight.status == PreflightStatus::DriverIncompatible)
        fail(ModelPackageErrorCode::DriverIncompatible, result.preflight.message);

    try {
        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::MediaInput, "Decoding media"));
        const auto media_started = Clock::now();
        const VideoInput video = decode_video_rgb24(
            helper, options.video, 0, options.cancellation_requested);
        const AudioInput audio = decode_audio_mono_f32_16khz(
            helper, options.audio, 0, options.cancellation_requested);
        result.media_input_seconds = elapsed_seconds(media_started);
        if (video.frames.empty() || video.fps_denominator != 1 ||
            video.fps_numerator != 25 || audio.samples.empty())
            fail(ModelPackageErrorCode::InvalidInput,
                 "lip-sync diffusion media must be RGB at 25 FPS with audio");
        const int64_t output_frames = lip_sync_output_frame_count(
            static_cast<int64_t>(audio.samples.size()), audio.sample_rate, 25);
        if (output_frames <= 0)
            fail(ModelPackageErrorCode::InvalidInput,
                 "driving audio is too short to produce an output frame");
        const std::vector<int64_t> cycle = plan_lip_sync_diffusion_source_frames(
            static_cast<int64_t>(video.frames.size()), output_frames);
        const SourceFrameDemandPlan source_demand =
            plan_source_frame_prefix_demand(
                static_cast<int64_t>(video.frames.size()), cycle);

        std::vector<RgbFrame> aligned_sources;
        std::vector<SimilarityAffine> transforms;
        aligned_sources.reserve(static_cast<size_t>(
            source_demand.required_prefix_frame_count));
        transforms.reserve(static_cast<size_t>(
            source_demand.required_prefix_frame_count));
        Tensor windows_host;
        size_t workflow_peak_device_bytes = 0;
        {
            // Face/audio conditioning is a completed component phase. Release
            // its device weights before the repeated diffusion chunks so
            // chunk count cannot grow persistent device residency.
            CudaBackend backend;
            backend.set_execution_dtype(DType::F32);
            backend.enable_weight_cache(true);
            AudioEncoderComponentExecutor audio_encoder(
                backend, WeightMap(whisper.bindings(whisper.graph())));
            VisionDetectorComponentExecutor detector(
                backend, WeightMap(detector_model.bindings(detector_model.graph())));
            PoseEstimator2DComponentExecutor pose(
                backend, WeightMap(pose_model.bindings(pose_model.graph())));

            if (progress)
                progress(RunEvent::stage_changed(
                    RunStage::Analysis,
                    "Analyzing and aligning " +
                        std::to_string(
                            source_demand.required_prefix_frame_count) +
                        " demanded source frames (" +
                        std::to_string(source_demand.source_frame_count) +
                        " decoded)"));
            const auto face_started = Clock::now();
            LipSyncDiffusionAlignmentState alignment_state;
            for (int64_t source_index = 0;
                 source_index < source_demand.required_prefix_frame_count;
                 ++source_index) {
                const RgbFrame& frame =
                    video.frames[static_cast<size_t>(source_index)];
                check_cancelled(options, "face analysis");
                const BlazeFacePreprocessResult prepared_detector =
                    preprocess_dense_face_detector_rgb_u8(
                        frame.pixels.data(), frame.height, frame.width);
                Tensor detector_input = Tensor::host({1, 3, 128, 128}, DType::F32);
                std::copy(prepared_detector.normalized_nchw.begin(),
                          prepared_detector.normalized_nchw.end(),
                          detector_input.data_as<float>());
                const DenseVisionDetectorResult dense = detector.execute_dense(
                    detector_model.graph(), detector_input);
                const DenseFaceDetectionStages decoded = decode_dense_face_detector(
                    {tensor_values(backend, dense.regressors),
                     tensor_values(backend, dense.classification_logits)},
                    prepared_detector);
                if (!decoded.selected)
                    fail(ModelPackageErrorCode::RuntimeError,
                         "face analysis found no usable source face");
                const std::vector<uint8_t> bgr = rgb_to_bgr(frame);
                const PosePreprocessResult pose_input =
                    preprocess_topdown_pose_bgr_u8(
                        bgr.data(), frame.height, frame.width);
                const auto original = pose.execute(
                    pose_model.graph(), pose_input.normalized_nchw);
                const auto mirrored = pose.execute(
                    pose_model.graph(), horizontal_flip(pose_input.normalized_nchw));
                const auto combined = combine_simcc_flip_tta(
                    backend.copy_to_host(original.simcc_x),
                    backend.copy_to_host(original.simcc_y),
                    backend.copy_to_host(mirrored.simcc_x),
                    backend.copy_to_host(mirrored.simcc_y), flip_indices());
                const KeypointSet keypoints = decode_simcc_keypoints(
                    combined.first, combined.second, pose_input.transform);
                const auto aligned = align_lip_sync_diffusion_face(
                    keypoints, *decoded.selected, alignment_state);
                if (!aligned)
                    fail(ModelPackageErrorCode::RuntimeError,
                         "DWPose alignment rejected unusable source geometry");
                transforms.push_back(aligned->smoothed);
                aligned_sources.push_back(
                    warp_lip_sync_diffusion_face(frame, aligned->smoothed));
                if (progress)
                    progress(RunEvent::progress(
                        RunStage::Analysis,
                        static_cast<uint64_t>(source_index + 1),
                        static_cast<uint64_t>(
                            source_demand.required_prefix_frame_count),
                        RunProgressUnit::Frame,
                        "Analyzing source frame " +
                            std::to_string(source_index + 1) + "/" +
                            std::to_string(
                                source_demand.required_prefix_frame_count)));
            }
            backend.synchronize();
            result.face_analysis_seconds = elapsed_seconds(face_started);

            if (progress)
                progress(RunEvent::stage_changed(
                    RunStage::Conditioning, "Encoding driving audio"));
            const auto audio_started = Clock::now();
            Tensor waveform = Tensor::host(
                {static_cast<int64_t>(audio.samples.size())}, DType::F32);
            std::copy(audio.samples.begin(), audio.samples.end(),
                      waveform.data_as<float>());
            const Json preprocessor = Json::parse(
                read_verified_text(model, "whisper-preprocessor"));
            const Tensor mel = whisper_log_mel_80_variable_audio(
                waveform, preprocessor);
            const auto encoded_audio = audio_encoder.execute(whisper.graph(), mel);
            std::vector<Tensor> states;
            for (const char* name : {"frontend_hidden", "encoder_block_1",
                                     "encoder_block_2", "encoder_block_3",
                                     "encoder_block_4"})
                states.push_back(encoded_audio.outputs.at(name));
            const Tensor stacked = stack_audio_encoder_states(backend, states);
            const Tensor windows = assemble_lip_sync_diffusion_audio_windows(
                backend, stacked, static_cast<int64_t>(audio.samples.size()),
                output_frames);
            windows_host = backend.copy_to_host(windows);
            backend.synchronize();
            result.conditioning_seconds = elapsed_seconds(audio_started);
            workflow_peak_device_bytes = backend.peak_device_bytes();
        }

        const VideoInput mask_media = decode_video_rgb24(
            helper, resolved_artifact(model, "fixed-mask").path, 1,
            options.cancellation_requested);
        if (mask_media.frames.size() != 1)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "fixed workflow mask failed bounded decode");
        const RgbFrame fixed_mask = prepare_lip_sync_diffusion_fixed_mask(
            mask_media.frames.front());
        const std::vector<TemporalFrameChunk> chunks =
            plan_temporal_frame_chunks(output_frames, 16);
        std::vector<RgbFrame> final_frames;
        final_frames.reserve(static_cast<size_t>(output_frames));

        // Component execution contexts are bounded to this Product invocation
        // and reused across every declared temporal chunk. The component
        // phases are kept sequential so their independent high-water device
        // working sets do not overlap. Only exact FP32 latent values cross a
        // phase boundary through host storage.
        struct PreparedChunk {
            TemporalFrameChunk plan;
            LipSyncDiffusionVaeInput vae_input;
            Tensor masked_latent;
            Tensor reference_latent;
        };
        std::vector<PreparedChunk> prepared_chunks;
        prepared_chunks.reserve(chunks.size());
        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::SourcePreparation, "Preparing source latents"));
        {
            CudaBackend backend;
            backend.set_execution_dtype(DType::F32);
            backend.enable_weight_cache(true);
            AutoencoderKLComponentExecutor autoencoder(
                backend, WeightMap(vae.bindings(vae.graph())));
            size_t chunk_index = 0;
            for (const TemporalFrameChunk& chunk : chunks) {
                check_cancelled(options, "source latent preparation");
                std::vector<RgbFrame> chunk_crops;
                chunk_crops.reserve(static_cast<size_t>(chunk.frames));
                for (int64_t local = 0; local < chunk.frames; ++local)
                    chunk_crops.push_back(aligned_sources[static_cast<size_t>(
                        cycle[static_cast<size_t>(chunk.start + local)])]);
                const auto source_started = Clock::now();
                LipSyncDiffusionVaeInput prepared =
                    prepare_lip_sync_diffusion_vae_input(chunk_crops, fixed_mask);
                const AutoencoderKLEncoderResult masked = autoencoder.encode(
                    vae.graph(), prepared.masked);
                const AutoencoderKLEncoderResult reference = autoencoder.encode(
                    vae.graph(), prepared.reference);
                std::vector<Tensor> masked_latents;
                std::vector<Tensor> reference_latents;
                masked_latents.reserve(static_cast<size_t>(chunk.frames));
                reference_latents.reserve(static_cast<size_t>(chunk.frames));
                for (int64_t local = 0; local < chunk.frames; ++local) {
                    const int64_t output_index = chunk.start + local;
                    const int64_t source_index =
                        cycle[static_cast<size_t>(output_index)];
                    RngState masked_rng = lip_sync_diffusion_rng(
                        seed, source_index,
                        LipSyncDiffusionRngBranch::MaskedSource);
                    RngState reference_rng = lip_sync_diffusion_rng(
                        seed, source_index,
                        LipSyncDiffusionRngBranch::ReferenceSource);
                    const AutoencoderKLPosteriorSample masked_sample =
                        autoencoder.sample(
                            vae.graph(),
                            backend.slice(
                                masked.posterior_mean, 0, local, local + 1),
                            backend.slice(
                                masked.posterior_logvar, 0, local, local + 1),
                            masked_rng);
                    const AutoencoderKLPosteriorSample reference_sample =
                        autoencoder.sample(
                            vae.graph(),
                            backend.slice(
                                reference.posterior_mean, 0, local, local + 1),
                            backend.slice(
                                reference.posterior_logvar, 0, local, local + 1),
                            reference_rng);
                    masked_latents.push_back(masked_sample.scaled_latent);
                    reference_latents.push_back(reference_sample.scaled_latent);
                }
                const Tensor masked_scaled = chunk.frames == 1
                    ? masked_latents.front()
                    : backend.concat(masked_latents, 0);
                const Tensor reference_scaled = chunk.frames == 1
                    ? reference_latents.front()
                    : backend.concat(reference_latents, 0);
                Tensor masked_host = backend.copy_to_host(masked_scaled);
                Tensor reference_host = backend.copy_to_host(reference_scaled);
                backend.synchronize();
                result.source_preparation_seconds +=
                    elapsed_seconds(source_started);

                prepared.masked = {};
                prepared_chunks.push_back({
                    chunk, std::move(prepared), std::move(masked_host),
                    std::move(reference_host)});
                ++chunk_index;
                if (progress)
                    progress(RunEvent::progress(
                        RunStage::SourcePreparation,
                        static_cast<uint64_t>(chunk_index),
                        static_cast<uint64_t>(chunks.size()),
                        RunProgressUnit::Chunk,
                        "Preparing source chunk " +
                            std::to_string(chunk_index) + "/" +
                            std::to_string(chunks.size())));
            }
            workflow_peak_device_bytes = std::max(
                workflow_peak_device_bytes, backend.peak_device_bytes());
        }

        std::vector<Tensor> sampled_latents;
        sampled_latents.reserve(chunks.size());
        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::Sampling, "Sampling temporal chunks"));
        {
            // Immutable parameters, dtype-specific device weights, shape-keyed
            // plans, handles, allocator state, and reusable workspaces survive
            // all chunk boundaries in this single generic execution context.
            CudaBackend backend;
            backend.set_execution_dtype(DType::F32);
            backend.enable_weight_cache(true);
            TemporalConditionalUNet2DComponentExecutor temporal_unet(
                backend, WeightMap(unet.bindings(unet.graph())),
                precision_policy);

            size_t chunk_index = 0;
            for (const PreparedChunk& prepared_chunk : prepared_chunks) {
                const TemporalFrameChunk& chunk = prepared_chunk.plan;
                check_cancelled(options, "temporal chunk");
                backend.set_execution_dtype(DType::F32);
                const Tensor masked_scaled = backend.copy_to_device(
                    prepared_chunk.masked_latent, DType::F32);
                const Tensor reference_scaled = backend.copy_to_device(
                    prepared_chunk.reference_latent, DType::F32);

                RngState noise_rng = lip_sync_diffusion_rng(
                    seed, chunk.start,
                    LipSyncDiffusionRngBranch::InitialDiffusionNoise);
                const Tensor base_noise = backend.rng_normal(
                    noise_rng, {1, 4, 1, 64, 64}, DType::F32);
                std::vector<Tensor> repeated_noise(
                    static_cast<size_t>(chunk.frames), base_noise);
                const Tensor initial_noise = chunk.frames == 1
                    ? base_noise
                    : backend.concat(repeated_noise, 2);
                const Tensor assembled =
                    assemble_lip_sync_diffusion_unet_input(
                        backend, initial_noise,
                        prepared_chunk.vae_input.mask, masked_scaled,
                        reference_scaled);
                const Tensor chunk_audio = backend.copy_to_device(
                    slice_host(windows_host, chunk.start,
                               chunk.start + chunk.frames),
                    DType::F32);
                TemporalWorkflowDenoiser denoiser(
                    backend, temporal_unet, unet.graph(), assembled,
                    chunk_audio);
                backend.set_execution_dtype(
                    precision_policy.requested_dtype());
                SamplingRuntime runtime(backend, precision_policy);
                runtime.set_cancellation_requested(
                    options.cancellation_requested);
                const auto sampling_started = Clock::now();
                const SamplingResult sampled = runtime.run_with_initial_state(
                    denoiser,
                    sampling_program(noise_rng.seed, chunk.frames),
                    initial_noise);
                backend.synchronize();
                result.sampling_seconds +=
                    elapsed_seconds(sampling_started);
                Tensor sampled_host =
                    backend.copy_to_host(sampled.final_latent);
                backend.synchronize();
                sampled_latents.push_back(std::move(sampled_host));
                workflow_peak_device_bytes = std::max(
                    workflow_peak_device_bytes,
                    backend.peak_device_bytes());
                ++chunk_index;
                if (progress)
                    progress(RunEvent::progress(
                        RunStage::Sampling,
                        static_cast<uint64_t>(chunk_index),
                        static_cast<uint64_t>(prepared_chunks.size()),
                        RunProgressUnit::Chunk,
                        "Sampling temporal chunk " +
                            std::to_string(chunk_index) + "/" +
                            std::to_string(prepared_chunks.size())));
            }
        }

        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::Decoding,
                "Decoding and compositing temporal chunks"));
        {
            CudaBackend backend;
            backend.set_execution_dtype(
                precision_policy.operation_compute_dtype(
                    PrecisionOperation::Vae));
            backend.enable_weight_cache(true);
            AutoencoderKLComponentExecutor autoencoder(
                backend, WeightMap(vae.bindings(vae.graph())));
            for (size_t chunk_index = 0;
                 chunk_index < prepared_chunks.size(); ++chunk_index) {
                check_cancelled(options, "source latent decode");
                const PreparedChunk& prepared_chunk =
                    prepared_chunks[chunk_index];
                const TemporalFrameChunk& chunk = prepared_chunk.plan;
                const auto decode_started = Clock::now();
                const Tensor sampled_device = backend.copy_to_device(
                    sampled_latents[chunk_index], DType::F32);
                const AutoencoderKLDecoderResult decoded = autoencoder.decode(
                    vae.graph(), temporal_latents_to_frame_batch(
                        backend, sampled_device));
                const Tensor reconstructed = reconstruct_lip_sync_diffusion_crop(
                    backend, decoded.decoded,
                    prepared_chunk.vae_input.reference,
                    prepared_chunk.vae_input.mask);
                const Tensor reconstructed_host =
                    backend.copy_to_host(reconstructed);
                backend.synchronize();
                result.decode_seconds += elapsed_seconds(decode_started);

                const auto composite_started = Clock::now();
                for (int64_t local = 0; local < chunk.frames; ++local) {
                    Tensor face = slice_host(
                        reconstructed_host, local, local + 1);
                    face = face.reshape({3, 512, 512});
                    const int64_t output_index = chunk.start + local;
                    const int64_t source_index = cycle[
                        static_cast<size_t>(output_index)];
                    final_frames.push_back(
                        composite_lip_sync_diffusion_face(
                            video.frames[static_cast<size_t>(source_index)],
                            face,
                            transforms[static_cast<size_t>(source_index)]));
                }
                result.composite_seconds +=
                    elapsed_seconds(composite_started);
                if (progress)
                    progress(RunEvent::progress(
                        RunStage::Composite,
                        static_cast<uint64_t>(chunk_index + 1),
                        static_cast<uint64_t>(prepared_chunks.size()),
                        RunProgressUnit::Chunk,
                        "Compositing temporal chunk " +
                            std::to_string(chunk_index + 1) + "/" +
                            std::to_string(prepared_chunks.size())));
            }
            workflow_peak_device_bytes = std::max(
                workflow_peak_device_bytes, backend.peak_device_bytes());
        }

        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::Encoding, "Encoding MP4"));
        const auto encode_started = Clock::now();
        const MediaEncodeResult encoded = encode_mux_mp4_atomic(
            helper, final_frames, 25, options.audio, output,
            options.cancellation_requested, options.overwrite, {16000, 1, 18});
        result.encoding_seconds = elapsed_seconds(encode_started);
        result.output_bytes = encoded.bytes;
        result.width = video.width;
        result.height = video.height;
        result.frames = output_frames;
        result.fps = 25;
        result.peak_device_bytes = workflow_peak_device_bytes;
        result.total_seconds = elapsed_seconds(process_started);
        if (progress)
            progress(RunEvent::stage_changed(RunStage::Finalizing, "Done"));
        return result;
    } catch (const ModelPackageError&) {
        throw;
    } catch (const std::exception& error) {
        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "lip-sync diffusion workflow cancelled");
        fail(ModelPackageErrorCode::RuntimeError, error.what());
    }
}

}  // namespace vrhino::product
