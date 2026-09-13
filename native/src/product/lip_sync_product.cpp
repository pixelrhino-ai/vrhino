#include "vrhino/product/run.h"
#ifdef _WIN32
#include "vrhino/product/windows_process.h"
#endif

#include <algorithm>
#include <array>
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
#include "vrhino/conditional_unet_2d.h"
#include "vrhino/face_mask.h"
#include "vrhino/face_roi.h"
#include "vrhino/lip_sync_workflow.h"
#include "vrhino/pose_estimator.h"
#include "vrhino/pose_geometry.h"
#include "vrhino/semantic_segmenter.h"
#include "vrhino/vision_detector.h"
#include "vrhino/product/vrm_verification.h"

namespace vrhino::product {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

VrmComponentIntegrityContract integrity_contract(
        const ComponentIdentityContract& contract) {
    return {contract.semantic_name, contract.architecture,
            contract.bytes, contract.sha256};
}

[[noreturn]] void fail(const ModelPackageErrorCode code,
                       const std::string& message) {
    throw ModelPackageError(code, message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "required product configuration is missing");
    return {std::istreambuf_iterator<char>(input), {}};
}

std::string read_verified_text(const ResolvedArtifact& artifact) {
    if (sha256_file(artifact.path) != artifact.declaration.sha256)
        fail(ModelPackageErrorCode::ChecksumMismatch,
             "required product configuration failed integrity validation: " +
                 artifact.declaration.id);
    return read_text(artifact.path);
}

const ResolvedArtifact& resolved_artifact(const ResolvedRunnableModel& model,
                                          const std::string& id) {
    const auto found = model.artifacts.find(id);
    if (found == model.artifacts.end())
        fail(ModelPackageErrorCode::ArtifactMissing,
             "required package artifact is missing: " + id);
    return found->second;
}

const ComponentDeclaration& component_role(const ResolvedRunnableModel& model,
                                            const std::string& role) {
    const auto found = std::find_if(
        model.manifest.components.begin(), model.manifest.components.end(),
        [&](const ComponentDeclaration& value) { return value.role == role; });
    if (found == model.manifest.components.end() || found->artifact_ids.size() != 1)
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync package component contract mismatch: " + role);
    return *found;
}

const ResolvedArtifact& component_artifact(const ResolvedRunnableModel& model,
                                           const std::string& role) {
    return resolved_artifact(
        model, component_role(model, role).artifact_ids.front());
}

std::vector<uint8_t> rgb_to_bgr(const RgbFrame& frame) {
    std::vector<uint8_t> output = frame.pixels;
    for (size_t index = 0; index < output.size(); index += 3)
        std::swap(output[index], output[index + 2]);
    return output;
}

Tensor detector_input(const RgbFrame& frame) {
    const std::vector<float> values = preprocess_detector_rgb_u8(
        frame.pixels.data(), frame.height, frame.width);
    Tensor output = Tensor::host({1, 3, frame.height, frame.width}, DType::F32);
    std::copy(values.begin(), values.end(), output.data_as<float>());
    return output;
}

std::vector<float> tensor_values(const Tensor& tensor) {
    return {tensor.data_as<float>(), tensor.data_as<float>() + tensor.numel()};
}

std::vector<DetectorScaleHost> detector_scales(
        CudaBackend& backend, const VisionDetectorResult& result) {
    std::vector<DetectorScaleHost> scales;
    for (const auto& scale : result.scales) {
        const Tensor confidence = backend.copy_to_host(scale.confidence_logits);
        const Tensor localization = backend.copy_to_host(scale.localization);
        scales.push_back({confidence.dim(0), confidence.dim(2), confidence.dim(3),
                          tensor_values(confidence), tensor_values(localization)});
    }
    return scales;
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
    static const std::vector<int64_t> values = {
        0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,20,21,22,17,18,19,
        39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,49,48,47,
        46,45,44,43,42,41,40,50,51,52,53,58,57,56,55,54,68,67,66,65,
        70,69,62,61,60,59,64,63,77,76,75,74,73,72,71,82,81,80,79,78,
        87,86,85,84,83,90,89,88,112,113,114,115,116,117,118,119,120,
        121,122,123,124,125,126,127,128,129,130,131,132,91,92,93,94,
        95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111};
    return values;
}

void check_cancelled(const RunOptions& options, const std::string& stage) {
    if (options.cancellation_requested && options.cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "lip-sync run cancelled before " + stage);
}

double elapsed_seconds(const Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

struct SourceMaterial {
    RgbFrame frame;
    FaceGeometry geometry;
    Tensor latent;
};

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

ModelPackageErrorCode workflow_error_code(const LipSyncStage stage) {
    switch (stage) {
        case LipSyncStage::MediaInput: return ModelPackageErrorCode::InvalidInput;
        case LipSyncStage::MediaOutput: return ModelPackageErrorCode::VideoEncodingFailed;
        case LipSyncStage::AudioConditioning:
        case LipSyncStage::FaceAnalysis:
        case LipSyncStage::SourcePreparation:
        case LipSyncStage::ComponentExecution:
        case LipSyncStage::Composite: return ModelPackageErrorCode::RuntimeError;
    }
    return ModelPackageErrorCode::RuntimeError;
}

}  // namespace

RunResult run_lip_sync_product(const ResolvedRunnableModel& model,
                               const RunOptions& options,
                               RunEventSink progress) {
    const auto process_started = Clock::now();
    if (model.manifest.product.family != "lip_sync" ||
        model.manifest.product.workflow_identity != "lip_sync_workflow_v1" ||
        (model.manifest.product.status != "technical_private" &&
         model.manifest.product.status != "public_supported"))
        fail(ModelPackageErrorCode::PackageInvalid,
             "installed package is not a qualified lip-sync product");
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
    const Json workflow = Json::parse(read_verified_text(resolved_artifact(
        model, model.manifest.product.workflow_artifact_id)));
    const Json execution = Json::parse(read_verified_text(resolved_artifact(
        model, model.manifest.product.execution_artifact_id)));
    if (workflow.at("family").string() != "lip_sync_workflow_v1" ||
        execution.at("product_family").string() != "lip_sync" ||
        execution.at("workflow_identity").string() != "lip_sync_workflow_v1" ||
        execution.at("execution_dtype").string() != "float32")
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync workflow/execution identity mismatch");
    if (product_schema != nullptr)
        validate_product_execution_consistency(
            model.manifest.product.family,
            model.manifest.product.workflow_identity,
            *product_schema,
            *model.manifest.product.frozen_profile,
            execution, &workflow);
    const Json& profile = workflow.at("profile");
    const std::string mask_mode = profile.at("mask_mode").string();
    const bool successor = mask_mode == "landmark_constrained_v1";
    if (profile.at("fps").integer() != 25 ||
        profile.at("bbox_shift").integer() != 0 ||
        profile.at("lower_margin").integer() != 10 ||
        profile.at("frame_cycle").string() != "forward_then_full_reverse" ||
        profile.at("crop_resize").string() != "lanczos4" ||
        (!successor && mask_mode != "jaw"))
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync frozen workflow profile drift");
    const uint64_t seed = resolve_product_seed(
        product_schema, options.seed,
        static_cast<uint64_t>(execution.at("default_seed").integer()));

    const ComponentDeclaration& detector_declaration =
        component_role(model, "face_detector");
    const ComponentDeclaration& parser_declaration =
        component_role(model, "semantic_segmenter");
    const std::string expected_detector_kind = successor
        ? "vision_detector_dense_anchors" : "vision_detector_multiscale";
    if (detector_declaration.kind != expected_detector_kind ||
        parser_declaration.kind != "semantic_segmenter_2d")
        fail(ModelPackageErrorCode::PackageInvalid,
             "lip-sync workflow component/profile contract mismatch");
    const ResolvedArtifact& audio_artifact = component_artifact(model, "audio_encoder");
    const ResolvedArtifact& vae_artifact = component_artifact(model, "image_autoencoder");
    const ResolvedArtifact& unet_artifact = component_artifact(model, "neural_edit");
    const ResolvedArtifact& detector_artifact = component_artifact(model, "face_detector");
    const ResolvedArtifact& pose_artifact = component_artifact(model, "pose_estimator");
    const ResolvedArtifact& parser_artifact = component_artifact(model, "semantic_segmenter");

    if (progress)
        progress(RunEvent::stage_changed(
            RunStage::Validation, "Validating six Native components"));
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
    auto parser_storage = load_component(parser_artifact, "semantic_segmenter");
    VrmModel& whisper = *whisper_storage;
    VrmModel& vae = *vae_storage;
    VrmModel& unet = *unet_storage;
    VrmModel& detector_model = *detector_storage;
    VrmModel& pose_model = *pose_storage;
    VrmModel& parser_model = *parser_storage;

    RunResult result;
    result.identity = model.manifest.identity;
    result.preset = options.preset.empty() ? model.manifest.default_preset : options.preset;
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
                 "lip-sync media must provide RGB frames at 25 FPS and non-empty audio");
        const int64_t output_frames = lip_sync_output_frame_count(
            static_cast<int64_t>(audio.samples.size()), audio.sample_rate, 25);
        if (output_frames <= 0)
            fail(ModelPackageErrorCode::InvalidInput,
                 "driving audio is too short to produce an output frame");
        check_cancelled(options, "audio conditioning");

        CudaBackend backend;
        backend.set_execution_dtype(DType::F32);
        backend.enable_weight_cache(true);
        AudioEncoderComponentExecutor audio_encoder(
            backend, WeightMap(whisper.bindings(whisper.graph())));
        AutoencoderKLComponentExecutor autoencoder(
            backend, WeightMap(vae.bindings(vae.graph())));
        ConditionalUNet2DComponentExecutor conditional_unet(
            backend, WeightMap(unet.bindings(unet.graph())));
        VisionDetectorComponentExecutor detector(
            backend, WeightMap(detector_model.bindings(detector_model.graph())));
        PoseEstimator2DComponentExecutor pose(
            backend, WeightMap(pose_model.bindings(pose_model.graph())));
        SemanticSegmenter2DComponentExecutor segmenter(
            backend, WeightMap(parser_model.bindings(parser_model.graph())));

        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::Conditioning, "Encoding driving audio"));
        const auto audio_started = Clock::now();
        Tensor waveform = Tensor::host(
            {static_cast<int64_t>(audio.samples.size())}, DType::F32);
        std::copy(audio.samples.begin(), audio.samples.end(), waveform.data_as<float>());
        Tensor mel = whisper_log_mel_80(waveform, Json::parse(read_text(
            resolved_artifact(model, "whisper-preprocessor").path)));
        const auto encoded_audio =
            audio_encoder.execute(whisper.graph(), mel);
        std::vector<Tensor> states;
        for (const char* name : {"frontend_hidden", "encoder_block_1",
                                 "encoder_block_2", "encoder_block_3",
                                 "encoder_block_4"})
            states.push_back(encoded_audio.outputs.at(name));
        Tensor stacked = stack_audio_encoder_states(backend, states);
        Tensor windowed = frame_audio_feature_windows(
            backend, stacked, static_cast<int64_t>(audio.samples.size()),
            output_frames);
        Tensor conditioning = add_sinusoidal_position_encoding(backend, windowed);
        backend.synchronize();
        result.conditioning_seconds = elapsed_seconds(audio_started);

        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::Analysis, "Analyzing source frames"));
        std::vector<SourceMaterial> sources;
        sources.reserve(video.frames.size());
        const auto face_started = Clock::now();
        double source_seconds = 0.0;
        for (size_t index = 0; index < video.frames.size(); ++index) {
            check_cancelled(options, "source-frame analysis");
            const RgbFrame& frame = video.frames[index];
            std::optional<FaceROI> selected_face;
            if (successor) {
                const BlazeFacePreprocessResult preparation =
                    preprocess_dense_face_detector_rgb_u8(
                        frame.pixels.data(), frame.height, frame.width);
                Tensor input = Tensor::host({1, 3, 128, 128}, DType::F32);
                std::copy(preparation.normalized_nchw.begin(),
                          preparation.normalized_nchw.end(), input.data_as<float>());
                const DenseVisionDetectorResult raw =
                    detector.execute_dense(detector_model.graph(), input);
                const Tensor regressors = backend.copy_to_host(raw.regressors);
                const Tensor logits = backend.copy_to_host(raw.classification_logits);
                DenseDetectorHost host{tensor_values(regressors), tensor_values(logits)};
                selected_face = decode_dense_face_detector(host, preparation).selected;
            } else {
                const VisionDetectorResult raw =
                    detector.execute(detector_model.graph(), detector_input(frame));
                selected_face = decode_multiscale_face_detector(
                    detector_scales(backend, raw)).selected;
            }
            if (!selected_face.has_value())
                fail(ModelPackageErrorCode::RuntimeError,
                     "face analysis found no usable source face");
            const std::vector<uint8_t> bgr = rgb_to_bgr(frame);
            const PosePreprocessResult pose_input = preprocess_topdown_pose_bgr_u8(
                bgr.data(), frame.height, frame.width);
            const auto original =
                pose.execute(pose_model.graph(), pose_input.normalized_nchw);
            const auto mirrored = pose.execute(
                pose_model.graph(), horizontal_flip(pose_input.normalized_nchw));
            const auto combined = combine_simcc_flip_tta(
                backend.copy_to_host(original.simcc_x),
                backend.copy_to_host(original.simcc_y),
                backend.copy_to_host(mirrored.simcc_x),
                backend.copy_to_host(mirrored.simcc_y), flip_indices());
            const KeypointSet keypoints = decode_simcc_keypoints(
                combined.first, combined.second, pose_input.transform);
            FaceGeometry geometry = select_musetalk_face_geometry(
                keypoints, selected_face, frame.height, 0, 10);
            if (!geometry.valid)
                fail(ModelPackageErrorCode::RuntimeError,
                     "face geometry is invalid for a source frame");

            const auto source_started = Clock::now();
            const RgbFrame crop = crop_resize_lanczos4(
                frame, geometry.crop_bbox, 256, 256);
            const AutoencoderKLEncoderResult masked = autoencoder.encode(
                vae.graph(), normalize_vae_rgb(crop, true));
            const AutoencoderKLEncoderResult full = autoencoder.encode(
                vae.graph(), normalize_vae_rgb(crop, false));
            RngState masked_rng = lip_sync_component_rng(seed,
                static_cast<int64_t>(index), 0);
            RngState full_rng = lip_sync_component_rng(seed,
                static_cast<int64_t>(index), 1);
            const AutoencoderKLPosteriorSample masked_sample = autoencoder.sample(
                vae.graph(), masked.posterior_mean, masked.posterior_logvar,
                masked_rng);
            const AutoencoderKLPosteriorSample full_sample = autoencoder.sample(
                vae.graph(), full.posterior_mean, full.posterior_logvar,
                full_rng);
            Tensor latent = concatenate_latent_branches(
                backend, masked_sample.scaled_latent, full_sample.scaled_latent);
            source_seconds += elapsed_seconds(source_started);
            sources.push_back({frame, std::move(geometry), std::move(latent)});
            if (progress)
                progress(RunEvent::progress(
                    RunStage::Analysis, static_cast<uint64_t>(index + 1),
                    static_cast<uint64_t>(video.frames.size()),
                    RunProgressUnit::Frame,
                    "Analyzing source frame " + std::to_string(index + 1) +
                        "/" + std::to_string(video.frames.size())));
        }
        result.face_analysis_seconds = elapsed_seconds(face_started) - source_seconds;
        result.source_preparation_seconds = source_seconds;

        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::Sampling, "Generating lip-synced frames"));
        const std::vector<int64_t> cycle = ping_pong_frame_cycle(
            static_cast<int64_t>(sources.size()), output_frames);
        std::vector<RgbFrame> final_frames;
        final_frames.reserve(static_cast<size_t>(output_frames));
        Tensor timestep = Tensor::host({1}, DType::I64);
        timestep.data_as<int64_t>()[0] = 0;
        uint64_t previous_alpha_support = 0;
        double previous_alpha_centroid_x = 0.0;
        double previous_alpha_centroid_y = 0.0;
        bool have_previous_alpha = false;
        for (int64_t index = 0; index < output_frames; ++index) {
            check_cancelled(options, "output-frame execution");
            SourceMaterial& source = sources[static_cast<size_t>(cycle[index])];
            Tensor frame_conditioning = backend.slice(
                conditioning, 0, index, index + 1);
            const auto unet_started = Clock::now();
            const ConditionalUNet2DResult predicted = conditional_unet.execute(
                unet.graph(), source.latent, timestep, frame_conditioning);
            backend.synchronize();
            result.sampling_seconds += elapsed_seconds(unet_started);
            const auto decode_started = Clock::now();
            const AutoencoderKLDecoderResult decoded = autoencoder.decode(
                vae.graph(), predicted.predicted_latent);
            const RgbFrame generated = vae_rgb_tensor_to_frame(
                backend, decoded.rgb_0_1);
            result.decode_seconds += elapsed_seconds(decode_started);

            const auto parser_started = Clock::now();
            const std::vector<uint8_t> bgr = rgb_to_bgr(source.frame);
            const auto legacy_preparation = preprocess_face_parser_bgr_u8(
                bgr.data(), source.frame.height, source.frame.width,
                source.geometry.crop_bbox);
            AlphaMask alpha;
            std::array<int32_t, 4> expanded_box = legacy_preparation.crop_box;
            if (successor) {
                const auto parser_input = preprocess_semantic_segmenter_rgb_u8(
                    legacy_preparation.resized_rgb.data(), 512, 512,
                    {0, 0, 512, 512});
                const auto logits = segmenter.execute(
                    parser_model.graph(), parser_input.normalized_nchw);
                SemanticLabelMap labels = semantic_argmax_first(
                    backend.copy_to_host(logits.logits));
                labels.coordinate_space = "successor_segmenter_pixels";
                alpha = build_landmark_constrained_alpha_mask(
                    labels, source.geometry.facial_landmarks, expanded_box);
            } else {
                const auto logits = segmenter.execute(
                    parser_model.graph(), legacy_preparation.normalized_nchw);
                const SemanticLabelMap labels = semantic_argmax_first(
                    backend.copy_to_host(logits.logits));
                alpha = build_musetalk_jaw_alpha_mask(
                    labels, source.geometry.crop_bbox, expanded_box, 90, 90, 0.5);
            }
            result.parser_mask_seconds += elapsed_seconds(parser_started);

            uint64_t alpha_support = 0;
            long double alpha_x = 0.0, alpha_y = 0.0;
            for (int32_t y = 0; y < alpha.height; ++y)
                for (int32_t x = 0; x < alpha.width; ++x) {
                    const bool selected = alpha.values[
                        static_cast<size_t>(y) * alpha.width + x] > 127;
                    alpha_support += selected;
                    alpha_x += selected ? x : 0;
                    alpha_y += selected ? y : 0;
                }
            if (alpha_support == 0) {
                ++result.invalid_alpha_masks;
            } else {
                const double centroid_x = static_cast<double>(alpha_x / alpha_support);
                const double centroid_y = static_cast<double>(alpha_y / alpha_support);
                if (have_previous_alpha) {
                    result.maximum_alpha_support_change = std::max(
                        result.maximum_alpha_support_change,
                        std::abs(static_cast<double>(alpha_support) -
                                 previous_alpha_support) / previous_alpha_support);
                    result.maximum_alpha_centroid_movement = std::max(
                        result.maximum_alpha_centroid_movement,
                        std::hypot(centroid_x - previous_alpha_centroid_x,
                                   centroid_y - previous_alpha_centroid_y));
                }
                previous_alpha_support = alpha_support;
                previous_alpha_centroid_x = centroid_x;
                previous_alpha_centroid_y = centroid_y;
                have_previous_alpha = true;
            }

            const auto composite_started = Clock::now();
            const RgbFrame placed = place_generated_crop(
                source.frame, generated, source.geometry.crop_bbox);
            final_frames.push_back(composite_rgb_alpha(
                source.frame, placed, alpha, expanded_box));
            result.composite_seconds += elapsed_seconds(composite_started);
            if (progress)
                progress(RunEvent::progress(
                    RunStage::Composite, static_cast<uint64_t>(index + 1),
                    static_cast<uint64_t>(output_frames),
                    RunProgressUnit::Frame,
                    "Generating lip-synced frame " +
                        std::to_string(index + 1) + "/" +
                        std::to_string(output_frames)));
        }
        backend.synchronize();
        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::Encoding, "Encoding MP4"));
        const auto encode_started = Clock::now();
        const MediaEncodeResult encoded = encode_mux_mp4_atomic(
            helper, final_frames, 25, options.audio, output,
            options.cancellation_requested, options.overwrite);
        result.encoding_seconds = elapsed_seconds(encode_started);
        result.output_bytes = encoded.bytes;
        result.width = video.width;
        result.height = video.height;
        result.frames = output_frames;
        result.fps = 25;
        result.peak_device_bytes = backend.peak_device_bytes();
        result.total_seconds = elapsed_seconds(process_started);
        if (progress)
            progress(RunEvent::stage_changed(RunStage::Finalizing, "Done"));
        return result;
    } catch (const LipSyncWorkflowError& error) {
        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled, "lip-sync workflow cancelled");
        fail(workflow_error_code(error.stage()), error.what());
    }
}

}  // namespace vrhino::product
