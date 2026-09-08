#include "vrhino/product/converter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
#include <sys/types.h>
#include <unistd.h>

#include "vrhino/error.h"

namespace vrhino::product {

ImportResult import_wan_model(const std::filesystem::path& source_directory,
                              LocalModelCache& cache,
                              const ImportOptions& options);
ImportResult import_mochi_model(const std::filesystem::path& source_directory,
                                LocalModelCache& cache,
                                const ImportOptions& options);

namespace {

namespace fs = std::filesystem;

constexpr const char* kLtxReference = "vrhino/ltx-video-v0.9.1:1.1.0";
constexpr const char* kLtxCheckpointSha256 =
    "a23200896c5eddf215c7cb9517820c5763a2b054eb62ba86cbce6b871a4577e3";
constexpr uint64_t kLtxCheckpointBytes = 5716863844ULL;
constexpr const char* kExpectedVrmSha256 =
    "267a95330f48dbe2134220e6116c60cddddf54f7548a661e20b18531fb70fa7d";

[[noreturn]] void fail(const ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

Json string_value(const std::string& value) { return Json(Json::Value(value)); }
Json integer_value(const int64_t value) { return Json(Json::Value(value)); }
Json number_value(const double value) { return Json(Json::Value(value)); }
Json boolean_value(const bool value) { return Json(Json::Value(value)); }
Json array_value(std::initializer_list<Json> values) {
    return Json(Json::Value(Json::Array(values)));
}
Json object_value(std::initializer_list<std::pair<const std::string, Json>> values) {
    Json::Object object;
    for (const auto& [key, value] : values) object.emplace(key, value);
    return Json(Json::Value(std::move(object)));
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
        fail(ModelPackageErrorCode::ArtifactMissing,
             "cannot open converter specification: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

std::vector<std::string> split(const std::string& value, const char separator) {
    std::vector<std::string> output;
    size_t begin = 0;
    while (true) {
        const size_t end = value.find(separator, begin);
        output.push_back(value.substr(begin, end == std::string::npos
                                                ? std::string::npos : end - begin));
        if (end == std::string::npos) return output;
        begin = end + 1;
    }
}

std::vector<int64_t> parse_shape(const std::string& text) {
    if (text.size() < 2 || text.front() != '[' || text.back() != ']')
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid converter tensor-map shape: " + text);
    const std::string body = text.substr(1, text.size() - 2);
    if (body.empty()) return {};
    std::vector<int64_t> shape;
    for (const std::string& field : split(body, ',')) {
        size_t consumed = 0;
        int64_t dimension = 0;
        try { dimension = std::stoll(field, &consumed); }
        catch (...) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid converter tensor-map dimension: " + field);
        }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid converter tensor-map dimension: " + field);
        shape.push_back(dimension);
    }
    return shape;
}

DType map_dtype(const std::string& value) {
    if (value == "BF16") return DType::BF16;
    if (value == "F32") return DType::F32;
    if (value == "F16") return DType::F16;
    if (value == "I64") return DType::I64;
    if (value == "I32") return DType::I32;
    if (value == "U8") return DType::U8;
    if (value == "BOOL") return DType::Bool;
    fail(ModelPackageErrorCode::PackageVersionUnsupported,
         "unsupported converter tensor-map dtype: " + value);
}

std::vector<TensorMapping> load_tensor_map(const fs::path& path) {
    std::ifstream input(path);
    if (!input.good())
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing LTX tensor mapping specification: " + path.string());
    std::string line;
    if (!std::getline(input, line) || line !=
            "source_name\tsource_dtype\tsource_shape\ttransformation\tdestination_name\t"
            "destination_dtype\tdestination_shape\tcomponent\trole")
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid LTX tensor mapping header");
    std::vector<TensorMapping> result;
    size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const std::vector<std::string> fields = split(line, '\t');
        if (fields.size() != 9)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid LTX tensor mapping row " + std::to_string(line_number));
        result.push_back(TensorMapping{
            fields[0], map_dtype(fields[1]), parse_shape(fields[2]), fields[3],
            fields[4], map_dtype(fields[5]), parse_shape(fields[6]), fields[7], fields[8],
        });
    }
    if (result.empty())
        fail(ModelPackageErrorCode::PackageInvalid, "empty LTX tensor mapping");
    return result;
}

Json::Object bindings_for(const std::vector<TensorMapping>& mappings,
                          const std::string& prefix) {
    Json::Object bindings;
    for (const TensorMapping& mapping : mappings) {
        if (!mapping.source_name.starts_with(prefix)) continue;
        const std::string logical = mapping.source_name.substr(prefix.size());
        if (!bindings.emplace(logical, string_value(mapping.destination_name)).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate LTX logical tensor binding: " + logical);
    }
    return bindings;
}

Json ltx_model_config() {
    return object_value({
        {"in_channels", integer_value(128)},
        {"out_channels", integer_value(128)},
        {"inner_dim", integer_value(2048)},
        {"caption_channels", integer_value(4096)},
        {"heads", integer_value(32)},
        {"layers", integer_value(28)},
        {"position_theta", number_value(10000.0)},
        {"position_max", array_value({number_value(20.0), number_value(2048.0),
                                       number_value(2048.0)})},
    });
}

Json ltx_latent_contract() {
    return object_value({
        {"channels", integer_value(128)},
        {"layout", string_value("BCTHW")},
        {"sampling_layout", string_value("BLC")},
        {"temporal_decode", string_value("1+8*(F-1)")},
        {"spatial_scale", integer_value(32)},
    });
}

Json ltx_vae(const Json& vae_config) {
    return object_value({
        {"decoder", string_value("ltx_v0_9_1.vae_decoder.v1")},
        {"config", vae_config},
        {"decode_timestep", number_value(0.0)},
        {"output_range", array_value({integer_value(0), integer_value(1)})},
    });
}

Json ltx_metadata(const Json& vae_config) {
    return object_value({
        {"architecture", string_value("ltx_v0_9_1")},
        {"default_dtype", string_value("bfloat16")},
        {"external_assets", object_value({
            {"text_encoder", object_value({{"embedded", boolean_value(false)},
                                             {"required", boolean_value(true)}})},
            {"tokenizer", object_value({{"embedded", boolean_value(false)},
                                          {"required", boolean_value(true)}})},
        })},
        {"latent_contract", ltx_latent_contract()},
        {"model", object_value({
            {"config", ltx_model_config()},
            {"name", string_value("LTX-Video-2B-v0.9.1")},
            {"source_commit", string_value(
                "ea6d5d64894d617ec16b7d70a44e85b8499b3225")},
            {"source_model", string_value("Lightricks/LTX-Video")},
            {"source_revision", string_value(
                "8984fa25007f376c1a299016d0957a37a2f797bb")},
        })},
        {"profile", string_value("dit-flow")},
        {"vae", ltx_vae(vae_config)},
        {"versions", object_value({
            {"architecture", integer_value(1)},
            {"backend_abi", integer_value(1)},
            {"component_schema", integer_value(1)},
            {"graph_schema", integer_value(1)},
            {"profile", integer_value(1)},
            {"sampling_schema", integer_value(1)},
            {"tensor_layout", integer_value(1)},
            {"tensor_primitives", integer_value(26)},
            {"sampling_primitives", integer_value(13)},
        })},
    });
}

Json string_array(const std::set<std::string>& values) {
    Json::Array result;
    for (const std::string& value : values) result.push_back(string_value(value));
    return Json(Json::Value(std::move(result)));
}

Json ltx_graph(const Json& vae_config, const std::vector<TensorMapping>& mappings) {
    const std::set<std::string> tensor_ops = {
        "activation", "add", "attention", "cast", "clamp", "conv3d",
        "layer_norm", "linear", "mul", "pad", "pixel_norm", "pixel_shuffle_nd",
        "reshape", "rms_norm", "rng_normal", "rope_nd", "sinusoidal_embedding",
        "split",
    };
    const std::set<std::string> sampling_ops = {
        "cfg_combine", "euler_update", "linear_schedule", "resolution_time_shift",
        "rng_normal", "schedule_lookup", "state_advance", "tensor_where",
    };
    const std::set<std::string> component_ops = {
        "activation", "clamp", "conv3d", "linear", "pad", "pixel_norm",
        "pixel_shuffle_nd", "rng_normal", "sinusoidal_embedding",
    };
    std::set<std::string> capabilities = {
        "profile.dit_flow", "layout.contiguous", "tensor.quantization.none",
        "dtype.float32", "dtype.bfloat16", "rng.pytorch_compat.v1",
        "component.explicit_state", "component.explicit_rng",
    };
    for (const std::string& operation : tensor_ops)
        capabilities.insert("tensor.op." + operation);
    for (const std::string& operation : sampling_ops)
        capabilities.insert("sampling." + operation);

    Json::Object architecture_config = ltx_model_config().object();
    architecture_config.emplace("rope_coordinate_scale", array_value({
        number_value(0.32), number_value(32.0), number_value(32.0)}));

    Json::Object denoiser_bindings =
        bindings_for(mappings, "model.diffusion_model.");
    Json::Object vae_bindings = bindings_for(mappings, "vae.");
    return object_value({
        {"architecture_graph", object_value({
            {"config", Json(Json::Value(std::move(architecture_config)))},
            {"implementation_id", string_value("dit_flow.ltx_v0_9_1.denoiser.v1")},
            {"required_primitives", string_array(tensor_ops)},
            {"runtime_tensor_bindings", Json(Json::Value(std::move(denoiser_bindings)))},
            {"schema_version", integer_value(1)},
        })},
        {"sampling_program", object_value({
            {"config", object_value({
                {"guidance_scale", number_value(4.5)},
                {"kernel", string_value("euler")},
                {"per_token_sigma", boolean_value(true)},
                {"steps", integer_value(3)},
            })},
            {"implementation_id", string_value("dit_flow.ltx_v0_9_1.sampling.v1")},
            {"required_primitives", string_array(sampling_ops)},
            {"schema_version", integer_value(1)},
        })},
        {"component_graphs", Json(Json::Value(Json::Array({object_value({
            {"config", ltx_vae(vae_config)},
            {"id", string_value("vae.decoder")},
            {"implementation_id", string_value("dit_flow.ltx_v0_9_1.vae_decoder.v1")},
            {"latent_contract", ltx_latent_contract()},
            {"output_contract", object_value({
                {"range", array_value({integer_value(0), integer_value(1)})},
            })},
            {"required_primitives", string_array(component_ops)},
            {"runtime_tensor_bindings", Json(Json::Value(std::move(vae_bindings)))},
            {"schema_version", integer_value(1)},
            {"state_requirements", array_value({string_value("explicit_rng")})},
        })})))},
        {"required_capabilities", string_array(capabilities)},
        {"rng", object_value({
            {"algorithm_id", string_value("pytorch_compat.v1")},
            {"device_semantics", string_value("explicit_per_device")},
            {"normal_distribution", string_value("torch.randn.compat")},
            {"seed_bits", integer_value(64)},
            {"state_bytes_encoding", string_value("torch_generator_state")},
            {"state_version", integer_value(1)},
        })},
        {"schema_version", integer_value(1)},
    });
}

fs::path checked_source(const fs::path& root, const fs::path& relative) {
    std::error_code error;
    const fs::path path = root / relative;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status))
        fail(ModelPackageErrorCode::ArtifactMissing,
             "required local-import source is missing or not a regular file: " +
                 relative.string());
    const fs::path canonical_root = fs::canonical(root, error);
    if (error) fail(ModelPackageErrorCode::PackageInvalid,
                    "cannot canonicalize local-import source root");
    const fs::path canonical_path = fs::canonical(path, error);
    if (error || canonical_path.native().size() < canonical_root.native().size() ||
        canonical_path.native().compare(0, canonical_root.native().size(),
                                        canonical_root.native()) != 0 ||
        (canonical_path.native().size() > canonical_root.native().size() &&
         canonical_path.native()[canonical_root.native().size()] != fs::path::preferred_separator))
        fail(ModelPackageErrorCode::PackageInvalid,
             "local-import source escapes source directory: " + relative.string());
    return canonical_path;
}

fs::path unique_import_path(const fs::path& directory, const std::string& label) {
    static std::atomic<uint64_t> sequence{0};
    const uint64_t tick = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return directory / (label + "-" + std::to_string(getpid()) + "-" +
                        std::to_string(tick) + "-" +
                        std::to_string(sequence.fetch_add(1)));
}

struct ArtifactSource {
    ArtifactDeclaration declaration;
    fs::path source;
};

uint64_t checked_work_add(const uint64_t left, const uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left)
        fail(ModelPackageErrorCode::PackageInvalid,
             "conversion progress work total overflows uint64");
    return left + right;
}

class WorkProgressTracker {
public:
    WorkProgressTracker(std::function<bool()> cancellation_requested,
                        std::function<void(uint64_t, uint64_t)> progress,
                        const uint64_t total, std::string interrupted_message)
        : cancellation_requested_(std::move(cancellation_requested)),
          progress_(std::move(progress)), total_(total),
          interrupted_message_(std::move(interrupted_message)) {}

    void begin() {
        check_cancelled();
        if (progress_) progress_(0, total_);
    }

    WorkProgressCallback work_callback() {
        return [this](const uint64_t bytes) {
            check_cancelled();
            completed_ = checked_work_add(completed_, bytes);
            if (progress_) {
                const uint64_t visible = total_ == 0
                    ? 0
                    : std::min(completed_, total_ - 1);
                progress_(visible, total_);
            }
            check_cancelled();
        };
    }

    void finish() {
        check_cancelled();
        finish_committed();
    }

    void finish_committed() {
        completed_ = total_;
        if (progress_) progress_(total_, total_);
    }

private:
    void check_cancelled() const {
        if (cancellation_requested_ && cancellation_requested_())
            fail(ModelPackageErrorCode::Cancelled, interrupted_message_);
    }

    std::function<bool()> cancellation_requested_;
    std::function<void(uint64_t, uint64_t)> progress_;
    uint64_t total_ = 0;
    uint64_t completed_ = 0;
    std::string interrupted_message_;
};

const ArtifactDeclaration& artifact_by_id(const ModelPackageManifest& manifest,
                                          const std::string& id) {
    for (const ArtifactDeclaration& artifact : manifest.artifacts)
        if (artifact.id == id) return artifact;
    fail(ModelPackageErrorCode::PackageInvalid,
         "converter package template is missing artifact: " + id);
}

void verify_artifact_source(const ArtifactSource& artifact,
                            const WorkProgressCallback& progress) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(artifact.source, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status))
        fail(ModelPackageErrorCode::ArtifactMissing,
             "conversion artifact source is missing: " + artifact.declaration.id);
    if (fs::file_size(artifact.source, error) != artifact.declaration.size || error)
        fail(ModelPackageErrorCode::PackageInvalid,
             "conversion artifact size mismatch: " + artifact.declaration.id);
    if (sha256_file(artifact.source, progress) != artifact.declaration.sha256)
        fail(ModelPackageErrorCode::ChecksumMismatch,
             "conversion artifact SHA256 mismatch: " + artifact.declaration.id);
}

uint64_t admit_artifact(const ArtifactSource& artifact, LocalModelCache& cache,
                        uint64_t& reused, const WorkProgressCallback& progress) {
    const BlobAdmissionResult admission = cache.admit_local_blob(
        artifact.source, artifact.declaration, progress);
    if (!admission.created) {
        reused += artifact.declaration.size;
        return 0;
    }
    return artifact.declaration.size;
}

ImportResult import_ltx(const fs::path& source_directory, LocalModelCache& cache,
                        const ImportOptions& options) {
    const auto conversion_started = std::chrono::steady_clock::now();
    const fs::path specification_root = options.converter_spec_root.empty()
        ? discover_converter_spec_root()
        : fs::canonical(options.converter_spec_root);
    const fs::path specification = specification_root / "ltx_v0_9_1";
    const fs::path manifest_path = options.package_manifest.empty()
        ? specification / "vrhino-model.json"
        : fs::canonical(options.package_manifest);
    const ModelPackageManifest manifest = load_model_package_manifest(manifest_path);
    const std::string expected_reference = options.expected_model_reference.empty()
        ? std::string(kLtxReference) : options.expected_model_reference;
    if (manifest.identity.reference() != expected_reference)
        fail(ModelPackageErrorCode::PackageInvalid,
             "LTX converter package template identity mismatch");

    for (const InstalledPackage& installed : cache.list())
        if (installed.identity.reference() == expected_reference)
            fail(ModelPackageErrorCode::InstallFailed,
                 "immutable package version is already installed: " +
                     expected_reference);

    const fs::path checkpoint = checked_source(
        source_directory, "ltx-video-2b-v0.9.1.safetensors");
    const std::vector<ArtifactSource> static_artifacts = {
        {artifact_by_id(manifest, "t5-index"), checked_source(
            source_directory, "text_encoder/model.safetensors.index.json")},
        {artifact_by_id(manifest, "t5-shard-1"), checked_source(
            source_directory, "text_encoder/model-00001-of-00002.safetensors")},
        {artifact_by_id(manifest, "t5-shard-2"), checked_source(
            source_directory, "text_encoder/model-00002-of-00002.safetensors")},
        {artifact_by_id(manifest, "tokenizer"), checked_source(
            source_directory, "tokenizer/spiece.model")},
        {artifact_by_id(manifest, "conditioning-graph"),
            specification / "conditioning/ltx-t5.json"},
        {artifact_by_id(manifest, "precision-policy"),
            specification / "policy/bf16-operation-contract-v1.json"},
        {artifact_by_id(manifest, "default-profile"),
            specification / "execution/default-preset.json"},
        {artifact_by_id(manifest, "license"),
            checked_source(source_directory,
                "ltx-video-2b-v0.9.1.license.txt")},
    };
    const ArtifactDeclaration& runtime_artifact = artifact_by_id(manifest, "runtime");
    if (runtime_artifact.sha256 != kExpectedVrmSha256)
        fail(ModelPackageErrorCode::PackageInvalid,
             "LTX converter package template runtime SHA256 drifted");
    const bool runtime_present = cache.contains_blob(runtime_artifact, false);
    uint64_t static_bytes = 0;
    for (const ArtifactSource& artifact : static_artifacts)
        static_bytes = checked_work_add(static_bytes, artifact.declaration.size);
    uint64_t conversion_work = checked_work_add(kLtxCheckpointBytes, static_bytes);
    conversion_work = checked_work_add(conversion_work, runtime_artifact.size);
    uint64_t finalization_work = checked_work_add(static_bytes, static_bytes);
    if (!runtime_present)
        finalization_work = checked_work_add(finalization_work, runtime_artifact.size);
    WorkProgressTracker conversion_progress(
        options.cancellation_requested, options.progress, conversion_work,
        "Conversion interrupted. Downloaded source retained.");
    conversion_progress.begin();
    const WorkProgressCallback conversion_work_progress =
        conversion_progress.work_callback();

    std::error_code error;
    if (fs::file_size(checkpoint, error) != kLtxCheckpointBytes || error)
        fail(ModelPackageErrorCode::PackageInvalid,
             "LTX source checkpoint size does not match frozen revision");
    if (sha256_file(checkpoint, conversion_work_progress) != kLtxCheckpointSha256)
        fail(ModelPackageErrorCode::ChecksumMismatch,
             "LTX source checkpoint SHA256 does not match frozen revision");

    SafeTensorReader reader(checkpoint);
    const std::vector<TensorMapping> mappings =
        load_tensor_map(specification / "tensor-mapping.tsv");
    if (mappings.size() != reader.tensors().size())
        fail(ModelPackageErrorCode::PackageVersionUnsupported,
             "LTX tensor count does not match converter specification");
    std::set<std::string> mapped_sources;
    for (const TensorMapping& mapping : mappings) {
        if (!mapped_sources.insert(mapping.source_name).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate LTX source tensor mapping: " + mapping.source_name);
        const auto found = reader.tensors().find(mapping.source_name);
        if (found == reader.tensors().end() || found->second.dtype != mapping.source_dtype ||
            found->second.shape != mapping.source_shape ||
            mapping.source_dtype != mapping.destination_dtype ||
            mapping.source_shape != mapping.destination_shape ||
            mapping.transformation != "identity_bytes")
            fail(ModelPackageErrorCode::PackageVersionUnsupported,
                 "LTX upstream tensor layout drift: " + mapping.source_name);
    }
    if (mapped_sources.size() != reader.tensors().size())
        fail(ModelPackageErrorCode::PackageVersionUnsupported,
             "LTX source contains an unmapped tensor");

    const Json* configuration = reader.metadata().find("config");
    if (configuration == nullptr || !configuration->is_string())
        fail(ModelPackageErrorCode::PackageInvalid,
             "LTX safetensors metadata is missing embedded config");
    const Json checkpoint_config = Json::parse(configuration->string());
    const Json vae_config = checkpoint_config.at("vae");
    const Json metadata = ltx_metadata(vae_config);
    const Json graph = ltx_graph(vae_config, mappings);
    const std::string metadata_serialized = canonical_json(metadata);
    const std::string graph_serialized = canonical_json(graph);
    if (metadata_serialized.size() != 1838 || graph_serialized.size() != 104219)
        fail(ModelPackageErrorCode::PackageInvalid,
             "native LTX declaration size differs from qualified artifact: metadata/graph=" +
                 std::to_string(metadata_serialized.size()) + "/" +
                 std::to_string(graph_serialized.size()) + " expected=1838/104219");

    for (const ArtifactSource& artifact : static_artifacts)
        verify_artifact_source(artifact, conversion_work_progress);

    uint64_t missing_bytes = runtime_present ? 0 : runtime_artifact.size;
    for (const ArtifactSource& artifact : static_artifacts)
        if (!cache.contains_blob(artifact.declaration, false))
            missing_bytes += artifact.declaration.size;
    fs::create_directories(cache.layout().root, error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create local cache root: " + error.message());
    const uint64_t available = options.available_space_override.value_or(
        fs::space(cache.layout().root, error).available);
    if (error)
        fail(ModelPackageErrorCode::CacheError,
             "cannot query local cache disk space: " + error.message());
    require_conversion_disk_space(missing_bytes, available,
                                  64ULL * 1024 * 1024, cache.layout().root);

    const fs::path staging = cache.layout().temporary / "imports";
    fs::create_directories(staging, error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create native converter staging directory: " +
                        error.message());
    const fs::path converted = unique_import_path(staging, "ltx-model.vrm.partial");
    uint64_t copied = 0;
    uint64_t reused = 0;
    VrmWriteResult write_result;
    try {
        if (!runtime_present) {
            write_result = write_vrm_streaming(
                converted, "dit-flow", "ltx_v0_9_1", metadata, graph, reader,
                mappings, options.cancellation_requested, conversion_work_progress);
            if (write_result.file_size != runtime_artifact.size)
                fail(ModelPackageErrorCode::PackageInvalid,
                     "native LTX VRM size differs from qualified artifact: " +
                         std::to_string(write_result.file_size) + " != " +
                         std::to_string(runtime_artifact.size) +
                         " sections metadata/table/graph/data_offset=" +
                         std::to_string(write_result.metadata_bytes) + "/" +
                         std::to_string(write_result.tensor_table_bytes) + "/" +
                         std::to_string(write_result.graph_bytes) + "/" +
                         std::to_string(write_result.data_offset));
        } else {
            if (!cache.contains_blob(runtime_artifact, true,
                                     conversion_work_progress))
                fail(ModelPackageErrorCode::CacheError,
                     "existing Runtime CAS blob is invalid");
            reused += runtime_artifact.size;
        }
        conversion_progress.finish();
        const auto conversion_finished = std::chrono::steady_clock::now();

        WorkProgressTracker finalization_progress(
            options.cancellation_requested, options.finalization_progress,
            finalization_work,
            "Finalization interrupted. Downloaded source retained.");
        finalization_progress.begin();
        const WorkProgressCallback finalization_work_progress =
            finalization_progress.work_callback();
        const auto finalization_started = std::chrono::steady_clock::now();
        if (!runtime_present) {
            cache.admit_downloaded_blob(
                converted, runtime_artifact, finalization_work_progress);
            copied += runtime_artifact.size;
        }
        for (const ArtifactSource& artifact : static_artifacts)
            copied += admit_artifact(
                artifact, cache, reused, finalization_work_progress);

        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "Finalization interrupted. Downloaded source retained.");

        const fs::path staged_manifest = unique_import_path(staging, "vrhino-model.json");
        {
            std::ofstream output(staged_manifest, std::ios::binary | std::ios::trunc);
            if (!output.good())
                fail(ModelPackageErrorCode::InstallFailed,
                     "cannot create native converter package manifest");
            output << read_text(manifest_path);
            output.flush();
            if (!output.good())
                fail(ModelPackageErrorCode::InstallFailed,
                     "cannot write native converter package manifest");
        }
        const InstallResult installation = cache.publish_manifest(staged_manifest);
        fs::remove(staged_manifest, error);
        // Publication is the commit point. A signal arriving after it must not
        // turn a successfully installed package into a reported failure.
        finalization_progress.finish_committed();
        const auto finalization_finished = std::chrono::steady_clock::now();
        ImportResult result;
        result.installation = installation;
        result.runtime_vrm_path = cache.artifact_path(runtime_artifact.sha256);
        result.source_checkpoint_bytes = kLtxCheckpointBytes;
        result.output_vrm_bytes = runtime_artifact.size;
        result.temporary_disk_peak_bytes = std::max<uint64_t>(
            runtime_artifact.size, artifact_by_id(manifest, "t5-shard-1").size);
        result.largest_temporary_buffer_bytes = write_result.largest_buffer_bytes;
        result.mapping_count = mappings.size();
        result.copied_artifact_bytes = copied;
        result.reused_artifact_bytes = reused;
        result.conversion_performed = !runtime_present;
        result.conversion_seconds = std::chrono::duration<double>(
            conversion_finished - conversion_started).count();
        result.finalization_seconds = std::chrono::duration<double>(
            finalization_finished - finalization_started).count();
        result.runtime_vrm_sha256 = runtime_artifact.sha256;
        return result;
    } catch (...) {
        fs::remove(converted, error);
        throw;
    }
}

}  // namespace

ImportResult import_local_model(const std::string& catalog_reference,
                                const std::filesystem::path& source_directory,
                                LocalModelCache& cache,
                                const ImportOptions& options) {
    ImportOptions resolved_options = options;
    if (resolved_options.expected_model_reference.empty())
        resolved_options.expected_model_reference = catalog_reference;
    if (resolved_options.package_manifest.empty()) {
        const fs::path specification_root = resolved_options.converter_spec_root.empty()
            ? discover_converter_spec_root()
            : fs::canonical(resolved_options.converter_spec_root);
        if (catalog_reference == "vrhino/ltx-video-v0.9.1:1.1.1")
            resolved_options.package_manifest = specification_root /
                "ltx_v0_9_1/successors/1.1.1/vrhino-model.json";
        else if (catalog_reference == "vrhino/wan2.1-t2v-1.3b:1.0.1")
            resolved_options.package_manifest = specification_root /
                "wan2_1_t2v_1_3b/successors/1.0.1/vrhino-model.json";
        else if (catalog_reference == "vrhino/mochi-1-preview:1.0.1")
            resolved_options.package_manifest = specification_root /
                "mochi_1_preview/successors/1.0.1/vrhino-model.json";
        else if (catalog_reference == "vrhino/musetalk-v1.5:1.0.1")
            resolved_options.package_manifest = specification_root /
                "public_musetalk_v15/successors/1.0.1/vrhino-model.json";
        else if (catalog_reference == "vrhino/latentsync-1.6:1.0.1")
            resolved_options.package_manifest = specification_root /
                "public_latentsync_16/successors/1.0.1/vrhino-model.json";
    }
    if (catalog_reference == kLtxReference)
        return import_ltx(source_directory, cache, resolved_options);
    if (catalog_reference == "vrhino/ltx-video-v0.9.1:1.1.1")
        return import_ltx(source_directory, cache, resolved_options);
    if (catalog_reference == "vrhino/wan2.1-t2v-1.3b:1.0.0")
        return import_wan_model(source_directory, cache, resolved_options);
    if (catalog_reference == "vrhino/wan2.1-t2v-1.3b:1.0.1")
        return import_wan_model(source_directory, cache, resolved_options);
    if (catalog_reference == "vrhino/mochi-1-preview:1.0.0")
        return import_mochi_model(source_directory, cache, resolved_options);
    if (catalog_reference == "vrhino/mochi-1-preview:1.0.1")
        return import_mochi_model(source_directory, cache, resolved_options);
    if (catalog_reference == "private/musetalk-v1.5:1.0.0")
        return import_private_musetalk_v15(source_directory, cache, resolved_options);
    if (catalog_reference == "private/musetalk-v1.5:2.0.0")
        return import_private_musetalk_v15_successor(
            source_directory, cache, resolved_options);
    if (catalog_reference == "vrhino/musetalk-v1.5:1.0.0")
        return import_public_musetalk_v15(source_directory, cache, resolved_options);
    if (catalog_reference == "vrhino/musetalk-v1.5:1.0.1")
        return import_public_musetalk_v15(source_directory, cache, resolved_options);
    if (catalog_reference == "private/latentsync-1.6:1.0.0")
        return import_private_latentsync_16(source_directory, cache, resolved_options);
    if (catalog_reference == "vrhino/latentsync-1.6:1.0.0")
        return import_public_latentsync_16(source_directory, cache, resolved_options);
    if (catalog_reference == "vrhino/latentsync-1.6:1.0.1")
        return import_public_latentsync_16(source_directory, cache, resolved_options);
    fail(ModelPackageErrorCode::PackageVersionUnsupported,
         "no native converter is registered for catalog model: " + catalog_reference);
}

}  // namespace vrhino::product
