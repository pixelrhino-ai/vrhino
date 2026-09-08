#include "vrhino/product/converter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
#include <unistd.h>

namespace vrhino::product {
namespace {

namespace fs = std::filesystem;

constexpr const char* kWanReference = "vrhino/wan2.1-t2v-1.3b:1.0.0";
constexpr const char* kDenoiserSha256 =
    "96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f";
constexpr const char* kUmt5SourceSha256 =
    "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d";
constexpr const char* kVaeSha256 =
    "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981";
constexpr const char* kConfigSha256 =
    "ab37994c43740513f94b3ba6233a784035a67b43c8cde83c8f31aa90468c67ce";
constexpr const char* kExpectedVrmSha256 =
    "5869f708f508ac647b4969f968e4794fe30f72183a9b326298f05a02347985b9";
constexpr const char* kExpectedUmt5Sha256 =
    "1c137395973a8a717bd5d467fec816e77af973474c6d4afd2cb83c78cc489bd6";
constexpr uint64_t kDenoiserBytes = 5676070424ULL;
constexpr uint64_t kUmt5SourceBytes = 11361920418ULL;
constexpr uint64_t kVaeBytes = 507609880ULL;
constexpr uint64_t kConfigBytes = 249ULL;

[[noreturn]] void fail(const ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
        fail(ModelPackageErrorCode::ArtifactMissing,
             "cannot open Wan converter specification: " + path.string());
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
    if (text.empty()) return {};
    std::vector<int64_t> shape;
    for (const std::string& field : split(text, ',')) {
        size_t consumed = 0;
        int64_t dimension = 0;
        try { dimension = std::stoll(field, &consumed); }
        catch (...) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid Wan tensor-map dimension: " + field);
        }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid Wan tensor-map dimension: " + field);
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
         "unsupported Wan tensor-map dtype: " + value);
}

uint64_t tensor_bytes(const DType dtype, const std::vector<int64_t>& shape) {
    uint64_t elements = 1;
    for (const int64_t dimension : shape) {
        if (dimension < 0 || (dimension != 0 &&
            elements > std::numeric_limits<uint64_t>::max() /
                       static_cast<uint64_t>(dimension)))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "Wan tensor-map shape overflows uint64");
        elements *= static_cast<uint64_t>(dimension);
    }
    const uint64_t width = dtype_size(dtype);
    if (width != 0 && elements > std::numeric_limits<uint64_t>::max() / width)
        fail(ModelPackageErrorCode::PackageInvalid,
             "Wan tensor-map byte count overflows uint64");
    return elements * width;
}

struct FrozenMappingSet {
    std::map<std::string, SourceTensorDescriptor> tensors;
    std::vector<TensorMapping> mappings;
};

FrozenMappingSet load_frozen_tensor_map(
        const fs::path& path, const std::map<std::string, size_t>& sources) {
    std::ifstream input(path);
    if (!input.good())
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing Wan tensor mapping specification: " + path.string());
    std::string line;
    if (!std::getline(input, line) || line !=
            "source_file\tsource_name\tsource_dtype\tsource_shape\tsource_offset\t"
            "transformation\tdestination_name\tdestination_dtype\tdestination_shape\t"
            "component\trole")
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid Wan tensor mapping header");
    FrozenMappingSet result;
    size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const std::vector<std::string> fields = split(line, '\t');
        if (fields.size() != 11)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid Wan tensor mapping row " + std::to_string(line_number));
        const auto source = sources.find(fields[0]);
        if (source == sources.end())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unknown Wan tensor-map source: " + fields[0]);
        size_t consumed = 0;
        uint64_t offset = 0;
        try { offset = std::stoull(fields[4], &consumed); }
        catch (...) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid Wan tensor-map offset at row " + std::to_string(line_number));
        }
        if (consumed != fields[4].size())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid Wan tensor-map offset at row " + std::to_string(line_number));
        const DType source_dtype = map_dtype(fields[2]);
        const std::vector<int64_t> source_shape = parse_shape(fields[3]);
        const std::string qualified_name = fields[0] + ":" + fields[1];
        SourceTensorDescriptor descriptor{
            qualified_name, source_dtype, fields[2], source_shape, source->second,
            offset, tensor_bytes(source_dtype, source_shape)};
        if (!result.tensors.emplace(qualified_name, std::move(descriptor)).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate Wan source tensor mapping: " + qualified_name);
        result.mappings.push_back(TensorMapping{
            qualified_name, source_dtype, source_shape, fields[5], fields[6],
            map_dtype(fields[7]), parse_shape(fields[8]), fields[9], fields[10]});
    }
    if (result.mappings.empty())
        fail(ModelPackageErrorCode::PackageInvalid, "empty Wan tensor mapping");
    return result;
}

fs::path checked_source(const fs::path& root, const fs::path& relative) {
    std::error_code error;
    const fs::path path = root / relative;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status))
        fail(ModelPackageErrorCode::ArtifactMissing,
             "required Wan source is missing or not a regular file: " + relative.string());
    const fs::path canonical_root = fs::canonical(root, error);
    if (error) fail(ModelPackageErrorCode::PackageInvalid,
                    "cannot canonicalize Wan source root");
    const fs::path canonical_path = fs::canonical(path, error);
    if (error || canonical_path.native().size() < canonical_root.native().size() ||
        canonical_path.native().compare(0, canonical_root.native().size(),
                                        canonical_root.native()) != 0 ||
        (canonical_path.native().size() > canonical_root.native().size() &&
         canonical_path.native()[canonical_root.native().size()] !=
             fs::path::preferred_separator))
        fail(ModelPackageErrorCode::PackageInvalid,
             "Wan source path escapes source directory: " + relative.string());
    return canonical_path;
}

fs::path unique_path(const fs::path& directory, const std::string& label) {
    static std::atomic<uint64_t> sequence{0};
    const uint64_t tick = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return directory / (label + "-" + std::to_string(getpid()) + "-" +
                        std::to_string(tick) + "-" +
                        std::to_string(sequence.fetch_add(1)));
}

uint64_t checked_add(const uint64_t left, const uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left)
        fail(ModelPackageErrorCode::PackageInvalid,
             "Wan conversion work total overflows uint64");
    return left + right;
}

class WorkTracker {
public:
    WorkTracker(std::function<bool()> cancelled,
                std::function<void(uint64_t, uint64_t)> progress,
                const uint64_t total, std::string interrupted)
        : cancelled_(std::move(cancelled)), progress_(std::move(progress)),
          total_(total), interrupted_(std::move(interrupted)) {}
    void begin() { check(); if (progress_) progress_(0, total_); }
    WorkProgressCallback callback() {
        return [this](const uint64_t bytes) {
            check();
            completed_ = checked_add(completed_, bytes);
            if (progress_) progress_(total_ == 0 ? 0 : std::min(completed_, total_ - 1), total_);
            check();
        };
    }
    void finish() { check(); committed(); }
    void committed() { completed_ = total_; if (progress_) progress_(total_, total_); }
private:
    void check() const {
        if (cancelled_ && cancelled_())
            fail(ModelPackageErrorCode::Cancelled, interrupted_);
    }
    std::function<bool()> cancelled_;
    std::function<void(uint64_t, uint64_t)> progress_;
    uint64_t total_ = 0;
    uint64_t completed_ = 0;
    std::string interrupted_;
};

const ArtifactDeclaration& artifact_by_id(const ModelPackageManifest& manifest,
                                          const std::string& id) {
    for (const ArtifactDeclaration& artifact : manifest.artifacts)
        if (artifact.id == id) return artifact;
    fail(ModelPackageErrorCode::PackageInvalid,
         "Wan package template is missing artifact: " + id);
}

struct ArtifactSource {
    ArtifactDeclaration declaration;
    fs::path source;
};

void verify_file(const fs::path& path, const uint64_t size, const std::string& sha,
                 const std::string& label, const WorkProgressCallback& progress) {
    std::error_code error;
    if (fs::file_size(path, error) != size || error)
        fail(ModelPackageErrorCode::PackageInvalid,
             label + " size does not match frozen revision");
    if (sha256_file(path, progress) != sha)
        fail(ModelPackageErrorCode::ChecksumMismatch,
             label + " SHA256 does not match frozen revision");
}

void verify_artifact(const ArtifactSource& artifact,
                     const WorkProgressCallback& progress) {
    verify_file(artifact.source, artifact.declaration.size,
                artifact.declaration.sha256,
                "Wan conversion artifact " + artifact.declaration.id, progress);
}

uint64_t admit_artifact(const ArtifactSource& artifact, LocalModelCache& cache,
                        uint64_t& reused, const WorkProgressCallback& progress) {
    const BlobAdmissionResult admitted = cache.admit_local_blob(
        artifact.source, artifact.declaration, progress);
    if (!admitted.created) {
        reused = checked_add(reused, artifact.declaration.size);
        return 0;
    }
    return artifact.declaration.size;
}

}  // namespace

ImportResult import_wan_model(const fs::path& source_directory,
                              LocalModelCache& cache,
                              const ImportOptions& options) {
    const auto conversion_started = std::chrono::steady_clock::now();
    const fs::path specification_root = options.converter_spec_root.empty()
        ? discover_converter_spec_root() : fs::canonical(options.converter_spec_root);
    const fs::path specification = specification_root / "wan2_1_t2v_1_3b";
    const fs::path manifest_path = options.package_manifest.empty()
        ? specification / "vrhino-model.json"
        : fs::canonical(options.package_manifest);
    const ModelPackageManifest manifest = load_model_package_manifest(manifest_path);
    const std::string expected_reference = options.expected_model_reference.empty()
        ? std::string(kWanReference) : options.expected_model_reference;
    if (manifest.identity.reference() != expected_reference)
        fail(ModelPackageErrorCode::PackageInvalid,
             "Wan converter package template identity mismatch");
    for (const InstalledPackage& installed : cache.list())
        if (installed.identity.reference() == expected_reference)
            fail(ModelPackageErrorCode::InstallFailed,
                 "immutable package version is already installed: " +
                     expected_reference);

    const fs::path denoiser = checked_source(
        source_directory, "diffusion_pytorch_model.safetensors");
    const fs::path umt5 = checked_source(
        source_directory, "models_t5_umt5-xxl-enc-bf16.pth");
    const fs::path vae = checked_source(source_directory, "Wan2.1_VAE.pth");
    const fs::path config = checked_source(source_directory, "config.json");
    const std::vector<ArtifactSource> static_artifacts = {
        {artifact_by_id(manifest, "umt5-index"),
            specification / "conditioning/model.safetensors.index.json"},
        {artifact_by_id(manifest, "tokenizer"), checked_source(
            source_directory, "google/umt5-xxl/tokenizer.json")},
        {artifact_by_id(manifest, "conditioning-graph"),
            specification / "conditioning/wan-umt5.json"},
        {artifact_by_id(manifest, "precision-policy"),
            specification / "policy/bf16-operation-contract-v1.json"},
        {artifact_by_id(manifest, "default-profile"),
            specification / "execution/default-preset.json"},
        {artifact_by_id(manifest, "license"), checked_source(
            source_directory, "LICENSE.txt")},
    };
    const ArtifactDeclaration& runtime_artifact = artifact_by_id(manifest, "runtime");
    const ArtifactDeclaration& umt5_artifact = artifact_by_id(manifest, "umt5-weights");
    if (runtime_artifact.sha256 != kExpectedVrmSha256 ||
        umt5_artifact.sha256 != kExpectedUmt5Sha256)
        fail(ModelPackageErrorCode::PackageInvalid,
             "Wan converter output identity drifted from qualified artifacts");
    const bool runtime_present = cache.contains_blob(runtime_artifact, false);
    const bool umt5_present = cache.contains_blob(umt5_artifact, false);

    uint64_t static_bytes = 0;
    for (const ArtifactSource& artifact : static_artifacts)
        static_bytes = checked_add(static_bytes, artifact.declaration.size);
    uint64_t conversion_work = checked_add(kDenoiserBytes, kUmt5SourceBytes);
    conversion_work = checked_add(conversion_work, kVaeBytes);
    conversion_work = checked_add(conversion_work, kConfigBytes);
    conversion_work = checked_add(conversion_work, static_bytes);
    conversion_work = checked_add(conversion_work, runtime_artifact.size);
    conversion_work = checked_add(conversion_work, umt5_artifact.size);
    uint64_t finalization_work = checked_add(static_bytes, static_bytes);
    if (!runtime_present) finalization_work = checked_add(finalization_work, runtime_artifact.size);
    if (!umt5_present) finalization_work = checked_add(finalization_work, umt5_artifact.size);
    WorkTracker conversion_progress(options.cancellation_requested, options.progress,
        conversion_work, "Conversion interrupted. Downloaded source retained.");
    conversion_progress.begin();
    const WorkProgressCallback work = conversion_progress.callback();

    verify_file(denoiser, kDenoiserBytes, kDenoiserSha256, "Wan denoiser", work);
    verify_file(umt5, kUmt5SourceBytes, kUmt5SourceSha256, "Wan text encoder", work);
    verify_file(vae, kVaeBytes, kVaeSha256, "Wan VAE", work);
    verify_file(config, kConfigBytes, kConfigSha256, "Wan config", work);
    for (const ArtifactSource& artifact : static_artifacts) verify_artifact(artifact, work);

    FrozenMappingSet runtime_map = load_frozen_tensor_map(
        specification / "tensor-mapping.tsv", {{"denoiser", 0}, {"vae", 1}});
    if (runtime_map.mappings.size() != 1019)
        fail(ModelPackageErrorCode::PackageVersionUnsupported,
             "Wan Runtime tensor count differs from qualified artifact");
    FrozenMappingSet umt5_map = load_frozen_tensor_map(
        specification / "umt5-mapping.tsv", {{"umt5", 0}});
    if (umt5_map.mappings.size() != 242)
        fail(ModelPackageErrorCode::PackageVersionUnsupported,
             "Wan UMT5 tensor count differs from qualified artifact");
    FrozenTensorSource runtime_source({denoiser, vae}, std::move(runtime_map.tensors));
    FrozenTensorSource umt5_source({umt5}, std::move(umt5_map.tensors));
    const Json metadata = Json::parse(read_text(specification / "metadata.json"));
    const Json graph = Json::parse(read_text(specification / "graph.json"));
    if (canonical_json(metadata).size() != 1025 || canonical_json(graph).size() != 75350)
        fail(ModelPackageErrorCode::PackageInvalid,
             "Wan declaration differs from qualified VRM metadata/graph");

    uint64_t missing_bytes = 0;
    if (!runtime_present) missing_bytes = checked_add(missing_bytes, runtime_artifact.size);
    if (!umt5_present) missing_bytes = checked_add(missing_bytes, umt5_artifact.size);
    for (const ArtifactSource& artifact : static_artifacts)
        if (!cache.contains_blob(artifact.declaration, false))
            missing_bytes = checked_add(missing_bytes, artifact.declaration.size);
    std::error_code error;
    fs::create_directories(cache.layout().root, error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create local cache root: " + error.message());
    const uint64_t available = options.available_space_override.value_or(
        fs::space(cache.layout().root, error).available);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot query local cache disk space: " + error.message());
    require_conversion_disk_space(missing_bytes, available,
                                  64ULL * 1024 * 1024, cache.layout().root);

    const fs::path staging = cache.layout().temporary / "imports";
    fs::create_directories(staging, error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create Wan converter staging directory: " + error.message());
    const fs::path converted_vrm = unique_path(staging, "wan-model.vrm.partial");
    const fs::path converted_umt5 = unique_path(staging, "wan-umt5.safetensors.partial");
    uint64_t copied = 0;
    uint64_t reused = 0;
    uint64_t largest_buffer = 0;
    try {
        if (!runtime_present) {
            const VrmWriteResult result = write_vrm_streaming(
                converted_vrm, "dit-flow", "wan", metadata, graph, runtime_source,
                runtime_map.mappings, options.cancellation_requested, work);
            if (result.file_size != runtime_artifact.size)
                fail(ModelPackageErrorCode::PackageInvalid,
                     "native Wan VRM size differs from qualified artifact");
            largest_buffer = std::max(largest_buffer, result.largest_buffer_bytes);
        } else {
            if (!cache.contains_blob(runtime_artifact, true, work))
                fail(ModelPackageErrorCode::CacheError,
                     "existing Wan Runtime CAS blob is invalid");
            reused = checked_add(reused, runtime_artifact.size);
        }
        if (!umt5_present) {
            const SafeTensorWriteResult result = write_safetensors_streaming(
                converted_umt5, umt5_source, umt5_map.mappings,
                {{"vrhino.phase", "13C"}, {"vrhino.component", "text_encoder"},
                 {"vrhino.extraction", "sorted-tensor-names-v1"}, {"format", "pt"}},
                options.cancellation_requested, work);
            if (result.file_size != umt5_artifact.size)
                fail(ModelPackageErrorCode::PackageInvalid,
                     "native Wan UMT5 size differs from qualified artifact");
            largest_buffer = std::max(largest_buffer, result.largest_buffer_bytes);
        } else {
            if (!cache.contains_blob(umt5_artifact, true, work))
                fail(ModelPackageErrorCode::CacheError,
                     "existing Wan UMT5 CAS blob is invalid");
            reused = checked_add(reused, umt5_artifact.size);
        }
        conversion_progress.finish();
        const auto conversion_finished = std::chrono::steady_clock::now();

        WorkTracker finalization_progress(options.cancellation_requested,
            options.finalization_progress, finalization_work,
            "Finalization interrupted. Downloaded source retained.");
        finalization_progress.begin();
        const WorkProgressCallback finalize = finalization_progress.callback();
        const auto finalization_started = std::chrono::steady_clock::now();
        if (!runtime_present) {
            cache.admit_downloaded_blob(converted_vrm, runtime_artifact, finalize);
            copied = checked_add(copied, runtime_artifact.size);
        }
        if (!umt5_present) {
            cache.admit_downloaded_blob(converted_umt5, umt5_artifact, finalize);
            copied = checked_add(copied, umt5_artifact.size);
        }
        for (const ArtifactSource& artifact : static_artifacts)
            copied = checked_add(copied, admit_artifact(artifact, cache, reused, finalize));
        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "Finalization interrupted. Downloaded source retained.");

        const fs::path staged_manifest = unique_path(staging, "vrhino-model.json");
        {
            std::ofstream output(staged_manifest, std::ios::binary | std::ios::trunc);
            if (!output.good())
                fail(ModelPackageErrorCode::InstallFailed,
                     "cannot create Wan package manifest");
            output << read_text(manifest_path);
            output.flush();
            if (!output.good())
                fail(ModelPackageErrorCode::InstallFailed,
                     "cannot write Wan package manifest");
        }
        const InstallResult installation = cache.publish_manifest(staged_manifest);
        fs::remove(staged_manifest, error);
        finalization_progress.committed();
        const auto finalization_finished = std::chrono::steady_clock::now();
        ImportResult result;
        result.installation = installation;
        result.runtime_vrm_path = cache.artifact_path(runtime_artifact.sha256);
        result.source_checkpoint_bytes = checked_add(checked_add(kDenoiserBytes,
            kUmt5SourceBytes), kVaeBytes);
        result.output_vrm_bytes = runtime_artifact.size;
        result.temporary_disk_peak_bytes = checked_add(runtime_artifact.size,
                                                       umt5_artifact.size);
        result.largest_temporary_buffer_bytes = largest_buffer;
        result.mapping_count = checked_add(runtime_map.mappings.size(),
                                           umt5_map.mappings.size());
        result.copied_artifact_bytes = copied;
        result.reused_artifact_bytes = reused;
        result.conversion_performed = !runtime_present || !umt5_present;
        result.conversion_seconds = std::chrono::duration<double>(
            conversion_finished - conversion_started).count();
        result.finalization_seconds = std::chrono::duration<double>(
            finalization_finished - finalization_started).count();
        result.runtime_vrm_sha256 = runtime_artifact.sha256;
        return result;
    } catch (...) {
        fs::remove(converted_vrm, error);
        fs::remove(converted_umt5, error);
        throw;
    }
}

}  // namespace vrhino::product
