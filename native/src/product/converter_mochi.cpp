#include "vrhino/product/converter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <vector>
#include <unistd.h>

namespace vrhino::product {
namespace {

namespace fs = std::filesystem;

constexpr const char* kMochiReference = "vrhino/mochi-1-preview:1.0.0";
constexpr const char* kExpectedVrmSha256 =
    "37fb840eaa8953ba28ccbfc80b3a601dbfc78c423ff10c40e8762ae4620525d0";
constexpr const char* kExpectedVrmPayloadBlake2b128 =
    "6cd15b7bd58a156f269e1ee44253e251";
constexpr uint64_t kExpectedVrmBytes = 20780961408ULL;
constexpr uint64_t kExpectedTensorCount = 1233ULL;
constexpr uint64_t kSourceLogicalBytes = 40025270143ULL;

struct FrozenFile {
    const char* relative_path;
    uint64_t size;
    const char* sha256;
};

constexpr std::array<FrozenFile, 20> kFrozenFiles{{
    {"model_index.json", 408ULL,
     "db470a8c28b77125d74cb955f277c20b9f629e01b177dd352d7ec396ccad3d85"},
    {"scheduler/scheduler_config.json", 299ULL,
     "e76276322d53cda452afe406d3c1e6a1a8bdc9de3fae3a32c831f4b8e41890f4"},
    {"text_encoder/config.json", 781ULL,
     "e2502471a5c7b13c12e62dd2f068d714a284b259a11ffbdefeb58af37cba1a60"},
    {"text_encoder/model-00001-of-00004.safetensors", 4989319680ULL,
     "7a68b2c8c080696a10109612a649bc69330991ecfea65930ccfdfbdb011f2686"},
    {"text_encoder/model-00002-of-00004.safetensors", 4999830656ULL,
     "b8ed6556d7507e38af5b428c605fb2a6f2bdb7e80bd481308b865f7a40c551ca"},
    {"text_encoder/model-00003-of-00004.safetensors", 4865612720ULL,
     "c831635f83041f83faf0024b39c6ecb21b45d70dd38a63ea5bac6c7c6e5e558c"},
    {"text_encoder/model-00004-of-00004.safetensors", 4194506688ULL,
     "02a5f2d69205be92ad48fe5d712d38c2ff55627969116aeffc58bd75a28da468"},
    {"text_encoder/model.safetensors.index.json", 19886ULL,
     "a545bb25dc0f423d84be7b577311bba8bb7c6931f1eefcea65fc8b0a61a60a76"},
    {"tokenizer/added_tokens.json", 2593ULL,
     "ea5a91a3234f66ea642c8e672d67f0f493759a9bee6910ae304ea9b9492118b5"},
    {"tokenizer/special_tokens_map.json", 2543ULL,
     "7a1985a994c41886db38c719d2a3d2f40606663cc19d7c5d6a85d349320e06d2"},
    {"tokenizer/spiece.model", 791656ULL,
     "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"},
    {"tokenizer/tokenizer_config.json", 20618ULL,
     "a821b0b127e1deb7bcdad0a493fb817d4efd90e6a46a2c3cdc21228d5492b2a9"},
    {"transformer/config.json", 396ULL,
     "592148b13b5b7ab182c266f73a8d35ef6a427775e45e573bc7359b01d8824d1a"},
    {"transformer/diffusion_pytorch_model.bf16-00001-of-00003.safetensors",
     9937333040ULL,
     "29f767d34af7bfba3a216def2d1a45c23f0a71ce69ead170af92328e0605eb1a"},
    {"transformer/diffusion_pytorch_model.bf16-00002-of-00003.safetensors",
     9929099008ULL,
     "fd0c821a4ed49847f30f4baed76136b5f0e3077d95538469e196ecca50259713"},
    {"transformer/diffusion_pytorch_model.bf16-00003-of-00003.safetensors",
     189051592ULL,
     "e94bb39dec286bafcd626252a2e26f00ea36a900bd813bf8b0e3cc17493a49db"},
    {"transformer/diffusion_pytorch_model.safetensors.index.bf16.json", 117567ULL,
     "710411b966b38d2f213e700da6db573df45dae85bf93e173fdde8db1577beed4"},
    {"vae/config.json", 1234ULL,
     "ef3433936ef0e152607387858b1a0e833cd7b8cd7253d31baa0f9d3f5975835c"},
    {"vae/diffusion_pytorch_model.bf16.safetensors", 919549966ULL,
     "fecceadc048487f07cad99f6f5fbbf691f7385573d53cea2f06e461b3a9721ea"},
    {"README.md", 8812ULL,
     "1f98f31b65b34445be9383b805ec61707a062067f77e21c2cdbdf91a744ecce4"},
}};

[[noreturn]] void fail(const ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
        fail(ModelPackageErrorCode::ArtifactMissing,
             "cannot open Mochi converter specification: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

std::vector<std::string> split(const std::string& value, const char separator) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (true) {
        const size_t end = value.find(separator, begin);
        result.push_back(value.substr(begin, end == std::string::npos
                                                ? std::string::npos : end - begin));
        if (end == std::string::npos) return result;
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
                 "invalid Mochi tensor-map dimension: " + field);
        }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid Mochi tensor-map dimension: " + field);
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
         "unsupported Mochi tensor-map dtype: " + value);
}

fs::path checked_source(const fs::path& root, const fs::path& relative) {
    std::error_code error;
    const fs::path path = root / relative;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status))
        fail(ModelPackageErrorCode::ArtifactMissing,
             "required Mochi source is missing or not a regular file: " +
                 relative.string());
    const fs::path canonical_root = fs::canonical(root, error);
    if (error)
        fail(ModelPackageErrorCode::PackageInvalid,
             "cannot canonicalize Mochi source root");
    const fs::path canonical_path = fs::canonical(path, error);
    if (error || canonical_path.native().size() < canonical_root.native().size() ||
        canonical_path.native().compare(0, canonical_root.native().size(),
                                        canonical_root.native()) != 0 ||
        (canonical_path.native().size() > canonical_root.native().size() &&
         canonical_path.native()[canonical_root.native().size()] !=
             fs::path::preferred_separator))
        fail(ModelPackageErrorCode::PackageInvalid,
             "Mochi source path escapes source directory: " + relative.string());
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
             "Mochi conversion work total overflows uint64");
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
            if (progress_)
                progress_(total_ == 0 ? 0 : std::min(completed_, total_ - 1), total_);
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

class CompositeSafeTensorSource final : public TensorSource {
public:
    explicit CompositeSafeTensorSource(std::vector<const SafeTensorReader*> readers)
        : readers_(std::move(readers)) {}

    const std::map<std::string, SourceTensorDescriptor>& tensors() const noexcept override {
        return tensors_;
    }

    void add(const std::string& qualified_name, const size_t source_index,
             const SourceTensorDescriptor& source) {
        SourceTensorDescriptor descriptor = source;
        descriptor.name = qualified_name;
        descriptor.source_index = source_index;
        if (!tensors_.emplace(qualified_name, std::move(descriptor)).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate Mochi mapped source tensor: " + qualified_name);
    }

    void read_tensor(const SourceTensorDescriptor& tensor,
                     const uint64_t relative_offset, void* destination,
                     const size_t bytes) const override {
        if (tensor.source_index >= readers_.size())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid Mochi source tensor reader index");
        readers_.at(tensor.source_index)->read_tensor(
            tensor, relative_offset, destination, bytes);
    }

private:
    std::vector<const SafeTensorReader*> readers_;
    std::map<std::string, SourceTensorDescriptor> tensors_;
};

struct MappingSet {
    CompositeSafeTensorSource source;
    std::vector<TensorMapping> mappings;
};

MappingSet load_tensor_map(
        const fs::path& path,
        const std::vector<std::pair<std::string, const SafeTensorReader*>>& readers) {
    std::ifstream input(path);
    if (!input.good())
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing Mochi tensor mapping specification: " + path.string());
    std::string line;
    if (!std::getline(input, line) || line !=
            "source_file\tsource_name\tsource_dtype\tsource_shape\ttransformation\t"
            "destination_name\tdestination_dtype\tdestination_shape\tcomponent\trole")
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid Mochi tensor mapping header");

    std::vector<const SafeTensorReader*> reader_values;
    std::map<std::string, size_t> reader_indices;
    for (const auto& [name, reader] : readers) {
        if (reader == nullptr || !reader_indices.emplace(name, reader_values.size()).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid Mochi tensor reader declaration: " + name);
        reader_values.push_back(reader);
    }
    MappingSet result{CompositeSafeTensorSource(std::move(reader_values)), {}};
    std::set<std::pair<size_t, std::string>> mapped_sources;
    std::set<std::string> mapped_destinations;
    size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const std::vector<std::string> fields = split(line, '\t');
        if (fields.size() != 10)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid Mochi tensor mapping row " + std::to_string(line_number));
        const auto source_file = reader_indices.find(fields[0]);
        if (source_file == reader_indices.end())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unknown Mochi tensor-map source: " + fields[0]);
        const SafeTensorReader& reader = *readers.at(source_file->second).second;
        const auto source_tensor = reader.tensors().find(fields[1]);
        if (source_tensor == reader.tensors().end())
            fail(ModelPackageErrorCode::ArtifactMissing,
                 "missing Mochi source tensor: " + fields[0] + ":" + fields[1]);
        const DType source_dtype = map_dtype(fields[2]);
        const std::vector<int64_t> source_shape = parse_shape(fields[3]);
        if (source_tensor->second.dtype != source_dtype ||
            source_tensor->second.shape != source_shape)
            fail(ModelPackageErrorCode::PackageVersionUnsupported,
                 "Mochi upstream tensor type/shape drift: " + fields[1]);
        if (!mapped_sources.emplace(source_file->second, fields[1]).second ||
            !mapped_destinations.insert(fields[5]).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate Mochi tensor mapping at row " +
                     std::to_string(line_number));
        const std::string qualified = fields[0] + ":" + fields[1];
        result.source.add(qualified, source_file->second, source_tensor->second);
        result.mappings.push_back(TensorMapping{
            qualified, source_dtype, source_shape, fields[4], fields[5],
            map_dtype(fields[6]), parse_shape(fields[7]), fields[8], fields[9]});
    }

    size_t expected = 0;
    for (size_t index = 0; index < readers.size(); ++index) {
        for (const auto& [name, unused] : readers[index].second->tensors()) {
            (void)unused;
            const bool required = readers[index].first != "vae" ||
                                  name.rfind("decoder.", 0) == 0;
            if (!required) continue;
            ++expected;
            if (!mapped_sources.contains({index, name}))
                fail(ModelPackageErrorCode::PackageVersionUnsupported,
                     "unmapped Mochi source tensor: " + readers[index].first + ":" + name);
        }
    }
    if (expected != kExpectedTensorCount ||
        result.mappings.size() != kExpectedTensorCount)
        fail(ModelPackageErrorCode::PackageVersionUnsupported,
             "Mochi Runtime tensor count differs from qualified artifact");
    return result;
}

const ArtifactDeclaration& artifact_by_id(const ModelPackageManifest& manifest,
                                          const std::string& id) {
    for (const ArtifactDeclaration& artifact : manifest.artifacts)
        if (artifact.id == id) return artifact;
    fail(ModelPackageErrorCode::PackageInvalid,
         "Mochi package template is missing artifact: " + id);
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
                "Mochi conversion artifact " + artifact.declaration.id, progress);
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

ImportResult import_mochi_model(const fs::path& source_directory,
                                LocalModelCache& cache,
                                const ImportOptions& options) {
    const auto conversion_started = std::chrono::steady_clock::now();
    const fs::path specification_root = options.converter_spec_root.empty()
        ? discover_converter_spec_root() : fs::canonical(options.converter_spec_root);
    const fs::path specification = specification_root / "mochi_1_preview";
    const fs::path manifest_path = options.package_manifest.empty()
        ? specification / "vrhino-model.json"
        : fs::canonical(options.package_manifest);
    const ModelPackageManifest manifest = load_model_package_manifest(manifest_path);
    const std::string expected_reference = options.expected_model_reference.empty()
        ? std::string(kMochiReference) : options.expected_model_reference;
    if (manifest.identity.reference() != expected_reference)
        fail(ModelPackageErrorCode::PackageInvalid,
             "Mochi converter package template identity mismatch");
    for (const InstalledPackage& installed : cache.list())
        if (installed.identity.reference() == expected_reference)
            fail(ModelPackageErrorCode::InstallFailed,
                 "immutable package version is already installed: " +
                     expected_reference);

    std::map<std::string, fs::path> sources;
    uint64_t source_total = 0;
    for (const FrozenFile& frozen : kFrozenFiles) {
        sources.emplace(frozen.relative_path,
                        checked_source(source_directory, frozen.relative_path));
        source_total = checked_add(source_total, frozen.size);
    }
    if (source_total != kSourceLogicalBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "Mochi fixed source logical size drifted");

    const std::vector<ArtifactSource> static_artifacts = {
        {artifact_by_id(manifest, "t5-index"),
            sources.at("text_encoder/model.safetensors.index.json")},
        {artifact_by_id(manifest, "t5-shard-1"),
            sources.at("text_encoder/model-00001-of-00004.safetensors")},
        {artifact_by_id(manifest, "t5-shard-2"),
            sources.at("text_encoder/model-00002-of-00004.safetensors")},
        {artifact_by_id(manifest, "t5-shard-3"),
            sources.at("text_encoder/model-00003-of-00004.safetensors")},
        {artifact_by_id(manifest, "t5-shard-4"),
            sources.at("text_encoder/model-00004-of-00004.safetensors")},
        {artifact_by_id(manifest, "tokenizer"), sources.at("tokenizer/spiece.model")},
        {artifact_by_id(manifest, "conditioning-graph"),
            specification / "conditioning/mochi-t5.json"},
        {artifact_by_id(manifest, "precision-policy"),
            specification / "policy/bf16-operation-contract-v1.json"},
        {artifact_by_id(manifest, "default-profile"),
            specification / "execution/default-preset.json"},
        {artifact_by_id(manifest, "license"), sources.at("README.md")},
    };
    const ArtifactDeclaration& runtime_artifact = artifact_by_id(manifest, "runtime");
    if (runtime_artifact.size != kExpectedVrmBytes ||
        runtime_artifact.sha256 != kExpectedVrmSha256)
        fail(ModelPackageErrorCode::PackageInvalid,
             "Mochi converter output identity drifted from qualified artifact");
    const bool runtime_present = cache.contains_blob(runtime_artifact, false);

    uint64_t static_bytes = 0;
    for (const ArtifactSource& artifact : static_artifacts)
        static_bytes = checked_add(static_bytes, artifact.declaration.size);
    uint64_t conversion_work = checked_add(kSourceLogicalBytes, runtime_artifact.size);
    conversion_work = checked_add(conversion_work,
        checked_add(artifact_by_id(manifest, "conditioning-graph").size,
        checked_add(artifact_by_id(manifest, "precision-policy").size,
                    artifact_by_id(manifest, "default-profile").size)));
    uint64_t finalization_work = checked_add(static_bytes, static_bytes);
    if (!runtime_present)
        finalization_work = checked_add(finalization_work, runtime_artifact.size);
    WorkTracker conversion_progress(options.cancellation_requested, options.progress,
        conversion_work, "Conversion interrupted. Downloaded source retained.");
    conversion_progress.begin();
    const WorkProgressCallback work = conversion_progress.callback();

    for (const FrozenFile& frozen : kFrozenFiles)
        verify_file(sources.at(frozen.relative_path), frozen.size, frozen.sha256,
                    std::string("Mochi source ") + frozen.relative_path, work);
    for (const ArtifactSource& artifact : static_artifacts)
        if (artifact.declaration.id == "conditioning-graph" ||
            artifact.declaration.id == "precision-policy" ||
            artifact.declaration.id == "default-profile")
            verify_artifact(artifact, work);

    SafeTensorReader transformer_1(sources.at(
        "transformer/diffusion_pytorch_model.bf16-00001-of-00003.safetensors"));
    SafeTensorReader transformer_2(sources.at(
        "transformer/diffusion_pytorch_model.bf16-00002-of-00003.safetensors"));
    SafeTensorReader transformer_3(sources.at(
        "transformer/diffusion_pytorch_model.bf16-00003-of-00003.safetensors"));
    SafeTensorReader vae(sources.at("vae/diffusion_pytorch_model.bf16.safetensors"));
    MappingSet mapping = load_tensor_map(specification / "tensor-mapping.tsv", {
        {"transformer-1", &transformer_1}, {"transformer-2", &transformer_2},
        {"transformer-3", &transformer_3}, {"vae", &vae}});
    const Json metadata = Json::parse(read_text(specification / "metadata.json"));
    const Json graph = Json::parse(read_text(specification / "graph.json"));
    if (canonical_json(metadata).size() != 2531 ||
        canonical_json(graph).size() != 129984)
        fail(ModelPackageErrorCode::PackageInvalid,
             "Mochi declaration differs from qualified VRM metadata/graph");

    uint64_t missing_bytes = runtime_present ? 0 : runtime_artifact.size;
    for (const ArtifactSource& artifact : static_artifacts)
        if (!cache.contains_blob(artifact.declaration, false))
            missing_bytes = checked_add(missing_bytes, artifact.declaration.size);
    std::error_code error;
    fs::create_directories(cache.layout().root, error);
    if (error)
        fail(ModelPackageErrorCode::CacheError,
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
    if (error)
        fail(ModelPackageErrorCode::CacheError,
             "cannot create Mochi converter staging directory: " + error.message());
    const fs::path converted = unique_path(staging, "mochi-model.vrm.partial");
    uint64_t copied = 0;
    uint64_t reused = 0;
    VrmWriteResult write_result;
    try {
        if (!runtime_present) {
            write_result = write_vrm_streaming(
                converted, "dit-flow", "mochi", metadata, graph, mapping.source,
                mapping.mappings, options.cancellation_requested, work);
            if (write_result.file_size != runtime_artifact.size ||
                write_result.tensor_count != kExpectedTensorCount ||
                write_result.payload_blake2b128 != kExpectedVrmPayloadBlake2b128)
                fail(ModelPackageErrorCode::PackageInvalid,
                     "native Mochi VRM differs from qualified artifact");
        } else {
            if (!cache.contains_blob(runtime_artifact, true, work))
                fail(ModelPackageErrorCode::CacheError,
                     "existing Mochi Runtime CAS blob is invalid");
            reused = checked_add(reused, runtime_artifact.size);
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
            cache.admit_downloaded_blob(converted, runtime_artifact, finalize);
            copied = checked_add(copied, runtime_artifact.size);
        }
        for (const ArtifactSource& artifact : static_artifacts)
            copied = checked_add(copied,
                                 admit_artifact(artifact, cache, reused, finalize));
        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "Finalization interrupted. Downloaded source retained.");

        const fs::path staged_manifest = unique_path(staging, "vrhino-model.json");
        {
            std::ofstream output(staged_manifest, std::ios::binary | std::ios::trunc);
            if (!output.good())
                fail(ModelPackageErrorCode::InstallFailed,
                     "cannot create Mochi package manifest");
            output << read_text(manifest_path);
            output.flush();
            if (!output.good())
                fail(ModelPackageErrorCode::InstallFailed,
                     "cannot write Mochi package manifest");
        }
        const InstallResult installation = cache.publish_manifest(staged_manifest);
        fs::remove(staged_manifest, error);
        finalization_progress.committed();
        const auto finalization_finished = std::chrono::steady_clock::now();

        ImportResult result;
        result.installation = installation;
        result.runtime_vrm_path = cache.artifact_path(runtime_artifact.sha256);
        result.source_checkpoint_bytes = kSourceLogicalBytes;
        result.output_vrm_bytes = runtime_artifact.size;
        result.temporary_disk_peak_bytes = runtime_artifact.size;
        result.largest_temporary_buffer_bytes = write_result.largest_buffer_bytes;
        result.mapping_count = mapping.mappings.size();
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

}  // namespace vrhino::product
