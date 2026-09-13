#include "vrhino/product/converter.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <sys/types.h>
#ifdef _WIN32
#include "windows_converter_io.h"
#else
#include <unistd.h>
#endif
#include <utility>
#include <vector>

#include "vrhino/error.h"

namespace vrhino::product {
namespace {
namespace fs = std::filesystem;

constexpr const char* kRevision =
    "169d4a4341b33bc18d8881c4b69c2e104e1cc0af";
constexpr uint64_t kCheckpointBytes = 151095027ULL;
constexpr const char* kCheckpointSha256 =
    "9607f98a2b22d9e229ae43c52ecea79dcede9e0c5cfae67e8da6eda86d8aac1d";

struct FrozenArtifact {
    const char* path;
    uint64_t bytes;
    const char* sha256;
};

constexpr std::array<FrozenArtifact, 4> kArtifacts = {{
    {"config.json", 1983ULL,
     "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98"},
    {"preprocessor_config.json", 184990ULL,
     "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666"},
    {"pytorch_model.bin", kCheckpointBytes, kCheckpointSha256},
    {"README.md", 19787ULL,
     "57a3bbbbf1e79369e4d1ced790812a1723b43a256c1b43fc70cc2f3339ae9881"},
}};

struct FrozenRange {
    const char* name;
    std::vector<int64_t> shape;
    uint64_t offset;
    uint64_t bytes;
};

const std::vector<FrozenRange>& encoder_ranges() {
    static const std::vector<FrozenRange> values = {
        {"model.encoder.conv1.weight", {384,80,3}, 21184ULL, 368640ULL},
        {"model.encoder.conv1.bias", {384}, 389888ULL, 1536ULL},
        {"model.encoder.conv2.weight", {384,384,3}, 31754176ULL, 1769472ULL},
        {"model.encoder.conv2.bias", {384}, 38251264ULL, 1536ULL},
        {"model.encoder.embed_positions.weight", {1500,384}, 42396736ULL, 2304000ULL},
        {"model.encoder.layers.0.self_attn.k_proj.weight", {384,384}, 50025856ULL, 589824ULL},
        {"model.encoder.layers.0.self_attn.v_proj.weight", {384,384}, 55343296ULL, 589824ULL},
        {"model.encoder.layers.0.self_attn.v_proj.bias", {384}, 139250176ULL, 1536ULL},
        {"model.encoder.layers.0.self_attn.q_proj.weight", {384,384}, 142209856ULL, 589824ULL},
        {"model.encoder.layers.0.self_attn.q_proj.bias", {384}, 148713088ULL, 1536ULL},
        {"model.encoder.layers.0.self_attn.out_proj.weight", {384,384}, 391552ULL, 589824ULL},
        {"model.encoder.layers.0.self_attn.out_proj.bias", {384}, 3351296ULL, 1536ULL},
        {"model.encoder.layers.0.self_attn_layer_norm.weight", {384}, 9266304ULL, 1536ULL},
        {"model.encoder.layers.0.self_attn_layer_norm.bias", {384}, 12226048ULL, 1536ULL},
        {"model.encoder.layers.0.fc1.weight", {1536,384}, 17552768ULL, 2359296ULL},
        {"model.encoder.layers.0.fc1.bias", {1536}, 22281984ULL, 6144ULL},
        {"model.encoder.layers.0.fc2.weight", {384,1536}, 27015808ULL, 2359296ULL},
        {"model.encoder.layers.0.fc2.bias", {384}, 31749248ULL, 1536ULL},
        {"model.encoder.layers.0.final_layer_norm.weight", {384}, 31750912ULL, 1536ULL},
        {"model.encoder.layers.0.final_layer_norm.bias", {384}, 31752576ULL, 1536ULL},
        {"model.encoder.layers.1.self_attn.k_proj.weight", {384,384}, 33523776ULL, 589824ULL},
        {"model.encoder.layers.1.self_attn.v_proj.weight", {384,384}, 34113728ULL, 589824ULL},
        {"model.encoder.layers.1.self_attn.v_proj.bias", {384}, 34703680ULL, 1536ULL},
        {"model.encoder.layers.1.self_attn.q_proj.weight", {384,384}, 34705344ULL, 589824ULL},
        {"model.encoder.layers.1.self_attn.q_proj.bias", {384}, 35295296ULL, 1536ULL},
        {"model.encoder.layers.1.self_attn.out_proj.weight", {384,384}, 35296960ULL, 589824ULL},
        {"model.encoder.layers.1.self_attn.out_proj.bias", {384}, 35886912ULL, 1536ULL},
        {"model.encoder.layers.1.self_attn_layer_norm.weight", {384}, 35888576ULL, 1536ULL},
        {"model.encoder.layers.1.self_attn_layer_norm.bias", {384}, 35890240ULL, 1536ULL},
        {"model.encoder.layers.1.fc1.weight", {1536,384}, 35891904ULL, 2359296ULL},
        {"model.encoder.layers.1.fc1.bias", {1536}, 38252928ULL, 6144ULL},
        {"model.encoder.layers.1.fc2.weight", {384,1536}, 38259200ULL, 2359296ULL},
        {"model.encoder.layers.1.fc2.bias", {384}, 40618624ULL, 1536ULL},
        {"model.encoder.layers.1.final_layer_norm.weight", {384}, 40620288ULL, 1536ULL},
        {"model.encoder.layers.1.final_layer_norm.bias", {384}, 40621952ULL, 1536ULL},
        {"model.encoder.layers.2.self_attn.k_proj.weight", {384,384}, 40623616ULL, 589824ULL},
        {"model.encoder.layers.2.self_attn.v_proj.weight", {384,384}, 41213568ULL, 589824ULL},
        {"model.encoder.layers.2.self_attn.v_proj.bias", {384}, 41803520ULL, 1536ULL},
        {"model.encoder.layers.2.self_attn.q_proj.weight", {384,384}, 41805184ULL, 589824ULL},
        {"model.encoder.layers.2.self_attn.q_proj.bias", {384}, 42395136ULL, 1536ULL},
        {"model.encoder.layers.2.self_attn.out_proj.weight", {384,384}, 44700864ULL, 589824ULL},
        {"model.encoder.layers.2.self_attn.out_proj.bias", {384}, 45290816ULL, 1536ULL},
        {"model.encoder.layers.2.self_attn_layer_norm.weight", {384}, 45292480ULL, 1536ULL},
        {"model.encoder.layers.2.self_attn_layer_norm.bias", {384}, 45294144ULL, 1536ULL},
        {"model.encoder.layers.2.fc1.weight", {1536,384}, 45295808ULL, 2359296ULL},
        {"model.encoder.layers.2.fc1.bias", {1536}, 47655232ULL, 6144ULL},
        {"model.encoder.layers.2.fc2.weight", {384,1536}, 47661504ULL, 2359296ULL},
        {"model.encoder.layers.2.fc2.bias", {384}, 50020928ULL, 1536ULL},
        {"model.encoder.layers.2.final_layer_norm.weight", {384}, 50022592ULL, 1536ULL},
        {"model.encoder.layers.2.final_layer_norm.bias", {384}, 50024256ULL, 1536ULL},
        {"model.encoder.layers.3.self_attn.k_proj.weight", {384,384}, 50615808ULL, 589824ULL},
        {"model.encoder.layers.3.self_attn.v_proj.weight", {384,384}, 51205760ULL, 589824ULL},
        {"model.encoder.layers.3.self_attn.v_proj.bias", {384}, 51795712ULL, 1536ULL},
        {"model.encoder.layers.3.self_attn.q_proj.weight", {384,384}, 51797376ULL, 589824ULL},
        {"model.encoder.layers.3.self_attn.q_proj.bias", {384}, 52387328ULL, 1536ULL},
        {"model.encoder.layers.3.self_attn.out_proj.weight", {384,384}, 52388992ULL, 589824ULL},
        {"model.encoder.layers.3.self_attn.out_proj.bias", {384}, 52978944ULL, 1536ULL},
        {"model.encoder.layers.3.self_attn_layer_norm.weight", {384}, 52980608ULL, 1536ULL},
        {"model.encoder.layers.3.self_attn_layer_norm.bias", {384}, 52982272ULL, 1536ULL},
        {"model.encoder.layers.3.fc1.weight", {1536,384}, 52983936ULL, 2359296ULL},
        {"model.encoder.layers.3.fc1.bias", {1536}, 55933248ULL, 6144ULL},
        {"model.encoder.layers.3.fc2.weight", {384,1536}, 55939520ULL, 2359296ULL},
        {"model.encoder.layers.3.fc2.bias", {384}, 58298944ULL, 1536ULL},
        {"model.encoder.layers.3.final_layer_norm.weight", {384}, 58300608ULL, 1536ULL},
        {"model.encoder.layers.3.final_layer_norm.bias", {384}, 58302272ULL, 1536ULL},
        {"model.encoder.layer_norm.weight", {384}, 58303936ULL, 1536ULL},
        {"model.encoder.layer_norm.bias", {384}, 58305600ULL, 1536ULL},
    };
    return values;
}

[[noreturn]] void fail(ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

Json string_value(const std::string& value) { return Json(Json::Value(value)); }
Json integer_value(int64_t value) { return Json(Json::Value(value)); }
Json number_value(double value) { return Json(Json::Value(value)); }
Json array_value(std::initializer_list<Json> values) {
    return Json(Json::Value(Json::Array(values)));
}
Json object_value(std::initializer_list<std::pair<const std::string, Json>> values) {
    Json::Object object;
    for (const auto& [key, value] : values) object.emplace(key, value);
    return Json(Json::Value(std::move(object)));
}

std::string logical_name(const std::string& source) {
    constexpr const char* prefix = "model.";
    require(source.starts_with(prefix), "Invalid Whisper source tensor prefix");
    return source.substr(std::char_traits<char>::length(prefix));
}

Json linear_descriptor(const std::string& prefix, bool bias = true) {
    Json::Object result;
    result.emplace("weight", string_value(prefix + ".weight"));
    if (bias) result.emplace("bias", string_value(prefix + ".bias"));
    else result.emplace("bias", Json(Json::Value(nullptr)));
    return Json(Json::Value(std::move(result)));
}

Json norm_descriptor(const std::string& prefix) {
    return object_value({
        {"kind", string_value("layer_norm")},
        {"eps", number_value(1.0e-5)},
        {"weight", string_value(prefix + ".weight")},
        {"bias", string_value(prefix + ".bias")},
    });
}

Json encoder_block(int index) {
    const std::string prefix = "encoder.layers." + std::to_string(index);
    return object_value({
        {"input_norm", norm_descriptor(prefix + ".self_attn_layer_norm")},
        {"attention", object_value({
            {"heads", integer_value(6)},
            {"scale", number_value(0.125)},
            {"query", linear_descriptor(prefix + ".self_attn.q_proj")},
            {"key", linear_descriptor(prefix + ".self_attn.k_proj", false)},
            {"value", linear_descriptor(prefix + ".self_attn.v_proj")},
            {"output", linear_descriptor(prefix + ".self_attn.out_proj")},
        })},
        {"post_attention_norm", norm_descriptor(prefix + ".final_layer_norm")},
        {"mlp", object_value({
            {"activation", string_value("gelu")},
            {"width", integer_value(1536)},
            {"input", linear_descriptor(prefix + ".fc1")},
            {"output", linear_descriptor(prefix + ".fc2")},
        })},
    });
}

Json graph(const std::vector<TensorMapping>& mappings) {
    Json::Object bindings;
    for (const TensorMapping& mapping : mappings)
        bindings.emplace(logical_name(mapping.source_name),
                         string_value(mapping.destination_name));
    Json::Array blocks;
    for (int index = 0; index < 4; ++index) blocks.push_back(encoder_block(index));
    Json::Array outputs;
    for (const auto& [name, source] : std::array<std::pair<const char*, const char*>, 5>{{
             {"frontend_hidden", "frontend"},
             {"encoder_block_1", "block.0"},
             {"encoder_block_2", "block.1"},
             {"encoder_block_3", "block.2"},
             {"encoder_block_4", "final"},
         }})
        outputs.push_back(object_value({{"name", string_value(name)},
                                        {"source", string_value(source)}}));
    return object_value({
        {"schema_version", integer_value(1)},
        {"kind", string_value("audio_encoder_transformer")},
        {"config", object_value({
            {"input_layout", string_value("NCL")},
            {"mel_bins", integer_value(80)},
            {"input_frames", integer_value(3000)},
            {"hidden_width", integer_value(384)},
            {"output_frames", integer_value(1500)},
            {"encoder_blocks", integer_value(4)},
            {"attention_heads", integer_value(6)},
            {"ffn_width", integer_value(1536)},
            {"dtype", string_value("float32")},
        })},
        {"frontend", object_value({
            {"conv1", object_value({
                {"weight", string_value("encoder.conv1.weight")},
                {"bias", string_value("encoder.conv1.bias")},
                {"kernel", integer_value(3)},
                {"stride", integer_value(1)},
                {"padding", integer_value(1)},
            })},
            {"activation1", string_value("gelu")},
            {"conv2", object_value({
                {"weight", string_value("encoder.conv2.weight")},
                {"bias", string_value("encoder.conv2.bias")},
                {"kernel", integer_value(3)},
                {"stride", integer_value(2)},
                {"padding", integer_value(1)},
            })},
            {"activation2", string_value("gelu")},
        })},
        {"absolute_position_embedding", object_value({
            {"weight", string_value("encoder.embed_positions.weight")},
        })},
        {"blocks", Json(Json::Value(std::move(blocks)))},
        {"final_norm", norm_descriptor("encoder.layer_norm")},
        {"outputs", Json(Json::Value(std::move(outputs)))},
        {"runtime_tensor_bindings", Json(Json::Value(std::move(bindings)))},
        {"required_primitives", array_value({
            string_value("conv1d"), string_value("conv2d"),
            string_value("activation"), string_value("permute"),
            string_value("add"), string_value("layer_norm"),
            string_value("linear"), string_value("attention"),
            string_value("reshape")})},
    });
}

Json metadata(uint64_t retained_bytes) {
    return object_value({
        {"architecture", string_value("audio-encoder")},
        {"component", object_value({
            {"kind", string_value("audio_encoder")},
            {"implementation", string_value("generic.conv_transformer_encoder.v1")},
            {"source_repository", string_value("openai/whisper-tiny")},
            {"source_revision", string_value(kRevision)},
            {"license", string_value("apache-2.0")},
            {"checkpoint_sha256", string_value(kCheckpointSha256)},
            {"checkpoint_bytes", integer_value(static_cast<int64_t>(kCheckpointBytes))},
            {"retained_encoder_bytes", integer_value(static_cast<int64_t>(retained_bytes))},
            {"decoder_included", Json(Json::Value(false))},
            {"tokenizer_included", Json(Json::Value(false))},
        })},
        {"profile", string_value("component")},
        {"default_dtype", string_value("float32")},
    });
}

fs::path checked_artifact(const fs::path& root, const FrozenArtifact& artifact,
                          const WorkProgressCallback& progress) {
    const fs::path path = root / artifact.path;
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing frozen Whisper source artifact: " + std::string(artifact.path));
    if (fs::file_size(path, error) != artifact.bytes || error)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen Whisper source size mismatch: " + std::string(artifact.path));
    if (sha256_file(path, progress) != artifact.sha256)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen Whisper source SHA256 mismatch: " + std::string(artifact.path));
    return path;
}

}  // namespace

FrozenComponentConversionResult convert_whisper_tiny_encoder_component(
        const fs::path& source_directory, const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "Conversion interrupted. Frozen source retained.");
    std::map<std::string, fs::path> paths;
    for (const FrozenArtifact& artifact : kArtifacts)
        paths.emplace(artifact.path, checked_artifact(source_directory, artifact, progress));

    std::map<std::string, SourceTensorDescriptor> tensors;
    std::vector<TensorMapping> mappings;
    uint64_t retained_bytes = 0;
    for (const FrozenRange& range : encoder_ranges()) {
        retained_bytes += range.bytes;
        tensors.emplace(range.name, SourceTensorDescriptor{
            range.name, DType::F32, "F32", range.shape, 0, range.offset, range.bytes});
        mappings.push_back(TensorMapping{
            range.name, DType::F32, range.shape, "identity_bytes",
            "audio_encoder." + logical_name(range.name).substr(
                std::char_traits<char>::length("encoder.")),
            DType::F32, range.shape, "audio_encoder", "weight"});
    }
    require(retained_bytes == 32833536ULL && mappings.size() == 67,
            "Frozen Whisper encoder inventory drift");
    FrozenTensorSource source({paths.at("pytorch_model.bin")}, std::move(tensors));

    std::error_code error;
    if (fs::exists(output, error))
        fail(ModelPackageErrorCode::InstallFailed,
             "Whisper component output already exists: " + output.string());
    fs::create_directories(output.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create Whisper component output directory");
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path temporary = output.parent_path() /
        (output.filename().string() + ".partial-" + std::to_string(getpid()) +
         "-" + std::to_string(tick));
    try {
        const VrmWriteResult written = write_vrm_streaming(
            temporary, "component", "audio-encoder", metadata(retained_bytes),
            graph(mappings), source, mappings, cancellation_requested, progress);
        fs::rename(temporary, output, error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish Whisper component: " + error.message());
        return FrozenComponentConversionResult{
            output, kCheckpointBytes, retained_bytes,
            static_cast<uint64_t>(mappings.size()), written.file_size,
            written.largest_buffer_bytes, written.seconds,
            sha256_file(output), written.payload_blake2b128};
    } catch (...) {
        fs::remove(temporary, error);
        throw;
    }
}

}  // namespace vrhino::product
