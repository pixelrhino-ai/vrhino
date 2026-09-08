#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vrhino/json.h"
#include "vrhino/product/model_package.h"
#include "vrhino/tensor.h"

namespace vrhino::product {

struct SourceTensorDescriptor {
    std::string name;
    DType dtype = DType::F32;
    std::string source_dtype;
    std::vector<int64_t> shape;
    size_t source_index = 0;
    uint64_t data_offset = 0;
    uint64_t byte_length = 0;
};

class TensorSource {
public:
    virtual ~TensorSource() = default;
    virtual const std::map<std::string, SourceTensorDescriptor>& tensors() const noexcept = 0;
    virtual void read_tensor(const SourceTensorDescriptor& tensor,
                             uint64_t relative_offset,
                             void* destination,
                             size_t bytes) const = 0;
};

// Streaming safetensors reader for product conversion. Only the bounded JSON
// header is resident; tensor bytes are read with pread into caller buffers.
class SafeTensorReader final : public TensorSource {
public:
    explicit SafeTensorReader(const std::filesystem::path& path);
    ~SafeTensorReader();
    SafeTensorReader(const SafeTensorReader&) = delete;
    SafeTensorReader& operator=(const SafeTensorReader&) = delete;

    const std::filesystem::path& path() const noexcept { return path_; }
    uint64_t file_size() const noexcept { return file_size_; }
    uint64_t header_bytes() const noexcept { return header_bytes_; }
    const Json& metadata() const noexcept { return metadata_; }
    const std::map<std::string, SourceTensorDescriptor>& tensors() const noexcept override {
        return tensors_;
    }
    void read_tensor(const SourceTensorDescriptor& tensor, uint64_t relative_offset,
                     void* destination, size_t bytes) const override;

private:
    int fd_ = -1;
    std::filesystem::path path_;
    uint64_t file_size_ = 0;
    uint64_t header_bytes_ = 0;
    uint64_t data_offset_ = 0;
    Json metadata_{Json::Value(Json::Object{})};
    std::map<std::string, SourceTensorDescriptor> tensors_;
};

// Streaming source for a fixed, checksum-verified upstream representation.
// Tensor ranges are declarative converter data; the production path never
// imports or executes the upstream serialization framework.
class FrozenTensorSource final : public TensorSource {
public:
    FrozenTensorSource(std::vector<std::filesystem::path> paths,
                       std::map<std::string, SourceTensorDescriptor> tensors);
    ~FrozenTensorSource();
    FrozenTensorSource(const FrozenTensorSource&) = delete;
    FrozenTensorSource& operator=(const FrozenTensorSource&) = delete;

    const std::map<std::string, SourceTensorDescriptor>& tensors() const noexcept override {
        return tensors_;
    }
    void read_tensor(const SourceTensorDescriptor& tensor, uint64_t relative_offset,
                     void* destination, size_t bytes) const override;

private:
    std::vector<int> descriptors_;
    std::vector<uint64_t> file_sizes_;
    std::vector<std::filesystem::path> paths_;
    std::map<std::string, SourceTensorDescriptor> tensors_;
};

struct TensorMapping {
    TensorMapping(std::string source_name_value, DType source_dtype_value,
                  std::vector<int64_t> source_shape_value,
                  std::string transformation_value,
                  std::string destination_name_value, DType destination_dtype_value,
                  std::vector<int64_t> destination_shape_value,
                  std::string component_value, std::string role_value,
                  Json quantization_value = {})
        : source_name(std::move(source_name_value)), source_dtype(source_dtype_value),
          source_shape(std::move(source_shape_value)),
          transformation(std::move(transformation_value)),
          destination_name(std::move(destination_name_value)),
          destination_dtype(destination_dtype_value),
          destination_shape(std::move(destination_shape_value)),
          component(std::move(component_value)), role(std::move(role_value)),
          quantization(std::move(quantization_value)) {}

    std::string source_name;
    DType source_dtype = DType::F32;
    std::vector<int64_t> source_shape;
    std::string transformation;
    std::string destination_name;
    DType destination_dtype = DType::F32;
    std::vector<int64_t> destination_shape;
    std::string component;
    std::string role;
    // Null preserves the canonical v0.1 unquantized tensor-table bytes.
    // A non-null object is emitted verbatim and is admitted fail-closed by the
    // generic VRM loader.
    Json quantization;
};

struct VrmWriteResult {
    uint64_t file_size = 0;
    uint64_t tensor_count = 0;
    uint64_t largest_buffer_bytes = 0;
    uint64_t metadata_bytes = 0;
    uint64_t tensor_table_bytes = 0;
    uint64_t graph_bytes = 0;
    uint64_t data_offset = 0;
    double seconds = 0.0;
    std::string payload_blake2b128;
};

struct FrozenComponentConversionResult {
    std::filesystem::path output_path;
    uint64_t source_checkpoint_bytes = 0;
    uint64_t retained_tensor_bytes = 0;
    uint64_t retained_tensor_count = 0;
    uint64_t output_bytes = 0;
    uint64_t largest_temporary_buffer_bytes = 0;
    double conversion_seconds = 0.0;
    std::string output_sha256;
    std::string payload_blake2b128;
};

// Fixed-source Native conversion for the generic Whisper Tiny AudioEncoder
// component used by research qualification. This does not register a MuseTalk
// product or package and retains no decoder/tokenizer tensors.
FrozenComponentConversionResult convert_whisper_tiny_encoder_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

// Fixed-source conversion for the OpenAI-format FP16 Whisper Tiny checkpoint
// used by temporal lip-sync workflows. The inert PyTorch ZIP is structurally
// validated and only the encoder tensors are retained as FP32. The graph stays
// in the generic AudioEncoder family and pickle is never executed.
FrozenComponentConversionResult convert_openai_whisper_tiny_encoder_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& tensor_map,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

// Fixed-source Native conversion for the generic SD image AutoencoderKL
// component used by MuseTalk research qualification. No MuseTalk product or
// model identity enters the component graph or Runtime.
FrozenComponentConversionResult convert_sd_vae_ft_mse_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& tensor_map,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

// Fixed-source Native conversion for a generic conditional 2D UNet component.
// The source topology is frozen by converter data; no upstream serialization
// framework or model-specific Runtime is involved.
FrozenComponentConversionResult convert_musetalk_v15_unet_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& tensor_map,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

// Fixed-source Native conversion for a generic temporal conditional UNet.
// The PyTorch ZIP64 container is parsed as inert data; pickle globals are
// validated but never resolved or executed.
FrozenComponentConversionResult convert_temporal_conditional_unet_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& tensor_map,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

// Fixed-source Native conversion for a generic multi-scale vision detector.
// The component emits raw neural heads; ROI policy remains bounded host work.
FrozenComponentConversionResult convert_s3fd_face_detector_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& tensor_map,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

// Fixed-source conversion for a generic dense-anchor vision detector. The
// bounded TFLite reader validates and extracts constants; TensorFlow Lite is
// never linked or executed in production.
FrozenComponentConversionResult convert_blazeface_short_range_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& tensor_map,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

// Fixed-source Native conversion for a generic top-down 2D pose estimator.
// Affine preprocessing, SimCC decode, and face geometry remain host workflow.
FrozenComponentConversionResult convert_dwpose_keypoint_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& tensor_map,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

// Fixed-source Native conversion for a generic 2D semantic segmenter. The
// component emits class logits; product mask policy remains bounded host work.
FrozenComponentConversionResult convert_bisenet_face_parser_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& tensor_map,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

// Fixed-source Native conversion for a generic six-class 2D semantic
// segmenter. The bounded TFLite reader supplies only audited graph/constant
// data; no TensorFlow Lite execution is part of the production path.
FrozenComponentConversionResult convert_selfie_multiclass_component(
    const std::filesystem::path& source_directory,
    const std::filesystem::path& tensor_map,
    const std::filesystem::path& output,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

VrmWriteResult write_vrm_streaming(
    const std::filesystem::path& output,
    const std::string& profile_id,
    const std::string& architecture_id,
    const Json& metadata,
    const Json& graph,
    const TensorSource& source,
    const std::vector<TensorMapping>& mappings,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

struct SafeTensorWriteResult {
    uint64_t file_size = 0;
    uint64_t tensor_count = 0;
    uint64_t largest_buffer_bytes = 0;
};

SafeTensorWriteResult write_safetensors_streaming(
    const std::filesystem::path& output,
    const TensorSource& source,
    const std::vector<TensorMapping>& mappings,
    const std::vector<std::pair<std::string, std::string>>& metadata,
    const std::function<bool()>& cancellation_requested = {},
    const WorkProgressCallback& progress = {});

std::string canonical_json(const Json& value);

void require_conversion_disk_space(uint64_t required_bytes,
                                   uint64_t available_bytes,
                                   uint64_t safety_margin_bytes = 64ULL * 1024 * 1024,
                                   const std::filesystem::path& cache_root = {});

struct ImportOptions {
    std::filesystem::path converter_spec_root;
    std::filesystem::path package_manifest;
    std::string expected_model_reference;
    std::function<bool()> cancellation_requested;
    std::function<void(uint64_t, uint64_t)> progress;
    std::function<void(uint64_t, uint64_t)> finalization_progress;
    std::optional<uint64_t> available_space_override;
};

struct ImportResult {
    InstallResult installation;
    std::filesystem::path runtime_vrm_path;
    uint64_t source_checkpoint_bytes = 0;
    uint64_t output_vrm_bytes = 0;
    uint64_t temporary_disk_peak_bytes = 0;
    uint64_t largest_temporary_buffer_bytes = 0;
    uint64_t mapping_count = 0;
    uint64_t copied_artifact_bytes = 0;
    uint64_t reused_artifact_bytes = 0;
    bool conversion_performed = false;
    double conversion_seconds = 0.0;
    double finalization_seconds = 0.0;
    std::string runtime_vrm_sha256;
};

// Product-layer converter registry. Architecture identity is resolved here,
// never in Shared Runtime, CUDA, or PrecisionPolicy.
ImportResult import_local_model(const std::string& catalog_reference,
                                const std::filesystem::path& source_directory,
                                LocalModelCache& cache,
                                const ImportOptions& options = {});

ImportResult import_private_musetalk_v15(
    const std::filesystem::path& source_directory,
    LocalModelCache& cache,
    const ImportOptions& options = {});
ImportResult import_private_musetalk_v15_successor(
    const std::filesystem::path& source_directory,
    LocalModelCache& cache,
    const ImportOptions& options = {});
ImportResult import_public_musetalk_v15(
    const std::filesystem::path& source_directory,
    LocalModelCache& cache,
    const ImportOptions& options = {});
ImportResult import_private_latentsync_16(
    const std::filesystem::path& source_directory,
    LocalModelCache& cache,
    const ImportOptions& options = {});
ImportResult import_public_latentsync_16(
    const std::filesystem::path& source_directory,
    LocalModelCache& cache,
    const ImportOptions& options = {});

std::filesystem::path discover_converter_spec_root(
    const std::filesystem::path& executable_path = {});

}  // namespace vrhino::product
