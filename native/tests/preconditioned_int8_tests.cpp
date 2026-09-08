#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/product/converter.h"
#include "vrhino/quantization/reference.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void rejects(Operation&& operation, const std::string& context) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(context + " unexpectedly succeeded");
}

class MemorySource final : public product::TensorSource {
public:
    void add(const std::string& name, vrhino::DType dtype,
             std::vector<int64_t> shape, std::vector<uint8_t> bytes) {
        const uint64_t length = bytes.size();
        descriptors_.emplace(name, product::SourceTensorDescriptor{
            name, dtype, vrhino::dtype_name(dtype), std::move(shape), 0, 0, length});
        bytes_.emplace(name, std::move(bytes));
    }
    const std::map<std::string, product::SourceTensorDescriptor>& tensors()
            const noexcept override { return descriptors_; }
    void read_tensor(const product::SourceTensorDescriptor& tensor,
                     uint64_t offset, void* destination, size_t bytes) const override {
        const auto& source = bytes_.at(tensor.name);
        check(offset <= source.size() && bytes <= source.size() - offset,
              "memory source range");
        std::memcpy(destination, source.data() + offset, bytes);
    }
private:
    std::map<std::string, product::SourceTensorDescriptor> descriptors_;
    std::map<std::string, std::vector<uint8_t>> bytes_;
};

std::vector<uint8_t> float_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> result(values.size() * sizeof(float));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

vrhino::Json quantization(
        const std::string& scale = "scale", const std::string& scheme =
            vrhino::kPreconditionedSymmetricInt8V1,
        const std::string& preconditioner = vrhino::kRegularBlockHadamardV1,
        int64_t scale_group = 512, int64_t transform_group = 256,
        const std::string& logical_shape = "[2,512]",
        const std::string& stored_shape = "[2,512]",
        const std::string& extra = "", const std::string& compute_dtype = "f32",
        const std::string& type = "int8_symmetric") {
    return vrhino::Json::parse(
        "{\"accumulation_dtype\":\"f32\",\"axis\":1,\"block_size\":0,"
        "\"compute_dtype\":\"" + compute_dtype + "\",\"dequantization\":"
        "\"scale_then_inverse_preconditioner\",\"granularity\":\"per_channel\","
        "\"group_size\":" + std::to_string(scale_group) +
        ",\"logical_shape\":" + logical_shape +
        ",\"original_dtype\":\"bf16\",\"packing_layout\":"
        "\"byte_twos_complement\",\"preconditioner\":{\"axis\":1,"
        "\"group_size\":" + std::to_string(transform_group) +
        ",\"identity\":\"" + preconditioner +
        "\",\"storage_basis\":\"preconditioned\"},\"scale_dtype\":\"f32\","
        "\"scale_tensor\":\"" + scale + "\",\"scheme\":\"" + scheme +
        "\",\"storage_dtype\":\"i8\",\"stored_shape\":" + stored_shape +
        ",\"symmetric\":true,\"type\":\"" + type + "\","
        "\"zero_point_mode\":\"none\",\"zero_point_tensor\":null" + extra + "}");
}

vrhino::Json metadata() {
    return vrhino::Json::parse("{\"architecture\":\"component-test\"}");
}
vrhino::Json graph() {
    return vrhino::Json::parse("{\"schema_version\":1}");
}

fs::path write_fixture(const fs::path& root, const std::string& name,
                       const vrhino::Json& quant, vrhino::DType scale_dtype = vrhino::DType::F32,
                       std::vector<int64_t> scale_shape = {2, 1}, bool include_scale = true,
                       bool duplicate_association = false, int64_t columns = 512,
                       std::vector<float> scale_values = {0.25f, 1.5f}) {
    MemorySource source;
    std::vector<uint8_t> packed(static_cast<size_t>(2 * columns));
    for (size_t index = 0; index < packed.size(); ++index) {
        const int value = static_cast<int>((index * 37 + index / columns * 11) % 255) - 127;
        packed[index] = static_cast<uint8_t>(static_cast<int8_t>(value));
    }
    packed[0] = 0;
    packed[1] = static_cast<uint8_t>(static_cast<int8_t>(-127));
    packed[2] = 127;
    source.add("weight", vrhino::DType::I8, {2, columns}, packed);
    if (duplicate_association) source.add("weight2", vrhino::DType::I8, {2, columns}, packed);
    if (include_scale) {
        std::vector<uint8_t> scales = scale_dtype == vrhino::DType::F32
            ? float_bytes(scale_values) : std::vector<uint8_t>{1, 2};
        source.add("scale", scale_dtype, scale_shape, std::move(scales));
    }
    std::vector<product::TensorMapping> mappings{{
        "weight", vrhino::DType::I8, {2, columns}, "identity_bytes", "weight",
        vrhino::DType::I8, {2, columns}, "test", "weight", quant}};
    if (duplicate_association) mappings.push_back({
        "weight2", vrhino::DType::I8, {2, columns}, "identity_bytes", "weight2",
        vrhino::DType::I8, {2, columns}, "test", "weight", quant});
    if (include_scale) mappings.push_back({
        "scale", scale_dtype, scale_shape, "identity_bytes", "scale",
        scale_dtype, scale_shape, "test", "quantization_scale"});
    const fs::path output = root / (name + ".vrm");
    product::write_vrm_streaming(output, "component", "component-test",
                                 metadata(), graph(), source, mappings);
    return output;
}

std::vector<int> independent_hadamard(int64_t size) {
    static constexpr int h4[4][4] = {
        {1,1,1,-1}, {1,1,-1,1}, {1,-1,1,1}, {-1,1,1,1}};
    std::vector<int> matrix{1};
    int64_t current = 1;
    while (current < size) {
        std::vector<int> expanded(static_cast<size_t>(current * 4 * current * 4));
        for (int64_t row = 0; row < current; ++row)
            for (int64_t column = 0; column < current; ++column)
                for (int64_t seed_row = 0; seed_row < 4; ++seed_row)
                    for (int64_t seed_column = 0; seed_column < 4; ++seed_column)
                        expanded[static_cast<size_t>((row * 4 + seed_row) *
                            (current * 4) + column * 4 + seed_column)] =
                            matrix[static_cast<size_t>(row * current + column)] *
                            h4[seed_row][seed_column];
        matrix = std::move(expanded);
        current *= 4;
    }
    return matrix;
}

void validate_rank_admission(const fs::path& root) {
    // All shape declarations agree, and flattening the trailing axes would
    // satisfy the old scale/group checks: K=512, two blocks per output row.
    const std::vector<int64_t> shape{2, 2, 256};
    MemorySource source;
    source.add("weight", vrhino::DType::I8, shape, std::vector<uint8_t>(1024, 1));
    source.add("scale", vrhino::DType::F32, {2, 1}, float_bytes({1.0f, 1.0f}));
    const auto quant = quantization("scale", vrhino::kPreconditionedSymmetricInt8V1,
        vrhino::kRegularBlockHadamardV1, 512, 256, "[2,2,256]", "[2,2,256]");
    const fs::path path = root / "matched-rank-3.vrm";
    product::write_vrm_streaming(path, "component", "component-test", metadata(),
        graph(), source, {
            {"weight", vrhino::DType::I8, shape, "identity_bytes", "weight",
             vrhino::DType::I8, shape, "test", "weight", quant},
            {"scale", vrhino::DType::F32, {2,1}, "identity_bytes", "scale",
             vrhino::DType::F32, {2,1}, "test", "quantization_scale"}});
    bool rejected = false;
    try {
        vrhino::VrmModel model(path.string(), true);
    } catch (const vrhino::Error& error) {
        check(std::string(error.what()) == "Preconditioned symmetric INT8 v1 requires rank-2 weights",
              "matched rank-3 tensor rejected for an unrelated reason");
        rejected = true;
    }
    check(rejected, "matched rank-3 tensor unexpectedly admitted");

    // The new executable-domain restriction must not become a global rank rule.
    for (const auto dtype : {vrhino::DType::F32, vrhino::DType::F16, vrhino::DType::BF16}) {
        MemorySource dense;
        const std::vector<uint8_t> bytes(1024 * vrhino::dtype_size(dtype), 0);
        dense.add("weight", dtype, shape, bytes);
        const fs::path dense_path = root / (vrhino::dtype_name(dtype) + "-rank-3.vrm");
        product::write_vrm_streaming(dense_path, "component", "component-test",
            metadata(), graph(), dense, {{"weight", dtype, shape, "identity_bytes",
                "weight", dtype, shape, "test", "weight"}});
        vrhino::VrmModel model(dense_path.string(), true);
        const auto& weight = model.tensor("weight");
        check(weight.shape() == shape && weight.dtype() == dtype && !weight.is_quantized() &&
                  weight.bytes() == bytes.size() &&
                  std::memcmp(weight.data(), bytes.data(), bytes.size()) == 0,
              "legacy dense rank-3 admission changed");
    }
}

void validate_golden(const fs::path& path) {
    vrhino::VrmModel model(path.string(), true);
    const vrhino::Tensor& weight = model.tensor("weight");
    check(weight.dtype() == vrhino::DType::I8 && weight.logical_dtype() == vrhino::DType::BF16,
          "storage/logical dtype separation");
    check(weight.memory_domain() == vrhino::MemoryDomain::HostPageable &&
              !weight.owns_storage() && weight.bytes() == 1024,
          "packed mmap storage preservation");
    const void* original_pointer = weight.data();
    const vrhino::Tensor first =
        vrhino::materialize_preconditioned_symmetric_int8_f32(weight, 0, 2, 0, 2);
    const vrhino::Tensor repeated =
        vrhino::materialize_preconditioned_symmetric_int8_f32(weight, 0, 2, 0, 2);
    check(first.dtype() == vrhino::DType::F32 && first.shape() == weight.shape(),
          "reference result shape/dtype");
    check(std::memcmp(first.data(), repeated.data(), first.bytes()) == 0,
          "deterministic repeated materialization");
    const auto* q = weight.data_as<const int8_t>();
    for (int64_t index = 0; index < 1024; ++index) {
        const int expected = index == 0 ? 0 : index == 1 ? -127 : index == 2 ? 127 :
            static_cast<int>((index * 37 + index / 512 * 11) % 255) - 127;
        check(q[index] == expected, "packed INT8 round-trip mismatch");
    }
    const float scales[2] = {0.25f, 1.5f};
    const auto* actual = first.data_as<const float>();
    const std::vector<int> hadamard = independent_hadamard(256);
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t block = 0; block < 2; ++block) {
            for (int64_t column = 0; column < 256; ++column) {
                double expected = 0.0;
                for (int64_t rotated = 0; rotated < 256; ++rotated)
                    expected += static_cast<double>(q[row * 512 + block * 256 + rotated]) *
                                scales[row] * hadamard[rotated * 256 + column] / 16.0;
                check(actual[row * 512 + block * 256 + column] ==
                          static_cast<float>(expected), "synthetic golden mismatch");
            }
        }
    }
    const vrhino::Tensor one_block =
        vrhino::materialize_preconditioned_symmetric_int8_f32(weight, 1, 1, 1, 1);
    check(one_block.bytes() == 256 * sizeof(float) && weight.data() == original_pointer,
          "bounded block materialization changed packed storage");
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-preconditioned-int8-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root);
    try {
        const fs::path valid = write_fixture(root, "valid", quantization());
        validate_golden(valid);
        validate_rank_admission(root);

        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "unknown-scheme", quantization("scale", "unknown.v1")).string(), false); },
            "unknown scheme");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "scheme-with-none", quantization("scale",
                vrhino::kPreconditionedSymmetricInt8V1,
                vrhino::kRegularBlockHadamardV1, 512, 256, "[2,512]", "[2,512]",
                "", "f32", "none")).string(), false); },
            "scheme with unquantized storage");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "unknown-preconditioner", quantization(
                "scale", vrhino::kPreconditionedSymmetricInt8V1, "unknown.v1")).string(), false); },
            "unknown preconditioner");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "unsupported-compute", quantization("scale",
                vrhino::kPreconditionedSymmetricInt8V1,
                vrhino::kRegularBlockHadamardV1, 512, 256, "[2,512]", "[2,512]",
                "", "bf16")).string(), false); },
            "unsupported compute dtype");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "missing-scale", quantization(), vrhino::DType::F32, {2,1}, false).string(), false); },
            "missing scale");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "wrong-scale-dtype", quantization(), vrhino::DType::I8).string(), false); },
            "wrong scale dtype");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "wrong-scale-shape", quantization(), vrhino::DType::F32, {1,2}).string(), false); },
            "wrong scale shape");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "wrong-scale-rank", quantization(), vrhino::DType::F32, {2}).string(), false); },
            "wrong scale rank");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "nonfinite-scale", quantization(), vrhino::DType::F32, {2,1},
                true, false, 512,
                {std::numeric_limits<float>::quiet_NaN(), 1.0f}).string(), false); },
            "non-finite scale");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "zero-scale", quantization(), vrhino::DType::F32, {2,1},
                true, false, 512, {0.0f, 1.0f}).string(), false); },
            "zero scale");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "wrong-logical-shape", quantization("scale",
                vrhino::kPreconditionedSymmetricInt8V1, vrhino::kRegularBlockHadamardV1,
                512, 256, "[1,1024]")).string(), false); }, "wrong logical shape");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "wrong-stored-shape", quantization("scale",
                vrhino::kPreconditionedSymmetricInt8V1, vrhino::kRegularBlockHadamardV1,
                512, 256, "[2,512]", "[1,1024]")).string(), false); },
            "wrong stored shape");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "invalid-scale-group", quantization("scale",
                vrhino::kPreconditionedSymmetricInt8V1,
                vrhino::kRegularBlockHadamardV1, 256)).string(), false); },
            "invalid scale group");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "invalid-transform-group", quantization("scale",
                vrhino::kPreconditionedSymmetricInt8V1,
                vrhino::kRegularBlockHadamardV1, 512, 128)).string(), false); },
            "invalid transform group");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "metadata-version", quantization("scale",
                vrhino::kPreconditionedSymmetricInt8V1,
                vrhino::kRegularBlockHadamardV1, 512, 256, "[2,512]", "[2,512]",
                ",\"version\":2")).string(), false); }, "unknown executable metadata");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "duplicate-association", quantization(), vrhino::DType::F32,
                {2,1}, true, true).string(), false); }, "duplicate association");
        rejects([&] { vrhino::VrmModel value(write_fixture(
            root, "incompatible-block", quantization("scale",
                vrhino::kPreconditionedSymmetricInt8V1,
                vrhino::kRegularBlockHadamardV1, 500, 256,
                "[2,500]", "[2,500]"), vrhino::DType::F32, {2,1}, true, false,
                500).string(), false); }, "incompatible block semantics");

        const fs::path truncated = root / "truncated.vrm";
        fs::copy_file(valid, truncated);
        fs::resize_file(truncated, fs::file_size(truncated) - 1);
        rejects([&] { vrhino::VrmModel value(truncated.string(), false); }, "truncated payload");
        rejects([] { (void)vrhino::shape_numel({std::numeric_limits<int64_t>::max(), 2}); },
                "tensor element-count overflow");
        rejects([&] {
            MemorySource source;
            source.add("overflow", vrhino::DType::F32,
                       {std::numeric_limits<int64_t>::max(), 2}, {0});
            const product::TensorMapping mapping{
                "overflow", vrhino::DType::F32,
                {std::numeric_limits<int64_t>::max(), 2}, "identity_bytes", "overflow",
                vrhino::DType::F32, {std::numeric_limits<int64_t>::max(), 2},
                "test", "weight"};
            product::write_vrm_streaming(root / "payload-overflow.vrm", "component",
                "component-test", metadata(), graph(), source, {mapping});
        }, "payload-size overflow");
        std::vector<int8_t> bare_bytes(256);
        const vrhino::Tensor bare = vrhino::Tensor::borrowed(
            bare_bytes.data(), bare_bytes.size(), {1,256}, vrhino::DType::I8);
        rejects([&] { (void)vrhino::materialize_preconditioned_symmetric_int8_f32(
            bare, 0, 1, 0, 1); }, "bare I8 arithmetic");

        fs::remove_all(root, error);
        std::cout << "preconditioned symmetric INT8 tests: PASS\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "preconditioned symmetric INT8 tests: FAIL: " << failure.what() << '\n';
        fs::remove_all(root, error);
        return 1;
    }
}
