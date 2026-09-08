#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/product/converter.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void put_u64(std::ostream& output, const uint64_t value) {
    std::array<char, 8> bytes{};
    for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<char>((value >> (8 * index)) & 0xff);
    output.write(bytes.data(), bytes.size());
}

void write_safe(const fs::path& path, std::string header,
                const std::vector<uint8_t>& payload) {
    while (header.size() % 8 != 0) header.push_back(' ');
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    put_u64(output, header.size());
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
}

template <typename Operation>
void expect_code(const product::ModelPackageErrorCode expected,
                 Operation&& operation, const std::string& context) {
    try {
        operation();
        throw std::runtime_error(context + ": operation unexpectedly succeeded");
    } catch (const product::ModelPackageError& error) {
        require_test(error.code() == expected, context + ": wrong error code");
    }
}

vrhino::Json object(std::initializer_list<std::pair<const std::string, vrhino::Json>> values) {
    vrhino::Json::Object result;
    for (const auto& [key, value] : values) result.emplace(key, value);
    return vrhino::Json(vrhino::Json::Value(std::move(result)));
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-native-converter-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root);
    try {
        const fs::path valid = root / "valid.safetensors";
        write_safe(valid,
            "{\"__metadata__\":{\"config\":\"{}\"},\"x\":{\"data_offsets\":[0,2],"
            "\"dtype\":\"BF16\",\"shape\":[1]}}",
            {0x34, 0x12});
        product::SafeTensorReader reader(valid);
        require_test(reader.tensors().size() == 1, "valid tensor count");
        require_test(reader.tensors().at("x").shape == std::vector<int64_t>({1}),
                     "valid tensor shape");
        std::array<uint8_t, 2> bytes{};
        reader.read_tensor(reader.tensors().at("x"), 0, bytes.data(), bytes.size());
        require_test(bytes[0] == 0x34 && bytes[1] == 0x12, "streamed tensor bytes");

        const fs::path signed_i8 = root / "signed-i8.safetensors";
        write_safe(signed_i8,
            "{\"x\":{\"data_offsets\":[0,3],\"dtype\":\"I8\",\"shape\":[3]}}",
            {0x81, 0x00, 0x7f});
        product::SafeTensorReader signed_reader(signed_i8);
        require_test(signed_reader.tensors().at("x").dtype == vrhino::DType::I8 &&
                         signed_reader.tensors().at("x").byte_length == 3,
                     "signed-I8 safetensors admission");

        expect_code(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            product::SafeTensorReader missing(root / "missing.safetensors");
        }, "missing source");

        const fs::path corrupt = root / "corrupt.safetensors";
        { std::ofstream output(corrupt, std::ios::binary); put_u64(output, 999999); }
        expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::SafeTensorReader value(corrupt);
        }, "corrupt header");

        const fs::path malformed = root / "malformed.safetensors";
        write_safe(malformed, "{not-json", {});
        expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::SafeTensorReader value(malformed);
        }, "malformed header JSON");

        const fs::path truncated = root / "truncated.safetensors";
        write_safe(truncated,
            "{\"x\":{\"data_offsets\":[0,4],\"dtype\":\"F32\",\"shape\":[1]}}",
            {0, 0});
        expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::SafeTensorReader value(truncated);
        }, "truncated payload");

        const fs::path unsupported = root / "unsupported.safetensors";
        write_safe(unsupported,
            "{\"x\":{\"data_offsets\":[0,1],\"dtype\":\"F8_E4M3\",\"shape\":[1]}}",
            {0});
        expect_code(product::ModelPackageErrorCode::PackageVersionUnsupported, [&] {
            product::SafeTensorReader value(unsupported);
        }, "unsupported dtype");

        const product::TensorMapping mapping{
            "x", vrhino::DType::BF16, {1}, "identity_bytes", "denoiser.x",
            vrhino::DType::BF16, {1}, "denoiser", "weight"};
        const vrhino::Json metadata = object({
            {"architecture", vrhino::Json(vrhino::Json::Value(std::string("ltx_v0_9_1")))},
        });
        const vrhino::Json graph = object({
            {"schema_version", vrhino::Json(vrhino::Json::Value(int64_t{1}))},
        });
        const fs::path output = root / "small.vrm";
        uint64_t converted_work = 0;
        std::vector<uint64_t> conversion_samples;
        const product::VrmWriteResult written = product::write_vrm_streaming(
            output, "dit-flow", "ltx_v0_9_1", metadata, graph, reader, {mapping}, {},
            [&](const uint64_t bytes_written) {
                require_test(bytes_written > 0, "conversion progress did not advance");
                converted_work += bytes_written;
                conversion_samples.push_back(converted_work);
            });
        require_test(written.tensor_count == 1, "small VRM tensor count");
        require_test(!conversion_samples.empty() &&
                         std::is_sorted(conversion_samples.begin(),
                                        conversion_samples.end()),
                     "conversion progress was not monotonic");
        require_test(converted_work == written.file_size,
                     "conversion progress does not reflect written output bytes");
        const vrhino::VrmModel model(output.string(), true);
        require_test(model.tensor("denoiser.x").data_as<const uint8_t>()[0] == 0x34,
                     "small VRM payload");

        const fs::path frozen_bytes = root / "fixed-source.bin";
        {
            std::ofstream raw(frozen_bytes, std::ios::binary | std::ios::trunc);
            const std::array<uint8_t, 4> payload{0xaa, 0xbb, 0x34, 0x12};
            raw.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        }
        std::map<std::string, product::SourceTensorDescriptor> frozen_tensors;
        frozen_tensors.emplace("fixed:x", product::SourceTensorDescriptor{
            "fixed:x", vrhino::DType::BF16, "BF16", {1}, 0, 2, 2});
        product::FrozenTensorSource frozen({frozen_bytes}, std::move(frozen_tensors));
        const product::TensorMapping frozen_mapping{
            "fixed:x", vrhino::DType::BF16, {1}, "identity_bytes", "x",
            vrhino::DType::BF16, {1}, "conditioning", "weight"};
        const fs::path safe_output = root / "fixed-output.safetensors";
        uint64_t safe_work = 0;
        const product::SafeTensorWriteResult safe_written =
            product::write_safetensors_streaming(safe_output, frozen,
                {frozen_mapping}, {{"format", "pt"}}, {},
                [&](const uint64_t bytes_written) { safe_work += bytes_written; });
        require_test(safe_work == safe_written.file_size,
                     "safetensors conversion progress did not reflect real writes");
        product::SafeTensorReader safe_reader(safe_output);
        std::array<uint8_t, 2> safe_bytes{};
        safe_reader.read_tensor(safe_reader.tensors().at("x"), 0,
                                safe_bytes.data(), safe_bytes.size());
        require_test(safe_bytes[0] == 0x34 && safe_bytes[1] == 0x12,
                     "fixed-range tensor bytes were not preserved");

        const fs::path cancelled_safe = root / "cancelled.safetensors";
        expect_code(product::ModelPackageErrorCode::Cancelled, [&] {
            product::write_safetensors_streaming(cancelled_safe, frozen,
                {frozen_mapping}, {{"format", "pt"}}, [] { return true; });
        }, "safetensors conversion interruption");
        require_test(!fs::exists(cancelled_safe),
                     "cancelled safetensors output was removed");

        product::TensorMapping wrong_shape = mapping;
        wrong_shape.source_shape = {2};
        expect_code(product::ModelPackageErrorCode::PackageVersionUnsupported, [&] {
            product::write_vrm_streaming(root / "wrong-shape.vrm", "dit-flow",
                "ltx_v0_9_1", metadata, graph, reader, {wrong_shape});
        }, "unexpected tensor shape");

        const fs::path cancelled = root / "cancelled.vrm";
        uint64_t cancelled_work = 0;
        expect_code(product::ModelPackageErrorCode::Cancelled, [&] {
            product::write_vrm_streaming(cancelled, "dit-flow", "ltx_v0_9_1",
                metadata, graph, reader, {mapping}, [] { return true; },
                [&](const uint64_t bytes_written) { cancelled_work += bytes_written; });
        }, "conversion interruption");
        require_test(!fs::exists(cancelled), "cancelled output was removed");
        require_test(cancelled_work > 0 && cancelled_work < written.file_size,
                     "cancelled conversion reported false completion");

        const fs::path parent_file = root / "not-a-directory";
        { std::ofstream file(parent_file); file << "x"; }
        expect_code(product::ModelPackageErrorCode::CacheError, [&] {
            product::write_vrm_streaming(parent_file / "output.vrm", "dit-flow",
                "ltx_v0_9_1", metadata, graph, reader, {mapping});
        }, "unwritable destination");

        expect_code(product::ModelPackageErrorCode::InsufficientDiskSpace, [&] {
            product::require_conversion_disk_space(1024, 1024, 1);
        }, "insufficient disk");

        expect_code(product::ModelPackageErrorCode::PackageVersionUnsupported, [&] {
            product::LocalModelCache cache(root / "cache");
            product::import_local_model("vrhino/unknown:1", root, cache);
        }, "incompatible catalog model");

        fs::remove_all(root, error);
        std::cout << "native converter tests: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove_all(root, error);
        std::cerr << "native converter tests: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
