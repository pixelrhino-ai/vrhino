#include "vrhino/bundle.h"

#include <array>
#include <cstring>
#include <fstream>

#include "vrhino/error.h"

namespace vrhino {
namespace {

constexpr std::array<char, 8> kMagic = {'V','R','I','N','P','U','T','1'};

template <typename T> void read_exact(std::istream& input, T* value, size_t count = 1) {
    input.read(reinterpret_cast<char*>(value), static_cast<std::streamsize>(sizeof(T) * count));
    require(input.good(), "Truncated tensor bundle");
}
template <typename T> void write_exact(std::ostream& output, const T* value, size_t count = 1) {
    output.write(reinterpret_cast<const char*>(value), static_cast<std::streamsize>(sizeof(T) * count));
    require(output.good(), "Tensor bundle write failed");
}

}  // namespace

TensorBundle read_bundle(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "Cannot open tensor bundle: " + path);
    std::array<char, 8> magic{}; read_exact(input, magic.data(), magic.size());
    require(magic == kMagic, "Bad tensor bundle magic");
    uint32_t count = 0; read_exact(input, &count);
    require(count <= 1024, "Tensor bundle count is unreasonable");
    TensorBundle result;
    for (uint32_t index = 0; index < count; ++index) {
        uint16_t name_length = 0; read_exact(input, &name_length);
        require(name_length > 0 && name_length <= 1024, "Invalid bundle tensor name length");
        std::string name(name_length, '\0'); read_exact(input, name.data(), name.size());
        uint8_t dtype_raw = 0, rank = 0; read_exact(input, &dtype_raw); read_exact(input, &rank);
        require(dtype_raw <= static_cast<uint8_t>(DType::Bool) && rank <= 12, "Invalid bundle tensor header");
        std::vector<int64_t> shape(rank); if (rank) read_exact(input, shape.data(), shape.size());
        uint64_t byte_length = 0; read_exact(input, &byte_length);
        const DType dtype = static_cast<DType>(dtype_raw);
        require(byte_length == static_cast<uint64_t>(shape_numel(shape)) * dtype_size(dtype), "Bundle byte length mismatch");
        Tensor tensor = Tensor::host(shape, dtype);
        if (byte_length) read_exact(input, tensor.data_as<uint8_t>(), byte_length);
        require(result.emplace(std::move(name), std::move(tensor)).second, "Duplicate bundle tensor name");
    }
    require(input.peek() == std::char_traits<char>::eof(), "Tensor bundle has trailing data");
    return result;
}

void write_bundle(const std::string& path, const TensorBundle& bundle) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(output.good(), "Cannot create tensor bundle: " + path);
    write_exact(output, kMagic.data(), kMagic.size());
    const uint32_t count = static_cast<uint32_t>(bundle.size()); write_exact(output, &count);
    for (const auto& [name, tensor] : bundle) {
        require(tensor.device() == Device::CPU, "Bundle writer requires CPU tensor");
        require(name.size() <= UINT16_MAX, "Bundle tensor name too long");
        const uint16_t name_length = static_cast<uint16_t>(name.size()); write_exact(output, &name_length);
        write_exact(output, name.data(), name.size());
        const uint8_t dtype = static_cast<uint8_t>(tensor.dtype());
        const uint8_t rank = static_cast<uint8_t>(tensor.ndim());
        write_exact(output, &dtype); write_exact(output, &rank);
        if (rank) write_exact(output, tensor.shape().data(), tensor.shape().size());
        const uint64_t bytes = tensor.bytes(); write_exact(output, &bytes);
        if (bytes) write_exact(output, tensor.data_as<uint8_t>(), bytes);
    }
}

}  // namespace vrhino
