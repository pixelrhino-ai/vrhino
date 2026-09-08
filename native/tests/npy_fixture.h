#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vrhino/tensor.h"

namespace vrhino::test {

inline float binary16_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fU;
    const uint32_t fraction = value & 0x03ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (fraction == 0) bits = sign;
        else {
            uint32_t normalized = fraction;
            int shift = 0;
            while ((normalized & 0x0400U) == 0) { normalized <<= 1; ++shift; }
            normalized &= 0x03ffU;
            bits = sign | static_cast<uint32_t>(127 - 14 - shift) << 23 |
                   normalized << 13;
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | fraction << 13;
    } else {
        bits = sign | (exponent + (127 - 15)) << 23 | fraction << 13;
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline std::pair<std::vector<int64_t>, std::string> read_npy_header(
        std::ifstream& input, const std::string& path) {
    uint8_t prefix[10]{};
    input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
    if (!input || std::memcmp(prefix, "\x93NUMPY", 6) != 0 ||
        prefix[6] != 1 || prefix[7] != 0)
        throw std::runtime_error("unsupported NPY fixture header: " + path);
    const uint16_t header_size = static_cast<uint16_t>(prefix[8]) |
                                 (static_cast<uint16_t>(prefix[9]) << 8);
    std::string header(header_size, '\0');
    input.read(header.data(), header.size());
    if (!input || header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("NPY fixture must be C-order: " + path);
    const size_t shape_key = header.find("'shape':");
    const size_t begin = header.find('(', shape_key);
    const size_t end = header.find(')', begin);
    if (shape_key == std::string::npos || begin == std::string::npos ||
        end == std::string::npos)
        throw std::runtime_error("NPY fixture shape is missing: " + path);
    std::vector<int64_t> shape;
    std::istringstream fields(header.substr(begin + 1, end - begin - 1));
    std::string field;
    while (std::getline(fields, field, ',')) {
        field.erase(std::remove_if(field.begin(), field.end(),
            [](unsigned char item) { return std::isspace(item); }), field.end());
        if (!field.empty()) shape.push_back(std::stoll(field));
    }
    if (shape.empty()) throw std::runtime_error("NPY fixture shape is empty: " + path);
    return {shape, header};
}

inline Tensor read_npy_f32(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open NPY fixture: " + path);
    const auto [shape, header] = read_npy_header(input, path);
    if (!input || header.find("'descr': '<f4'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("NPY fixture must be C-order little-endian FP32: " + path);
    Tensor tensor = Tensor::host(shape, DType::F32);
    input.read(static_cast<char*>(tensor.data()), static_cast<std::streamsize>(tensor.bytes()));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("NPY fixture payload length mismatch: " + path);
    return tensor;
}

inline Tensor read_npy_f16_as_f32(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open NPY fixture: " + path);
    const auto [shape, header] = read_npy_header(input, path);
    if (header.find("'descr': '<f2'") == std::string::npos)
        throw std::runtime_error(
            "NPY fixture must be C-order little-endian FP16: " + path);
    Tensor output = Tensor::host(shape, DType::F32);
    std::vector<uint16_t> source(static_cast<size_t>(output.numel()));
    input.read(reinterpret_cast<char*>(source.data()),
               static_cast<std::streamsize>(source.size() * sizeof(uint16_t)));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("NPY fixture payload length mismatch: " + path);
    for (size_t index = 0; index < source.size(); ++index)
        output.data_as<float>()[index] = binary16_to_float(source[index]);
    return output;
}

inline Tensor read_npy_f64_as_f32(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open NPY fixture: " + path);
    const auto [shape, header] = read_npy_header(input, path);
    if (header.find("'descr': '<f8'") == std::string::npos)
        throw std::runtime_error(
            "NPY fixture must be C-order little-endian FP64: " + path);
    Tensor output = Tensor::host(shape, DType::F32);
    std::vector<double> source(static_cast<size_t>(output.numel()));
    input.read(reinterpret_cast<char*>(source.data()),
               static_cast<std::streamsize>(source.size() * sizeof(double)));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("NPY fixture payload length mismatch: " + path);
    for (size_t index = 0; index < source.size(); ++index)
        output.data_as<float>()[index] = static_cast<float>(source[index]);
    return output;
}

inline std::pair<std::vector<int64_t>, std::vector<uint8_t>> read_npy_u8(
        const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open NPY fixture: " + path);
    const auto [shape, header] = read_npy_header(input, path);
    if (header.find("'descr': '|u1'") == std::string::npos)
        throw std::runtime_error("NPY fixture must be C-order uint8: " + path);
    int64_t elements = 1;
    for (int64_t dimension : shape) elements *= dimension;
    std::vector<uint8_t> output(static_cast<size_t>(elements));
    input.read(reinterpret_cast<char*>(output.data()),
               static_cast<std::streamsize>(output.size()));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("NPY fixture payload length mismatch: " + path);
    return {shape, std::move(output)};
}

}  // namespace vrhino::test
