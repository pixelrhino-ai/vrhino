#include "vrhino/loader.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "vrhino/error.h"
#include "vrhino/quantization/reference.h"

namespace vrhino {
namespace {

constexpr size_t kHeaderSize = 128;
constexpr size_t kAlignment = 64;
constexpr std::array<uint8_t, 8> kMagic = {'V','R','H','I','N','O',0,1};

uint16_t u16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}
uint64_t u64(const uint8_t* data) {
    uint64_t value = 0;
    for (int index = 7; index >= 0; --index) value = (value << 8) | data[index];
    return value;
}
std::string identifier(const uint8_t* data) {
    size_t length = 0; while (length < 16 && data[length] != 0) ++length;
    for (size_t index = 0; index < length; ++index)
        require(data[index] >= 0x20 && data[index] <= 0x7e, "Invalid VRM identifier");
    return std::string(reinterpret_cast<const char*>(data), length);
}
size_t aligned(size_t value) {
    require(value <= std::numeric_limits<size_t>::max() - (kAlignment - 1),
            "VRM tensor alignment overflow");
    return (value + kAlignment - 1) / kAlignment * kAlignment;
}

size_t nonnegative_size(const Json& value, const char* message) {
    const int64_t integer = value.integer();
    require(integer >= 0, message);
    return static_cast<size_t>(integer);
}

bool is_power_of_four(int64_t value) {
    if (value < 4) return false;
    while (value > 1) {
        if (value % 4 != 0) return false;
        value /= 4;
    }
    return true;
}

size_t dense_bytes(const std::vector<int64_t>& shape, DType dtype) {
    const int64_t elements = shape_numel(shape);
    require(static_cast<uint64_t>(elements) <=
                std::numeric_limits<size_t>::max() / dtype_size(dtype),
            "Tensor byte size overflow");
    return static_cast<size_t>(elements) * dtype_size(dtype);
}

void require_exact_keys(const Json& object, const std::set<std::string>& expected,
                        const char* message) {
    require(object.is_object() && object.object().size() == expected.size(), message);
    for (const auto& [key, unused] : object.object()) {
        (void)unused;
        require(expected.contains(key), message);
    }
}

#ifdef _WIN32
int open_vrm(const std::wstring& path) {
    require(path.find(L'\0') == std::wstring::npos, "Invalid VRM path");
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw Error("Cannot open VRM");
    // Retain the existing stable-open descriptor boundary. Only _close may
    // close the file HANDLE after a successful ownership transfer.
    const int descriptor = _open_osfhandle(reinterpret_cast<intptr_t>(file),
        _O_RDONLY | _O_BINARY | _O_NOINHERIT);
    if (descriptor < 0) {
        CloseHandle(file);
        throw Error("Cannot own VRM file handle");
    }
    return descriptor;
}

int open_vrm(const std::string& path) {
    require(!path.empty() && path.find('\0') == std::string::npos &&
                path.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
            "Invalid VRM UTF-8 path");
    const int length = static_cast<int>(path.size());
    const int wide_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
        path.data(), length, nullptr, 0);
    require(wide_length > 0, "Invalid VRM UTF-8 path");
    std::wstring wide(static_cast<size_t>(wide_length), L'\0');
    require(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(), length,
                wide.data(), wide_length) == wide_length, "Invalid VRM UTF-8 path");
    return open_vrm(wide);
}
#else
int open_vrm(const std::string& path) {
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(descriptor >= 0, "Cannot open VRM: " + path);
    return descriptor;
}
#endif

constexpr std::array<uint64_t, 8> kIv = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};
constexpr uint8_t kSigma[12][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
    {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
    {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
    {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
    {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
    {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},
    {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
    {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},
    {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
};

uint64_t rotate(uint64_t value, int bits) { return (value >> bits) | (value << (64 - bits)); }

void compress(std::array<uint64_t, 8>& h, const uint8_t block[128], uint64_t low,
              uint64_t high, bool last) {
    uint64_t m[16];
    for (int index = 0; index < 16; ++index) m[index] = u64(block + index * 8);
    uint64_t v[16];
    for (int index = 0; index < 8; ++index) { v[index] = h[index]; v[index + 8] = kIv[index]; }
    v[12] ^= low; v[13] ^= high; if (last) v[14] = ~v[14];
    auto g = [&](int a, int b, int c, int d, uint64_t x, uint64_t y) {
        v[a] = v[a] + v[b] + x; v[d] = rotate(v[d] ^ v[a], 32);
        v[c] += v[d]; v[b] = rotate(v[b] ^ v[c], 24);
        v[a] = v[a] + v[b] + y; v[d] = rotate(v[d] ^ v[a], 16);
        v[c] += v[d]; v[b] = rotate(v[b] ^ v[c], 63);
    };
    for (int round = 0; round < 12; ++round) {
        const uint8_t* s = kSigma[round];
        g(0,4,8,12,m[s[0]],m[s[1]]); g(1,5,9,13,m[s[2]],m[s[3]]);
        g(2,6,10,14,m[s[4]],m[s[5]]); g(3,7,11,15,m[s[6]],m[s[7]]);
        g(0,5,10,15,m[s[8]],m[s[9]]); g(1,6,11,12,m[s[10]],m[s[11]]);
        g(2,7,8,13,m[s[12]],m[s[13]]); g(3,4,9,14,m[s[14]],m[s[15]]);
    }
    for (int index = 0; index < 8; ++index) h[index] ^= v[index] ^ v[index + 8];
}

std::array<uint8_t, 16> blake2b128(const uint8_t* data, size_t length) {
    std::array<uint64_t, 8> h = kIv;
    h[0] ^= 0x01010000U ^ 16U;
    uint64_t low = 0, high = 0;
    while (length > 128) {
        const uint64_t before = low; low += 128; if (low < before) ++high;
        compress(h, data, low, high, false); data += 128; length -= 128;
    }
    uint8_t block[128]{}; if (length) std::memcpy(block, data, length);
    const uint64_t before = low; low += length; if (low < before) ++high;
    compress(h, block, low, high, true);
    std::array<uint8_t, 16> output{};
    for (int word = 0; word < 2; ++word)
        for (int byte = 0; byte < 8; ++byte) output[word * 8 + byte] = (h[word] >> (8 * byte)) & 0xff;
    return output;
}

std::vector<int64_t> shape_from_json(const Json& value) {
    std::vector<int64_t> shape;
    for (const Json& dimension : value.array()) {
        const int64_t size = dimension.integer(); require(size >= 0, "Negative VRM tensor shape");
        shape.push_back(size);
    }
    return shape;
}

}  // namespace

VrmModel::VrmModel(const std::string& path, bool verify_checksum)
    : VrmModel(open_vrm(path), verify_checksum) {}

#ifdef _WIN32
VrmModel::OwnedDescriptor::~OwnedDescriptor() {
    if (value >= 0) _close(value);
}

VrmModel::VrmModel(const std::wstring& path, bool verify_checksum)
    : VrmModel(open_vrm(path), verify_checksum) {}
#endif

VrmModel::VrmModel(const int owned_descriptor, bool verify_checksum)
#ifdef _WIN32
    : fd_{owned_descriptor}
#endif
{
    const auto started = std::chrono::steady_clock::now();
    require(owned_descriptor >= 0, "Invalid VRM file descriptor");
#ifndef _WIN32
    fd_ = owned_descriptor;
#endif
    try {
#ifdef _WIN32
        const HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd_.value));
        LARGE_INTEGER size{};
        require(GetFileSizeEx(file, &size) != 0 &&
                    size.QuadPart >= static_cast<LONGLONG>(kHeaderSize),
                "Truncated VRM file");
        require(static_cast<uint64_t>(size.QuadPart) <=
                    std::numeric_limits<size_t>::max(),
                "VRM file is too large for this host");
        mapping_size_ = static_cast<size_t>(size.QuadPart);
        const HANDLE section = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (section == nullptr) throw Error("VRM file mapping failed");
        mapping_ = MapViewOfFile(section, FILE_MAP_READ, 0, 0, mapping_size_);
        // The mapped view retains the section independently of this handle.
        // Close it on both success and failure; release_storage owns the view.
        CloseHandle(section);
        require(mapping_ != nullptr, "VRM mmap failed");
#else
        struct stat info{};
        require(fstat(fd_, &info) == 0 && info.st_size >= static_cast<off_t>(kHeaderSize), "Truncated VRM file");
        require(static_cast<uintmax_t>(info.st_size) <=
                    std::numeric_limits<size_t>::max(),
                "VRM file is too large for this host");
        mapping_size_ = static_cast<size_t>(info.st_size);
        mapping_ = mmap(nullptr, mapping_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        require(mapping_ != MAP_FAILED, "VRM mmap failed");
#endif
    mmap_setup_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto* bytes = static_cast<const uint8_t*>(mapping_);
    require(std::equal(kMagic.begin(), kMagic.end(), bytes), "Bad VRM magic");
    require(u16(bytes + 8) == 0 && u16(bytes + 10) == 1, "Unsupported VRM format version");
    require(bytes[12] == 1 && bytes[13] == 6 && u16(bytes + 14) == 0,
            "Unsupported VRM endian/alignment/flags");
    profile_id_ = identifier(bytes + 16);
    architecture_id_ = identifier(bytes + 32);
    require(profile_id_ == "dit-flow" || profile_id_ == "component",
            "Unsupported VRM profile");
    const uint64_t metadata_offset = u64(bytes + 48), metadata_length = u64(bytes + 56);
    const uint64_t table_offset = u64(bytes + 64), table_length = u64(bytes + 72);
    const uint64_t graph_offset = u64(bytes + 80), graph_length = u64(bytes + 88);
    const uint64_t data_offset = u64(bytes + 96);
    file_size_ = u64(bytes + 104);
    require(file_size_ == mapping_size_, "VRM file size/header mismatch");
    require(metadata_offset == kHeaderSize && metadata_length <= file_size_ - metadata_offset &&
            table_offset == metadata_offset + metadata_length && table_offset <= file_size_ &&
            table_length <= file_size_ - table_offset &&
            graph_offset == table_offset + table_length && graph_offset <= file_size_ &&
            graph_length <= file_size_ - graph_offset &&
            graph_offset + graph_length <= data_offset &&
            data_offset % kAlignment == 0 && data_offset <= file_size_,
            "Invalid VRM section layout");
    if (verify_checksum) {
        const auto checksum_started = std::chrono::steady_clock::now();
        const auto digest = blake2b128(bytes + kHeaderSize, mapping_size_ - kHeaderSize);
        require(std::equal(digest.begin(), digest.end(), bytes + 112), "VRM payload checksum mismatch");
        checksum_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - checksum_started).count();
    }
    const auto parse_started = std::chrono::steady_clock::now();
    metadata_ = Json::parse(std::string(reinterpret_cast<const char*>(bytes + metadata_offset), metadata_length));
    const Json table = Json::parse(std::string(reinterpret_cast<const char*>(bytes + table_offset), table_length));
    graph_ = Json::parse(std::string(reinterpret_cast<const char*>(bytes + graph_offset), graph_length));
    require(table.at("schema_version").integer() == 1, "Unsupported tensor table schema");
    require(graph_.at("schema_version").integer() == 1, "Unsupported graph schema");
    size_t expected_relative = 0;
    std::string previous_name;
    struct PendingQuantization { std::string name; Json metadata; };
    std::vector<PendingQuantization> pending_quantization;
    for (const Json& item : table.at("tensors").array()) {
        const std::string name = item.at("name").string();
        require(previous_name.empty() || previous_name < name, "VRM tensor names are not strictly sorted");
        previous_name = name;
        const DType dtype = dtype_from_vrm(item.at("dtype").string());
        const std::vector<int64_t> shape = shape_from_json(item.at("shape"));
        const size_t relative = nonnegative_size(item.at("offset"), "Negative tensor offset");
        const size_t byte_length = nonnegative_size(
            item.at("byte_length"), "Negative tensor byte length");
        require(item.at("alignment").integer() == static_cast<int64_t>(kAlignment), "Unsupported tensor alignment");
        require(item.at("layout").string() == "contiguous", "Unsupported tensor layout");
        const Json& quantization = item.at("quantization");
        const std::string quant_type = quantization.at("type").string();
        const bool quantized = quant_type != "none";
        const Json* scheme_value = quantization.find("scheme");
        const bool preconditioned = scheme_value != nullptr;
        require(!preconditioned || quantized,
                "Executable quantization scheme requires quantized storage");
        require(preconditioned || quantization.find("preconditioner") == nullptr,
                "Preconditioner metadata requires a recognized executable scheme");
        if (quantized) {
            require(shape.size() >= 2 && shape[0] > 0, "Invalid quantized tensor shape");
            require(quant_type == "fp8_e4m3fn" || quant_type == "int8_symmetric" ||
                    quant_type == "int4_symmetric", "Unsupported tensor quantization");
            require(dtype == (preconditioned ? DType::I8 : DType::U8),
                    "Quantized tensor storage dtype mismatch");
            require(quantization.at("storage_dtype").string() ==
                        (preconditioned ? "i8" : "u8"),
                    "Quantization storage dtype mismatch");
            require(quantization.at("scale_dtype").string() == "f32",
                    "Quantization scale dtype must be f32");
            require(quantization.at("axis").integer() == 1 &&
                    quantization.at("group_size").integer() > 0,
                    "Unsupported quantization axis/group size");
            require(quantization.at("block_size").integer() == 0 &&
                    (quantization.at("granularity").string() == "per_group" ||
                     quantization.at("granularity").string() == "per_channel"),
                    "Unsupported quantization block/granularity");
            require(quantization.at("symmetric").boolean(),
                    "Only symmetric quantization is supported");
            require(quantization.at("zero_point_mode").string() == "none" &&
                    quantization.at("zero_point_tensor").is_null(),
                    "Quantization zero points are unsupported");
            const size_t elements = static_cast<size_t>(shape_numel(shape));
            const int64_t inner = static_cast<int64_t>(elements) / shape.at(0);
            const bool per_channel = quantization.at("group_size").integer() >= inner;
            require(quantization.at("granularity").string() ==
                        (per_channel ? "per_channel" : "per_group"),
                    "Quantization granularity/group mismatch");
            const size_t expected_bytes = quant_type == "int4_symmetric"
                ? elements / 2 + elements % 2 : elements;
            require(byte_length == expected_bytes, "Quantized tensor byte length mismatch");
            if (preconditioned) {
                require_exact_keys(quantization, {
                    "accumulation_dtype", "axis", "block_size", "compute_dtype",
                    "dequantization", "granularity", "group_size", "logical_shape",
                    "original_dtype", "packing_layout", "preconditioner", "scale_dtype",
                    "scale_tensor", "scheme", "storage_dtype", "stored_shape", "symmetric",
                    "type", "zero_point_mode", "zero_point_tensor",
                }, "Unknown or incomplete executable quantization metadata");
                require(scheme_value->string() == kPreconditionedSymmetricInt8V1 &&
                            quant_type == "int8_symmetric",
                        "Unsupported executable quantization scheme");
                require(shape.size() == 2,
                        "Preconditioned symmetric INT8 v1 requires rank-2 weights");
                require(shape_from_json(quantization.at("logical_shape")) == shape &&
                            shape_from_json(quantization.at("stored_shape")) == shape,
                        "Quantized logical/stored shape mismatch");
                require(quantization.at("group_size").integer() == inner &&
                            quantization.at("granularity").string() == "per_channel",
                        "Preconditioned INT8 scale group must span the input axis");
                const Json& preconditioner = quantization.at("preconditioner");
                require_exact_keys(preconditioner,
                    {"axis", "group_size", "identity", "storage_basis"},
                    "Unknown or incomplete executable preconditioner metadata");
                const int64_t transform_group = preconditioner.at("group_size").integer();
                require(preconditioner.at("identity").string() == kRegularBlockHadamardV1,
                        "Unsupported quantization preconditioner");
                require(preconditioner.at("axis").integer() == 1 &&
                            preconditioner.at("storage_basis").string() == "preconditioned" &&
                            is_power_of_four(transform_group) && inner % transform_group == 0,
                        "Incompatible regular block-Hadamard semantics");
                require(quantization.at("dequantization").string() ==
                            "scale_then_inverse_preconditioner",
                        "Unsupported preconditioned dequantization semantics");
            }
            pending_quantization.push_back({name, quantization});
        }
        require(relative == aligned(expected_relative), "Overlapping/non-canonical tensor offset");
        if (!quantized)
            require(byte_length == dense_bytes(shape, dtype),
                    "Tensor byte length mismatch");
        require(relative <= file_size_ - data_offset &&
                    byte_length <= file_size_ - data_offset - relative,
                "Tensor range exceeds VRM");
        Tensor tensor = Tensor::borrowed(const_cast<uint8_t*>(bytes + data_offset + relative), byte_length, shape, dtype);
        require(tensors_.emplace(name, TensorRecord{name, std::move(tensor), relative, byte_length}).second,
                "Duplicate tensor name");
        expected_relative = relative + byte_length;
    }
    std::set<std::string> associated_scales;
    for (const PendingQuantization& pending : pending_quantization) {
        Tensor& tensor = tensors_.at(pending.name).tensor;
        const Json& q = pending.metadata;
        const std::string scale_name = q.at("scale_tensor").string();
        const bool preconditioned = q.find("scheme") != nullptr;
        if (preconditioned)
            require(associated_scales.insert(scale_name).second,
                    "Duplicate quantization scale association");
        const auto scale_found = tensors_.find(scale_name);
        require(scale_found != tensors_.end(), "Missing quantization scale tensor");
        const Tensor& scales = scale_found->second.tensor;
        require(!scales.is_quantized() && scales.dtype() == DType::F32,
                "Invalid quantization scale tensor");
        const int64_t outer = tensor.dim(0);
        const int64_t inner = tensor.numel() / outer;
        const int64_t scale_group = q.at("group_size").integer();
        const int64_t groups = inner / scale_group + (inner % scale_group != 0);
        require(scales.shape() == std::vector<int64_t>({outer, groups}),
                "Quantization scale shape mismatch");
        if (preconditioned)
            for (int64_t index = 0; index < scales.numel(); ++index)
                require(std::isfinite(scales.data_as<const float>()[index]) &&
                            scales.data_as<const float>()[index] > 0.0f,
                        "Quantization scales must be finite and positive");
        auto info = std::make_shared<QuantizationInfo>();
        const std::string type = q.at("type").string();
        info->type = type == "fp8_e4m3fn" ? QuantType::FP8E4M3FN :
                     type == "int8_symmetric" ? QuantType::INT8Symmetric :
                                                QuantType::INT4Symmetric;
        info->logical_dtype = dtype_from_vrm(q.at("original_dtype").string());
        info->compute_dtype = dtype_from_vrm(q.at("compute_dtype").string());
        info->accumulation_dtype = dtype_from_vrm(q.at("accumulation_dtype").string());
        require((info->logical_dtype == DType::F32 || info->logical_dtype == DType::F16 ||
                 info->logical_dtype == DType::BF16) &&
                ((!preconditioned && info->compute_dtype == DType::BF16) ||
                 (preconditioned && info->compute_dtype == DType::F32)) &&
                info->accumulation_dtype == DType::F32,
                "Unsupported quantization logical/compute/accumulation dtype");
        info->group_size = q.at("group_size").integer();
        info->block_size = q.at("block_size").integer();
        info->axis = q.at("axis").integer();
        info->symmetric = q.at("symmetric").boolean();
        info->granularity = q.at("granularity").string();
        info->zero_point_mode = q.at("zero_point_mode").string();
        info->packing_layout = q.at("packing_layout").string();
        const std::string dequantization = q.at("dequantization").string();
        const Json* legacy_capability = q.find("kernel_capability");
        if (legacy_capability) {
            require(dequantization == "cuda_to_compute" &&
                    legacy_capability->string() == "sm80",
                    "Unsupported legacy quantization execution capability");
        } else if (!preconditioned) {
            require(dequantization == "scale_then_cast",
                    "Unsupported dequantization semantics");
        }
        info->dequantization = preconditioned
            ? DequantizationSemantics::ScaleThenInversePreconditioner
            : DequantizationSemantics::ScaleThenCast;
        if (info->type == QuantType::FP8E4M3FN)
            require(info->packing_layout == "byte_e4m3fn", "Invalid FP8 packing layout");
        if (info->type == QuantType::INT8Symmetric)
            require(info->packing_layout == "byte_twos_complement", "Invalid INT8 packing layout");
        if (info->type == QuantType::INT4Symmetric)
            require(info->packing_layout == "nibble_low_first_twos_complement", "Invalid INT4 packing layout");
        if (preconditioned) {
            info->scheme = q.at("scheme").string();
            info->stored_shape = shape_from_json(q.at("stored_shape"));
            info->preconditioner = PreconditionerType::RegularBlockHadamardV1;
            info->preconditioner_axis = q.at("preconditioner").at("axis").integer();
            info->preconditioner_group_size =
                q.at("preconditioner").at("group_size").integer();
        }
        info->scales = scales;
        tensor.set_quantization(std::move(info));
    }
    require(data_offset + expected_relative == file_size_, "Tensor table does not cover data section");
    require(metadata_.at("architecture").string() == architecture_id_, "Metadata/header architecture mismatch");
    metadata_parse_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - parse_started).count();
        load_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    } catch (...) {
        release_storage();
        throw;
    }
}

VrmModel::~VrmModel() {
    release_storage();
}

void VrmModel::release_storage() noexcept {
    tensors_.clear();
#ifdef _WIN32
    if (mapping_ != nullptr) UnmapViewOfFile(mapping_);
    // The descriptor member closes the file after view cleanup, including
    // constructor unwinding before this function can ever be entered.
#else
    if (mapping_ != nullptr && mapping_ != MAP_FAILED) munmap(mapping_, mapping_size_);
    if (fd_ >= 0) close(fd_);
#endif
    mapping_ = nullptr;
    mapping_size_ = 0;
#ifndef _WIN32
    fd_ = -1;
#endif
}

const Tensor& VrmModel::tensor(const std::string& canonical_name) const {
    const auto found = tensors_.find(canonical_name);
    require(found != tensors_.end(), "Unknown VRM tensor: " + canonical_name);
    return found->second.tensor;
}

std::map<std::string, const Tensor*> VrmModel::bindings(const Json& descriptor) const {
    require(descriptor.at("schema_version").integer() == 1, "Unsupported descriptor schema");
    std::map<std::string, const Tensor*> result;
    for (const auto& [logical, canonical] : descriptor.at("runtime_tensor_bindings").object())
        result.emplace(logical, &tensor(canonical.string()));
    require(!result.empty(), "Empty runtime tensor binding map");
    return result;
}

}  // namespace vrhino
