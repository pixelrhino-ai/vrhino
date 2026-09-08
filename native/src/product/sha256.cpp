#include "vrhino/product/model_package.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <unistd.h>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define VRHINO_HAS_X86_SHA_NI_INTRINSICS 1
#include <immintrin.h>
#else
#define VRHINO_HAS_X86_SHA_NI_INTRINSICS 0
#endif

namespace vrhino::product {
namespace {

constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
    0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
    0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
    0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

using Sha256State = std::array<uint32_t, 8>;
using TransformBlocks = void (*)(Sha256State&, const uint8_t*, size_t);

void transform_scalar(Sha256State& state, const uint8_t* block) {
    std::array<uint32_t, 64> words{};
    for (size_t index = 0; index < 16; ++index) {
        const uint8_t* word = block + index * 4;
        words[index] = (static_cast<uint32_t>(word[0]) << 24) |
                       (static_cast<uint32_t>(word[1]) << 16) |
                       (static_cast<uint32_t>(word[2]) << 8) |
                       static_cast<uint32_t>(word[3]);
    }
    for (size_t index = 16; index < words.size(); ++index) {
        const uint32_t s0 = std::rotr(words[index - 15], 7) ^
                            std::rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
        const uint32_t s1 = std::rotr(words[index - 2], 17) ^
                            std::rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    auto [a, b, c, d, e, f, g, h] = state;
    for (size_t index = 0; index < words.size(); ++index) {
        const uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const uint32_t choice = (e & f) ^ (~e & g);
        const uint32_t temporary1 = h + sum1 + choice + kRoundConstants[index] + words[index];
        const uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void transform_blocks_scalar(Sha256State& state, const uint8_t* blocks,
                             const size_t block_count) {
    for (size_t index = 0; index < block_count; ++index)
        transform_scalar(state, blocks + index * 64);
}

#if VRHINO_HAS_X86_SHA_NI_INTRINSICS

#define VRHINO_X86_SHA_TARGET __attribute__((target("sha,ssse3,sse4.1")))

VRHINO_X86_SHA_TARGET
void sha_ni_rounds4(__m128i& state0, __m128i& state1, __m128i message,
                    const size_t constant_offset) {
    message = _mm_add_epi32(
        message,
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(
            kRoundConstants.data() + constant_offset)));
    state1 = _mm_sha256rnds2_epu32(state1, state0, message);
    message = _mm_shuffle_epi32(message, 0x0e);
    state0 = _mm_sha256rnds2_epu32(state0, state1, message);
}

VRHINO_X86_SHA_TARGET
void transform_blocks_sha_ni(Sha256State& state, const uint8_t* blocks,
                             const size_t block_count) {
    const __m128i byte_swap = _mm_set_epi64x(
        0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    for (size_t block_index = 0; block_index < block_count; ++block_index) {
        const uint8_t* block = blocks + block_index * 64;
        __m128i messages[16];
        for (size_t group = 0; group < 4; ++group) {
            messages[group] = _mm_shuffle_epi8(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    block + group * 16)), byte_swap);
        }
        for (size_t group = 4; group < 16; ++group) {
            __m128i message = _mm_sha256msg1_epu32(
                messages[group - 4], messages[group - 3]);
            message = _mm_add_epi32(
                message,
                _mm_alignr_epi8(messages[group - 1], messages[group - 2], 4));
            messages[group] = _mm_sha256msg2_epu32(
                message, messages[group - 1]);
        }

        __m128i temporary = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(state.data()));
        __m128i state1 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(state.data() + 4));
        temporary = _mm_shuffle_epi32(temporary, 0xb1);
        state1 = _mm_shuffle_epi32(state1, 0x1b);
        __m128i state0 = _mm_alignr_epi8(temporary, state1, 8);
        state1 = _mm_blend_epi16(state1, temporary, 0xf0);
        const __m128i saved0 = state0;
        const __m128i saved1 = state1;
        for (size_t group = 0; group < 16; ++group)
            sha_ni_rounds4(state0, state1, messages[group], group * 4);

        state0 = _mm_add_epi32(state0, saved0);
        state1 = _mm_add_epi32(state1, saved1);
        temporary = _mm_shuffle_epi32(state0, 0x1b);
        state1 = _mm_shuffle_epi32(state1, 0xb1);
        state0 = _mm_blend_epi16(temporary, state1, 0xf0);
        state1 = _mm_alignr_epi8(state1, temporary, 8);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(state.data()), state0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(state.data() + 4), state1);
    }
}

bool cpu_supports_sha_ni() noexcept {
    static const bool supported = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("sha");
    }();
    return supported;
}

#undef VRHINO_X86_SHA_TARGET

#else

bool cpu_supports_sha_ni() noexcept { return false; }

#endif

enum class Sha256Implementation { Automatic, Scalar, ShaNi };

TransformBlocks select_transform(const Sha256Implementation implementation) {
#if VRHINO_HAS_X86_SHA_NI_INTRINSICS
    if (implementation == Sha256Implementation::ShaNi) {
        if (!cpu_supports_sha_ni())
            throw std::runtime_error("SHA-NI is unavailable on this CPU");
        return transform_blocks_sha_ni;
    }
    if (implementation == Sha256Implementation::Automatic && cpu_supports_sha_ni())
        return transform_blocks_sha_ni;
#else
    if (implementation == Sha256Implementation::ShaNi)
        throw std::runtime_error("SHA-NI is unavailable in this build");
#endif
    return transform_blocks_scalar;
}

class Sha256 {
public:
    explicit Sha256(const Sha256Implementation implementation =
                        Sha256Implementation::Automatic)
        : transform_blocks_(select_transform(implementation)) {}

    void update(const uint8_t* data, size_t size) {
        total_bytes_ += size;
        if (block_size_ != 0) {
            const size_t count = std::min(size, block_.size() - block_size_);
            std::copy_n(data, count, block_.data() + block_size_);
            data += count;
            size -= count;
            block_size_ += count;
            if (block_size_ == block_.size()) {
                transform_blocks_(state_, block_.data(), 1);
                block_size_ = 0;
            }
        }
        const size_t block_count = size / block_.size();
        if (block_count != 0) {
            transform_blocks_(state_, data, block_count);
            const size_t consumed = block_count * block_.size();
            data += consumed;
            size -= consumed;
        }
        if (size != 0) {
            std::copy_n(data, size, block_.data());
            block_size_ = size;
        }
    }

    std::array<uint8_t, 32> finish() {
        const uint64_t total_bits = total_bytes_ * 8U;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
            transform_blocks_(state_, block_.data(), 1);
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0);
        for (size_t index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<uint8_t>(total_bits >> (index * 8));
        }
        transform_blocks_(state_, block_.data(), 1);

        std::array<uint8_t, 32> result{};
        for (size_t word = 0; word < state_.size(); ++word) {
            for (size_t byte = 0; byte < 4; ++byte) {
                result[word * 4 + byte] =
                    static_cast<uint8_t>(state_[word] >> ((3 - byte) * 8));
            }
        }
        return result;
    }

private:
    std::array<uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<uint8_t, 64> block_{};
    size_t block_size_ = 0;
    uint64_t total_bytes_ = 0;
    TransformBlocks transform_blocks_ = transform_blocks_scalar;
};

std::string hex_digest(const std::array<uint8_t, 32>& digest) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const uint8_t byte : digest) stream << std::setw(2) << static_cast<unsigned>(byte);
    return stream.str();
}

std::string sha256_file_with_implementation(
        const std::filesystem::path& path,
        const Sha256Implementation implementation,
        const size_t buffer_size) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open SHA-256 test file");
    Sha256 hash(implementation);
    std::vector<char> buffer(buffer_size);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0)
            hash.update(reinterpret_cast<const uint8_t*>(buffer.data()),
                        static_cast<size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("failed while reading SHA-256 test file");
    return hex_digest(hash.finish());
}

}  // namespace

namespace sha256_testing {

bool sha_ni_supported() noexcept { return cpu_supports_sha_ni(); }

bool select_sha_ni(const bool feature_available) noexcept {
    return VRHINO_HAS_X86_SHA_NI_INTRINSICS != 0 && feature_available;
}

std::array<uint8_t, 32> digest(const uint8_t* data, const size_t size,
                               const bool use_sha_ni) {
    Sha256 hash(use_sha_ni ? Sha256Implementation::ShaNi :
                            Sha256Implementation::Scalar);
    hash.update(data, size);
    return hash.finish();
}

std::string file_digest(const std::filesystem::path& path,
                        const bool use_sha_ni, const size_t buffer_size) {
    return sha256_file_with_implementation(
        path, use_sha_ni ? Sha256Implementation::ShaNi :
                           Sha256Implementation::Scalar,
        buffer_size);
}

}  // namespace sha256_testing

std::string sha256_file(const std::filesystem::path& path,
                        const WorkProgressCallback& progress) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw ModelPackageError(ModelPackageErrorCode::ArtifactMissing,
                                "cannot open artifact: " + path.string());
    }
    Sha256 hash;
    std::vector<char> buffer(8 * 1024 * 1024);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            hash.update(reinterpret_cast<const uint8_t*>(buffer.data()),
                        static_cast<size_t>(count));
            if (progress) progress(static_cast<uint64_t>(count));
        }
    }
    if (!input.eof()) {
        throw ModelPackageError(ModelPackageErrorCode::CacheError,
                                "failed while reading artifact: " + path.string());
    }
    return hex_digest(hash.finish());
}

std::string sha256_file_descriptor(const int descriptor, const uint64_t size,
                                   const WorkProgressCallback& progress) {
    if (descriptor < 0) {
        throw ModelPackageError(ModelPackageErrorCode::CacheError,
                                "invalid artifact descriptor");
    }
    Sha256 hash;
    std::vector<uint8_t> buffer(8 * 1024 * 1024);
    uint64_t offset = 0;
    while (offset < size) {
        const size_t requested = static_cast<size_t>(
            std::min<uint64_t>(buffer.size(), size - offset));
        ssize_t count = -1;
        do {
            count = pread(descriptor, buffer.data(), requested,
                          static_cast<off_t>(offset));
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            throw ModelPackageError(ModelPackageErrorCode::CacheError,
                                    "failed while reading artifact descriptor");
        }
        if (count == 0) {
            throw ModelPackageError(ModelPackageErrorCode::CacheError,
                                    "premature EOF while reading artifact descriptor");
        }
        hash.update(buffer.data(), static_cast<size_t>(count));
        offset += static_cast<uint64_t>(count);
        if (progress) progress(static_cast<uint64_t>(count));
    }
    return hex_digest(hash.finish());
}

std::string copy_file_and_sha256(const std::filesystem::path& source,
                                 const std::filesystem::path& destination,
                                 const WorkProgressCallback& progress) {
    std::ifstream input(source, std::ios::binary);
    if (!input) {
        throw ModelPackageError(ModelPackageErrorCode::ArtifactMissing,
                                "cannot open artifact: " + source.string());
    }
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw ModelPackageError(ModelPackageErrorCode::CacheError,
                                "cannot create staged blob: " + destination.string());
    }
    Sha256 hash;
    std::vector<char> buffer(8 * 1024 * 1024);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            hash.update(reinterpret_cast<const uint8_t*>(buffer.data()),
                        static_cast<size_t>(count));
            output.write(buffer.data(), count);
            if (!output) {
                throw ModelPackageError(ModelPackageErrorCode::CacheError,
                                        "failed while writing staged blob: " +
                                            destination.string());
            }
            if (progress) progress(static_cast<uint64_t>(count));
        }
    }
    if (!input.eof()) {
        throw ModelPackageError(ModelPackageErrorCode::CacheError,
                                "failed while reading artifact: " + source.string());
    }
    output.flush();
    if (!output) {
        throw ModelPackageError(ModelPackageErrorCode::CacheError,
                                "failed to flush staged blob: " + destination.string());
    }
    return hex_digest(hash.finish());
}

}  // namespace vrhino::product
