// Test the private production primitives without adding a public hashing API.
#include "../src/loader.cpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

namespace {
namespace fs = std::filesystem;
using Digest = std::array<uint8_t, 16>;

Digest memory_digest(const uint8_t* input, size_t length, vrhino::CompressBlock compress) {
    auto h = vrhino::kIv;
    // The only production variant: unkeyed, sequential BLAKE2b, 16-byte digest.
    h[0] ^= 0x01010010U;
    size_t consumed = 0;
    alignas(32) uint8_t tail[129]{};
    do {
        const size_t bytes = std::min<size_t>(128, length - consumed);
        const uint8_t* block = input + consumed;
        if (bytes < 128) {
            std::memcpy(tail + 1, block, bytes);
            block = tail + 1; // Also exercise an unaligned padded final block.
        }
        consumed += bytes;
        compress(h, block, consumed, 0, consumed == length);
    } while (consumed < length);
    Digest result{};
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = static_cast<uint8_t>(h[i / 8] >> (8 * (i % 8)));
    return result;
}

std::string hex(const Digest& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (const uint8_t byte : digest) {
        result += digits[byte >> 4]; result += digits[byte & 15];
    }
    return result;
}

struct TestFile {
    fs::path path = fs::temp_directory_path() / ("vrhino-blake2b-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TestFile() { std::error_code error; fs::remove(path, error); }
    void write(const uint8_t* input, size_t length, size_t offset) const {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        const std::string prefix(offset, 'x');
        output.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
        output.write(reinterpret_cast<const char*>(input), static_cast<std::streamsize>(length));
        output << "excluded suffix";
        output.close();
        vrhino::require(static_cast<bool>(output), "cannot write checksum fixture");
    }
};

struct Descriptor {
    int value;
    explicit Descriptor(const fs::path& path)
#ifdef _WIN32
        : value(vrhino::open_vrm(path.wstring())) {}
#else
        : value(vrhino::open_vrm(path.string())) {}
#endif
    ~Descriptor() {
#ifdef _WIN32
        _close(value);
#else
        close(value);
#endif
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
};
} // namespace

int main() {
    try {
        struct Case { size_t length; std::string_view digest; };
        // hashlib.blake2b(bytes((i*17+3)&255 for i in range(n)), digest_size=16).
        const Case cases[] = {
            {0, "cae66941d9efbd404e4d88758ea67670"},
            {1, "71b186b851e866e71be237342976049a"},
            {127, "2e1c55f3b67cd9bec47c0bc6fc7f22f4"},
            {128, "9a1656851f7c741c8758868e53e82acb"},
            {129, "c47902f20a439ade21136a7f4f1d4cca"},
            {255, "0ed23e599174fb08c19b99c6cdb325df"},
            {256, "b786bd29617274d8a015f30499a98553"},
            {257, "f02e1d4a9576b94496ac2503d66e38df"},
            {1031, "d17a7e79e7b92ee0c4b362f53b6bf86b"},
        };
        bool has_avx2 = false;
#if VRHINO_HAS_AVX2_BLAKE2B
        __builtin_cpu_init();
        has_avx2 = __builtin_cpu_supports("avx2");
        vrhino::require(vrhino::select_compress() == (has_avx2 ?
            vrhino::compress_avx2 : vrhino::compress_scalar), "incorrect CPU dispatch");
#else
        vrhino::require(vrhino::select_compress() == vrhino::compress_scalar,
                        "unsupported target did not select scalar");
#endif
        TestFile file;
        for (const auto& item : cases) {
            for (const size_t offset : {size_t{0}, size_t{1}, size_t{31}}) {
                alignas(32) std::array<uint8_t, 1088> storage{};
                uint8_t* input = storage.data() + offset;
                for (size_t i = 0; i < item.length; ++i)
                    input[i] = static_cast<uint8_t>(i * 17 + 3);
                const auto scalar = memory_digest(input, item.length, vrhino::compress_scalar);
                vrhino::require(hex(scalar) == item.digest, "scalar/reference digest mismatch");
#if VRHINO_HAS_AVX2_BLAKE2B
                if (has_avx2) vrhino::require(
                    memory_digest(input, item.length, vrhino::compress_avx2) == scalar,
                    "AVX2/scalar digest bytes differ");
#endif
                file.write(input, item.length, offset);
                const Descriptor descriptor(file.path);
                vrhino::require(vrhino::blake2b128(descriptor.value, offset, item.length) == scalar,
                                "production descriptor digest/range mismatch");
            }
        }
        // Keep the original requested range after truncating the last fixture.
        fs::resize_file(file.path, 31 + 1031 - 1);
        const Descriptor descriptor(file.path);
        bool rejected = false;
        try { (void)vrhino::blake2b128(descriptor.value, 31, 1031); }
        catch (const vrhino::Error& error) {
            rejected = std::string_view(error.what()).find(
                "Truncated or unreadable VRM checksum range") != std::string_view::npos;
        }
        vrhino::require(rejected, "truncated checksum range was not rejected");
        std::cout << "BLAKE2b-128 reference/range/unaligned: PASS cases=27 scalar=PASS avx2="
                  << (has_avx2 ? "PASS" : "NOT_RUN") << " truncated=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
