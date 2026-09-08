#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#include "vrhino/product/model_package.h"

namespace vrhino::product::sha256_testing {
bool sha_ni_supported() noexcept;
bool select_sha_ni(bool feature_available) noexcept;
std::array<uint8_t, 32> digest(const uint8_t* data, size_t size,
                               bool use_sha_ni);
std::string file_digest(const std::filesystem::path& path,
                        bool use_sha_ni, size_t buffer_size);
}  // namespace vrhino::product::sha256_testing

namespace {

namespace fs = std::filesystem;
namespace sha = vrhino::product::sha256_testing;

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string hex(const std::array<uint8_t, 32>& digest) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const uint8_t byte : digest)
        stream << std::setw(2) << static_cast<unsigned>(byte);
    return stream.str();
}

std::vector<uint8_t> deterministic_bytes(const size_t size) {
    std::vector<uint8_t> result(size);
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (uint8_t& byte : result) {
        state ^= state << 7;
        state ^= state >> 9;
        state ^= state << 8;
        byte = static_cast<uint8_t>(state);
    }
    return result;
}

double cpu_seconds() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<double>(usage.ru_utime.tv_sec) + usage.ru_utime.tv_usec / 1e6 +
           static_cast<double>(usage.ru_stime.tv_sec) + usage.ru_stime.tv_usec / 1e6;
}

struct Measurement {
    double wall_seconds = 0.0;
    double cpu_seconds = 0.0;
    std::string digest;
};

template <typename Operation>
Measurement measure(Operation&& operation) {
    const double cpu_before = cpu_seconds();
    const auto started = std::chrono::steady_clock::now();
    const std::string digest = operation();
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return {wall, cpu_seconds() - cpu_before, digest};
}

void print_measurement(const std::string& name, const Measurement& value,
                       const uint64_t bytes) {
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(6)
              << "path=" << name
              << " bytes=" << bytes
              << " wall_s=" << value.wall_seconds
              << " cpu_s=" << value.cpu_seconds
              << " cpu_pct=" << 100.0 * value.cpu_seconds / value.wall_seconds
              << " mib_s=" << mib / value.wall_seconds
              << " digest=" << value.digest << '\n';
}

void run_known_answers() {
    struct Vector {
        size_t size;
        const char* expected;
    };
    constexpr std::array<Vector, 11> vectors{{
        {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {55, "7b661b0ac996dff354f4dcde9f18cdbe90757190a473a065adab71a05611c6bc"},
        {56, "ae66612c33d8f5a9a3ce521fb55452b24e9e740b7ec806550991ca23ab10a6ae"},
        {63, "812757cd2d72ffb60676939dd80cd88aa1508fe8c90cfcdb3046d106a7ae1eb5"},
        {64, "65e1336298094606fa596c0515d7ad02aa83db3523153b9a63ca28365ce532b6"},
        {65, "9b80e1c94797b5ec774d98d50b52a7d8b292a8dcd376f6ca0c436c7c0968cbd4"},
        {127, "36bcbdda102a592beab911b6db4f238254dc33afa5c82874a474cabf83e76ab9"},
        {128, "c8efe9d886fe9a78b33e4c18fab4c79102d55fb064ea91498c6d08415cb8c15c"},
        {129, "cf9377cba79a6ed950dd7a0d5693b0e534c6d216a10f01fe4a3ef470c57eea52"},
        {1000, "b9343837425c602dcf6957b73639be0d2cb00b866eefaf8d5199ad4949535883"},
        {1048576, "4502127e093f104662708b73dfc925ecc3cabefab1af84ea275460105c400027"},
    }};
    for (const Vector& vector : vectors) {
        const auto bytes = deterministic_bytes(vector.size);
        const auto scalar = sha::digest(bytes.data(), bytes.size(), false);
        require_test(hex(scalar) == vector.expected,
                     "scalar deterministic known-answer mismatch at " +
                         std::to_string(vector.size));
        if (sha::sha_ni_supported()) {
            const auto accelerated = sha::digest(bytes.data(), bytes.size(), true);
            require_test(accelerated == scalar,
                         "SHA-NI differential mismatch at " +
                             std::to_string(vector.size) + ": scalar=" +
                             hex(scalar) + " accelerated=" + hex(accelerated));
        }
    }

    const std::string abc = "abc";
    require_test(hex(sha::digest(
                     reinterpret_cast<const uint8_t*>(abc.data()), abc.size(), false)) ==
                     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                 "abc known-answer mismatch");
    const std::string long_message =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    require_test(hex(sha::digest(
                     reinterpret_cast<const uint8_t*>(long_message.data()),
                     long_message.size(), false)) ==
                     "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
                 "standard long-message known-answer mismatch");
    if (sha::sha_ni_supported()) {
        require_test(sha::digest(
                         reinterpret_cast<const uint8_t*>(long_message.data()),
                         long_message.size(), true) ==
                         sha::digest(reinterpret_cast<const uint8_t*>(long_message.data()),
                                     long_message.size(), false),
                     "long-message SHA-NI differential mismatch");
        for (size_t size = 0; size <= 257; ++size) {
            const auto bytes = deterministic_bytes(size);
            require_test(sha::digest(bytes.data(), bytes.size(), true) ==
                             sha::digest(bytes.data(), bytes.size(), false),
                         "dense small-buffer differential mismatch at " +
                             std::to_string(size));
        }
        for (const size_t size : {4097U, 65537U, 1048583U}) {
            const auto bytes = deterministic_bytes(size);
            require_test(sha::digest(bytes.data(), bytes.size(), true) ==
                             sha::digest(bytes.data(), bytes.size(), false),
                         "multi-block differential mismatch at " +
                             std::to_string(size));
        }
    }
}

void run_dispatch_and_file_tests() {
    require_test(!sha::select_sha_ni(false),
                 "feature-unavailable dispatch did not select scalar fallback");
    if (sha::sha_ni_supported())
        require_test(sha::select_sha_ni(true),
                     "feature-available dispatch did not select SHA-NI");

    const fs::path path = fs::temp_directory_path() /
        ("vrhino-sha256-tests-" + std::to_string(getpid()));
    const auto bytes = deterministic_bytes(4097);
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        require_test(static_cast<bool>(output), "cannot write SHA file fixture");
    }
    const std::string scalar = sha::file_digest(path, false, 257);
    require_test(vrhino::product::sha256_file(path) == scalar,
                 "automatic file SHA differs from forced scalar");
    if (sha::sha_ni_supported())
        require_test(sha::file_digest(path, true, 257) == scalar,
                     "SHA-NI file digest differs from scalar");
    std::error_code error;
    fs::remove(path, error);
}

int run_memory_benchmark(const int argc, char** argv) {
    require_test(argc == 4, "usage: --benchmark-memory BYTES REPEATS");
    const size_t bytes = std::stoull(argv[2]);
    const int repeats = std::stoi(argv[3]);
    const auto input = deterministic_bytes(bytes);
    const uint64_t total_bytes = static_cast<uint64_t>(bytes) * repeats;
    const auto benchmark = [&](const bool accelerated) {
        return measure([&] {
            std::array<uint8_t, 32> result{};
            for (int repeat = 0; repeat < repeats; ++repeat)
                result = sha::digest(input.data(), input.size(), accelerated);
            return hex(result);
        });
    };
    const Measurement scalar = benchmark(false);
    print_measurement("scalar", scalar, total_bytes);
    if (sha::sha_ni_supported()) {
        const Measurement accelerated = benchmark(true);
        print_measurement("sha-ni", accelerated, total_bytes);
        require_test(accelerated.digest == scalar.digest,
                     "memory benchmark digest mismatch");
    }
    return 0;
}

int run_file_benchmark(const int argc, char** argv) {
    require_test(argc == 4, "usage: --benchmark-file PATH EXPECTED_SHA256");
    const fs::path path = argv[2];
    const std::string expected = argv[3];
    const uint64_t bytes = fs::file_size(path);
    const Measurement scalar = measure([&] {
        return sha::file_digest(path, false, 8 * 1024 * 1024);
    });
    print_measurement("scalar", scalar, bytes);
    require_test(scalar.digest == expected, "real-file scalar digest mismatch");
    if (sha::sha_ni_supported()) {
        const Measurement accelerated = measure([&] {
            return sha::file_digest(path, true, 8 * 1024 * 1024);
        });
        print_measurement("sha-ni", accelerated, bytes);
        require_test(accelerated.digest == expected,
                     "real-file SHA-NI digest mismatch");
    }
    const Measurement automatic = measure([&] {
        return vrhino::product::sha256_file(path);
    });
    print_measurement("automatic", automatic, bytes);
    require_test(automatic.digest == expected,
                 "real-file automatic digest mismatch");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--benchmark-memory")
            return run_memory_benchmark(argc, argv);
        if (argc > 1 && std::string(argv[1]) == "--benchmark-file")
            return run_file_benchmark(argc, argv);
        require_test(argc == 1, "unknown SHA-256 test mode");
        run_known_answers();
        run_dispatch_and_file_tests();
        std::cout << "SHA-256 scalar/SHA-NI tests: PASS sha_ni="
                  << (sha::sha_ni_supported() ? "supported" : "unavailable") << '\n';
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "SHA-256 scalar/SHA-NI tests: FAIL: " << failure.what() << '\n';
        return 1;
    }
}
