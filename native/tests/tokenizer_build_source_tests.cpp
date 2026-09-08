#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "vrhino/product/model_package.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require_test(static_cast<bool>(input), "cannot read tokenizer source fixture");
    return {std::istreambuf_iterator<char>(input), {}};
}

std::size_t count_occurrences(const std::string& text,
                              const std::string& needle) {
    std::size_t count = 0;
    std::size_t offset = 0;
    while ((offset = text.find(needle, offset)) != std::string::npos) {
        ++count;
        offset += needle.size();
    }
    return count;
}

}  // namespace

int main() {
    try {
        const fs::path native_root = VRHINO_TEST_SOURCE_ROOT;
        const fs::path repository = VRHINO_REPOSITORY_ROOT;
        const fs::path sources = native_root /
            "third_party/tokenizer-build-sources";
        const fs::path inventories = sources / "inventories";

        require_test(product::sha256_file(
                         inventories / "tokenizers-cpp-expanded.sha256") ==
                         "63c37881d795895fc65322f648552c5cfeaa2db8438cb0c59fa9537d2ddbc04e",
                     "expanded tokenizers inventory drift");
        require_test(product::sha256_file(
                         inventories / "abseil-cpp.sha256") ==
                         "f5ff13fdf9f770f0f07c295b0ca1e9fca2d88038f49ebf20e83616c2c8f22346",
                     "Abseil inventory drift");
        require_test(product::sha256_file(
                         inventories / "cargo-vendor.sha256") ==
                         "ebb8d6b25210908e1c45b0a585819c8fff177240d356bc5c76600628db1a76dd",
                     "Cargo vendor inventory drift");
        require_test(product::sha256_file(
                         sources / "tokenizers-cpp/rust/Cargo.lock") ==
                         "ea028c09e0ac3a242df874856b34b1fbedd1eda0dfb6b6160c0e4598229669a1",
                     "qualified Cargo.lock drift");

        const std::string manifest = read_text(sources / "SOURCE-MANIFEST.json");
        for (const char* required : {
                 "c586c52f93f7b060753bd2388eb96a105cb7374d",
                 "e0f0f966959108415183d6cbe7a9051ca8bc2da1",
                 "092bc69b6e815980bce7808595c914dd3a29f905",
                 "255c84dadd029fd8ad25c5efb5933e47beaa00c7",
                 "ea028c09e0ac3a242df874856b34b1fbedd1eda0dfb6b6160c0e4598229669a1",
                 "\"registry_crate_count\": 80"}) {
            require_test(manifest.find(required) != std::string::npos,
                         "tokenizer source manifest is incomplete");
        }

        const std::string cargo_lock = read_text(
            sources / "tokenizers-cpp/rust/Cargo.lock");
        require_test(count_occurrences(cargo_lock, "[[package]]") == 81,
                     "Cargo.lock package count drift");
        require_test(count_occurrences(
                         cargo_lock,
                         "source = \"registry+https://github.com/rust-lang/crates.io-index\"") ==
                         80,
                     "Cargo.lock registry package count drift");

        std::size_t crate_directories = 0;
        for (const fs::directory_entry& entry :
             fs::directory_iterator(sources / "cargo-vendor")) {
            if (!entry.is_directory()) continue;
            ++crate_directories;
            require_test(fs::is_regular_file(entry.path() / ".cargo-checksum.json"),
                         "vendored crate lacks Cargo checksum metadata");
        }
        require_test(crate_directories == 80,
                     "Cargo vendor crate count drift");

        const fs::path legal_root = repository /
            "release/compliance-overlay/licenses/static";
        for (const auto& pair : {
                 std::pair{sources / "tokenizers-cpp/LICENSE",
                           legal_root / "tokenizers-cpp/LICENSE"},
                 std::pair{sources / "tokenizers-cpp/sentencepiece/LICENSE",
                           legal_root / "sentencepiece/LICENSE"},
                 std::pair{sources / "abseil-cpp/LICENSE",
                           legal_root / "sentencepiece/abseil-cpp/LICENSE"},
                 std::pair{sources / "tokenizers-cpp/msgpack/LICENSE_1_0.txt",
                           legal_root / "msgpack/LICENSE_1_0.txt"}}) {
            require_test(product::sha256_file(pair.first) ==
                             product::sha256_file(pair.second),
                         "release tokenizer license differs from source bytes");
        }

        const std::string cmake = read_text(native_root / "CMakeLists.txt");
        const std::string cargo_wrapper = read_text(
            native_root / "cmake/tokenizer-cargo-offline.sh.in");
        const std::string cargo_config = read_text(
            native_root / "cmake/tokenizer-cargo-config.toml.in");
        require_test(cmake.find("FetchContent_Declare(tokenizers_cpp") ==
                         std::string::npos,
                     "production CMake retains tokenizer network FetchContent");
        require_test(cmake.find("SPM_ABSL_PROVIDER \"package\"") !=
                         std::string::npos,
                     "SentencePiece local Abseil provider is not forced");
        require_test(cargo_wrapper.find("--locked --offline") !=
                         std::string::npos,
                     "Cargo wrapper is not locked/offline");
        require_test(cargo_config.find("replace-with = \"vendored-sources\"") !=
                         std::string::npos,
                     "Cargo source replacement is absent");

        const std::string notices = read_text(repository /
            "release/compliance-overlay/THIRD_PARTY_NOTICES.txt");
        for (const char* required : {
                 "c586c52f93f7b060753bd2388eb96a105cb7374d",
                 "e0f0f966959108415183d6cbe7a9051ca8bc2da1",
                 "092bc69b6e815980bce7808595c914dd3a29f905",
                 "255c84dadd029fd8ad25c5efb5933e47beaa00c7"}) {
            require_test(notices.find(required) != std::string::npos,
                         "release legal inventory lacks tokenizer source pin");
        }

        std::cout << "hermetic tokenizer source/lock/vendor/legal tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hermetic tokenizer source tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
