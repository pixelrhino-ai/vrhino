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
    require_test(static_cast<bool>(input), "cannot read legal fixture");
    return {std::istreambuf_iterator<char>(input), {}};
}

}  // namespace

int main() {
    try {
        const fs::path source = fs::path(VRHINO_TEST_SOURCE_ROOT) /
                                "third_party/cpp-httplib";
        const fs::path repository = VRHINO_REPOSITORY_ROOT;
        const fs::path release_license = repository /
            "release/compliance-overlay/licenses/static/cpp-httplib/LICENSE";
        const fs::path notices = repository /
            "release/compliance-overlay/THIRD_PARTY_NOTICES.txt";

        require_test(fs::file_size(source / "httplib.h") == 784467,
                     "vendored cpp-httplib source size drift");
        require_test(product::sha256_file(source / "httplib.h") ==
                         "5933c14b2d0f45212925ed18ca579841f5fce717f431fc20cec712423e905b10",
                     "vendored cpp-httplib source hash drift");
        require_test(product::sha256_file(source / "LICENSE") ==
                         "4b45cbe16d7b71b89ae6127e26e0d90a029198ca5e958ad8e3d0b8bbed364d8b",
                     "vendored cpp-httplib license hash drift");
        require_test(product::sha256_file(release_license) ==
                         product::sha256_file(source / "LICENSE"),
                     "release cpp-httplib license differs from upstream bytes");

        const std::string upstream = read_text(source / "UPSTREAM.md");
        const std::string notice = read_text(notices);
        for (const char* required : {
                 "v0.54.1",
                 "9d6a7ee2c1aaeb1fd9ae15d14f06f487149d147f",
                 "License: MIT"}) {
            require_test(upstream.find(required) != std::string::npos,
                         "upstream pin record is incomplete");
            require_test(notice.find(required) != std::string::npos,
                         "release third-party notice is incomplete");
        }

        std::cout << "cpp-httplib pin/license/release inventory tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpp-httplib pin/license/release inventory tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
