#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "vrhino/product/version.h"

namespace {

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read release identity file");
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::string one_line(const std::string& path) {
    std::istringstream input(read_file(path));
    std::string value;
    std::getline(input, value);
    std::string extra;
    if (value.empty() || std::getline(input, extra)) {
        throw std::runtime_error("release identity file is not exactly one line");
    }
    return value;
}

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        const std::string canonical = one_line(VRHINO_TEST_CANONICAL_VERSION_FILE);
        const std::string package_version = one_line(VRHINO_TEST_PACKAGE_VERSION_FILE);
        const std::string archive_name = one_line(VRHINO_TEST_ARCHIVE_NAME_FILE);
        const vrhino::product::VersionInfo version =
            vrhino::product::current_version_info();

        require(canonical == "v0.7.0-alpha", "unexpected frozen release identity");
        require(std::string(VRHINO_TEST_PROJECT_VERSION) == "0.7.0",
                "CMake project version diverges from the release identity");
        require(version.version == canonical,
                "VersionInfo diverges from the canonical release identity");
        require(vrhino::product::build_version_info().at("version").string() ==
                    canonical,
                "Native API version projection diverges from VersionInfo");
        require(vrhino::product::format_cli_version(version).starts_with(
                    "VRhino " + canonical + "\n"),
                "CLI version projection diverges from VersionInfo");
        require(package_version == canonical,
                "package VERSION diverges from the canonical release identity");
        require(archive_name ==
                    "vrhino-linux-x86_64-cuda-" + canonical + ".tar.gz",
                "archive name diverges from the canonical release identity");

        std::cout << "release-version-consistency: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
