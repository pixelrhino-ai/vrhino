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

std::string parse_one_line(const std::string& contents) {
    std::istringstream input(contents);
    std::string value;
    std::getline(input, value);
    // Only a CR immediately before the consumed LF belongs to the line ending.
    // A bare CR at EOF and all other whitespace remain part of the value.
    if (!input.eof() && !value.empty() && value.back() == '\r') {
        value.pop_back();
    }
    std::string extra;
    if (value.empty() || std::getline(input, extra)) {
        throw std::runtime_error("release identity file is not exactly one line");
    }
    return value;
}

std::string one_line(const std::string& path) {
    return parse_one_line(read_file(path));
}

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_one_line_parser() {
    const std::string expected = "v0.8.0-alpha";
    for (const char* ending : {"\n", "\r\n", ""}) {
        require(parse_one_line(expected + ending) == expected,
                "logical release line differs across supported line endings");
    }
    for (const char* invalid : {"", "\n", "\r\n", "v0.8.0-alpha\nextra",
                                "v0.8.0-alpha\r\nextra", "v0.8.0-alpha\n\n",
                                "v0.8.0-alpha\r\n\r\n"}) {
        bool rejected = false;
        try {
            (void)parse_one_line(invalid);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "empty or multiline release metadata was accepted");
    }
    for (const char* significant : {" v0.8.0-alpha", "v0.8.0-alpha ",
                                    "\tv0.8.0-alpha", "v0.8.0-alpha\t"}) {
        for (const char* ending : {"\n", "\r\n", ""}) {
            const std::string value = significant;
            const std::string parsed = parse_one_line(value + ending);
            require(parsed == value && parsed != expected,
                    "significant release metadata whitespace was trimmed");
        }
    }
    require(parse_one_line(expected + "\r") == expected + "\r",
            "bare CR at EOF was removed");
    require(parse_one_line(expected + "\r\r\n") == expected + "\r",
            "more than one CR removed from the line ending");
    std::cout << "release-line-parser: PASS\n";
}

}  // namespace

int main() {
    try {
        test_one_line_parser();
        const std::string canonical = one_line(VRHINO_TEST_CANONICAL_VERSION_FILE);
        const std::string package_version = one_line(VRHINO_TEST_PACKAGE_VERSION_FILE);
        const std::string archive_name = one_line(VRHINO_TEST_ARCHIVE_NAME_FILE);
        const vrhino::product::VersionInfo version =
            vrhino::product::current_version_info();

        require(canonical == "v0.9.0-alpha", "unexpected frozen release identity");
        require(std::string(VRHINO_TEST_PROJECT_VERSION) == "0.9.0",
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
