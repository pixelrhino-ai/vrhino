#pragma once
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "vrhino/product/model_package.h"
namespace cache_fixture {
namespace fs=std::filesystem;
namespace product=vrhino::product;
inline void require_test(bool b,const std::string& m){if(!b)throw std::runtime_error(m);}
inline void write_file(const fs::path& path, const std::string& contents) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
    require_test(static_cast<bool>(output), "failed to write test file: " + path.string());
}

inline std::string artifact_json(const std::string& id, const std::string& role,
                          const std::string& path, const uint64_t size,
                          const std::string& sha256) {
    std::ostringstream stream;
    stream << "{\"id\":\"" << id << "\",\"role\":\"" << role
           << "\",\"path\":\"" << path << "\",\"size\":" << size
           << ",\"sha256\":\"" << sha256 << "\",\"required\":true}";
    return stream.str();
}

inline fs::path make_package(const fs::path& parent, const std::string& version,
                      const int64_t schema = 1, const bool missing_profile = false,
                      const bool bad_runtime_hash = false) {
    const fs::path package = parent / ("package-" + version);
    fs::create_directories(package);
    write_file(package / "model/model.vrm", "abc");
    if (!missing_profile) write_file(package / "execution/default.json", "{}\n");
    write_file(package / "legal/LICENSE.txt", "test license\n");
    const std::string runtime_hash = bad_runtime_hash
                                         ? std::string(64, '0')
                                         : product::sha256_file(package / "model/model.vrm");
    const std::string profile_hash = missing_profile
                                         ? product::sha256_file(package / "model/model.vrm")
                                         : product::sha256_file(package / "execution/default.json");
    const std::string license_hash = product::sha256_file(package / "legal/LICENSE.txt");
    std::ostringstream manifest;
    manifest
        << "{\n"
        << "  \"schema_version\": " << schema << ",\n"
        << "  \"identity\": {\"namespace\":\"vrhino\",\"name\":\"fixture\","
           "\"version\":\"" << version
        << "\",\"architecture\":\"test_arch\",\"publisher\":\"VRhino\"},\n"
        << "  \"compatibility\": {\"runtime_contract\":\"cuda-v1\","
           "\"vrm_schema\":{\"format_major\":0,\"format_minor\":1,"
           "\"metadata_schema\":1}},\n"
        << "  \"artifacts\": [\n"
        << "    " << artifact_json("runtime", "runtime.vrm", "model/model.vrm", 3,
                                      runtime_hash) << ",\n"
        << "    " << artifact_json("profile", "execution.profile", "execution/default.json",
                                      missing_profile ? 3 : 3, profile_hash) << ",\n"
        << "    " << artifact_json("license", "legal.license", "legal/LICENSE.txt", 13,
                                      license_hash) << "\n"
        << "  ],\n"
        << "  \"entrypoint\": {\"runtime_artifact\":\"runtime\",\"components\":[],"
           "\"default_preset\":\"default\"},\n"
        << "  \"defaults\": {\"default_preset\":\"default\",\"presets\":{"
           "\"default\":{\"profile_artifact\":\"profile\",\"inputs\":{}}}},\n"
        << "  \"hardware\": {\"presets\":{\"default\":{"
           "\"minimum_vram_bytes\":null}}},\n"
        << "  \"source\": {\"repository\":\"example/fixture\","
           "\"revision\":\"0123456789abcdef\",\"converter_version\":\"test\"},\n"
        << "  \"license\": {\"identifier\":\"LicenseRef-Test\","
           "\"artifact\":\"license\",\"upstream_notice\":\"example.invalid\"}\n"
        << "}\n";
    write_file(package / product::kModelManifestName, manifest.str());
    return package;
}

}
