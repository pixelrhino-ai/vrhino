#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "vrhino/product/model_package.h"

namespace vrhino::product {

inline constexpr int64_t kComponentPackageSchemaVersion = 1;
inline constexpr const char* kMediaComponentReference =
    "vrhino/media-linux-x86_64:1.0.0";
inline constexpr const char* kMediaComponentContract =
    "vrhino.media.rgb24-h264-mp4.cli";

struct ComponentArtifact {
    std::filesystem::path relative_path;
    uint64_t size = 0;
    std::string sha256;
    bool executable = false;
};

struct ComponentPackageManifest {
    int64_t schema_version = 0;
    PackageIdentity identity;
    std::string contract_name;
    int64_t contract_major = 0;
    int64_t contract_minor = 0;
    std::string operating_system;
    std::string machine_architecture;
    std::filesystem::path entrypoint;
    std::vector<ComponentArtifact> artifacts;
    std::string raw_json;
};

struct ResolvedComponent {
    ComponentPackageManifest manifest;
    std::filesystem::path root;
    std::filesystem::path entrypoint;
};

struct ComponentInstallResult {
    PackageIdentity identity;
    std::filesystem::path root;
    bool already_installed = false;
};

ComponentPackageManifest load_component_package_manifest(
    const std::filesystem::path& path);

class LocalComponentCache {
public:
    explicit LocalComponentCache(std::filesystem::path root = {});

    const CacheLayout& layout() const noexcept { return layout_; }
    std::filesystem::path components_root() const;
    ResolvedComponent resolve(const std::string& exact_reference,
                              bool verify_hashes = true) const;
    ComponentInstallResult install_archive(const std::filesystem::path& archive);

private:
    CacheLayout layout_;
};

}  // namespace vrhino::product
