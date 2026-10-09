#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "vrhino/architecture.h"

namespace vrhino::product {

// Read-only Native loader for conditioning assets kept outside .vrm v0.1.
// Logical shard names come from the index while physical paths come from the
// package resolver, so immutable CAS filenames never leak into model semantics.
struct VerifiedSafeTensorArtifact {
    std::filesystem::path path;
    uint64_t bytes;
    std::string sha256;
};

class SafeTensorAsset {
public:
    static SafeTensorAsset single(const std::filesystem::path& path);
    static SafeTensorAsset indexed(
        const std::filesystem::path& index_path,
        const std::map<std::string, std::filesystem::path>& shard_paths);
    // Hash the same stable mapping used by borrowed tensors; strict complete index.
    static SafeTensorAsset indexed_verified(const Json& index,
        const std::map<std::string, VerifiedSafeTensorArtifact>& shards);
    ~SafeTensorAsset();
    SafeTensorAsset(SafeTensorAsset&& other) noexcept;
    SafeTensorAsset& operator=(SafeTensorAsset&& other) noexcept;
    SafeTensorAsset(const SafeTensorAsset&) = delete;
    SafeTensorAsset& operator=(const SafeTensorAsset&) = delete;

    WeightMap weights() const;
    size_t mapped_bytes() const;
    void verify_backing_identity() const;

private:
    struct Mapping {
        int fd = -1;
        void* data = nullptr;
        size_t bytes = 0;
        std::filesystem::path path;
        bool verified = false;
        // Opaque OS timestamp units; compared only on the same platform.
        int64_t modified_stamp = 0, changed_stamp = 0;
        Mapping() = default;
        ~Mapping();
        Mapping(Mapping&& other) noexcept;
        Mapping(const Mapping&) = delete;
        Mapping& operator=(const Mapping&) = delete;
    };
    SafeTensorAsset() = default;
    void add_file(const std::filesystem::path& path, const std::string& logical_name,
                  const std::map<std::string, std::string>* owners = nullptr,
                  const VerifiedSafeTensorArtifact* verified = nullptr);
    void clear();

    std::vector<Mapping> mappings_;
    std::map<std::string, Tensor> tensors_;
};

}  // namespace vrhino::product
