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
class SafeTensorAsset {
public:
    static SafeTensorAsset single(const std::filesystem::path& path);
    static SafeTensorAsset indexed(
        const std::filesystem::path& index_path,
        const std::map<std::string, std::filesystem::path>& shard_paths);
    ~SafeTensorAsset();
    SafeTensorAsset(SafeTensorAsset&& other) noexcept;
    SafeTensorAsset& operator=(SafeTensorAsset&& other) noexcept;
    SafeTensorAsset(const SafeTensorAsset&) = delete;
    SafeTensorAsset& operator=(const SafeTensorAsset&) = delete;

    WeightMap weights() const;
    size_t mapped_bytes() const;

private:
    struct Mapping {
        int fd = -1;
        void* data = nullptr;
        size_t bytes = 0;
        std::filesystem::path path;
    };
    SafeTensorAsset() = default;
    void add_file(const std::filesystem::path& path, const std::string& logical_name,
                  const std::map<std::string, std::string>* owners = nullptr);
    void clear();

    std::vector<Mapping> mappings_;
    std::map<std::string, Tensor> tensors_;
};

}  // namespace vrhino::product
