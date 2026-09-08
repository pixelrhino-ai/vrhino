#pragma once

#include <map>
#include <string>
#include <vector>

#include "vrhino/architecture.h"

namespace vrhino::phase13 {

// Research-only mmap loader for fixed extracted/reference safetensors.  Final
// one-container execution remains VRM-owned; this keeps Phase 13C conversion
// validation independent from a premature VRM format migration.
class SafeTensorSet {
public:
    static SafeTensorSet single(const std::string& path);
    static SafeTensorSet indexed(const std::string& index_path);
    ~SafeTensorSet();
    SafeTensorSet(SafeTensorSet&& other) noexcept;
    SafeTensorSet& operator=(SafeTensorSet&& other) noexcept;
    SafeTensorSet(const SafeTensorSet&) = delete;
    SafeTensorSet& operator=(const SafeTensorSet&) = delete;

    WeightMap weights() const;
    size_t mapped_bytes() const;

private:
    struct Mapping { int fd = -1; void* data = nullptr; size_t bytes = 0; std::string path; };
    SafeTensorSet() = default;
    void add_file(const std::string& path, const std::map<std::string, std::string>* owners = nullptr);
    void clear();

    std::vector<Mapping> mappings_;
    std::map<std::string, Tensor> tensors_;
};

}  // namespace vrhino::phase13
