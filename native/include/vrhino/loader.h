#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "vrhino/json.h"
#include "vrhino/tensor.h"

namespace vrhino {

struct TensorRecord {
    std::string name;
    Tensor tensor;
    size_t relative_offset = 0;
    size_t byte_length = 0;
};

class VrmModel {
public:
    // On Windows, narrow paths are UTF-8; native UTF-16 paths use the overload
    // below. The model owns the mapping: it must outlive all borrowed tensors,
    // including their copies and reshape views.
    explicit VrmModel(const std::string& path, bool verify_checksum = true);
#ifdef _WIN32
    explicit VrmModel(const std::wstring& path, bool verify_checksum = true);
#endif
    // Takes ownership of an already-open read-only descriptor. This permits
    // callers to bind parsing and payload verification to a stable open-file
    // identity rather than reopening a pathname.
    // On Windows this is a CRT descriptor owning a native HANDLE (for example,
    // from _open_osfhandle); ownership is consumed even if construction fails.
    explicit VrmModel(int owned_descriptor, bool verify_checksum = true);
    ~VrmModel();
    VrmModel(const VrmModel&) = delete;
    VrmModel& operator=(const VrmModel&) = delete;

    const std::string& profile_id() const { return profile_id_; }
    const std::string& architecture_id() const { return architecture_id_; }
    const Json& metadata() const { return metadata_; }
    const Json& graph() const { return graph_; }
    const std::map<std::string, TensorRecord>& tensors() const { return tensors_; }
    const Tensor& tensor(const std::string& canonical_name) const;
    std::map<std::string, const Tensor*> bindings(const Json& descriptor) const;
    uint64_t file_size() const { return file_size_; }
    double load_seconds() const { return load_seconds_; }
    double mmap_setup_seconds() const { return mmap_setup_seconds_; }
    double checksum_seconds() const { return checksum_seconds_; }
    double metadata_parse_seconds() const { return metadata_parse_seconds_; }

private:
    void release_storage() noexcept;

#ifdef _WIN32
    // Constructed before potentially allocating JSON/map members, so an
    // exception during member initialization also closes the consumed file.
    struct OwnedDescriptor {
        int value = -1;
        ~OwnedDescriptor();
    };
    OwnedDescriptor fd_;
#else
    int fd_ = -1;
#endif
    void* mapping_ = nullptr;
    size_t mapping_size_ = 0;
    std::string profile_id_;
    std::string architecture_id_;
    Json metadata_;
    Json graph_;
    std::map<std::string, TensorRecord> tensors_;
    uint64_t file_size_ = 0;
    double load_seconds_ = 0.0;
    double mmap_setup_seconds_ = 0.0;
    double checksum_seconds_ = 0.0;
    double metadata_parse_seconds_ = 0.0;
};

}  // namespace vrhino
