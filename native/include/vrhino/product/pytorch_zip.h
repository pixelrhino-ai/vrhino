#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace vrhino::product {

// Read-only structural view of the stored (uncompressed) members in a
// PyTorch ZIP64 checkpoint. This is deliberately not a pickle executor or a
// general archive extraction API: converters use the audited member ranges as
// immutable tensor sources.
struct PytorchZipEntry {
    std::string name;
    uint64_t data_offset = 0;
    uint64_t byte_length = 0;
    uint32_t crc32 = 0;
};

class PytorchZipArchive {
public:
    explicit PytorchZipArchive(const std::filesystem::path& path,
                               size_t maximum_entries = 4096);
    ~PytorchZipArchive();
    PytorchZipArchive(const PytorchZipArchive&) = delete;
    PytorchZipArchive& operator=(const PytorchZipArchive&) = delete;

    uint64_t file_size() const noexcept { return file_size_; }
    const std::map<std::string, PytorchZipEntry>& entries() const noexcept {
        return entries_;
    }
    const PytorchZipEntry& at(const std::string& name) const;
    std::vector<uint8_t> read(const std::string& name,
                              uint64_t maximum_bytes) const;

private:
    int fd_ = -1;
    uint64_t file_size_ = 0;
    std::map<std::string, PytorchZipEntry> entries_;
};

struct RestrictedPickleInventory {
    uint64_t opcode_count = 0;
    std::vector<std::string> globals;
    std::multiset<std::string> unicode_strings;
};

// Parses only the opcode grammar present in the fixed tensor checkpoints.
// GLOBAL targets are data, never resolved or invoked. Any other opcode or
// global fails closed.
RestrictedPickleInventory inspect_restricted_tensor_pickle(
    const std::vector<uint8_t>& pickle,
    const std::set<std::string>& allowed_globals);

}  // namespace vrhino::product
