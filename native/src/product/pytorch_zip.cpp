#include "vrhino/product/pytorch_zip.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#ifdef _WIN32
#include "windows_converter_io.h"
#else
#include <unistd.h>
#endif

#include "vrhino/error.h"

namespace vrhino::product {
#ifdef _WIN32
using namespace windows_converter_io;
static_assert(sizeof(off_t) == 8 && sizeof(ssize_t) == 8);
#endif
namespace {

constexpr uint32_t kLocalHeader = 0x04034b50U;
constexpr uint32_t kCentralHeader = 0x02014b50U;
constexpr uint32_t kEndOfCentralDirectory = 0x06054b50U;
constexpr uint32_t kZip64End = 0x06064b50U;
constexpr uint32_t kZip64Locator = 0x07064b50U;

uint16_t u16(const uint8_t* value) {
    return static_cast<uint16_t>(value[0]) |
           static_cast<uint16_t>(value[1]) << 8;
}

uint32_t u32(const uint8_t* value) {
    return static_cast<uint32_t>(value[0]) |
           static_cast<uint32_t>(value[1]) << 8 |
           static_cast<uint32_t>(value[2]) << 16 |
           static_cast<uint32_t>(value[3]) << 24;
}

uint64_t u64(const uint8_t* value) {
    uint64_t result = 0;
    for (int index = 7; index >= 0; --index)
        result = (result << 8) | value[index];
    return result;
}

void read_exact(int fd, uint64_t file_size, uint64_t offset, void* destination,
                size_t bytes, const std::string& context) {
    require(offset <= file_size && bytes <= file_size - offset,
            "PyTorch ZIP range exceeds checkpoint while " + context);
    auto* output = static_cast<uint8_t*>(destination);
    size_t completed = 0;
    while (completed < bytes) {
        const ssize_t count = pread(fd, output + completed, bytes - completed,
                                    static_cast<off_t>(offset + completed));
        require(count > 0, "truncated PyTorch ZIP while " + context);
        completed += static_cast<size_t>(count);
    }
}

std::vector<uint8_t> read_range(int fd, uint64_t file_size, uint64_t offset,
                                size_t bytes, const std::string& context) {
    std::vector<uint8_t> result(bytes);
    read_exact(fd, file_size, offset, result.data(), bytes, context);
    return result;
}

uint64_t zip64_value(const uint8_t*& cursor, const uint8_t* end,
                     const std::string& context) {
    require(static_cast<size_t>(end - cursor) >= 8,
            "truncated ZIP64 extra field while " + context);
    const uint64_t result = u64(cursor);
    cursor += 8;
    return result;
}

void skip(const std::vector<uint8_t>& bytes, size_t& cursor, size_t count,
          const char* context) {
    require(cursor <= bytes.size() && count <= bytes.size() - cursor,
            std::string("truncated restricted pickle ") + context);
    cursor += count;
}

std::string newline_string(const std::vector<uint8_t>& bytes, size_t& cursor) {
    const size_t begin = cursor;
    while (cursor < bytes.size() && bytes[cursor] != '\n') ++cursor;
    require(cursor < bytes.size(), "unterminated restricted pickle GLOBAL");
    std::string result(reinterpret_cast<const char*>(bytes.data() + begin),
                       cursor - begin);
    ++cursor;
    return result;
}

}  // namespace

PytorchZipArchive::PytorchZipArchive(const std::filesystem::path& path,
                                     size_t maximum_entries) {
    require(maximum_entries > 0 && maximum_entries <= 65536,
            "PyTorch ZIP entry bound is invalid");
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(fd_ >= 0, "cannot open PyTorch ZIP checkpoint");
#ifdef _WIN32
    struct _stat64 status{};
#else
    struct stat status{};
#endif
    require(fstat(fd_, &status) == 0 && status.st_size >= 98,
            "truncated PyTorch ZIP checkpoint");
    file_size_ = static_cast<uint64_t>(status.st_size);

    const size_t tail_bytes = static_cast<size_t>(std::min<uint64_t>(
        file_size_, 65535ULL + 22ULL + 20ULL));
    const uint64_t tail_offset = file_size_ - tail_bytes;
    const std::vector<uint8_t> tail = read_range(
        fd_, file_size_, tail_offset, tail_bytes, "reading end record");
    size_t eocd = std::numeric_limits<size_t>::max();
    for (size_t index = tail.size() - 22;; --index) {
        if (u32(tail.data() + index) == kEndOfCentralDirectory &&
            index + 22 + u16(tail.data() + index + 20) == tail.size()) {
            eocd = index;
            break;
        }
        if (index == 0) break;
    }
    require(eocd != std::numeric_limits<size_t>::max(),
            "PyTorch ZIP end record is missing");
    require(u16(tail.data() + eocd + 4) == 0 &&
                u16(tail.data() + eocd + 6) == 0,
            "multi-disk PyTorch ZIP is unsupported");

    uint64_t entry_count = u16(tail.data() + eocd + 10);
    uint64_t central_size = u32(tail.data() + eocd + 12);
    uint64_t central_offset = u32(tail.data() + eocd + 16);
    if (entry_count == 0xffffU || central_size == 0xffffffffU ||
        central_offset == 0xffffffffU) {
        const uint64_t absolute_eocd = tail_offset + eocd;
        require(absolute_eocd >= 20, "PyTorch ZIP64 locator is missing");
        std::array<uint8_t, 20> locator{};
        read_exact(fd_, file_size_, absolute_eocd - locator.size(),
                   locator.data(), locator.size(), "reading ZIP64 locator");
        require(u32(locator.data()) == kZip64Locator &&
                    u32(locator.data() + 4) == 0 &&
                    u32(locator.data() + 16) == 1,
                "unsupported PyTorch ZIP64 locator");
        const uint64_t zip64_offset = u64(locator.data() + 8);
        std::array<uint8_t, 56> record{};
        read_exact(fd_, file_size_, zip64_offset, record.data(), record.size(),
                   "reading ZIP64 end record");
        require(u32(record.data()) == kZip64End && u64(record.data() + 4) >= 44 &&
                    u32(record.data() + 16) == 0 && u32(record.data() + 20) == 0,
                "unsupported PyTorch ZIP64 end record");
        require(u64(record.data() + 24) == u64(record.data() + 32),
                "PyTorch ZIP64 entry counts disagree");
        entry_count = u64(record.data() + 32);
        central_size = u64(record.data() + 40);
        central_offset = u64(record.data() + 48);
    }
    require(entry_count > 0 && entry_count <= maximum_entries,
            "PyTorch ZIP entry count exceeds fixed bound");
    require(central_offset <= file_size_ && central_size <= file_size_ - central_offset,
            "PyTorch ZIP central directory range is invalid");

    uint64_t cursor = central_offset;
    const uint64_t central_end = central_offset + central_size;
    for (uint64_t index = 0; index < entry_count; ++index) {
        std::array<uint8_t, 46> header{};
        read_exact(fd_, file_size_, cursor, header.data(), header.size(),
                   "reading central directory");
        require(u32(header.data()) == kCentralHeader,
                "bad PyTorch ZIP central-directory signature");
        const uint16_t flags = u16(header.data() + 8);
        const uint16_t method = u16(header.data() + 10);
        require((flags & 1U) == 0 && method == 0,
                "compressed/encrypted PyTorch ZIP member is unsupported");
        const uint16_t name_bytes = u16(header.data() + 28);
        const uint16_t extra_bytes = u16(header.data() + 30);
        const uint16_t comment_bytes = u16(header.data() + 32);
        require(name_bytes > 0 && name_bytes <= 4096 && extra_bytes <= 4096,
                "PyTorch ZIP member metadata exceeds fixed bound");
        const uint64_t variable_bytes = static_cast<uint64_t>(name_bytes) +
            extra_bytes + comment_bytes;
        require(cursor + header.size() <= central_end &&
                    variable_bytes <= central_end - cursor - header.size(),
                "PyTorch ZIP central member exceeds directory");
        const std::vector<uint8_t> variable = read_range(
            fd_, file_size_, cursor + header.size(),
            static_cast<size_t>(variable_bytes), "reading member metadata");
        const std::string name(reinterpret_cast<const char*>(variable.data()),
                               name_bytes);
        require(name.find('\0') == std::string::npos && name.front() != '/' &&
                    name.find("../") == std::string::npos,
                "unsafe PyTorch ZIP member name");

        uint64_t compressed = u32(header.data() + 20);
        uint64_t uncompressed = u32(header.data() + 24);
        uint64_t local_offset = u32(header.data() + 42);
        const uint8_t* extra = variable.data() + name_bytes;
        const uint8_t* extra_end = extra + extra_bytes;
        bool found_zip64 = false;
        while (extra < extra_end) {
            require(static_cast<size_t>(extra_end - extra) >= 4,
                    "truncated PyTorch ZIP extra field");
            const uint16_t kind = u16(extra);
            const uint16_t bytes = u16(extra + 2);
            extra += 4;
            require(static_cast<size_t>(extra_end - extra) >= bytes,
                    "PyTorch ZIP extra field exceeds member");
            if (kind == 1) {
                const uint8_t* value = extra;
                const uint8_t* value_end = extra + bytes;
                if (uncompressed == 0xffffffffU)
                    uncompressed = zip64_value(value, value_end, "reading member size");
                if (compressed == 0xffffffffU)
                    compressed = zip64_value(value, value_end, "reading compressed size");
                if (local_offset == 0xffffffffU)
                    local_offset = zip64_value(value, value_end, "reading local offset");
                found_zip64 = true;
            }
            extra += bytes;
        }
        require((u32(header.data() + 20) != 0xffffffffU &&
                 u32(header.data() + 24) != 0xffffffffU &&
                 u32(header.data() + 42) != 0xffffffffU) || found_zip64,
                "PyTorch ZIP64 member extra is missing");
        require(compressed == uncompressed,
                "stored PyTorch ZIP member size mismatch");

        std::array<uint8_t, 30> local{};
        read_exact(fd_, file_size_, local_offset, local.data(), local.size(),
                   "reading local member header");
        require(u32(local.data()) == kLocalHeader &&
                    u16(local.data() + 6) == flags &&
                    u16(local.data() + 8) == method,
                "invalid PyTorch ZIP local member header");
        const uint16_t local_name_bytes = u16(local.data() + 26);
        const uint16_t local_extra_bytes = u16(local.data() + 28);
        require(local_name_bytes == name_bytes,
                "PyTorch ZIP local/central name length mismatch");
        const std::vector<uint8_t> local_name = read_range(
            fd_, file_size_, local_offset + local.size(), local_name_bytes,
            "reading local member name");
        require(std::equal(local_name.begin(), local_name.end(), variable.begin()),
                "PyTorch ZIP local/central member name mismatch");
        const uint64_t data_offset = local_offset + local.size() +
            local_name_bytes + local_extra_bytes;
        require(data_offset <= file_size_ && uncompressed <= file_size_ - data_offset,
                "PyTorch ZIP member data range exceeds checkpoint");
        require(entries_.emplace(name, PytorchZipEntry{
            name, data_offset, uncompressed, u32(header.data() + 16)}).second,
            "duplicate PyTorch ZIP member");
        cursor += header.size() + variable_bytes;
    }
    require(cursor == central_end, "PyTorch ZIP central-directory size mismatch");
}

PytorchZipArchive::~PytorchZipArchive() {
    if (fd_ >= 0) close(fd_);
}

const PytorchZipEntry& PytorchZipArchive::at(const std::string& name) const {
    const auto found = entries_.find(name);
    require(found != entries_.end(), "missing PyTorch ZIP member: " + name);
    return found->second;
}

std::vector<uint8_t> PytorchZipArchive::read(const std::string& name,
                                             uint64_t maximum_bytes) const {
    const PytorchZipEntry& entry = at(name);
    require(entry.byte_length <= maximum_bytes &&
                entry.byte_length <= std::numeric_limits<size_t>::max(),
            "PyTorch ZIP member exceeds read bound: " + name);
    return read_range(fd_, file_size_, entry.data_offset,
                      static_cast<size_t>(entry.byte_length), "reading " + name);
}

RestrictedPickleInventory inspect_restricted_tensor_pickle(
        const std::vector<uint8_t>& pickle,
        const std::set<std::string>& allowed_globals) {
    require(!pickle.empty() && pickle.size() <= 16ULL * 1024 * 1024,
            "restricted tensor pickle size is invalid");
    RestrictedPickleInventory result;
    size_t cursor = 0;
    bool stopped = false;
    while (cursor < pickle.size()) {
        const uint8_t opcode = pickle[cursor++];
        ++result.opcode_count;
        switch (opcode) {
            case 0x80:
                require(cursor < pickle.size() && pickle[cursor] == 2,
                        "unsupported restricted pickle protocol");
                skip(pickle, cursor, 1, "PROTO");
                break;
            case '}': case '(': case 't': case 'Q': case ')': case 'R':
            case 'u': case 's': case 0x85: case 0x86: case 0x87:
            case 0x89: break;
            case 'q': case 'h': case 'K': skip(pickle, cursor, 1, "byte operand"); break;
            case 'r': case 'J': skip(pickle, cursor, 4, "integer operand"); break;
            case 'M': skip(pickle, cursor, 2, "integer operand"); break;
            case 'X': {
                require(cursor + 4 <= pickle.size(),
                        "truncated restricted pickle BINUNICODE size");
                const uint32_t bytes = u32(pickle.data() + cursor);
                cursor += 4;
                require(bytes <= 1ULL * 1024 * 1024,
                        "restricted pickle unicode exceeds bound");
                require(cursor <= pickle.size() && bytes <= pickle.size() - cursor,
                        "truncated restricted pickle BINUNICODE");
                result.unicode_strings.emplace(
                    reinterpret_cast<const char*>(pickle.data() + cursor), bytes);
                cursor += bytes;
                break;
            }
            case 'c': {
                const std::string module = newline_string(pickle, cursor);
                const std::string name = newline_string(pickle, cursor);
                const std::string global = module + "." + name;
                require(allowed_globals.contains(global),
                        "unsupported restricted pickle global: " + global);
                result.globals.push_back(global);
                break;
            }
            case '.':
                require(cursor == pickle.size(),
                        "trailing data after restricted pickle STOP");
                stopped = true;
                break;
            default:
                throw Error("unsupported restricted tensor pickle opcode: " +
                            std::to_string(opcode));
        }
        if (stopped) break;
    }
    require(stopped, "restricted tensor pickle STOP is missing");
    return result;
}

}  // namespace vrhino::product
