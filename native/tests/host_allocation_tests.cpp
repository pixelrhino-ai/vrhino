#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/tensor.h"

namespace {

using vrhino::DType;
using vrhino::Tensor;
using vrhino::require;

void allocation_contract() {
    for (const auto dtype : {DType::F32, DType::F16, DType::BF16, DType::I64,
                            DType::I32, DType::U8, DType::Bool, DType::I8}) {
        for (const int64_t elements : {1, 7, 63, 64, 65, 129}) {
            auto tensor = Tensor::host({elements}, dtype);
            require(tensor.data() != nullptr &&
                        reinterpret_cast<uintptr_t>(tensor.data()) % 64 == 0 &&
                        tensor.alignment() >= 64, "host alignment mismatch");
            require(tensor.bytes() == static_cast<size_t>(elements) * vrhino::dtype_size(dtype) &&
                        tensor.storage_bytes() == tensor.bytes(), "logical allocation size mismatch");
            require(tensor.owns_storage() && !tensor.is_view() &&
                        tensor.device() == vrhino::DeviceId::host() &&
                        tensor.memory_domain() == vrhino::MemoryDomain::HostPageable &&
                        tensor.dtype() == dtype, "host storage metadata mismatch");
            std::memset(tensor.data(), 0x5a, tensor.bytes());
        }
        for (const auto& shape : {std::vector<int64_t>{0}, {2, 0, 3}}) {
            auto empty = Tensor::host(shape, dtype);
            require(empty.defined() && empty.owns_storage() && empty.numel() == 0 &&
                        empty.bytes() == 0 && empty.storage_bytes() == 0 &&
                        empty.shape() == shape && empty.dtype() == dtype,
                    "zero-element tensor metadata mismatch");
            require(empty.data() == nullptr || empty.alignment() >= 64,
                    "zero-element storage alignment mismatch");
            const auto expected_strides = shape.size() == 1
                ? std::vector<int64_t>{1} : std::vector<int64_t>{0, 3, 1};
            require(empty.strides() == expected_strides, "zero-element strides changed");
            auto alias = empty.reshape({0});
            empty = {};
            require(alias.defined() && alias.numel() == 0 && alias.bytes() == 0,
                    "zero-element alias lifetime mismatch");
        }
    }
    auto scalar = Tensor::host({}, DType::F32);
    require(scalar.numel() == 1 && scalar.bytes() == sizeof(float) &&
                scalar.strides().empty(), "rank-zero scalar became empty");
}

int releases = 0;
void counted_free(void* data) {
    ++releases;
    std::free(data);
}

void ownership_contract() {
    Tensor alias;
    {
        auto original = Tensor::host({2, 3}, DType::I32);
        for (int i = 0; i < 6; ++i) original.data_as<int32_t>()[i] = i + 10;
        alias = original.reshape({3, 2});
        auto copy = original;
        require(copy.data() == alias.data(), "host copy lost shared storage");
    }
    require(alias.strides() == std::vector<int64_t>({2, 1}), "reshape strides changed");
    for (int i = 0; i < 6; ++i)
        require(alias.data_as<int32_t>()[i] == i + 10, "host storage released before alias");
    alias.data_as<int32_t>()[5] = 42;
    alias = {}; // Sanitizers check the real host allocator/deallocator pair.

    {
        auto storage = std::make_shared<vrhino::Storage>();
        storage->data = std::malloc(64);
        require(storage->data != nullptr, "test allocation failed");
        storage->bytes = 64;
        storage->owner = true;
        storage->deleter = counted_free;
        Tensor original(storage, 0, {16}, DType::I32);
        alias = original.reshape({4, 4});
    }
    require(releases == 0, "deleter ran before final alias release");
    alias = {};
    require(releases == 1, "deleter did not run exactly once");
    alias = {};
    require(releases == 1, "deleter ran twice");

    int32_t borrowed_value = 7;
    { auto borrowed = Tensor::borrowed(&borrowed_value, sizeof(borrowed_value), {1}, DType::I32);
      require(!borrowed.owns_storage(), "borrowed tensor became owning"); }
    require(borrowed_value == 7, "borrowed storage changed on destruction");
}

void rejects(const std::vector<int64_t>& shape, DType dtype, const char* message) {
    try { (void)Tensor::host(shape, dtype); }
    catch (const vrhino::Error& error) {
        require(error.what() == std::string(message), "wrong overflow rejection");
        return;
    }
    throw vrhino::Error(std::string("missing rejection: ") + message);
}

void overflow_contract() {
    rejects({-1}, DType::U8, "Negative tensor dimension");
    rejects({INT64_MAX, 2}, DType::U8, "Tensor numel overflow");
    rejects({static_cast<int64_t>(std::numeric_limits<size_t>::max() / 8) + 1},
            DType::I64, "Tensor byte size overflow");
    if constexpr (sizeof(size_t) == 8)
        rejects({INT64_MAX}, DType::F16, "Host tensor allocation size overflow");
    // Zero bytes pass allocation, then construction rejects overflowing strides.
    // This also exercises allocation cleanup during constructor unwinding.
    rejects({0, INT64_MAX, 2}, DType::U8, "Tensor stride overflow");
}

} // namespace

int main() {
    try {
        allocation_contract();
        ownership_contract();
        overflow_contract();
        std::cout << "host allocation alignment/empty/lifetime/overflow: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "host allocation: FAIL: " << error.what() << '\n';
        return 1;
    }
}
