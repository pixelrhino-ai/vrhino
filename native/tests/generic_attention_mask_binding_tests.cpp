#include <cstdint>
#include <iostream>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {

void expect_bool(const vrhino::Tensor& tensor,
                 const std::vector<int64_t>& shape,
                 const std::vector<uint8_t>& expected,
                 const char* label) {
    vrhino::require(tensor.device() == vrhino::DeviceId::host() &&
                        tensor.dtype() == vrhino::DType::Bool &&
                        tensor.shape() == shape,
                    std::string(label) + " descriptor mismatch");
    const uint8_t* actual = tensor.data_as<uint8_t>();
    vrhino::require(expected.size() == static_cast<size_t>(tensor.numel()),
                    std::string(label) + " expected size mismatch");
    for (size_t index = 0; index < expected.size(); ++index)
        vrhino::require(actual[index] == expected[index],
                        std::string(label) + " value mismatch");
}

}  // namespace

int main() {
    try {
        const vrhino::Tensor validity = vrhino::host_bool({1, 3}, {1, 1, 0});
        expect_bool(vrhino::boolean_self_attention_mask(validity, true),
                    {1, 1, 3, 3},
                    {1, 1, 0,
                     1, 1, 0,
                     1, 0, 0},
                    "sentinel self mask");
        expect_bool(vrhino::boolean_self_attention_mask(validity, false),
                    {1, 1, 3, 3},
                    {1, 1, 0,
                     1, 1, 0,
                     0, 0, 0},
                    "strict self mask");
        expect_bool(vrhino::boolean_key_padding_mask(validity, 2),
                    {1, 1, 5}, {1, 1, 1, 1, 0},
                    "key padding mask");

        const vrhino::Tensor batches = vrhino::host_bool(
            {2, 2}, {1, 0, 0, 1});
        expect_bool(vrhino::boolean_self_attention_mask(batches, true),
                    {2, 1, 2, 2},
                    {1, 0, 1, 0,
                     1, 0, 1, 1},
                    "batched self mask");
        expect_bool(vrhino::boolean_key_padding_mask(batches, 1),
                    {2, 1, 3}, {1, 1, 0, 1, 0, 1},
                    "batched key padding mask");

        std::cout << "status=PASS\n"
                  << "semantic=generic_boolean_validity_mask\n"
                  << "sentinel_key=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generic_attention_mask_binding_tests: "
                  << error.what() << '\n';
        return 1;
    }
}
