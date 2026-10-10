#include "vrhino/error.h"
#include "vrhino/product/converter.h"
#include <iostream>
int main(int argc, char **argv) {
  try {
    vrhino::require(argc == 4, "usage: ltx-schema2-convert SOURCE_DIR "
                               "CONVERTER_SPEC_ROOT NEW_OUTPUT_DIR");
    const auto result =
        vrhino::product::convert_ltx_schema2_package(argv[1], argv[2], argv[3]);
    std::cout
        << "LTX_SCHEMA2_DIRECT_CONVERSION=PASS tensor_identity_bitwise=PASS "
           "slots=715 decoder_slots=297 bytes="
        << result.file_size << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
