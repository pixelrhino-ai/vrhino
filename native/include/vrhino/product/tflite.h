#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vrhino::product {

// Read-only structural inventory for bounded fixed-source TFLite converters.
// This is deliberately not an interpreter: it validates FlatBuffer ranges and
// exposes tensors/operators so a fixed converter can fail closed on drift.
struct TfliteTensorRecord {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint32_t buffer_index = 0;
    uint64_t data_offset = 0;
    uint64_t data_bytes = 0;
};

struct TfliteOperatorRecord {
    std::string type;
    int32_t version = 0;
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;
    // Canonical scalar/vector option values for the small supported builtin
    // option tables. Fixed converters validate their exact meaning.
    std::vector<int64_t> options;
};

struct TfliteModelInventory {
    int32_t schema_version = 0;
    uint64_t file_bytes = 0;
    uint32_t subgraph_count = 0;
    uint32_t buffer_count = 0;
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;
    std::vector<TfliteTensorRecord> tensors;
    std::vector<TfliteOperatorRecord> operators;
};

TfliteModelInventory inspect_tflite_flatbuffer(
    const std::filesystem::path& path);

}  // namespace vrhino::product
