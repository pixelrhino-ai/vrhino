#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "vrhino/product/safetensors.h"

namespace fs = std::filesystem;
using vrhino::product::SafeTensorAsset;

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void write_asset(const fs::path& path, float value, bool invalid_range = false) {
    std::string header = invalid_range
        ? R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,12]}})"
        : R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    while (header.size() % 8) header.push_back(' ');
    std::ofstream output(path, std::ios::binary);
    const uint64_t length = header.size();
    for (unsigned byte = 0; byte < 8; ++byte)
        output.put(static_cast<char>((length >> (byte * 8)) & 255));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    const float values[] = {value, -2.5f};
    output.write(reinterpret_cast<const char*>(values), sizeof(values));
    check(output.good(), "fixture write failed");
}

template<class Action> void rejects(Action action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    check(rejected, "invalid asset accepted");
}

#ifdef _WIN32
DWORD handle_count() {
    DWORD count = 0;
    check(GetProcessHandleCount(GetCurrentProcess(), &count) != 0, "handle query failed");
    return count;
}
#endif
}

int main() {
    const fs::path root = fs::current_path() / ("safetensors-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directory(root);
        const auto asset_path = root / "physical-cas-blob";
        const auto invalid_path = root / "invalid";
        const auto short_path = root / "short";
        const auto index_path = root / "index.json";
        write_asset(asset_path, 1.25f);
        write_asset(invalid_path, 1.25f, true);
        { std::ofstream output(short_path); output << "short"; }
        { std::ofstream output(index_path);
          output << R"({"weight_map":{"x":"logical-shard.safetensors"}})"; }

        {
            auto asset = SafeTensorAsset::single(asset_path);
            const auto& tensor = asset.weights().at("x");
            const float* values = tensor.data_as<float>();
            check(values[0] == 1.25f && values[1] == -2.5f, "mapped values changed");
            check(tensor.dtype() == vrhino::DType::F32 && tensor.numel() == 2,
                  "tensor metadata changed");
            check(asset.mapped_bytes() == fs::file_size(asset_path), "mapped size mismatch");
#ifdef _WIN32
            MEMORY_BASIC_INFORMATION memory{};
            check(VirtualQuery(values, &memory, sizeof(memory)) != 0 &&
                  memory.Protect == PAGE_READONLY && memory.Type == MEM_MAPPED,
                  "asset is not a read-only mapped view");
#endif
            auto moved = std::move(asset);
            auto assigned = SafeTensorAsset::single(asset_path);
            assigned = std::move(moved);
            check(assigned.weights().at("x").data_as<float>() == values,
                  "move lost the borrowed view");
            fs::rename(asset_path, root / "renamed");
            write_asset(asset_path, 7.0f);
            check(values[0] == 1.25f, "path replacement changed an open mapping");
            auto replacement = SafeTensorAsset::single(asset_path);
            check(replacement.weights().at("x").data_as<float>()[0] == 7.0f,
                  "replacement not independently mapped");
        }
        {
            auto indexed = SafeTensorAsset::indexed(index_path,
                {{"logical-shard.safetensors", asset_path}});
            check(indexed.weights().at("x").data_as<float>()[0] == 7.0f,
                  "logical shard to physical path resolution failed");
            rejects([&] { SafeTensorAsset::indexed(index_path, {}); });
        }
        // Warm exception machinery before measuring file/section ownership.
        rejects([&] { SafeTensorAsset::single(invalid_path); });
        rejects([&] { SafeTensorAsset::single(short_path); });
        rejects([&] { SafeTensorAsset::single(root / "missing"); });
#ifdef _WIN32
        const DWORD before = handle_count();
#endif
        for (int iteration = 0; iteration < 64; ++iteration) {
            auto valid = SafeTensorAsset::single(asset_path);
            rejects([&] { SafeTensorAsset::single(invalid_path); });
            rejects([&] { SafeTensorAsset::single(short_path); });
            rejects([&] { SafeTensorAsset::single(root / "missing"); });
        }
#ifdef _WIN32
        check(handle_count() == before, "file or section handle leak");
#endif
        // Delete only this test's explicitly named fixtures and empty directory.
        for (const char* name : {"physical-cas-blob", "invalid", "short", "index.json", "renamed"})
            fs::remove(root / name);
        check(fs::remove(root), "fixture directory not empty");
        std::cout << "safetensors read-only mapping, moves, replacement, index, rejection, cleanup: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
