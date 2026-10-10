#include "../src/product/converter_ltx_internal.h"
#include "vrhino/error.h"
#include <fstream>
#include <iostream>
namespace fs = std::filesystem;
namespace p = vrhino::product;
using namespace vrhino;
int main(int argc, char **argv) {
  try {
    require(argc == 3,
            "usage: ltx-schema2-converter-tests FRESH_ROOT SPEC_ROOT");
    const fs::path root = argv[1], specs = argv[2];
    require(!fs::exists(root), "fresh fixture required");
    fs::create_directories(root / "source");
    int count = 0;
    auto reject = [&](auto operation) {
      bool rejected = false;
      try {
        operation();
      } catch (const std::exception &) {
        rejected = true;
      }
      require(rejected, "negative converter case admitted");
      ++count;
    };
    const auto output = root / "output";
    p::ImportOptions cancelled;
    cancelled.cancellation_requested = [] { return true; };
    reject([&] {
      p::convert_ltx_schema2_package(root / "source", specs, output, cancelled);
    });
    require(!fs::exists(output), "cancellation left output");
    p::ImportOptions no_space;
    no_space.available_space_override = 0;
    reject([&] {
      p::convert_ltx_schema2_package(root / "source", specs, output, no_space);
    });
    require(!fs::exists(output), "disk failure left output");
    fs::create_directory(output);
    std::ofstream(output / "sentinel") << "preserve";
    reject([&] {
      p::convert_ltx_schema2_package(root / "source", specs, output);
    });
    require(fs::file_size(output / "sentinel") == 8,
            "existing output destroyed");
    reject([&] { p::checked_ltx_source(root / "source", "missing"); });
    std::ofstream(root / "outside") << "outside";
    reject([&] { p::checked_ltx_source(root / "source", "../outside"); });
    fs::create_symlink(root / "outside", root / "source/link");
    reject([&] { p::checked_ltx_source(root / "source", "link"); });
    fs::create_directories(root / "changed-specs/ltx_schema2_v1");
    std::ofstream(root /
                  "changed-specs/ltx_schema2_v1/direct-source-contract.json")
        << "{}";
    reject([&] {
      p::convert_ltx_schema2_package(root / "source", root / "changed-specs",
                                     root / "changed-output");
    });
    require(!fs::exists(root / "changed-output"), "spec drift left output");
    const auto checkpoint = root / "source/ltx-video-2b-v0.9.1.safetensors";
    std::ofstream(checkpoint) << "wrong checkpoint";
    p::ImportOptions resources;
    resources.available_space_override = UINT64_MAX;
    reject([&] {
      p::convert_ltx_schema2_package(root / "source", specs,
                                     root / "bad-checkpoint-output", resources);
    });
    require(!fs::exists(root / "bad-checkpoint-output"),
            "bad checkpoint left output");
    // Mapping admission reads a real bounded source, including complete source
    // inventory.
    const auto safe = root / "tiny.safetensors";
    const std::string header =
        R"({"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})";
    {
      std::ofstream file(safe, std::ios::binary);
      uint64_t bytes = header.size();
      for (int i = 0; i < 8; ++i)
        file.put(char(bytes >> (8 * i)));
      file << header;
      file.write("\0\0\0\0", 4);
    }
    p::SafeTensorReader reader(safe);
    std::vector<p::TensorMapping> maps{{"x",
                                        DType::BF16,
                                        {2},
                                        "identity_bytes",
                                        "denoiser.x",
                                        DType::BF16,
                                        {2},
                                        "denoiser",
                                        "weight"}};
    p::validate_ltx_source_mappings(reader, maps);
    auto bad = maps;
    bad[0].source_dtype = DType::F32;
    reject([&] { p::validate_ltx_source_mappings(reader, bad); });
    bad = maps;
    bad[0].destination_shape = {1, 2};
    reject([&] { p::validate_ltx_source_mappings(reader, bad); });
    bad = maps;
    bad[0].transformation = "cast";
    reject([&] { p::validate_ltx_source_mappings(reader, bad); });
    bad = maps;
    bad[0].source_name = "missing";
    reject([&] { p::validate_ltx_source_mappings(reader, bad); });
    bad = maps;
    bad.push_back(maps[0]);
    reject([&] { p::validate_ltx_source_mappings(reader, bad); });
    std::cout << "LTX_SCHEMA2_CONVERTER_NEGATIVES=PASS cases=" << count << '\n';
    fs::remove_all(root);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
