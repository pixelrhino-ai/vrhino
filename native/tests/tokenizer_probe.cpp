#include <fstream>
#include <iostream>
#include <string>

#include "vrhino/error.h"
#include "vrhino/tokenizer.h"

namespace {

std::string read(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open tokenizer asset: " + path);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void array(const std::vector<int32_t>& values) {
    std::cout << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void array(const std::vector<uint8_t>& values) {
    std::cout << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << static_cast<int>(values[index]);
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 4, "usage: tokenizer-probe FAMILY ASSET TEXT");
        const std::string family = argv[1];
        vrhino::TokenizerSpec spec;
        if (family == "wan") {
            spec.max_length = 512; spec.pad_id = 0; spec.suffix_ids = {1};
        } else if (family == "llama") {
            spec.max_length = 351; spec.pad_id = 128258; spec.prefix_ids = {128000};
        } else if (family == "clip") {
            spec.max_length = 77; spec.pad_id = 49407;
            spec.prefix_ids = {49406}; spec.suffix_ids = {49407};
        } else if (family == "t5") {
            spec.format = vrhino::TokenizerAssetFormat::SentencePiece;
            spec.max_length = 128; spec.pad_id = 0; spec.suffix_ids = {1};
            spec.added_tokens = {{"<extra_id_0>", 32099, true, true}};
        } else if (family == "mochi") {
            spec.format = vrhino::TokenizerAssetFormat::SentencePiece;
            spec.max_length = 256; spec.pad_id = 0; spec.suffix_ids = {1};
            spec.added_tokens = {{"<extra_id_0>", 32099, true, false}};
            spec.suppress_metaspace_after_added_token = true;
            spec.empty_input = vrhino::EmptyInputPolicy::AllPadding;
        } else {
            throw vrhino::Error("Unknown tokenizer family: " + family);
        }
        vrhino::NativeTokenizer tokenizer(spec, read(argv[2]));
        const auto result = tokenizer.encode(argv[3]);
        std::cout << "{\"input_ids\":"; array(result.input_ids);
        std::cout << ",\"attention_mask\":"; array(result.attention_mask);
        std::cout << ",\"valid_length\":" << result.valid_length << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
