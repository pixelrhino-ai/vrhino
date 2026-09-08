#include <fstream>
#include <iostream>
#include <string>

#include "vrhino/error.h"
#include "vrhino/input.h"
#include "vrhino/tokenizer.h"

namespace {

std::string read(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open tokenizer asset: " + path);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void check(const vrhino::TokenizedInput& value, int64_t size, int64_t valid,
           int32_t first, int32_t last_valid) {
    vrhino::require(static_cast<int64_t>(value.input_ids.size()) == size, "Token ID length mismatch");
    vrhino::require(value.input_ids.size() == value.attention_mask.size(), "Mask length mismatch");
    vrhino::require(value.valid_length == valid, "Valid length mismatch");
    vrhino::require(value.input_ids.front() == first, "First token mismatch");
    vrhino::require(value.input_ids[static_cast<size_t>(valid - 1)] == last_valid,
                    "Last valid token mismatch");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 5, "usage: tokenizer-tests WAN_JSON LLAMA_JSON CLIP_JSON T5_MODEL");

        vrhino::TokenizerSpec wan;
        wan.max_length = 512; wan.pad_id = 0; wan.suffix_ids = {1};
        vrhino::NativeTokenizer wan_tokenizer(wan, read(argv[1]));
        const auto wan_result = wan_tokenizer.encode("A red panda runs through snow.");
        check(wan_result, 512, 8, 320, 1);

        vrhino::TokenizerSpec llama;
        llama.max_length = 351; llama.pad_id = 128258; llama.prefix_ids = {128000};
        vrhino::NativeTokenizer llama_tokenizer(llama, read(argv[2]));
        const auto llama_result = llama_tokenizer.encode("Describe this video in detail.");
        check(llama_result, 351, 7, 128000, 13);

        vrhino::TokenizerSpec clip;
        clip.max_length = 77; clip.pad_id = 49407; clip.prefix_ids = {49406}; clip.suffix_ids = {49407};
        vrhino::NativeTokenizer clip_tokenizer(clip, read(argv[3]));
        const auto clip_result = clip_tokenizer.encode("A red panda runs through snow.");
        check(clip_result, 77, 9, 49406, 49407);

        vrhino::TokenizerSpec t5;
        t5.format = vrhino::TokenizerAssetFormat::SentencePiece;
        t5.max_length = 128; t5.pad_id = 0; t5.suffix_ids = {1};
        vrhino::NativeTokenizer t5_tokenizer(t5, read(argv[4]));
        const auto t5_result = t5_tokenizer.encode("A red panda runs through snow.");
        check(t5_result, 128, 10, 71, 1);
        const auto unicode = t5_tokenizer.encode("一只猫在雪地里行走 🐈");
        vrhino::require(unicode.valid_length > 1 && unicode.input_ids.back() == 0 &&
                        unicode.attention_mask.front() == 1,
                        "Valid Unicode prompt did not tokenize");

        t5.max_length = 256;
        vrhino::NativeTokenizer mochi_tokenizer(t5, read(argv[4]));
        const auto empty = mochi_tokenizer.encode("");
        vrhino::require(empty.valid_length == 1 && empty.input_ids.front() == 1 &&
                        empty.attention_mask.front() == 1 &&
                        empty.input_ids[1] == 0 && empty.attention_mask[1] == 0,
                        "T5 empty input must retain one valid EOS token");

        t5.empty_input = vrhino::EmptyInputPolicy::AllPadding;
        vrhino::NativeTokenizer all_padding_tokenizer(t5, read(argv[4]));
        const auto all_padding = all_padding_tokenizer.encode("");
        vrhino::require(all_padding.valid_length == 0 &&
                        all_padding.input_ids.front() == 0 &&
                        all_padding.attention_mask.front() == 0,
                        "Explicit generic all-padding policy mismatch");

        bool invalid_failed = false;
        try { (void)t5_tokenizer.encode(std::string("\xc3\x28", 2)); }
        catch (const vrhino::Error&) { invalid_failed = true; }
        vrhino::require(invalid_failed, "Malformed UTF-8 did not fail");
        std::cout << "tokenizer-contract: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
