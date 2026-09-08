#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vrhino {

enum class TokenizerAssetFormat {
    HuggingFaceJson,
    SentencePiece,
};

enum class SequenceSide {
    Left,
    Right,
};

enum class EmptyInputPolicy {
    EncodeNormally,
    AllPadding,
};

struct AddedTokenSpec {
    std::string content;
    int32_t id = -1;
    bool lstrip = false;
    bool rstrip = false;
};

struct TokenizerSpec {
    TokenizerAssetFormat format = TokenizerAssetFormat::HuggingFaceJson;
    int64_t max_length = 0;
    int32_t pad_id = 0;
    SequenceSide padding_side = SequenceSide::Right;
    SequenceSide truncation_side = SequenceSide::Right;
    std::vector<int32_t> prefix_ids;
    std::vector<int32_t> suffix_ids;
    std::vector<AddedTokenSpec> added_tokens;
    bool suppress_metaspace_after_added_token = false;
    EmptyInputPolicy empty_input = EmptyInputPolicy::EncodeNormally;
};

struct TokenizedInput {
    std::vector<int32_t> input_ids;
    std::vector<uint8_t> attention_mask;
    int64_t valid_length = 0;
};

class NativeTokenizer {
public:
    NativeTokenizer(TokenizerSpec spec, const std::string& asset_blob);
    ~NativeTokenizer();
    NativeTokenizer(NativeTokenizer&&) noexcept;
    NativeTokenizer& operator=(NativeTokenizer&&) noexcept;
    NativeTokenizer(const NativeTokenizer&) = delete;
    NativeTokenizer& operator=(const NativeTokenizer&) = delete;

    TokenizedInput encode(const std::string& text) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vrhino
