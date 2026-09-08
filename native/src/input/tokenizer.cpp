#include "vrhino/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <utility>

#include <tokenizers_cpp.h>

#include "vrhino/error.h"

namespace vrhino {
namespace {

bool valid_utf8(const std::string& text) {
    size_t index = 0;
    while (index < text.size()) {
        const uint8_t first = static_cast<uint8_t>(text[index++]);
        if (first <= 0x7f) continue;
        int continuation = 0;
        uint32_t codepoint = 0;
        if (first >= 0xc2 && first <= 0xdf) { continuation = 1; codepoint = first & 0x1f; }
        else if (first >= 0xe0 && first <= 0xef) { continuation = 2; codepoint = first & 0x0f; }
        else if (first >= 0xf0 && first <= 0xf4) { continuation = 3; codepoint = first & 0x07; }
        else return false;
        if (index + static_cast<size_t>(continuation) > text.size()) return false;
        for (int offset = 0; offset < continuation; ++offset) {
            const uint8_t next = static_cast<uint8_t>(text[index++]);
            if ((next & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        if ((continuation == 2 && codepoint < 0x800) ||
            (continuation == 3 && codepoint < 0x10000) ||
            codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) return false;
    }
    return true;
}

}  // namespace

class NativeTokenizer::Impl {
public:
    Impl(TokenizerSpec spec, const std::string& blob) : spec_(std::move(spec)) {
        require(spec_.max_length > 0, "Tokenizer max_length must be positive");
        require(static_cast<int64_t>(spec_.prefix_ids.size() + spec_.suffix_ids.size()) <=
                    spec_.max_length, "Tokenizer special tokens exceed max_length");
        for (const auto& token : spec_.added_tokens) {
            require(!token.content.empty() && token.id >= 0, "Invalid tokenizer added-token spec");
        }
        if (spec_.format == TokenizerAssetFormat::HuggingFaceJson) {
            tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
        } else {
            tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
        }
        require(tokenizer_ != nullptr, "Tokenizer asset could not be loaded");
    }

    TokenizedInput encode(const std::string& text) const {
        require(valid_utf8(text), "Tokenizer input is not valid UTF-8");
        TokenizedInput result;
        if (text.empty() && spec_.empty_input == EmptyInputPolicy::AllPadding) {
            result.input_ids.assign(static_cast<size_t>(spec_.max_length), spec_.pad_id);
            result.attention_mask.assign(static_cast<size_t>(spec_.max_length), 0);
            return result;
        }

        std::vector<int32_t> body = encode_body(text);
        const size_t reserved = spec_.prefix_ids.size() + spec_.suffix_ids.size();
        const size_t capacity = static_cast<size_t>(spec_.max_length) - reserved;
        if (body.size() > capacity) {
            if (spec_.truncation_side == SequenceSide::Right) body.resize(capacity);
            else body.erase(body.begin(), body.end() - static_cast<std::ptrdiff_t>(capacity));
        }
        result.input_ids.reserve(static_cast<size_t>(spec_.max_length));
        result.input_ids.insert(result.input_ids.end(), spec_.prefix_ids.begin(), spec_.prefix_ids.end());
        result.input_ids.insert(result.input_ids.end(), body.begin(), body.end());
        result.input_ids.insert(result.input_ids.end(), spec_.suffix_ids.begin(), spec_.suffix_ids.end());
        result.valid_length = static_cast<int64_t>(result.input_ids.size());
        result.attention_mask.assign(result.input_ids.size(), 1);

        const size_t padding = static_cast<size_t>(spec_.max_length) - result.input_ids.size();
        if (spec_.padding_side == SequenceSide::Right) {
            result.input_ids.insert(result.input_ids.end(), padding, spec_.pad_id);
            result.attention_mask.insert(result.attention_mask.end(), padding, 0);
        } else {
            result.input_ids.insert(result.input_ids.begin(), padding, spec_.pad_id);
            result.attention_mask.insert(result.attention_mask.begin(), padding, 0);
        }
        return result;
    }

private:
    std::vector<int32_t> encode_body(const std::string& text) const {
        if (spec_.added_tokens.empty()) return tokenizer_->Encode(text);
        std::vector<int32_t> output;
        size_t position = 0;
        bool after_added_token = false;
        while (position < text.size()) {
            size_t match_position = std::string::npos;
            const AddedTokenSpec* match = nullptr;
            for (const auto& token : spec_.added_tokens) {
                const size_t candidate = text.find(token.content, position);
                if (candidate == std::string::npos) continue;
                if (match == nullptr || candidate < match_position ||
                    (candidate == match_position && token.content.size() > match->content.size())) {
                    match_position = candidate;
                    match = &token;
                }
            }
            if (match == nullptr) {
                auto ids = tokenizer_->Encode(text.substr(position));
                suppress_metaspace(ids, after_added_token);
                output.insert(output.end(), ids.begin(), ids.end());
                break;
            }
            size_t prefix_end = match_position;
            if (match->lstrip) {
                while (prefix_end > position &&
                       std::isspace(static_cast<unsigned char>(text[prefix_end - 1]))) --prefix_end;
            }
            if (prefix_end > position) {
                auto ids = tokenizer_->Encode(text.substr(position, prefix_end - position));
                suppress_metaspace(ids, after_added_token);
                output.insert(output.end(), ids.begin(), ids.end());
            }
            output.push_back(match->id);
            after_added_token = true;
            position = match_position + match->content.size();
            if (match->rstrip) {
                while (position < text.size() &&
                       std::isspace(static_cast<unsigned char>(text[position]))) ++position;
            }
        }
        return output;
    }

    void suppress_metaspace(std::vector<int32_t>& ids, bool enabled) const {
        if (!enabled || !spec_.suppress_metaspace_after_added_token || ids.empty()) return;
        const std::string piece = tokenizer_->IdToToken(ids.front());
        constexpr const char* marker = "▁";
        const std::string prefix(marker);
        if (piece.rfind(prefix, 0) != 0) return;
        const int32_t replacement = tokenizer_->TokenToId(piece.substr(prefix.size()));
        if (replacement >= 0) ids.front() = replacement;
    }

    TokenizerSpec spec_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

NativeTokenizer::NativeTokenizer(TokenizerSpec spec, const std::string& asset_blob)
    : impl_(std::make_unique<Impl>(std::move(spec), asset_blob)) {}
NativeTokenizer::~NativeTokenizer() = default;
NativeTokenizer::NativeTokenizer(NativeTokenizer&&) noexcept = default;
NativeTokenizer& NativeTokenizer::operator=(NativeTokenizer&&) noexcept = default;
TokenizedInput NativeTokenizer::encode(const std::string& text) const { return impl_->encode(text); }

}  // namespace vrhino
