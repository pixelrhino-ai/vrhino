// Copyright 2024 nlcodec / Thamme Gowda
// Fast BPE merge loop using max-heap + doubly-linked list + lazy deletion.
// https://github.com/isi-nlp/nlcodec
// Paper: "Many-to-English Machine Translation Tools, Data, and Pretrained Models"
//        Gowda et al., ACL 2021. https://arxiv.org/abs/2104.00290v2
//
// Algorithm: O(log N) per merge via:
//   1. Max-heap for finding the best pair (O(log N) pop vs O(N) scan)
//   2. Doubly-linked lists for O(1) merge of adjacent tokens
//   3. Lazy deletion: frequency decrements stored in a "dirty" map,
//      applied on pop, avoiding O(N) heap search for updates

#ifndef BPE_MODEL_TRAINER_NLCODEC_H_
#define BPE_MODEL_TRAINER_NLCODEC_H_

#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "trainer_interface.h"
#include "util.h"

namespace sentencepiece {
namespace nlcodec {

// ─── Data Structures ─────────────────────────────────────────────────────────

// Doubly-linked list node for token sequences.
struct LnNode {
  int32_t val;
  LnNode *left = nullptr, *right = nullptr;
  int64_t freq = 1;

  void unlink() {
    if (left) left->right = right;
    if (right) right->left = left;
    left = right = nullptr;
  }
  bool is_unlinked() const { return !left && !right; }
};

// Arena allocator for LnNode: bulk alloc, no individual frees.
class NodeArena {
 public:
  explicit NodeArena(size_t cap = 1 << 20)
      : blocks_(1, std::vector<LnNode>(cap)) {}

  LnNode *alloc(int32_t val, int64_t freq) {
    if (pos_ >= blocks_[blk_].size()) {
      blocks_.push_back(std::vector<LnNode>(blocks_[blk_].size() * 2));
      blk_++;
      pos_ = 0;
    }
    auto *n = &blocks_[blk_][pos_++];
    n->val = val;
    n->freq = freq;
    n->left = n->right = nullptr;
    return n;
  }

  auto from_seq(const std::vector<int32_t> &seq, int64_t freq)
      -> std::vector<LnNode *> {
    std::vector<LnNode *> nodes;
    nodes.reserve(seq.size());
    for (auto tok : seq) nodes.push_back(alloc(tok, freq));
    for (size_t i = 0; i < nodes.size(); i++) {
      if (i > 0) nodes[i]->left = nodes[i - 1];
      if (i + 1 < nodes.size()) nodes[i]->right = nodes[i + 1];
    }
    return nodes;
  }

 private:
  std::vector<std::vector<LnNode>> blocks_;
  size_t blk_ = 0, pos_ = 0;
};

using Bigram = std::pair<int32_t, int32_t>;

struct BigramHash {
  size_t operator()(const Bigram &b) const {
    return std::hash<size_t>{}(
        (static_cast<size_t>(b.first) << 32) |
        static_cast<uint32_t>(b.second));
  }
};

using BigramIndex =
    std::unordered_map<Bigram, std::unordered_set<LnNode *>, BigramHash>;
using HeapDirty = std::unordered_map<Bigram, int64_t, BigramHash>;

class MaxHeap {
 public:
  explicit MaxHeap(
      const std::unordered_map<Bigram, int64_t, BigramHash> &items) {
    for (auto &[bg, f] : items) heap_.push({f, bg});
  }
  void push(const Bigram &bg, int64_t freq) { heap_.push({freq, bg}); }
  auto pop() -> std::pair<Bigram, int64_t> {
    auto [f, bg] = heap_.top();
    heap_.pop();
    return {bg, f};
  }
  auto empty() const -> bool { return heap_.empty(); }

 private:
  std::priority_queue<std::pair<int64_t, Bigram>> heap_;
};

// ─── Fast BPE Training Function ─────────────────────────────────────────────

// Runs the fast BPE merge loop on SP's loaded sentences.
// Expects: sentences_, meta_pieces_, required_chars_ are populated
//          (via LoadSentences + SplitSentencesByWhitespace).
// Populates: final_pieces_ with merged subword pieces.
//
// Uses SP's IsValidSentencePiece for validation and GetCharSymbol
// for required_chars output.
//
// trainer: the bpe::Trainer instance (provides access to SP internals)
// sentences: word → frequency pairs (already whitespace-split with ▁)
// vocab_size: number of merge pieces to produce
// final_pieces: output vector of (piece, score) pairs
// is_valid: validation function for candidate pieces
auto RunFastBPEMerges(
    const TrainerInterface::Sentences &sentences,
    int vocab_size,
    std::vector<std::pair<std::string, float>> *final_pieces,
    std::function<bool(const string_util::UnicodeText &)> is_valid)
    -> util::Status;

}  // namespace nlcodec
}  // namespace sentencepiece

#endif  // BPE_MODEL_TRAINER_NLCODEC_H_
