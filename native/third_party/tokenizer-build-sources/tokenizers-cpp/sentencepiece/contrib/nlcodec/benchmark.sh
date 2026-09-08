#!/bin/bash
# Benchmark: default BPE vs nlcodec_bpe in SentencePiece
#
# Self-contained: downloads CC-100 multilingual data, builds SentencePiece,
# and runs the benchmark. No external dependencies beyond standard tools.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
DATA_DIR="${SCRIPT_DIR}/data"
SPM_TRAIN="${BUILD_DIR}/src/spm_train"
SPM_ENCODE="${BUILD_DIR}/src/spm_encode"

# ─── Defaults ─────────────────────────────────────────────────────────────────
NUM_LINES=200000
VOCAB_SIZE=32000
INPUT=""
SKIP_ENCODE=false
BUILD_JOBS="$(nproc)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark default BPE vs nlcodec_bpe (--nlcodec_bpe) training speed.
Auto-downloads CC-100 multilingual data and builds SentencePiece if needed.

Options:
  -n, --lines NUM       Number of lines to download from CC-100 (default: 200000)
  -v, --vocab NUM       Vocabulary size (default: 32000)
  -i, --input FILE      Use custom input file instead of auto-download
  -j, --jobs NUM        Parallel build jobs (default: nproc)
  -s, --skip-encode     Skip encoding comparison (faster)
  -h, --help            Show this help

Examples:
  $(basename "$0")                          # 200k lines, 32k vocab
  $(basename "$0") -n 1000000              # 1M lines
  $(basename "$0") -n 200000 -v 16000      # 200k lines, 16k vocab
  $(basename "$0") -i data.txt -v 32000    # custom input
  $(basename "$0") -n 50000 -s             # quick run, skip encoding
EOF
  exit 0
}

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--lines)   NUM_LINES="$2"; shift 2 ;;
    -v|--vocab)   VOCAB_SIZE="$2"; shift 2 ;;
    -i|--input)   INPUT="$2"; shift 2 ;;
    -j|--jobs)    BUILD_JOBS="$2"; shift 2 ;;
    -s|--skip-encode) SKIP_ENCODE=true; shift ;;
    -h|--help)    usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# ─── Step 1: Build SentencePiece if needed ────────────────────────────────────
if [[ ! -x "$SPM_TRAIN" ]]; then
  echo "Building SentencePiece (with -DSPM_NLCODEC_BPE=ON)..."
  mkdir -p "$BUILD_DIR"
  (cd "$BUILD_DIR" && cmake "$ROOT_DIR" -DSPM_BUILD_TEST=OFF -DSPM_NLCODEC_BPE=ON \
    && cmake --build . -j"$BUILD_JOBS") \
    2>&1 | tail -5
  echo ""
fi

if [[ ! -x "$SPM_TRAIN" ]]; then
  echo "ERROR: Build failed. spm_train not found at $SPM_TRAIN"
  exit 1
fi

# ─── Step 2: Download CC-100 data if needed ───────────────────────────────────
if [[ -z "$INPUT" ]]; then
  LINES_PER_LANG=$(( NUM_LINES / 5 ))
  INPUT="${DATA_DIR}/train_${NUM_LINES}.txt"

  if [[ -f "$INPUT" ]]; then
    echo "Data already exists: $INPUT ($(wc -l < "$INPUT") lines)"
  else
    echo "Downloading CC-100 multilingual data ($NUM_LINES lines total)..."
    echo "  Languages: en, de, zh-Hans, ar, hi ($LINES_PER_LANG lines each)"
    mkdir -p "$DATA_DIR"

    LANGS=(en de zh-Hans ar hi)
    for lang in "${LANGS[@]}"; do
      LANG_FILE="$DATA_DIR/${lang}.txt"
      if [[ -f "$LANG_FILE" ]] && [[ $(wc -l < "$LANG_FILE") -ge $LINES_PER_LANG ]]; then
        echo "  $lang: already downloaded"
      else
        URL="https://data.statmt.org/cc-100/${lang}.txt.xz"
        echo "  $lang: streaming from $URL..."
        # Stream + decompress + take first N lines
        # CC-100 contains Unicode line/paragraph separators (U+2028/U+2029); normalize to newlines
        # Disable pipefail: head causes SIGPIPE when it has enough lines, which is expected
        set +o pipefail
        rm -f "$LANG_FILE.tmp" $LANG_FILE
        curl -sL "$URL" | xzcat \
          | perl -CSD -pe 's/[\x{2028}\x{2029}]/\n/g' \
          | head -n "$LINES_PER_LANG" > "$LANG_FILE.tmp" || true
        mv "$LANG_FILE.tmp" "$LANG_FILE"
        set -o pipefail
        echo "  $lang: $(wc -l < "$LANG_FILE") lines"
      fi
    done

    echo "Combining and shuffling..."
    for lang in "${LANGS[@]}"; do
      head -n "$LINES_PER_LANG" "$DATA_DIR/${lang}.txt"
    done | shuf > "$INPUT"

    echo "Created: $INPUT ($(wc -l < "$INPUT") lines)"
  fi
  echo ""
fi

# ─── Step 3: Run benchmark ───────────────────────────────────────────────────
RESULT_DIR=$(mktemp -d)
trap "rm -rf $RESULT_DIR" EXIT

LINES=$(wc -l < "$INPUT")
echo "=============================================="
echo "  BPE Training Benchmark"
echo "  Input: $(basename "$INPUT") ($LINES lines)"
echo "  Vocab: $VOCAB_SIZE"
echo "=============================================="

# --- Default BPE ---
echo ""
echo "--- Default BPE ---"
START=$(date +%s%N)
$SPM_TRAIN \
  --model_prefix="$RESULT_DIR/default" \
  --input="$INPUT" \
  --vocab_size="$VOCAB_SIZE" \
  --model_type=bpe \
  --max_sentence_length=4192 \
  2>&1 | grep -E "LOG\(INFO\) (Saving|Loaded|Done!)" || true
END=$(date +%s%N)
DEFAULT_MS=$(( (END - START) / 1000000 ))
echo "Time: ${DEFAULT_MS}ms ($(python3 -c "print(f'{${DEFAULT_MS}/1000:.1f}s')"))"

# --- Nlcodec BPE ---
echo ""
echo "--- Nlcodec BPE (--nlcodec_bpe) ---"
START=$(date +%s%N)
$SPM_TRAIN \
  --nlcodec_bpe \
  --model_prefix="$RESULT_DIR/nlcodec" \
  --input="$INPUT" \
  --vocab_size="$VOCAB_SIZE" \
  --model_type=bpe \
  --max_sentence_length=4192 \
  2>&1 | grep -E "LOG\(INFO\) (Saving|Loaded|Done!|nlcodec)" || true
END=$(date +%s%N)
NLCODEC_MS=$(( (END - START) / 1000000 ))
echo "Time: ${NLCODEC_MS}ms ($(python3 -c "print(f'{${NLCODEC_MS}/1000:.1f}s')"))"

# --- Vocab comparison ---
echo ""
echo "--- Vocab Comparison ---"
DEFAULT_VOCAB=$(wc -l < "$RESULT_DIR/default.vocab")
NLCODEC_VOCAB=$(wc -l < "$RESULT_DIR/nlcodec.vocab")
echo "Default vocab: $DEFAULT_VOCAB"
echo "Nlcodec vocab: $NLCODEC_VOCAB"

OVERLAP=$(comm -12 \
  <(cut -f1 "$RESULT_DIR/default.vocab" | sort) \
  <(cut -f1 "$RESULT_DIR/nlcodec.vocab" | sort) | wc -l)
echo "Overlap: $OVERLAP / $DEFAULT_VOCAB ($(python3 -c "print(f'{${OVERLAP}/${DEFAULT_VOCAB}*100:.1f}%')"))"

# --- Encoding comparison ---
if [[ "$SKIP_ENCODE" == false ]] && [[ -x "$SPM_ENCODE" ]]; then
  echo ""
  echo "--- Encoding Comparison ---"
  DEFAULT_TOKS=$($SPM_ENCODE --model="$RESULT_DIR/default.model" --input="$INPUT" | awk '{n+=NF} END{print n}')
  NLCODEC_TOKS=$($SPM_ENCODE --model="$RESULT_DIR/nlcodec.model" --input="$INPUT" | awk '{n+=NF} END{print n}')
  echo "Default total tokens: $DEFAULT_TOKS"
  echo "Nlcodec total tokens: $NLCODEC_TOKS"
  echo "Mean sent len (default): $(python3 -c "print(f'{${DEFAULT_TOKS}/${LINES}:.2f}')")"
  echo "Mean sent len (nlcodec): $(python3 -c "print(f'{${NLCODEC_TOKS}/${LINES}:.2f}')")"
fi

# --- Summary ---
echo ""
echo "=============================================="
if [[ $NLCODEC_MS -gt 0 ]]; then
  SPEEDUP=$(python3 -c "print(f'{${DEFAULT_MS}/${NLCODEC_MS}:.1f}')")
  echo "  Default:  $(python3 -c "print(f'{${DEFAULT_MS}/1000:.1f}s')")"
  echo "  Nlcodec:  $(python3 -c "print(f'{${NLCODEC_MS}/1000:.1f}s')")"
  echo "  Speedup:  ${SPEEDUP}x"
else
  echo "  Speedup: (nlcodec too fast to measure)"
fi
echo "=============================================="
