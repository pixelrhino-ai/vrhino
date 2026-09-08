#!/bin/sh
set -eu

source_root=$1
verification_script=$2
scratch=$(mktemp -d /tmp/vrhino-tokenizer-source-fail-closed.XXXXXX)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM

cp -a --reflink=auto "$source_root" "$scratch/sources"

cmake -DVRHINO_TOKENIZER_BUILD_SOURCES="$scratch/sources" \
  -P "$verification_script" >/dev/null

rm "$scratch/sources/tokenizers-cpp/LICENSE"
if cmake -DVRHINO_TOKENIZER_BUILD_SOURCES="$scratch/sources" \
    -P "$verification_script" >/dev/null 2>&1; then
  echo "missing tokenizer source did not fail closed" >&2
  exit 1
fi

cp "$source_root/tokenizers-cpp/LICENSE" \
  "$scratch/sources/tokenizers-cpp/LICENSE"
printf '\ncorruption-test\n' >> "$scratch/sources/tokenizers-cpp/LICENSE"
if cmake -DVRHINO_TOKENIZER_BUILD_SOURCES="$scratch/sources" \
    -P "$verification_script" >/dev/null 2>&1; then
  echo "corrupt tokenizer source did not fail closed" >&2
  exit 1
fi

cp "$source_root/tokenizers-cpp/LICENSE" \
  "$scratch/sources/tokenizers-cpp/LICENSE"
printf '# corrupt inventory\n' >> \
  "$scratch/sources/inventories/tokenizers-cpp-expanded.sha256"
if cmake -DVRHINO_TOKENIZER_BUILD_SOURCES="$scratch/sources" \
    -P "$verification_script" >/dev/null 2>&1; then
  echo "corrupt tokenizer inventory did not fail closed" >&2
  exit 1
fi

echo "tokenizer source missing/corrupt/hash fail-closed tests: PASS"
