#!/usr/bin/env bash
# Build smart_grader.ankiaddon and book_to_cards.ankiaddon from source.
# Anki requires files at the ROOT of the .ankiaddon zip, not nested in a folder.
set -euo pipefail

cd "$(dirname "$0")"

build_addon() {
  local name="$1"
  local out="${name}.ankiaddon"
  rm -f "$out"

  # Compile only the package's own .py files (skip vendored deps).
  find "$name" -name "*.py" -not -path "*/_vendor/*" -print0 \
    | xargs -0 python3 -m py_compile

  (cd "$name" && zip -qr "../$out" . \
    -x "*.DS_Store" \
    -x "__pycache__/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc" \
    -x "embedding_cache.sqlite" \
    -x "user_files/*")

  echo "Built $out:"
  unzip -l "$out" | tail -5
}

build_addon smart_grader

if [ -d book_to_cards ]; then
  build_addon book_to_cards
fi
