#!/usr/bin/env bash
# Build smart_grader.ankiaddon from the smart_grader/ source tree.
# Anki requires files at the ROOT of the .ankiaddon zip, not nested in a folder.
set -euo pipefail

cd "$(dirname "$0")"
OUT="smart_grader.ankiaddon"
rm -f "$OUT"

# Sanity-check that the Python modules at least parse before we ship.
python3 -m py_compile smart_grader/*.py

# `cd smart_grader && zip ..` so paths in the zip are root-relative.
(cd smart_grader && zip -qr "../$OUT" . \
  -x "*.DS_Store" \
  -x "__pycache__/*" \
  -x "*.pyc" \
  -x "embedding_cache.sqlite")

echo "Built $OUT:"
unzip -l "$OUT"
