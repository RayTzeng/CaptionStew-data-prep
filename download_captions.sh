#!/usr/bin/env bash
# Download the CaptionStew caption manifests from Hugging Face.
#
# Usage:
#   ./download_captions.sh [output_dir] [subset ...]
#
# Folders: 10M (CaptionStew 10M, sharded), dataset_splits (one file per caption dataset),
#          400k, 1M, 4M (smaller data-scaling subsets). Default: 10M dataset_splits.
# The audio recipes read dataset_splits/ to find the clips they need, so keep it in the list.
# Examples:
#   ./download_captions.sh captions
#   ./download_captions.sh captions dataset_splits
#
# Set CAPTIONSTEW_REVISION to pin a release revision (default: main).

set -euo pipefail

out="${1:-captions}"
shift || true
subsets=("$@")
if [ ${#subsets[@]} -eq 0 ]; then
  subsets=(10M dataset_splits)
fi

include=(--include "README.md")
for s in "${subsets[@]}"; do
  include+=(--include "${s}/*.jsonl.gz")
done

hf download RayTseng0406/CaptionStew \
  --repo-type dataset \
  --revision "${CAPTIONSTEW_REVISION:-main}" \
  --local-dir "${out}" \
  "${include[@]}"

echo "Captions downloaded to ${out}/"
