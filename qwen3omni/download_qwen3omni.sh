#!/usr/bin/env bash
# Download the Qwen3-Omni captions of CaptionStew 10M from Hugging Face
# (eureka1500/CaptionStew10M-Qwen3Omni, about 8.3 GB).
#
# Usage:
#   ./qwen3omni/download_qwen3omni.sh [output_dir] [shard ...]
#
# Shards: 00 ... 09 (default: all). Examples:
#   ./qwen3omni/download_qwen3omni.sh captions_qwen3omni
#   ./qwen3omni/download_qwen3omni.sh captions_qwen3omni 00 01

set -euo pipefail

out="${1:-captions_qwen3omni}"
shift || true
shards=("$@")
if [ ${#shards[@]} -eq 0 ]; then
  shards=(00 01 02 03 04 05 06 07 08 09)
fi

include=()
for s in "${shards[@]}"; do
  include+=(--include "manifest/CaptionStew_10M_30s_${s}.jsonl.gz")
done

hf download eureka1500/CaptionStew10M-Qwen3Omni \
  --repo-type dataset \
  --local-dir "${out}" \
  "${include[@]}"

echo "Qwen3-Omni captions downloaded to ${out}/manifest/"
