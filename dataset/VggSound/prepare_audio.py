#!/usr/bin/env python3
"""
Download the VGGSound clips used by CaptionStew from Hugging Face.

Audio comes from the 16 kHz WAV copy of VGGSound on Hugging Face
(txya900619/vggsound-16k). Each parquet shard of the train split is downloaded,
the clips referenced by the CaptionStew manifests are written as individual WAV
files, and the shard is deleted again.

Output layout (matches the "vggsound/..." keys in the CaptionStew manifests):

    ${vggsound_root}/<youtube_id>_<start>.wav

Usage:
    # Test mode (process only 2 shards)
    python prepare_audio.py --captions-dir /path/to/captions --vggsound-root /path/to/vggsound --limit-shards 2

    # Full run
    python prepare_audio.py --captions-dir /path/to/captions --vggsound-root /path/to/vggsound --workers 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.hf_parquet import collect_keys, count_files, extract_shards, list_shards  # noqa: E402

REPO = "txya900619/vggsound-16k"


def clip_name(path: str) -> str:
    """'--0PQM4-hqg_000030.wav' -> '--0PQM4-hqg_000030' (also drops a trailing '-0' channel suffix)."""
    uid = path.rsplit(".", 1)[0]
    return uid[:-2] if uid.endswith("-0") else uid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vggsound-root", type=Path, required=True, help="Output folder (the 'vggsound' source root)")
    parser.add_argument("--captions-dir", type=Path, required=True,
                        help="CaptionStew captions folder; clips referenced by dataset_splits/VggSound.jsonl.gz are kept")
    parser.add_argument("--workers", type=int, default=4, help="Shards processed in parallel (default: 4)")
    parser.add_argument("--limit-shards", type=int, default=None, help="Process only the first N shards (for testing)")
    parser.add_argument("--keep-parquet", action="store_true", help="Keep downloaded parquet shards in the HF cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache dir for the parquet shards")
    args = parser.parse_args()

    manifests = sorted((args.captions_dir / "dataset_splits").glob("*.jsonl.gz"))
    if not manifests:
        parser.error(f"No manifests found in {args.captions_dir / 'dataset_splits'}")
    needed = collect_keys(manifests, "vggsound")
    print(f"{len(needed):,} VGGSound clips referenced by the captions")

    # CaptionStew only uses the VGGSound train split.
    shards = [s for s in list_shards(REPO, "data") if "/train-" in s][: args.limit_shards]
    extract_shards(REPO, [(s, "") for s in shards], args.vggsound_root, {"": needed}, clip_name,
                   key_column="audio.path", default_ext=".wav", workers=args.workers,
                   keep_parquet=args.keep_parquet, cache_dir=args.cache_dir)

    n_files = count_files(args.vggsound_root)
    print(f"{n_files:,} clips on disk ({100 * n_files / max(1, len(needed)):.1f}% of the {len(needed):,} referenced)")


if __name__ == "__main__":
    main()
