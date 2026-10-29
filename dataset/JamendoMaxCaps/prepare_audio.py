#!/usr/bin/env python3
"""
Download the JamendoMaxCaps tracks used by CaptionStew from Hugging Face.

Audio comes from amaai-lab/JamendoMaxCaps (full tracks, MP3 inside parquet
shards). Each shard is downloaded, the tracks referenced by the CaptionStew
manifests are written as individual MP3 files (bytes copied as-is), and the
shard is deleted again.

CaptionStew captions describe 30-second windows of these tracks; the manifests
refer to the full track and select each window with the cut's start offset, so
no segment files need to be cut.

Output layout (matches the "jamendo/<track_id>" keys in the CaptionStew manifests):

    ${jamendo_root}/<track_id>.mp3

Usage:
    # Test mode (process only 2 shards)
    python prepare_audio.py --captions-dir /path/to/captions --jamendo-root /path/to/jamendo --limit-shards 2

    # Full run
    python prepare_audio.py --captions-dir /path/to/captions --jamendo-root /path/to/jamendo --workers 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.hf_parquet import collect_keys, count_files, extract_shards, list_shards  # noqa: E402

REPO = "amaai-lab/JamendoMaxCaps"


def track_name(path: str) -> str:
    """'1000306.mp3' -> '1000306'"""
    return path.rsplit(".", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jamendo-root", type=Path, required=True, help="Output folder (the 'jamendo' source root)")
    parser.add_argument("--captions-dir", type=Path, default=None,
                        help="CaptionStew captions folder; tracks referenced by its dataset_splits/*.jsonl.gz are kept")
    parser.add_argument("--all-tracks", action="store_true", help="Keep every track instead of only the referenced ones")
    parser.add_argument("--workers", type=int, default=4, help="Shards processed in parallel (default: 4)")
    parser.add_argument("--limit-shards", type=int, default=None, help="Process only the first N shards (for testing)")
    parser.add_argument("--downmix", action="store_true",
                        help="Decode stereo tracks and store them as mono FLAC (about 1.5-2x the size of the MP3s)")
    parser.add_argument("--keep-parquet", action="store_true", help="Keep downloaded parquet shards in the HF cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache dir for the parquet shards")
    args = parser.parse_args()

    if args.all_tracks:
        needed = None
    else:
        if args.captions_dir is None:
            parser.error("--captions-dir is required unless --all-tracks is given")
        manifests = sorted((args.captions_dir / "dataset_splits").glob("*.jsonl.gz"))
        if not manifests:
            parser.error(f"No manifests found in {args.captions_dir / 'dataset_splits'}")
        needed = collect_keys(manifests, "jamendo")
        print(f"{len(needed):,} JamendoMaxCaps tracks referenced by the captions")

    shards = list_shards(REPO, "data")[: args.limit_shards]
    extract_shards(REPO, [(s, "") for s in shards], args.jamendo_root, {"": needed}, track_name,
                   key_column="audio.path", default_ext=".mp3", workers=args.workers, downmix=args.downmix,
                   keep_parquet=args.keep_parquet, cache_dir=args.cache_dir)

    n_files = count_files(args.jamendo_root)
    msg = f"{n_files:,} tracks on disk"
    if needed is not None:
        msg += f" ({100 * n_files / max(1, len(needed)):.1f}% of the {len(needed):,} referenced)"
    print(msg)


if __name__ == "__main__":
    main()
