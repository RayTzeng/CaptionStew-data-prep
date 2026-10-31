#!/usr/bin/env python3
"""
Download the AudioSet clips used by CaptionStew from Hugging Face.

Audio comes from Muno459/AudioSet: 48 kHz FLAC clips packed in WebDataset tar
shards (about 4 GB each), indexed by metadata/clips.parquet. The index is used
to find the shards that hold the clips referenced by the CaptionStew manifests;
each of those shards is downloaded, the referenced clips are written as
individual FLAC files, and the shard is deleted again, so peak extra disk usage
stays around (workers x 4 GB).

Output layout (matches the "audioset/..." keys in the CaptionStew manifests):

    ${audioset_root}/
    ├── unbalanced/Y<youtube_id>.flac     (AudioCaps, AudioSetCaps, FusionAudio)
    └── balanced/Y<youtube_id>.flac       (MusicCaps)

Usage:
    # Test run: one shard of the balanced split
    python prepare_audio.py --captions-dir /path/to/captions --audioset-root /path/to/audioset \
        --splits balanced --limit-shards 1

    # Full run
    python prepare_audio.py --captions-dir /path/to/captions --audioset-root /path/to/audioset --workers 8
"""

import argparse
import os
import sys
import tarfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.hf_parquet import collect_keys, count_files, write_audio  # noqa: E402

REPO = "Muno459/AudioSet"
# CaptionStew key folder -> Muno459/AudioSet split
SPLITS = {"unbalanced": "unbalanced_train", "balanced": "balanced_train"}

_state = {}


def load_index(cache_dir):
    """Available clips of the train splits with their shard and tar member."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "metadata/clips.parquet", repo_type="dataset", cache_dir=cache_dir)
    df = pq.read_table(path, columns=["ytid", "split", "available", "shard", "member"]).to_pandas()
    return df[df.available & df.split.isin(list(SPLITS.values()))]


def init_worker(state):
    _state.update(state)


def process_shard(job: Tuple[str, List[Tuple[str, str]]]):
    """Download one tar shard, write the wanted members, delete the shard. Returns (shard, written, existing)."""
    from huggingface_hub import hf_hub_download

    shard, members = job
    wanted: Dict[str, Path] = {}
    existing = 0
    for member, out_path in members:
        out_path = Path(out_path)
        if out_path.exists():
            existing += 1
        else:
            wanted[member] = out_path
    if not wanted:
        return shard, 0, existing

    local = hf_hub_download(REPO, shard, repo_type="dataset", cache_dir=_state["cache_dir"])
    written = 0
    with tarfile.open(local) as tar:
        for info in tar:
            out_path = wanted.get(info.name)
            if out_path is None:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_audio(tar.extractfile(info).read(), out_path, _state["downmix"])
            written += 1
    if not _state["keep_shards"]:
        os.remove(os.path.realpath(local))
    return shard, written, existing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audioset-root", type=Path, required=True, help="Output folder (the 'audioset' source root)")
    parser.add_argument("--captions-dir", type=Path, default=None,
                        help="CaptionStew captions folder; only clips referenced by its dataset_splits/*.jsonl.gz are kept")
    parser.add_argument("--all-clips", action="store_true", help="Keep every train clip instead of only the referenced ones")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--workers", type=int, default=4, help="Shards processed in parallel (default: 4)")
    parser.add_argument("--limit-shards", type=int, default=None, help="Process only the first N shards per split")
    parser.add_argument("--downmix", action="store_true",
                        help="Average multi-channel clips to mono (still lossless FLAC at 48 kHz)")
    parser.add_argument("--keep-shards", action="store_true", help="Keep downloaded tar shards in the HF cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache dir for the index and shards")
    args = parser.parse_args()

    if not args.all_clips:
        if args.captions_dir is None:
            parser.error("--captions-dir is required unless --all-clips is given")
        manifests = sorted((args.captions_dir / "dataset_splits").glob("*.jsonl.gz"))
        if not manifests:
            parser.error(f"No manifests found in {args.captions_dir / 'dataset_splits'}")
        print(f"Collecting AudioSet clip IDs from {len(manifests)} manifests...")
        keys = collect_keys(manifests, "audioset")
        needed = {split: {k.split("/", 1)[1] for k in keys if k.startswith(split + "/")} for split in SPLITS}

    print(f"Loading the {REPO} clip index...")
    index = load_index(args.cache_dir)

    jobs = []
    for split in args.splits:
        rows = index[index.split == SPLITS[split]]
        names = "Y" + rows.ytid
        if not args.all_clips:
            keep = names.isin(needed[split])
            print(f"  {split}: {len(needed[split]):,} clips referenced, {int(keep.sum()):,} available "
                  f"({100 * keep.sum() / max(1, len(needed[split])):.1f}%)")
            rows, names = rows[keep], names[keep]
        by_shard = defaultdict(list)
        for shard, member, name in zip(rows.shard, rows.member, names):
            by_shard[shard].append((member, str(args.audioset_root / split / f"{name}.flac")))
        shards = sorted(by_shard)[: args.limit_shards]
        jobs += [(s, by_shard[s]) for s in shards]

    args.audioset_root.mkdir(parents=True, exist_ok=True)
    done_log = args.audioset_root / ".done_shards"
    done = set(done_log.read_text().split()) if done_log.exists() else set()
    todo = [j for j in jobs if j[0] not in done]
    print(f"{len(todo)} shards to process ({len(jobs) - len(todo)} already done)")

    state = dict(downmix=args.downmix, keep_shards=args.keep_shards, cache_dir=args.cache_dir)
    with Pool(args.workers, initializer=init_worker, initargs=(state,)) as pool, open(done_log, "a") as log:
        for shard, _, _ in tqdm(pool.imap_unordered(process_shard, todo), total=len(todo), desc="Shards",
                                unit=" shards"):
            log.write(shard + "\n")
            log.flush()

    for split in args.splits:
        n_files = count_files(args.audioset_root / split)
        msg = f"{split}: {n_files:,} clips on disk"
        if not args.all_clips:
            msg += f" ({100 * n_files / max(1, len(needed[split])):.1f}% of the {len(needed[split]):,} referenced)"
        print(msg)


if __name__ == "__main__":
    main()
