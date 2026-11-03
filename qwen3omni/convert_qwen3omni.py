#!/usr/bin/env python3
"""
Convert the Qwen3-Omni captions of CaptionStew 10M into CaptionStew manifests.

The captions come from eureka1500/CaptionStew10M-Qwen3Omni (generated with
Qwen3-Omni-30B-A3B-Captioner for "Unlocking Strong Supervision: A Data-Centric
Study of General-Purpose Audio Pre-Training Methods", arXiv:2603.25767). Its
manifests point to placeholder paths such as
"/path/to/audioset/unbalanced_train_segments/unbalanced_train_segments_part06/Y4K8P4BoufRM.wav".
This script rewrites them into the same format as the CaptionStew release on
Hugging Face, so that `link_captions.py` can link them to the audio prepared
with the recipes in `dataset/`:

1. Recording sources become CaptionStew path keys ("audioset/unbalanced/Y4K8P4BoufRM").
2. JamendoMaxCaps windows become cuts on the full track ("jamendo/<track_id>", start = offset).
3. SoundBible IDs are prefixed ("SoundBible_<id>"), as in the CaptionStew release.
4. Cuts sharing an ID across the input files are merged into one cut that keeps the
   first audio path and all of their captions.
5. `custom.caption` holds the selected Qwen3-Omni text as a list (see --text), and
   `custom.original_caption` keeps the CaptionStew captions.

Usage:
    python qwen3omni/convert_qwen3omni.py captions_qwen3omni/manifest/*.jsonl.gz \
        --output-dir captions/qwen3omni --text full

    python link_captions.py captions/qwen3omni/*.jsonl.gz --audio-root "${audio_root}" --output-dir manifests
"""

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_release_v2 import RAW, USER, as_list, merge_captions, new_id, to_key  # noqa: E402

PLACEHOLDER = "/path/to/"
TEXT_CHOICES = ["full", "summary", "full+original", "summary+original"]


def qwen_key(src: str) -> Tuple[str, Optional[Tuple[str, int]]]:
    """Map a placeholder path (or a v1 absolute path) to a CaptionStew path key."""
    if not src.startswith(PLACEHOLDER):
        return to_key(src)
    rel = src[len(PLACEHOLDER):]
    try:
        return to_key(RAW + rel)
    except ValueError:
        return to_key(USER + rel)


def summary_paragraph(caption: str) -> str:
    """Last paragraph of a Qwen3-Omni caption (the summary used for training in arXiv:2603.25767)."""
    paragraphs = [p.strip() for p in caption.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else caption.strip()


def select_text(qwen_caption: str, original: List[str], text: str) -> List[str]:
    chosen = [summary_paragraph(qwen_caption) if text.startswith("summary") else qwen_caption.strip()]
    return merge_captions(chosen, original) if text.endswith("+original") else chosen


def iter_cuts(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def convert(cut: dict, text: str) -> Optional[Tuple[dict, List[str], List[str]]]:
    """Return (CaptionStew cut, selected captions, original captions), or None for an empty cut."""
    rec = cut["recording"]
    duration = cut["duration"]
    custom = cut["supervisions"][0].get("custom", {})
    qwen_caption = custom.get("caption") or ""
    if duration <= 0 or not qwen_caption.strip():
        return None
    key, jamendo = qwen_key(rec["sources"][0]["source"])
    cid = new_id(cut, key)
    original = as_list(custom.get("original_caption"))
    captions = select_text(qwen_caption, original, text)

    if jamendo:
        track, offset = jamendo
        rec_id, start = track, float(offset)
        sr = rec["sampling_rate"]
        new_rec = {"id": track, "sources": [{"type": "file", "channels": [0], "source": key}], "sampling_rate": sr,
                   "num_samples": round((start + duration) * sr), "duration": start + duration, "channel_ids": [0]}
    else:
        rec_id, start = cid, cut["start"]
        new_rec = {"id": cid, "sources": [{"type": "file", "channels": rec["sources"][0]["channels"], "source": key}],
                   "sampling_rate": rec["sampling_rate"], "num_samples": rec["num_samples"],
                   "duration": rec["duration"], "channel_ids": rec["channel_ids"]}

    new = {
        "id": cid,
        "start": start,
        "duration": duration,
        "channel": 0,
        "supervisions": [{"id": cid, "recording_id": rec_id, "start": 0.0, "duration": duration, "channel": 0,
                          "custom": {"caption": captions, "original_caption": original}}],
        "recording": new_rec,
        "type": "MonoCut",
    }
    return new, captions, original


def ids_of(job) -> List[str]:
    path, text = job
    out = []
    for cut in iter_cuts(Path(path)):
        key, _ = qwen_key(cut["recording"]["sources"][0]["source"])
        out.append(new_id(cut, key))
    return out


def duplicates_of(job) -> Dict[str, list]:
    """For duplicated IDs in one file: [(line index, captions, original captions), ...]."""
    path, text, dup_ids = job
    found = defaultdict(list)
    for i, cut in enumerate(iter_cuts(Path(path))):
        key, _ = qwen_key(cut["recording"]["sources"][0]["source"])
        cid = new_id(cut, key)
        if cid in dup_ids:
            res = convert(cut, text)
            if res:
                found[cid].append((i, res[1], res[2]))
    return dict(found)


def write_file(job) -> dict:
    path, out_path, text, file_index, merged = job
    stats = {"file": out_path.name, "input_cuts": 0, "cuts": 0, "seconds": 0.0, "captions": 0,
             "merged_duplicates": 0, "dropped": 0, "sources": Counter()}
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for i, cut in enumerate(iter_cuts(Path(path))):
            stats["input_cuts"] += 1
            res = convert(cut, text)
            if res is None:
                stats["dropped"] += 1
                continue
            new = res[0]
            m = merged.get(new["id"])
            if m is not None:
                if (file_index, i) != m["first"]:
                    stats["merged_duplicates"] += 1
                    continue
                custom = new["supervisions"][0]["custom"]
                custom["caption"], custom["original_caption"] = m["captions"], m["original"]
            fout.write(json.dumps(new, ensure_ascii=False) + "\n")
            stats["cuts"] += 1
            stats["seconds"] += new["duration"]
            stats["captions"] += len(new["supervisions"][0]["custom"]["caption"])
            stats["sources"][new["recording"]["sources"][0]["source"].split("/")[0]] += 1
    stats["sources"] = dict(stats["sources"])
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+", help="Qwen3-Omni manifests (manifest/*.jsonl.gz)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for the converted manifests")
    parser.add_argument("--text", choices=TEXT_CHOICES, default="full",
                        help="Caption text stored in custom.caption: the full Qwen3-Omni caption, its final summary "
                             "paragraph (as used for training in the paper), optionally followed by the original "
                             "CaptionStew captions (default: full)")
    parser.add_argument("--workers", type=int, default=4, help="Files processed in parallel (default: 4)")
    args = parser.parse_args()

    files = sorted(args.manifests)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pass 1/3: collecting cut IDs from {len(files)} files...", flush=True)
    counts: Counter = Counter()
    with Pool(args.workers) as pool:
        for ids in pool.imap(ids_of, [(str(f), args.text) for f in files]):
            counts.update(ids)
    dup_ids = {i for i, n in counts.items() if n > 1}
    print(f"  {len(counts):,} unique cut IDs, {len(dup_ids):,} appear more than once", flush=True)
    del counts

    print("Pass 2/3: gathering captions of duplicated cuts...", flush=True)
    merged: Dict[str, dict] = {}
    if dup_ids:
        with Pool(args.workers) as pool:
            per_file = pool.map(duplicates_of, [(str(f), args.text, dup_ids) for f in files], chunksize=1)
        for file_index, found in enumerate(per_file):
            for cid, occurrences in found.items():
                for line, captions, original in occurrences:
                    m = merged.setdefault(cid, {"first": (file_index, line), "captions": [], "original": []})
                    m["captions"] = merge_captions(m["captions"], captions)
                    m["original"] = merge_captions(m["original"], original)

    print("Pass 3/3: writing converted manifests...", flush=True)
    jobs = [(str(f), args.output_dir / f.name, args.text, k, merged) for k, f in enumerate(files)]
    with Pool(args.workers) as pool:
        stats = pool.map(write_file, jobs, chunksize=1)

    (args.output_dir / "convert_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"\n{'file':36s} {'cuts':>10s} {'hours':>8s} {'captions':>10s} {'merged':>8s} {'dropped':>8s}")
    for s in stats:
        print(f"{s['file']:36s} {s['cuts']:>10,} {s['seconds'] / 3600:>8,.0f} {s['captions']:>10,} "
              f"{s['merged_duplicates']:>8,} {s['dropped']:>8,}")
    print(f"{'total':36s} {sum(s['cuts'] for s in stats):>10,} {sum(s['seconds'] for s in stats) / 3600:>8,.0f}")


if __name__ == "__main__":
    main()
