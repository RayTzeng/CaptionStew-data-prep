#!/usr/bin/env python3
"""
Validate linked CaptionStew manifests (the output of link_captions.py).

Checks every cut for:
  - unique cut IDs within the manifest
  - a non-empty caption list in supervisions[0].custom.caption
  - a cut that lies inside its recording, with consistent recording metadata
  - (with --check-files) an existing audio file
and decodes a random sample of cuts per manifest with Lhotse to confirm the
audio loads with the expected number of samples.

Usage:
    python validate.py manifests/dataset_splits/*.jsonl.gz
    python validate.py manifests/10M/*.jsonl.gz --num-decode 50 --check-files
"""

import argparse
import gzip
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

TOLERANCE = 0.025  # seconds


def check_cut(cut: dict, check_files: bool) -> list:
    problems = []
    rec = cut.get("recording", {})
    sources = rec.get("sources") or [{}]
    if not sources[0].get("source", "").startswith("/"):
        problems.append("source is not an absolute path (manifest not linked?)")
    elif check_files and not os.path.isfile(sources[0]["source"]):
        problems.append("audio file does not exist")
    sr, n = rec.get("sampling_rate", 0), rec.get("num_samples", 0)
    if sr <= 0 or n <= 0:
        problems.append("recording has no samples")
    elif abs(n / sr - rec.get("duration", 0)) > TOLERANCE:
        problems.append("recording duration != num_samples / sampling_rate")
    if cut["duration"] <= 0:
        problems.append("cut has non-positive duration")
    if cut["start"] < 0 or cut["start"] + cut["duration"] > rec.get("duration", 0) + TOLERANCE:
        problems.append("cut extends beyond its recording")
    if cut.get("channel", 0) not in rec.get("channel_ids", [0]):
        problems.append("cut channel not in recording channels")
    sups = cut.get("supervisions") or []
    caption = sups[0].get("custom", {}).get("caption") if sups else None
    if not isinstance(caption, list) or not caption or not all(isinstance(c, str) and c.strip() for c in caption):
        problems.append("caption is not a non-empty list of strings")
    return problems


def decode(cut_dict: dict) -> str:
    """Load the cut's audio with Lhotse; return an error string or ''."""
    from lhotse.cut import MonoCut

    try:
        cut = MonoCut.from_dict(cut_dict)
        audio = cut.load_audio()
        expected = cut.num_samples
        if audio.shape[-1] < expected - 2:
            return f"decoded {audio.shape[-1]} samples, expected {expected}"
        return ""
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:200]}"


def validate(path: Path, num_decode: int, check_files: bool, rng: random.Random) -> dict:
    ids = set()
    problems = Counter()
    examples = {}
    sample, n = [], 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Checking {path.name}", unit=" cuts", leave=False):
            cut = json.loads(line)
            n += 1
            if cut["id"] in ids:
                problems["duplicate cut id"] += 1
                examples.setdefault("duplicate cut id", cut["id"])
            ids.add(cut["id"])
            for p in check_cut(cut, check_files):
                problems[p] += 1
                examples.setdefault(p, cut["id"])
            # reservoir sampling of cuts to decode
            if len(sample) < num_decode:
                sample.append(cut)
            elif num_decode:
                j = rng.randrange(n)
                if j < num_decode:
                    sample[j] = cut
    for cut in sample:
        err = decode(cut)
        if err:
            problems["decode failed"] += 1
            examples.setdefault("decode failed", f"{cut['id']}: {err}")
    return {"manifest": str(path), "cuts": n, "decoded": len(sample), "problems": dict(problems), "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+", help="Linked manifests (.jsonl.gz)")
    parser.add_argument("--num-decode", type=int, default=20, help="Random cuts per manifest to decode (default: 20)")
    parser.add_argument("--check-files", action="store_true", help="Check that every audio file exists (slow)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    results = [validate(p, args.num_decode, args.check_files, rng) for p in args.manifests]
    failed = False
    for r in results:
        status = "OK" if not r["problems"] else "PROBLEMS"
        failed |= bool(r["problems"])
        print(f"{status:8s} {r['manifest']}: {r['cuts']:,} cuts, decoded {r['decoded']}")
        for p, count in r["problems"].items():
            print(f"         {count:,} x {p} (e.g. {r['examples'][p]})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
