#!/usr/bin/env python3
"""
Build the CaptionStew v2 caption release from the v1 files on Hugging Face.

v1 manifests point to absolute paths on the authors' cluster. v2 rewrites every
manifest so that it can be linked to a local copy of the audio:

1. Recording sources become source-relative path keys without extension,
   e.g. "audioset/unbalanced/Y4K8P4BoufRM" or "wavcaps/FreeSound/180799".
   The first component names the audio source (audioset, wavcaps, clotho,
   vggsound, youtube4m, voxceleb1, voxceleb2, ears, expresso, emilia, jamendo),
   which `link_captions.py` maps to a local root.
2. JamendoMaxCaps windows become virtual cuts: the recording is the full track
   ("jamendo/<track_id>") and each cut starts at its window offset.
3. SoundBible IDs are prefixed ("SoundBible_<id>") because they collide with
   FreeSound / Clotho IDs that refer to different audio.
4. Cuts sharing an ID within a file (or within a subset spread over shards)
   are merged into one cut that keeps the first audio path and the union of
   all captions.
5. Cuts with zero duration are dropped, and captions are always a list.

Usage:
    # 1. Download v1 (tag it first so it stays reachable, see tools/README.md)
    hf download RayTseng0406/CaptionStew --repo-type dataset --revision v1.0 --local-dir captionstew_v1

    # 2. Build v2
    python tools/build_release_v2.py captionstew_v1 captionstew_v2 --workers 8

    # 3. Upload (see tools/README.md)
"""

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAW = "/apdcephfs_cq12/share_302080740/data/audio_train_data/raw/"
USER = "/apdcephfs_cq12/share_302080740/user/raytseng/data/"

# (pattern on the v1 absolute path, v2 key template)
KEY_RULES = [
    (RAW + r"audioset/unbalanced_train_segments/unbalanced_train_segments_part\d+/(?P<f>[^/]+)\.\w+", "audioset/unbalanced/{f}"),
    (RAW + r"audioset/balanced_train_segments/(?P<f>[^/]+)\.\w+", "audioset/balanced/{f}"),
    (RAW + r"wavcaps/Zip_files/(?P<s>[^/]+)/(?P=s)_wav/(?P<f>[^/]+)\.\w+", "wavcaps/{s}/{f}"),
    (RAW + r"clotho/wav/train/(?P<f>[^/]+)\.\w+", "clotho/development/{f}"),
    (RAW + r"vggsound/wav/train/(?P<f>[^/]+)\.\w+", "vggsound/{f}"),
    (USER + r"AudioSetCaps/raw/yt8m_audios_packed_4m/(?P<p>batch\d+/[^/]+/[^/]+)\.\w+", "youtube4m/{p}"),
    (USER + r"ParaSpeechCaps/vox1/(?P<p>.+)\.\w+", "voxceleb1/{p}"),
    (USER + r"ParaSpeechCaps/vox2/dev/wav/(?P<p>.+)\.\w+", "voxceleb2/dev/aac/{p}"),
    (USER + r"ParaSpeechCaps/ears/(?P<p>.+)\.\w+", "ears/{p}"),
    (USER + r"ParaSpeechCaps/expresso/expresso/(?P<p>.+)\.\w+", "expresso/{p}"),
    (USER + r"Emilia/(?P<p>.+)\.\w+", "emilia/{p}"),
]
KEY_RULES = [(re.compile(p + "$"), t) for p, t in KEY_RULES]
JAMENDO_RULE = re.compile(re.escape(USER) + r"JamendoMaxCaps/cut/shard_\d+/wavs/(?P<t>\d+)_(?P<o>\d+)\.\w+$")

# Output file name for each v1 file name
SUBSET_DIRS = ["400k", "1M", "4M", "10M"]


def v2_name(v1_rel: str) -> str:
    """dataset_splits/X.jsonl.gz stays; subset shards drop the `_corrected` suffix."""
    if v1_rel.startswith("400k/"):
        return "400k/shard_0.jsonl.gz"
    return v1_rel.replace("_corrected", "")


def to_key(src: str) -> Tuple[str, Optional[Tuple[str, int]]]:
    """Map a v1 absolute path to a v2 key. Returns (key, (track, offset)) for Jamendo windows."""
    m = JAMENDO_RULE.match(src)
    if m:
        return f"jamendo/{m['t']}", (m["t"], int(m["o"]))
    for pattern, template in KEY_RULES:
        m = pattern.match(src)
        if m:
            return template.format(**m.groupdict()), None
    raise ValueError(f"No v2 key rule for source path: {src}")


def as_list(caption) -> List[str]:
    if caption is None:
        return []
    return list(caption) if isinstance(caption, (list, tuple)) else [caption]


def merge_captions(*caption_lists: List[str]) -> List[str]:
    """Ordered union of captions."""
    out, seen = [], set()
    for caps in caption_lists:
        for c in caps:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def iter_cuts(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def new_id(cut: dict, key: str) -> str:
    cid = cut["id"]
    if key.startswith("wavcaps/SoundBible/") and not cid.startswith("SoundBible_"):
        cid = f"SoundBible_{cid}"
    return cid


def convert(cut: dict, jamendo_durations: Dict[str, float], canonical: Dict[str, str]) -> Optional[dict]:
    """Convert one v1 cut into a v2 cut (or None if it has zero duration)."""
    rec = cut["recording"]
    duration = cut["duration"]
    if duration <= 0 or rec.get("num_samples", 1) <= 0:
        return None
    key, jamendo = to_key(rec["sources"][0]["source"])
    cid = new_id(cut, key)
    key = canonical.get(cid, key)
    caption = as_list(cut["supervisions"][0].get("custom", {}).get("caption"))

    if jamendo:
        track, offset = jamendo
        rec_id, start = track, float(offset)
        track_duration = jamendo_durations.get(track, start + duration)
        sampling_rate = rec["sampling_rate"]
        new_rec = {
            "id": track,
            "sources": [{"type": "file", "channels": [0], "source": key}],
            "sampling_rate": sampling_rate,
            "num_samples": round(track_duration * sampling_rate),
            "duration": track_duration,
            "channel_ids": [0],
        }
    else:
        rec_id, start = cid, cut["start"]
        new_rec = {
            "id": cid,
            "sources": [{"type": "file", "channels": rec["sources"][0]["channels"], "source": key}],
            "sampling_rate": rec["sampling_rate"],
            "num_samples": rec["num_samples"],
            "duration": rec["duration"],
            "channel_ids": rec["channel_ids"],
        }

    return {
        "id": cid,
        "start": start,
        "duration": duration,
        "channel": 0,
        "supervisions": [
            {
                "id": cid,
                "recording_id": rec_id,
                "start": 0.0,
                "duration": duration,
                "channel": 0,
                "custom": {"caption": caption},
            }
        ],
        "recording": new_rec,
        "type": "MonoCut",
    }


def jamendo_track_durations(v1_root: Path) -> Dict[str, float]:
    """Lower bound of each Jamendo track's duration: end of its last listed window."""
    durations: Dict[str, float] = {}
    for path in sorted(v1_root.glob("**/*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = (line for line in f if "/JamendoMaxCaps/" in line)
            for cut in map(json.loads, lines):
                m = JAMENDO_RULE.match(cut["recording"]["sources"][0]["source"])
                if m:
                    end = int(m["o"]) + cut["duration"]
                    if end > durations.get(m["t"], 0.0):
                        durations[m["t"]] = end
    return durations


def first_keys_of_conflicting_ids(path: Path) -> Dict[str, str]:
    """For cut IDs that appear with several audio paths in one file, return the key of the first occurrence."""
    first: Dict[str, int] = {}
    conflicts = set()
    for cut in iter_cuts(path):
        key, _ = to_key(cut["recording"]["sources"][0]["source"])
        cid = new_id(cut, key)
        h = hash(key)
        if first.setdefault(cid, h) != h:
            conflicts.add(cid)
    del first
    keys: Dict[str, str] = {}
    if conflicts:
        for cut in iter_cuts(path):
            key, _ = to_key(cut["recording"]["sources"][0]["source"])
            cid = new_id(cut, key)
            if cid in conflicts and cid not in keys:
                keys[cid] = key
    return keys


def build_group(args) -> dict:
    """Convert one group of v1 files (a single file, or all shards of a subset) with shared deduplication."""
    v1_root, v2_root, rel_paths, jamendo_durations, canonical = args
    v1_root, v2_root = Path(v1_root), Path(v2_root)

    # Pass 1: count new IDs across the group.
    counts: Counter = Counter()
    for rel in rel_paths:
        for cut in iter_cuts(v1_root / rel):
            key, _ = to_key(cut["recording"]["sources"][0]["source"])
            counts[new_id(cut, key)] += 1
    dup_ids = {i for i, n in counts.items() if n > 1}
    del counts

    # Pass 2: gather captions of duplicated IDs.
    dup_captions: Dict[str, List[str]] = defaultdict(list)
    if dup_ids:
        for rel in rel_paths:
            for cut in iter_cuts(v1_root / rel):
                key, _ = to_key(cut["recording"]["sources"][0]["source"])
                cid = new_id(cut, key)
                if cid in dup_ids and cut["duration"] > 0:
                    caps = as_list(cut["supervisions"][0].get("custom", {}).get("caption"))
                    dup_captions[cid] = merge_captions(dup_captions[cid], caps)

    # Pass 3: convert and write.
    stats = []
    written = set()
    for rel in rel_paths:
        out_path = v2_root / v2_name(rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        s = {"file": v2_name(rel), "v1_file": rel, "v1_cuts": 0, "cuts": 0, "seconds": 0.0, "captions": 0,
             "dropped_zero_duration": 0, "merged_duplicates": 0, "sources": Counter()}
        with gzip.open(out_path, "wt", encoding="utf-8") as fout:
            for cut in iter_cuts(v1_root / rel):
                s["v1_cuts"] += 1
                new = convert(cut, jamendo_durations, canonical)
                if new is None:
                    s["dropped_zero_duration"] += 1
                    continue
                cid = new["id"]
                if cid in dup_ids:
                    if cid in written:
                        s["merged_duplicates"] += 1
                        continue
                    written.add(cid)
                    new["supervisions"][0]["custom"]["caption"] = dup_captions[cid]
                fout.write(json.dumps(new, ensure_ascii=False) + "\n")
                s["cuts"] += 1
                s["seconds"] += new["duration"]
                s["captions"] += len(new["supervisions"][0]["custom"]["caption"])
                s["sources"][new["recording"]["sources"][0]["source"].split("/")[0]] += 1
        s["sources"] = dict(s["sources"])
        stats.append(s)
        print(f"  {rel} -> {s['file']}: {s['cuts']:,} cuts ({s['seconds'] / 3600:,.0f} h), "
              f"merged {s['merged_duplicates']:,}, dropped {s['dropped_zero_duration']:,}", flush=True)
    return {"files": stats, "renamed_soundbible": sum(1 for i in dup_captions if i.startswith("SoundBible_"))}


def write_dataset_card(v2_root: Path, stats: List[dict]):
    """Fill tools/dataset_card.md with the release statistics and write it as the dataset README."""
    template = (Path(__file__).parent / "dataset_card.md").read_text()
    per_source = [s for s in stats if s["file"].startswith("dataset_splits/")]
    configs = "\n".join(
        f"- config_name: {Path(s['file']).name.split('.')[0]}\n  data_files: \"{s['file']}\"" for s in per_source
    )

    rows = ["| File | Cuts | Hours | Captions |", "|---|---:|---:|---:|"]
    groups = defaultdict(lambda: [0, 0.0, 0])
    for s in stats:
        name = s["file"] if s["file"].startswith("dataset_splits/") else s["file"].split("/")[0] + "/"
        g = groups[name]
        g[0] += s["cuts"]
        g[1] += s["seconds"]
        g[2] += s["captions"]
    order = {f"{d}/": i for i, d in enumerate(SUBSET_DIRS)}
    for name, (cuts, seconds, captions) in sorted(groups.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])):
        rows.append(f"| `{name}` | {cuts:,} | {seconds / 3600:,.0f} | {captions:,} |")

    example_file = v2_root / "dataset_splits" / "clotho_train.jsonl.gz"
    example = next(iter_cuts(example_file)) if example_file.exists() else next(iter_cuts(v2_root / stats[0]["file"]))
    card = (template.replace("{per_source_configs}", configs)
            .replace("{stats_table}", "\n".join(rows))
            .replace("{example}", json.dumps(example, indent=2, ensure_ascii=False))
            .replace("{merged:,}", f"{sum(s['merged_duplicates'] for s in stats):,}")
            .replace("{dropped:,}", f"{sum(s['dropped_zero_duration'] for s in stats):,}"))
    (v2_root / "README.md").write_text(card)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("v1_root", type=Path, help="Local copy of the v1 release")
    parser.add_argument("v2_root", type=Path, help="Output directory for the v2 release")
    parser.add_argument("--workers", type=int, default=4, help="Number of file groups converted in parallel")
    args = parser.parse_args()

    v1_files = sorted(str(p.relative_to(args.v1_root)) for p in args.v1_root.glob("**/*.jsonl.gz"))
    if not v1_files:
        raise SystemExit(f"No .jsonl.gz files found under {args.v1_root}")

    print("Computing JamendoMaxCaps track durations...", flush=True)
    jamendo_durations = jamendo_track_durations(args.v1_root)
    print(f"  {len(jamendo_durations):,} tracks", flush=True)

    # Some clips are listed under several audio paths (e.g. YouTube-4M clips stored in two category
    # folders). The first path in the per-source file is used everywhere, including the subsets.
    print("Finding clips listed under several audio paths...", flush=True)
    per_source = [args.v1_root / f for f in v1_files if f.startswith("dataset_splits/")]
    canonical: Dict[str, str] = {}
    with Pool(args.workers) as pool:
        for keys in pool.map(first_keys_of_conflicting_ids, per_source, chunksize=1):
            for cid, key in keys.items():
                canonical.setdefault(cid, key)
    print(f"  {len(canonical):,} clips", flush=True)

    groups = [[f] for f in v1_files if f.split("/")[0] not in SUBSET_DIRS]
    for d in SUBSET_DIRS:
        shards = [f for f in v1_files if f.split("/")[0] == d]
        if shards:
            groups.append(shards)

    print(f"Converting {len(v1_files)} files in {len(groups)} groups...", flush=True)
    jobs = [(str(args.v1_root), str(args.v2_root), g, jamendo_durations, canonical) for g in groups]
    with Pool(args.workers) as pool:
        results = pool.map(build_group, jobs, chunksize=1)

    stats = sorted((s for r in results for s in r["files"]), key=lambda s: s["file"])
    (args.v2_root / "stats.json").write_text(json.dumps(stats, indent=1))
    write_dataset_card(args.v2_root, stats)
    print(f"\nWrote {len(stats)} files, stats.json and README.md to {args.v2_root}")
    print(f"{'file':42s} {'cuts':>11s} {'hours':>8s} {'captions':>11s} {'merged':>8s} {'dropped':>8s}")
    for s in stats:
        print(f"{s['file']:42s} {s['cuts']:>11,} {s['seconds'] / 3600:>8,.0f} {s['captions']:>11,} "
              f"{s['merged_duplicates']:>8,} {s['dropped_zero_duration']:>8,}")


if __name__ == "__main__":
    main()
