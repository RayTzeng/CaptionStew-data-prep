#!/usr/bin/env python3
"""
Link CaptionStew caption manifests to a local copy of the audio.

Every recording in a CaptionStew v2 manifest points to a path key such as
"audioset/unbalanced/Y4K8P4BoufRM" or "emilia/EN/EN_B00009/.../EN_B00009_S03733_W000058".
The first component is the audio source; the rest is the file's path relative to
that source's root folder, without extension. This script:

1. collects the keys needed by the given manifests,
2. scans each source root once and resolves every key to an existing audio file
   (any supported extension; .wav > .flac > .mp3 > .m4a > ... if several exist),
3. reads sampling rate / length / channels from each file (cached across runs),
4. writes the manifests with absolute local paths and refreshed metadata,
   dropping cuts whose audio is missing,
5. prints and saves a coverage report.

Multi-channel files are kept as they are; the cut uses the first channel.

Usage:
    # All sources under one root (<audio_root>/<source>/...)
    python link_captions.py captions/dataset_splits/*.jsonl.gz \
        --audio-root /data/captionstew_audio \
        --output-dir manifests

    # Some sources stored elsewhere
    python link_captions.py captions/10M/*.jsonl.gz \
        --audio-root /data/captionstew_audio \
        --root voxceleb1=/data/voxceleb1 --root emilia=/mnt/emilia \
        --output-dir manifests

    # Only report coverage, write nothing
    python link_captions.py captions/dataset_splits/*.jsonl.gz --audio-root /data/captionstew_audio --check-only
"""

import argparse
import gzip
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

SOURCES = ["audioset", "vggsound", "wavcaps", "clotho", "youtube4m", "voxceleb1", "voxceleb2",
           "ears", "expresso", "emilia", "jamendo"]
AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"]
EXT_RANK = {ext: i for i, ext in enumerate(AUDIO_EXTENSIONS)}
CACHE_NAME = ".audio_info_cache.jsonl"


def iter_cuts(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def source_of(key: str) -> str:
    return key.split("/", 1)[0]


def collect_keys(manifests: List[Path]) -> Dict[str, Set[str]]:
    """Return {source: {key, ...}} for all recordings referenced by the manifests."""
    needed: Dict[str, Set[str]] = defaultdict(set)
    for path in manifests:
        for cut in tqdm(iter_cuts(path), desc=f"Reading {path.name}", unit=" cuts", leave=False):
            key = cut["recording"]["sources"][0]["source"]
            if key.startswith("/"):
                raise SystemExit(
                    f"{path} holds absolute paths ({key[:60]}...). It looks like a v1 manifest or an "
                    "already-linked one; link_captions.py expects the v2 release (see download_captions.sh)."
                )
            needed[source_of(key)].add(key)
    return needed


def scan_source(source: str, root: Path, keys: Set[str]) -> Dict[str, str]:
    """Walk `root` once and map each needed key to an audio file."""
    found: Dict[str, str] = {}
    if not root.is_dir():
        return found
    prefix = source + "/"
    with tqdm(desc=f"Scanning {source} ({root})", unit=" files", leave=False) as bar:
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            rel_dir = os.path.relpath(dirpath, root)
            rel_dir = "" if rel_dir == "." else rel_dir + "/"
            for name in filenames:
                stem, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext not in EXT_RANK:
                    continue
                bar.update()
                key = prefix + rel_dir + stem
                if key not in keys:
                    continue
                path = os.path.join(dirpath, name)
                old = found.get(key)
                if old is None or EXT_RANK[ext] < EXT_RANK[os.path.splitext(old)[1].lower()]:
                    found[key] = path
    return found


def silence_stderr():
    """Pool initializer: libmpg123 prints warnings about malformed ID3 tags; read errors are still reported."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)


def probe(path: str) -> Tuple[str, Optional[Tuple[int, int, int]], Optional[str]]:
    """Return (path, (sampling_rate, num_samples, channels), error)."""
    from lhotse.audio.backend import info

    try:
        i = info(path)
        return path, (int(i.samplerate), int(i.frames), int(i.channels)), None
    except Exception as e:  # unreadable or corrupt file
        return path, None, f"{type(e).__name__}: {e}"


def load_cache(cache_path: Path) -> Dict[str, list]:
    cache = {}
    if cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                cache[entry[0]] = entry[1:]
    return cache


def probe_all(paths: List[str], cache_path: Path, workers: int) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, str]]:
    """Read audio metadata for all paths, reusing cached entries whose size and mtime are unchanged."""
    cache = load_cache(cache_path)
    info, errors, todo = {}, {}, []
    for p in paths:
        entry = cache.get(p)
        if entry is not None:
            st = os.stat(p)
            if entry[0] == st.st_size and entry[1] == int(st.st_mtime):
                info[p] = tuple(entry[2:])
                continue
        todo.append(p)

    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with Pool(workers, initializer=silence_stderr) as pool, open(cache_path, "a", encoding="utf-8") as cache_file:
            for path, meta, err in tqdm(pool.imap_unordered(probe, todo, chunksize=64), total=len(todo),
                                        desc="Reading audio metadata", unit=" files"):
                if meta is None:
                    errors[path] = err
                    continue
                info[path] = meta
                st = os.stat(path)
                cache_file.write(json.dumps([path, st.st_size, int(st.st_mtime), *meta]) + "\n")
    return info, errors


def link_cut(cut: dict, path: str, meta: Tuple[int, int, int]) -> Tuple[Optional[dict], str]:
    """Point the cut at the local file and refresh recording metadata. Returns (cut or None, status)."""
    sampling_rate, num_samples, channels = meta
    rec = cut["recording"]
    rec["sources"] = [{"type": "file", "channels": list(range(channels)), "source": path}]
    rec["sampling_rate"] = sampling_rate
    rec["num_samples"] = num_samples
    rec["duration"] = num_samples / sampling_rate
    rec["channel_ids"] = list(range(channels))

    start, duration = cut["start"], cut["duration"]
    if start >= rec["duration"] - 0.01:
        return None, "outside_audio"
    status = "ok"
    if start + duration > rec["duration"]:
        new_duration = rec["duration"] - start
        if duration - new_duration > 0.1:
            status = "trimmed"
        cut["duration"] = new_duration
        for sup in cut["supervisions"]:
            sup["duration"] = min(sup["duration"], new_duration - sup["start"])
    return cut, status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+", help="CaptionStew v2 manifests (.jsonl.gz)")
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="Folder holding one sub-folder per source (audioset/, wavcaps/, ...)")
    parser.add_argument("--root", action="append", default=[], metavar="SOURCE=PATH",
                        help=f"Root folder of one source, overriding --audio-root. Sources: {', '.join(SOURCES)}")
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"), help="Where linked manifests are written")
    parser.add_argument("--num-workers", type=int, default=min(32, os.cpu_count() or 8),
                        help="Processes used to read audio metadata")
    parser.add_argument("--check-only", action="store_true", help="Only report coverage; write no manifests")
    args = parser.parse_args()

    roots: Dict[str, Path] = {}
    if args.audio_root:
        roots = {s: args.audio_root / s for s in SOURCES}
    for item in args.root:
        source, _, path = item.partition("=")
        if source not in SOURCES or not path:
            parser.error(f"--root expects SOURCE=PATH with SOURCE in {SOURCES}, got {item!r}")
        roots[source] = Path(path)

    # 1-2. Resolve keys to local files.
    needed = collect_keys(args.manifests)
    resolved: Dict[str, str] = {}
    print(f"{'source':10s} {'root':50s} {'needed':>10s} {'found':>10s}")
    for source in sorted(needed):
        root = roots.get(source)
        found = scan_source(source, root, needed[source]) if root else {}
        resolved.update(found)
        print(f"{source:10s} {str(root) if root else '(no root given)':50s} {len(needed[source]):>10,} {len(found):>10,}")

    # 3. Audio metadata.
    meta, errors = ({}, {})
    if not args.check_only and resolved:
        meta, errors = probe_all(sorted(set(resolved.values())), args.output_dir / CACHE_NAME, args.num_workers)
        if errors:
            print(f"WARNING: {len(errors):,} files could not be read and are treated as missing, e.g.:")
            for p, e in list(errors.items())[:5]:
                print(f"  {p}: {e}")

    # 4. Write linked manifests (keeping their folder name, e.g. 10M/shard_0.jsonl.gz) and collect the report.
    report = []
    for path in args.manifests:
        rel = f"{path.resolve().parent.name}/{path.name}"
        out_path = args.output_dir / rel
        stats = defaultdict(Counter)  # source -> counters
        fout = None
        if not args.check_only:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fout = gzip.open(out_path, "wt", encoding="utf-8")
        for cut in tqdm(iter_cuts(path), desc=f"Linking {rel}", unit=" cuts", leave=False):
            key = cut["recording"]["sources"][0]["source"]
            s = stats[source_of(key)]
            s["cuts"] += 1
            s["expected_seconds"] += cut["duration"]
            local = resolved.get(key)
            if local is None or local in errors:
                s["missing_audio"] += 1
                continue
            if args.check_only:
                s["linked"] += 1
                s["linked_seconds"] += cut["duration"]
                continue
            cut, status = link_cut(cut, local, meta[local])
            s[status] += 1
            if cut is None:
                continue
            s["linked"] += 1
            s["linked_seconds"] += cut["duration"]
            if len(cut["recording"]["channel_ids"]) > 1:
                s["multi_channel"] += 1
            fout.write(json.dumps(cut, ensure_ascii=False) + "\n")
        if fout:
            fout.close()
        report.append({"manifest": rel, "output": None if args.check_only else str(out_path),
                       "sources": {k: dict(v) for k, v in stats.items()}})

    # 5. Report.
    print(f"\n{'manifest':40s} {'source':10s} {'cuts':>10s} {'linked':>10s} {'missing':>9s} {'linked %':>8s} {'hours':>9s}")
    for r in report:
        for source, s in sorted(r["sources"].items()):
            pct = 100 * s.get("linked", 0) / max(1, s["cuts"])
            print(f"{r['manifest']:40s} {source:10s} {s['cuts']:>10,} {s.get('linked', 0):>10,} "
                  f"{s.get('missing_audio', 0):>9,} {pct:>7.1f}% {s.get('linked_seconds', 0) / 3600:>9,.1f}")
    extra = Counter()
    for r in report:
        for s in r["sources"].values():
            extra.update({k: s.get(k, 0) for k in ("trimmed", "outside_audio", "multi_channel")})
    if any(extra.values()):
        print(f"\nCuts trimmed to the audio length: {extra['trimmed']:,}; dropped as starting past the end: "
              f"{extra['outside_audio']:,}; using channel 0 of a multi-channel file: {extra['multi_channel']:,}")

    missing_keys = {k for keys in needed.values() for k in keys if k not in resolved or resolved[k] in errors}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / ("coverage_report.json" if args.check_only else "link_report.json")
    report_path.write_text(json.dumps(report, indent=1))
    missing_path = args.output_dir / "missing_audio.txt"
    missing_path.write_text("".join(k + "\n" for k in sorted(missing_keys)))
    print(f"\nReport: {report_path}\nMissing audio keys ({len(missing_keys):,}): {missing_path}")


if __name__ == "__main__":
    main()
