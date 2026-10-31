"""
Extract individual audio files from Hugging Face parquet datasets.

Several sources (AudioSet, VGGSound, JamendoMaxCaps) are hosted on Hugging Face as
parquet shards whose `audio` column holds the original encoded file (FLAC, WAV
or MP3 bytes). `extract_shards` downloads one shard at a time, writes the rows
that CaptionStew needs as individual files (bytes copied as-is, or downmixed to
mono FLAC), and deletes the shard again so that disk usage stays bounded.
Completed shards are logged so that interrupted runs can resume.
"""

import gzip
import io
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm

_state = {}


def collect_keys(manifests: Iterable[Path], source: str) -> Set[str]:
    """Return the path keys (without the "<source>/" prefix) referenced by CaptionStew manifests."""
    prefix = f'"{source}/'
    keys = set()
    for path in manifests:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if prefix not in line:
                    continue
                key = json.loads(line)["recording"]["sources"][0]["source"]
                if key.startswith(source + "/"):
                    keys.add(key[len(source) + 1:])
    return keys


def list_shards(repo: str, prefix: str) -> List[str]:
    from huggingface_hub import HfApi

    return sorted(f.path for f in HfApi().list_repo_tree(repo, repo_type="dataset", path_in_repo=prefix, recursive=True)
                  if f.path.endswith(".parquet"))


def write_audio(data: bytes, out_path: Path, downmix: bool) -> Path:
    """Write encoded audio bytes as-is, or (with downmix) multi-channel audio as mono FLAC. Returns the written path."""
    if downmix:
        import soundfile as sf

        info = sf.info(io.BytesIO(data))
        if info.channels > 1:
            audio, sr = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
            subtype = info.subtype if info.subtype.startswith("PCM_") and info.subtype != "PCM_32" else "PCM_16"
            out_path = out_path.with_suffix(".flac")
            tmp = out_path.with_name(out_path.name + ".tmp")
            sf.write(tmp, audio.mean(axis=1), sr, format="FLAC", subtype=subtype)
            os.replace(tmp, out_path)
            return out_path
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out_path)
    return out_path


def _init(state):
    _state.update(state)


def _process(job: Tuple[str, str]):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shard, group = job
    out_dir = Path(_state["out_root"]) / group if group else Path(_state["out_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = _state["needed"].get(group)
    name_fn = _state["name_fn"]

    local = hf_hub_download(_state["repo"], shard, repo_type="dataset", cache_dir=_state["cache_dir"])
    written = existing = 0
    pf = pq.ParquetFile(local)
    for rg in range(pf.num_row_groups):
        keys = pf.read_row_group(rg, columns=[_state["key_column"]]).column(0).to_pylist()
        if _state["key_column"] == "audio.path":
            keys = [k["path"] if isinstance(k, dict) else k for k in keys]
        names = [name_fn(k) for k in keys]
        wanted = [i for i, n in enumerate(names) if n is not None and (needed is None or n in needed)]
        if not wanted:
            continue
        audio = pf.read_row_group(rg, columns=["audio"]).column(0)
        for i in wanted:
            src_ext = os.path.splitext(audio[i]["path"].as_py() or "")[1] or _state["default_ext"]
            out_path = out_dir / f"{names[i]}{src_ext}"
            if out_path.exists() or out_path.with_suffix(".flac").exists():
                existing += 1
                continue
            write_audio(audio[i]["bytes"].as_py(), out_path, _state["downmix"])
            written += 1

    if not _state["keep_parquet"]:
        os.remove(os.path.realpath(local))
    return shard, written, existing


def extract_shards(
    repo: str,
    jobs: List[Tuple[str, str]],
    out_root: Path,
    needed: Dict[str, Optional[Set[str]]],
    name_fn: Callable[[str], Optional[str]],
    key_column: str = "audio.path",
    default_ext: str = ".wav",
    workers: int = 4,
    downmix: bool = False,
    keep_parquet: bool = False,
    cache_dir: Optional[str] = None,
) -> Dict[str, int]:
    """
    Process parquet shards in parallel.

    jobs:       (shard path in the repo, output sub-folder) pairs
    needed:     output sub-folder -> set of file names (without extension) to keep, or None to keep all
    name_fn:    maps the value of `key_column` of a row to its output file name (without extension), or None to skip
    key_column: a cheap column identifying each row ("video_id", or "audio.path" for the original file name)
    """
    out_root.mkdir(parents=True, exist_ok=True)
    done_log = out_root / ".done_shards"
    done = set(done_log.read_text().split()) if done_log.exists() else set()
    todo = [j for j in jobs if j[0] not in done]
    print(f"{len(todo)} shards to process ({len(jobs) - len(todo)} already done)")

    state = dict(repo=repo, out_root=str(out_root), needed=needed, name_fn=name_fn, key_column=key_column,
                 default_ext=default_ext, downmix=downmix, keep_parquet=keep_parquet, cache_dir=cache_dir)
    totals = {"written": 0, "existing": 0}
    with Pool(workers, initializer=_init, initargs=(state,)) as pool, open(done_log, "a") as log:
        for shard, written, existing in tqdm(pool.imap_unordered(_process, todo), total=len(todo), desc="Shards",
                                             unit=" shards"):
            totals["written"] += written
            totals["existing"] += existing
            log.write(shard + "\n")
            log.flush()
    return totals


def count_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for e in os.scandir(folder)
               if e.is_file() and not e.name.startswith(".") and not e.name.endswith(".tmp"))
