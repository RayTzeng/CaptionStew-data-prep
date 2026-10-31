# Clotho Preparing Script

This guide describes how to download the Clotho audio used by CaptionStew (`dataset_splits/clotho_train.jsonl.gz`, 3,740 clips from the Clotho v2.1 development split).

## Dataset Source

The audio is downloaded from the official Clotho v2.1 release on Zenodo:

<https://zenodo.org/records/4783391>

The Hugging Face copies of Clotho rename the audio files, so they cannot be matched to the CaptionStew captions; use the Zenodo release.

## Workflow

1. Download the development audio archive.
2. Extract it into the CaptionStew layout.
3. Link and validate with the top-level scripts.

## 1. Download the Audio

    mkdir -p "${audio_root}/clotho"
    cd "${audio_root}/clotho"
    wget -O clotho_audio_development.7z "https://zenodo.org/records/4783391/files/clotho_audio_development.7z?download=1"
    md5sum clotho_audio_development.7z   # c8b05bc7acdb13895bb3c6a29608667e

The archive is about 4.5 GB.

## 2. Extract

`prepare_audio.py` extracts the archive (requires `7z`) and renames the files the way CaptionStew refers to them (spaces replaced by underscores). The archive is deleted afterwards unless `--keep-archive` is given.

    python prepare_audio.py "${audio_root}/clotho/clotho_audio_development.7z" "${audio_root}/clotho"

## 3. Expected Output

    ${audio_root}/clotho/
    └── development/
        └── <file name with spaces replaced by _>.wav   (3,837 files, 44.1 kHz)

Two pairs of development files get the same name once spaces are replaced (`Car Keys.wav` / `Car_Keys.wav` and `cow bells.wav` / `cow_bells.wav`). CaptionStew uses one file of each pair (`Car_Keys.wav` and `cow bells.wav`, identified by their captions); the script keeps those and skips the other two.

## 4. Link and Validate

These commands link the per-dataset caption file(s) this source serves (Outcome B in the top-level README). To build CaptionStew 10M (Outcome A), prepare every audio source first and then link `captions/10M/shard_*.jsonl.gz` as described in the [top-level README](../../README.md#outcome-a-captionstew-10m).

    python ../../link_captions.py /path/to/captions/dataset_splits/clotho_train.jsonl.gz \
      --root clotho="${audio_root}/clotho" \
      --output-dir /path/to/manifests

    python ../../validate.py /path/to/manifests/dataset_splits/clotho_train.jsonl.gz

## Caption Reference

The captions are provided in the following work:

> Drossos, Konstantinos, Samuel Lipping, and Tuomas Virtanen. *Clotho: An Audio Captioning Dataset.* ICASSP, 2020.

## Notes

- Only the development split is used; the validation and evaluation splits are excluded from CaptionStew to keep evaluation sets clean.
- 1,484 CaptionStew Clotho cut IDs are FreeSound IDs, but their path keys still use the Clotho file name; `link_captions.py` resolves clips by path key, so this needs no extra handling.
