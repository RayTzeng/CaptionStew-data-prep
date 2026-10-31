# JamendoMaxCaps Preparing Script

This guide describes how to download the JamendoMaxCaps audio used by CaptionStew (`dataset_splits/JamendoMaxCaps.jsonl.gz`, 1.81M captions over 360,731 instrumental tracks).

## Dataset Source

The audio is downloaded from [`amaai-lab/JamendoMaxCaps`](https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps) on Hugging Face: 362,298 full tracks stored as MP3 (mostly 44.1 kHz stereo) inside 2,272 parquet shards, about 1.06 TB in total.

CaptionStew captions describe 30-second windows of each track (windows start every 30 s and are 30.08 s long; the last one is shorter). The manifests point every caption to the full track and select its window with the cut's `start` offset, so the tracks are used as they are and no segment files are written. Cut IDs are `<track_id>_<offset in seconds>`.

## Workflow

1. Download the CaptionStew captions (see the top-level README).
2. Run a small test job to verify the pipeline.
3. Launch the full run.
4. Link and validate with the top-level scripts.

## 1. Download the Audio

`prepare_audio.py` downloads one parquet shard at a time, writes the tracks referenced by the CaptionStew captions as individual `.mp3` files (bytes copied as-is), and deletes the shard again. Completed shards are logged in `${audio_root}/jamendo/.done_shards`, so an interrupted run can be restarted with the same command.

### Test Run

Use a small test run for debugging purposes:

    python prepare_audio.py \
      --captions-dir /path/to/captions \
      --jamendo-root "${audio_root}/jamendo" \
      --limit-shards 2

### Full Run

Once the test run succeeds, launch the full run:

    python prepare_audio.py \
      --captions-dir /path/to/captions \
      --jamendo-root "${audio_root}/jamendo" \
      --workers 8

Optional flags:

- `--downmix` to decode the stereo MP3s and store mono FLAC instead (about 1.5-2x the disk space). Without it, the linked cuts use the first channel.
- `--all-tracks` to keep every track, not just the ones CaptionStew references.

## 2. Expected Output

    ${audio_root}/jamendo/
    └── <track_id>.mp3   (~360k files, ~1 TB)

## 3. Link and Validate

These commands link the per-dataset caption file(s) this source serves (Outcome B in the top-level README). To build CaptionStew 10M (Outcome A), prepare every audio source first and then link `captions/10M/shard_*.jsonl.gz` as described in the [top-level README](../../README.md#outcome-a-captionstew-10m).

    python ../../link_captions.py /path/to/captions/dataset_splits/JamendoMaxCaps.jsonl.gz \
      --root jamendo="${audio_root}/jamendo" \
      --output-dir /path/to/manifests

    python ../../validate.py /path/to/manifests/dataset_splits/JamendoMaxCaps.jsonl.gz

## Caption Reference

The audio and metadata are from the following work; the captions used in CaptionStew were generated for 30-second windows of each track:

> Roy, Abhinaba, et al. *JamendoMaxCaps: A Large Scale Music-caption Dataset with Imputed Metadata.* IJCNN, 2025.

## Notes

- Decoding a window from an MP3 seeks into the full track, which is slower than reading a short WAV file. If data loading becomes a bottleneck, consider `--downmix` (FLAC seeks quickly) or caching features.
- Some MP3s carry malformed ID3 tags; `libmpg123` prints warnings for them while the audio still decodes correctly.
- Set `HF_TOKEN` to avoid anonymous rate limits, and lower `--workers` if you see HTTP 429 errors.
- The tracks are distributed under Creative Commons licenses chosen by each artist; check the license of each track before use.
