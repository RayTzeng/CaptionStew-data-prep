# VGGSound Preparing Script

This guide describes how to download the VGGSound audio used by CaptionStew (`dataset_splits/VggSound.jsonl.gz`, 181,913 clips with AudioSetCaps captions).

The CaptionStew captions are pre-filtered to remove overlap with common evaluation benchmarks, including the AudioCaps evaluation/test set and the MusicCaps evaluation set.

## Dataset Source

The audio is downloaded from the 16,000 Hz version of VGGSound hosted on Hugging Face, [`txya900619/vggsound-16k`](https://huggingface.co/datasets/txya900619/vggsound-16k). All 181,913 clips referenced by CaptionStew are in its train split.

## Workflow

1. Download the CaptionStew captions (see the top-level README).
2. Run a small test job to verify the pipeline.
3. Launch the full run.
4. Link and validate with the top-level scripts.

## 1. Download the Audio

`prepare_audio.py` downloads one parquet shard of the train split at a time (118 shards, about 62 GB in total), writes the clips referenced by the CaptionStew captions as individual `.wav` files (bytes copied as-is, no re-encoding), and deletes the shard again. Completed shards are logged in `${audio_root}/vggsound/.done_shards`, so an interrupted run can be restarted with the same command.

### Test Run

Use a small test run for debugging purposes:

    python prepare_audio.py \
      --captions-dir /path/to/captions \
      --vggsound-root "${audio_root}/vggsound" \
      --limit-shards 2

### Full Run

Once the test run succeeds, launch the full run:

    python prepare_audio.py \
      --captions-dir /path/to/captions \
      --vggsound-root "${audio_root}/vggsound" \
      --workers 8

## 2. Expected Output

    ${audio_root}/vggsound/
    └── <youtube_id>_<start>.wav   (181,913 files, 16 kHz mono)

## 3. Link and Validate

These commands link the per-dataset caption file(s) this source serves (Outcome B in the top-level README). To build CaptionStew 10M (Outcome A), prepare every audio source first and then link `captions/10M/shard_*.jsonl.gz` as described in the [top-level README](../../README.md#outcome-a-captionstew-10m).

    python ../../link_captions.py /path/to/captions/dataset_splits/VggSound.jsonl.gz \
      --root vggsound="${audio_root}/vggsound" \
      --output-dir /path/to/manifests

    python ../../validate.py /path/to/manifests/dataset_splits/VggSound.jsonl.gz

## Caption Reference

The captions are provided in the following work:

> Bai, Jisheng, et al. *AudioSetCaps: An Enriched Audio-Caption Dataset Using Automated Generation Pipeline with Large Audio and Language Models.* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2025.

The audio is from:

> Chen, Honglie, et al. *VGGSound: A Large-scale Audio-Visual Dataset.* ICASSP, 2020.

## Notes

- This version is pre-filtered to remove overlap with common captioning evaluation benchmarks.
- It is recommended to run the small test job first before launching the full pipeline.
- Parquet shards are deleted after processing unless `--keep-parquet` is given, so no separate cache cleanup is needed.
- Set `HF_TOKEN` to avoid anonymous rate limits, and lower `--workers` if you see HTTP 429 errors.
