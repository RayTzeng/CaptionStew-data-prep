# AudioSet Preparing Script

This guide describes how to download the AudioSet clips used by CaptionStew. AudioSet audio is shared by four caption sources:

| Caption file | AudioSet split | Clips referenced |
|---|---|---:|
| `dataset_splits/audiocaps_train.jsonl.gz` | unbalanced | 90,859 |
| `dataset_splits/AudioSetCaps.jsonl.gz` | unbalanced | 1,833,428 |
| `dataset_splits/FusionAudio.jsonl.gz` | unbalanced | 1,175,190 |
| `dataset_splits/MusicCaps_train.jsonl.gz` | balanced | 2,632 |

AudioSetCaps and FusionAudio also reference clips from the AudioSet strongly-labelled subset (`wavcaps/AudioSet_SL/...`); those come from the WavCaps recipe.

## Dataset Source

The audio is downloaded from the Hugging Face copy of AudioSet, [`Muno459/AudioSet`](https://huggingface.co/datasets/Muno459/AudioSet): 48 kHz FLAC clips (mostly stereo) packed in WebDataset tar shards of about 4 GB, with an index (`metadata/clips.parquet`) that lists the shard and file of every clip. Access is granted for non-commercial research: accept the terms on the dataset page and log in with `hf auth login` before running the script.

Not every AudioSet clip is still available on YouTube, so the copy does not hold all of them:

| Split | Clips referenced by CaptionStew | Available in `Muno459/AudioSet` |
|---|---:|---:|
| unbalanced | 1,834,030 | 1,660,238 (90.5%) |
| balanced | 2,632 | 2,626 (99.8%) |

Missing clips are dropped (and listed) by `link_captions.py`.

## Workflow

1. Download the CaptionStew captions (see the top-level README).
2. Run a small test job to verify the pipeline.
3. Launch the full run.
4. Link and validate with the top-level scripts.

## 1. Download the Audio

`prepare_audio.py` reads the clip index, downloads only the shards that hold clips referenced by the CaptionStew manifests (one shard at a time per worker), writes those clips as individual FLAC files named after their YouTube ID, and deletes the shard again. Peak extra disk usage is about `workers x 4 GB`. Completed shards are logged in `${audioset_root}/.done_shards`, so an interrupted run can be restarted with the same command.

### Test Run

Use a small test run for debugging purposes (one balanced shard, about 4 GB):

    python prepare_audio.py \
      --captions-dir /path/to/captions \
      --audioset-root "${audio_root}/audioset" \
      --splits balanced \
      --limit-shards 1

### Full Run

Once the test run succeeds, launch the full run:

    python prepare_audio.py \
      --captions-dir /path/to/captions \
      --audioset-root "${audio_root}/audioset" \
      --workers 8

Optional flags:

- `--splits balanced` to download only the MusicCaps clips (7 shards, about 26 GB).
- `--downmix` to average multi-channel clips to mono while writing (still lossless FLAC at 48 kHz). About 85% of the clips are stereo (and a few are 5.1); without this flag the linked cuts use the first channel.
- `--all-clips` to keep every train clip, not just the ones CaptionStew references.

## 2. Expected Output

    ${audio_root}/audioset/
    ├── unbalanced/
    │   └── Y<youtube_id>.flac      (~1.66M files)
    └── balanced/
        └── Y<youtube_id>.flac      (~2.6k files)

## 3. Link and Validate

These commands link the per-dataset caption file(s) this source serves (Outcome B in the top-level README). To build CaptionStew 10M (Outcome A), prepare every audio source first and then link `captions/10M/shard_*.jsonl.gz` as described in the [top-level README](../../README.md#outcome-a-captionstew-10m).

    python ../../link_captions.py \
      /path/to/captions/dataset_splits/{audiocaps_train,MusicCaps_train,AudioSetCaps,FusionAudio}.jsonl.gz \
      --root audioset="${audio_root}/audioset" \
      --root wavcaps="${audio_root}/wavcaps" \
      --output-dir /path/to/manifests

    python ../../validate.py /path/to/manifests/dataset_splits/*.jsonl.gz

## Caption Reference

The captions are provided in the following works:

> Kim, Chris Dongjoo, et al. *AudioCaps: Generating Captions for Audios in The Wild.* NAACL-HLT, 2019.

> Agostinelli, Andrea, et al. *MusicLM: Generating Music From Text.* arXiv:2301.11325, 2023. (MusicCaps)

> Bai, Jisheng, et al. *AudioSetCaps: An Enriched Audio-Caption Dataset Using Automated Generation Pipeline with Large Audio and Language Models.* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2025.

> Chen, Shunian, et al. *FusionAudio-1.2M: Towards Fine-grained Audio Captioning with Multimodal Contextual Fusion.* arXiv:2506.01111, 2025.

The audio is from:

> Gemmeke, Jort F., et al. *Audio Set: An Ontology and Human-Labeled Dataset for Audio Events.* ICASSP, 2017.

## Notes

- The shards holding the referenced unbalanced clips total about 2.4 TB; the extracted clips take roughly the same space.
- It is recommended to run the small test job first before launching the full pipeline.
- The audio is YouTube content shared for non-commercial research; do not redistribute it.
- Lower `--workers` if you see HTTP 429 (rate limit) errors.
