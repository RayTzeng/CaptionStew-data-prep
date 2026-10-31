# WavCaps Preparing Script

This guide describes how to download the WavCaps audio used by CaptionStew.

WavCaps contains LLM-generated captions for audio from AudioSet (strongly-labelled subset), BBC Sound Effects, FreeSound, and SoundBible. CaptionStew uses all four subsets:

| Subset | Clips referenced by `dataset_splits/WavCaps.jsonl.gz` | Archive size |
|---|---:|---:|
| FreeSound | 146,609 | 653 GB |
| AudioSet_SL | 84,150 | 36 GB |
| BBC_Sound_Effects | 7,609 | 130 GB |
| SoundBible | 1,140 | 0.6 GB |

The AudioSet_SL audio is also used by `dataset_splits/AudioSetCaps.jsonl.gz` (75,718 clips) and `dataset_splits/FusionAudio.jsonl.gz` (52,838 clips). Clips longer than about 30 s are not used by CaptionStew, which is why only part of FreeSound and BBC_Sound_Effects is referenced.

## Dataset Source

The audio is downloaded from the official WavCaps release on Hugging Face, [`cvssp/WavCaps`](https://huggingface.co/datasets/cvssp/WavCaps). Each subset is a split zip archive of 32 kHz FLAC files.

## Recommended Workflow

1. Download and extract each subset.
2. Link and validate with the top-level scripts.

## 1. Download and Extract

`prepare_audio.sh` downloads each subset's archive with the `hf` CLI, extracts the FLAC files with `7z` (split archives are read directly, no merging needed), and deletes the archive afterwards. Finished subsets are marked with `<Subset>/.done`, so the script can be re-run after an interruption.

### Test Run

Start with the smallest subset to verify the pipeline:

    ./prepare_audio.sh "${audio_root}/wavcaps" SoundBible

### Full Run

Once the test run succeeds, prepare all subsets:

    ./prepare_audio.sh "${audio_root}/wavcaps"

Peak disk usage for a subset is its archive plus its extracted audio (about 1.3 TB for FreeSound). Set `WAVCAPS_DOWNLOAD_DIR` to place the archives on a different disk.

## 2. Expected Output

    ${audio_root}/wavcaps/
    ├── AudioSet_SL/<youtube_id>.flac
    ├── BBC_Sound_Effects/<id>.flac
    ├── FreeSound/<id>.flac
    └── SoundBible/<id>.flac

## 3. Link and Validate

These commands link the per-dataset caption file(s) this source serves (Outcome B in the top-level README). To build CaptionStew 10M (Outcome A), prepare every audio source first and then link `captions/10M/shard_*.jsonl.gz` as described in the [top-level README](../../README.md#outcome-a-captionstew-10m).

    python ../../link_captions.py /path/to/captions/dataset_splits/WavCaps.jsonl.gz \
      --root wavcaps="${audio_root}/wavcaps" \
      --output-dir /path/to/manifests

    python ../../validate.py /path/to/manifests/dataset_splits/WavCaps.jsonl.gz

## Caption Reference

The captions are provided in the following work:

> Mei, Xinhao, et al. *WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research.* IEEE/ACM Transactions on Audio, Speech, and Language Processing 32 (2024): 3339-3354.

## Notes

- Requires the `hf` CLI (`pip install -U huggingface_hub`) and `7z` (`p7zip`).
- SoundBible IDs collide with FreeSound IDs of different sounds, so CaptionStew names SoundBible cuts `SoundBible_<id>`; the audio files keep their original names.
- Unreferenced clips (longer FreeSound/BBC recordings) are extracted as well; they are simply not linked.
- Make sure you have sufficient disk space for both the downloaded archives and the extracted audio.
