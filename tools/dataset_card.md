---
pretty_name: CaptionStew
language:
- en
license: other
tags:
- audio
- audio-captioning
- audio-language
- lhotse
size_categories:
- 1M<n<10M
configs:
- config_name: CS10M
  data_files: "10M/*.jsonl.gz"
- config_name: CS4M
  data_files: "4M/*.jsonl.gz"
- config_name: CS1M
  data_files: "1M/*.jsonl.gz"
- config_name: CS400k
  data_files: "400k/*.jsonl.gz"
{per_source_configs}
---

# CaptionStew

CaptionStew is the audio-caption corpus introduced in
[Revisiting Audio-language Pretraining for Learning General-purpose Audio Representation](https://arxiv.org/abs/2511.16757).
It aggregates eight open-source audio caption datasets (AudioCaps, Clotho, MusicCaps, WavCaps, AudioSetCaps,
FusionAudio, JamendoMaxCaps and ParaSpeechCaps) spanning environmental sound, music and expressive speech.
Captions of the same audio clip coming from different datasets are merged into one caption list.

This repository holds **captions only**. The audio has to be downloaded from the original sources; the
[CaptionStew-data-prep](https://github.com/RayTzeng/CaptionStew-data-prep) repository has one recipe per audio
source and a script (`link_captions.py`) that links these manifests to your local audio.

## Files

| Folder | Content |
|---|---|
| `10M/`, `4M/`, `1M/`, `400k/` | The data-scaling subsets used in the paper (Sec. 4.5), sharded. `10M` is the full corpus (CS10M). Captions of identical clips are merged. |
| `dataset_splits/` | One file per source caption dataset, captions not merged across datasets. |

{stats_table}

## Format

Every file is a gzipped [Lhotse](https://github.com/lhotse-speech/lhotse) `CutSet` (one `MonoCut` per line).
Captions are stored as a list in `supervisions[0].custom.caption`:

```json
{example}
```

`recording.sources[0].source` is a **path key**, not a file path: `<source>/<path relative to the source root>`
without extension. `link_captions.py` resolves each key to a local file, refreshes the sampling rate and length
from the file, and writes ready-to-use manifests. The recording metadata stored here describes the authors' copy
of the audio and is replaced during linking.

| Key prefix | Audio source | Recipe |
|---|---|---|
| `audioset/{unbalanced,balanced}/Y<youtube_id>` | AudioSet (`Muno459/AudioSet`) | `dataset/AudioSet` |
| `vggsound/<youtube_id>_<start>` | VGGSound (`txya900619/vggsound-16k`) | `dataset/VggSound` |
| `wavcaps/{FreeSound,BBC_Sound_Effects,SoundBible,AudioSet_SL}/<id>` | WavCaps (`cvssp/WavCaps`) | `dataset/WavCaps` |
| `clotho/development/<file name, spaces replaced by _>` | Clotho v2.1 (Zenodo) | `dataset/Clotho` |
| `youtube4m/batch<N>/<category>/<id>` | AudioSetCaps YouTube-8M audio | `dataset/YouTube4M` |
| `voxceleb1/...`, `voxceleb2/...`, `ears/...`, `expresso/...`, `emilia/...` | ParaSpeechCaps sources | `dataset/ParaSpeechCaps` |
| `jamendo/<track_id>` | JamendoMaxCaps (`amaai-lab/JamendoMaxCaps`) | `dataset/JamendoMaxCaps` |

JamendoMaxCaps captions describe 30-second windows of a track. Their cuts point to the full track and use
`start` = window offset; the cut ID is `<track_id>_<offset>`.

## Changes from v1

v1 (tag `v1.0`) stored absolute paths from the authors' cluster. v2:

- replaces absolute paths with path keys (see above),
- turns JamendoMaxCaps windows into cuts on the full track,
- prefixes SoundBible IDs (`SoundBible_<id>`), which collided with FreeSound/Clotho IDs of different audio,
- merges cuts that shared an ID within a file or subset into one cut with all captions ({merged:,} cuts merged),
- drops cuts with zero duration ({dropped:,} cuts),
- always stores captions as a list.

## Notes

- Clips are at most ~30 s long (JamendoMaxCaps windows are 30.08 s).
- The audio of each source keeps its original license; check the source datasets before use.

## Citation

```bibtex
@article{tseng2025revisiting,
  title={Revisiting Audio-language Pretraining for Learning General-purpose Audio Representation},
  author={Tseng, Wei-Cheng and Zhou, Xuanru and Huo, Mingyue and Shao, Yiwen and Zhang, Hao and Yu, Dong},
  journal={arXiv preprint arXiv:2511.16757},
  year={2025}
}
```
