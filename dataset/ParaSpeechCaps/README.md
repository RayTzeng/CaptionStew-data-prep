
# ParaSpeechCaps Preparing Script

This guide describes how to prepare the ParaSpeechCaps audio by downloading each source dataset and applying the required preprocessing steps, so that it can be linked to the CaptionStew captions (`dataset_splits/ParaSpeechCaps.jsonl.gz`).

ParaSpeechCaps contains speaking-style captions paired with audio from VoxCeleb, EARS, Expresso, and Emilia. CaptionStew uses its `train_base` and `train_scaled` splits:

| Source dataset | Clips referenced | CaptionStew source name |
|---|---:|---|
| VoxCeleb1 (dev) | 22,257 | `voxceleb1` |
| VoxCeleb2 (dev) | 61,797 | `voxceleb2` |
| EARS | 13,983 | `ears` |
| Expresso | 12,001 | `expresso` |
| Emilia (EN) | 924,651 | `emilia` |

## Dataset Source

ParaSpeechCaps is constructed from the following source datasets:

- VoxCeleb1
- VoxCeleb2
- Expresso
- EARS
- Emilia

You may choose where to store the audio files for each source dataset. In the commands below, we denote their root directories as:

- `${voxceleb1_root}`
- `${voxceleb2_root}`
- `${expresso_root}`
- `${ears_root}`
- `${emilia_root}`

If you keep all CaptionStew audio under one `${audio_root}`, use `${audio_root}/voxceleb1`, `${audio_root}/voxceleb2`, `${audio_root}/expresso`, `${audio_root}/ears` and `${audio_root}/emilia`; otherwise pass each root to `link_captions.py` with `--root`.

## Recommended Workflow

1. Download and prepare each source dataset.
2. Link the captions to the audio.
3. Validate the linked manifests.

## 1. Download and Prepare Each Source Dataset

### 1.1 VoxCeleb

Request access to the VoxCeleb dataset here:

<https://mm.kaist.ac.kr/datasets/voxceleb/>

CaptionStew only uses the dev sets of VoxCeleb1 and VoxCeleb2 (the test sets are kept for evaluation), so the test archives are not needed.

#### VoxCeleb1

A minimal preparation script looks like:

    mkdir -p "${voxceleb1_root}/dev"
    cd "${voxceleb1_root}/dev"
    wget "link to VoxCeleb1 dev zip parts"
    cat vox1_dev_wav* > vox1_dev_wav.zip
    unzip vox1_dev_wav.zip

After extraction, the directory structure should look like:

    ${voxceleb1_root}/
    └── dev/
        └── wav/
            └── id10001/<video_id>/<utterance>.wav

#### VoxCeleb2

A minimal preparation script looks like:

    mkdir -p "${voxceleb2_root}/dev"
    cd "${voxceleb2_root}/dev"
    wget "link to VoxCeleb2 dev zip parts"
    cat vox2_dev_aac* > vox2_aac.zip
    unzip vox2_aac.zip

After extraction, the directory structure should look like:

    ${voxceleb2_root}/
    └── dev/
        └── aac/
            └── id00012/<video_id>/<utterance>.m4a

Convert the `.m4a` files to `.wav` files using the following script (requires `ffmpeg`). This creates a copy of each audio file with a `.wav` extension in the same directory, which is the file that gets linked:

    python ./preprocessing/convert_m4a_to_wav.py "${voxceleb2_root}"

### 1.2 Expresso

Download the Expresso dataset and extract it:

    wget https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar
    tar -xvf expresso.tar

The extracted `expresso/` folder is `${expresso_root}`. Its directory structure should look like:

    ${expresso_root}/
    ├── README.txt
    ├── LICENSE.txt
    ├── read_transcriptions.txt
    ├── VAD_segments.txt
    ├── splits/
    └── audio_48khz/
        ├── conversational/
        └── read/

Apply VAD segmentation to the Expresso conversational audio files. This will create an `audio_48khz/conversational_vad_segmented` directory containing the segmented audio files:

    python ./preprocessing/apply_expresso_vad.py "${expresso_root}"

### 1.3 EARS

Download and extract all EARS partitions:

    mkdir -p "${ears_root}"
    cd "${ears_root}"

    for X in $(seq -w 001 107); do
      curl -L "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p${X}.zip" -o "p${X}.zip"
      unzip "p${X}.zip"
      rm "p${X}.zip"
    done

After extraction, the directory structure should look like:

    ${ears_root}/
    ├── p001/
    ├── p002/
    ├── ...
    └── p107/

### 1.4 Emilia

Emilia is downloaded from Hugging Face. Make sure you have access to the repository before running the commands below.

CaptionStew uses the English batches `EN_B00000` to `EN_B00092`, except `EN_B00090` and `EN_B00091`. Create the target directory and download them:

    mkdir -p "${emilia_root}"
    cd "${emilia_root}"

    repo="amphion/Emilia-Dataset"
    rev="fc71e07e8572f5f3be1dbd02ed3172a4d298f152"

    for i in $(seq 0 92); do
      [ "$i" -eq 90 ] || [ "$i" -eq 91 ] && continue
      tag=$(printf "B%05d" "$i")

      if [ "$tag" = "B00008" ]; then
        hf download "$repo" "EN/EN_${tag}.tar.gz.0" --repo-type dataset --revision "$rev" --local-dir .
        hf download "$repo" "EN/EN_${tag}.tar.gz.1" --repo-type dataset --revision "$rev" --local-dir .
        cat "EN/EN_${tag}.tar.gz.0" "EN/EN_${tag}.tar.gz.1" > "EN/EN_${tag}.tar.gz"
      else
        hf download "$repo" "EN/EN_${tag}.tar.gz" --repo-type dataset --revision "$rev" --local-dir .
      fi
    done

Then extract the downloaded archives:

    cd ./EN

    for f in EN_B*.tar.gz; do
      tar -xzf "$f" && rm "$f"
    done

After extraction, the directory structure should look like:

    ${emilia_root}/
    └── EN/
        ├── EN_B00000/
        │   └── EN_B00000_S00000/mp3/EN_B00000_S00000_W000000.mp3
        ├── EN_B00001/
        ├── ...
        └── EN_B00092/

## 2. Link the Captions

These commands link the per-dataset caption file(s) this source serves (Outcome B in the top-level README). To build CaptionStew 10M (Outcome A), prepare every audio source first and then link `captions/10M/shard_*.jsonl.gz` as described in the [top-level README](../../README.md#outcome-a-captionstew-10m).

After all source datasets are prepared, link the ParaSpeechCaps captions to the audio:

    python ../../link_captions.py /path/to/captions/dataset_splits/ParaSpeechCaps.jsonl.gz \
      --root voxceleb1="${voxceleb1_root}" \
      --root voxceleb2="${voxceleb2_root}" \
      --root ears="${ears_root}" \
      --root expresso="${expresso_root}" \
      --root emilia="${emilia_root}" \
      --output-dir /path/to/manifests

The script reports, per source, how many clips were found, and lists missing ones in `/path/to/manifests/missing_audio.txt`.

## 3. Validate the Output

    python ../../validate.py /path/to/manifests/dataset_splits/ParaSpeechCaps.jsonl.gz

## Caption Reference

The captions are provided in the following work, and some of the preprocessing scripts are derived from their repository:

> Diwan, Anuj, et al. *Scaling Rich Style-Prompted Text-to-Speech Datasets.* Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025.

## Notes

- Make sure all source datasets are downloaded and organized under the expected directory structure before linking.
- VoxCeleb requires access approval before download.
- Emilia requires access to the Hugging Face repository and a working `hf download` setup.
- ParaSpeechCaps' dev and test splits are not part of CaptionStew; they are used for evaluation in the paper.
