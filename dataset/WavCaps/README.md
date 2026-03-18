# WavCaps (BBC_Sound_Effects + SoundBible) Preparing Script

This guide describes how to download and prepare the BBC_Sound_Effects and SoundBible subsets from WavCaps, and generate Lhotse manifests for training use.

WavCaps contains LLM-generated captions for audio from AudioSet, BBC_Sound_Effects, FreeSound, and SoundBible. In this folder, we only process the BBC_Sound_Effects and SoundBible subsets, since they contain no overlap with other common sources.

## Dataset Source

We download the dataset via the `aac-datasets` package.

Suppose you have a `${root_folder}` argument. Running the commands below will download the dataset under the Hugging Face home directory `$HF_HOME` and create a symbolic link under `${root_folder}`.

## Recommended Workflow

1. Download each source dataset.
2. Run a small test job to verify the pipeline.
3. Generate Lhotse manifests.
4. Validate the generated Lhotse manifests and inspect basic statistics.

## 1. Download Each Source Dataset

Download the required subsets with:

    aac-datasets-download --root "${root_folder}" wavcaps --subsets "bbc"
    aac-datasets-download --root "${root_folder}" wavcaps --subsets "soundbible"

After downloading, `${root_folder}` should have the following structure:

    ${root_folder}/
    └── WavCaps
        ├── Audio
        │   ├── BBC_Sound_Effects
        │   │   └── (31201 flac files, ~142GB)
        │   └── SoundBible
        │       └── (1232 flac files, ~884MB)
        ├── Zip_files
        │   ├── BBC_Sound_Effects
        │   │   └── (26 zip files, ~562GB)
        │   └── SoundBible
        │       └── (1 zip file, ~624GB)
        ├── json_files
        │   ├── BBC_Sound_Effects
        │   │   └── bbc_final.json
        │   ├── SoundBible
        │   │   └── sb_final.json
        │   └── blacklist
        │       ├── blacklist_exclude_all_ac.json
        │       ├── blacklist_exclude_test_ac.json
        │       └── blacklist_exclude_ubs8k_esc50_vggsound.json
        ├── .gitattributes
        └── README.md

## 2. Generate Lhotse Manifests

After downloading the datasets, generate the manifests with `generate_manifest.py`.

### Test Run

Use a small test run for debugging purposes:

    python3 generate_manifest.py \
      "${root_folder}" \
      /path/to/output/manifest \
      --subsets BBC_Sound_Effects SoundBible \
      --limit 100 \
      --workers 8

### Full Run

Once the test run succeeds, launch the full run:

    python3 generate_manifest.py \
      "${root_folder}" \
      /path/to/output/manifest \
      --subsets BBC_Sound_Effects SoundBible \
      --workers 8

## 3. Validate the Output

After manifest generation, validate the output manifest(s):

    lhotse validate /path/to/output/manifest

You can also inspect basic statistics with:

    lhotse cut describe /path/to/output/manifest

## 4. (Optional) Remove Zip Files

After preprocessing, you may ought to remove the zip files under `$HF_HOME`. The actual path can be easily find via the following command:

    ls -ld '${root_folder}/WavCaps/Zip_files'

## Caption Reference

The captions are provided in the following work:

> Mei, Xinhao, et al. *WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research.* IEEE/ACM Transactions on Audio, Speech, and Language Processing 32 (2024): 3339-3354.

The downloading script uses:

> Labbé, E. (2025). *aac-datasets: Audio Captioning datasets for PyTorch.*

## Notes

- This preparation only uses the BBC_Sound_Effects and SoundBible subsets from WavCaps.
- The downloaded data are managed through `aac-datasets`, which stores the actual files under `HF_HOME` and creates symbolic links under `${root_folder}`.
- It is recommended to run the small test job first before launching the full preprocessing pipeline.
- Make sure you have sufficient disk space for both the downloaded audio and the generated manifest files.