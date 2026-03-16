# VGGSound Preparing Script

This guide describes how to download the VGGSound dataset and generate Lhotse manifests for training use.

We pre-filtered the VGGSound dataset to remove overlap with common evaluation benchmarks, including the AudioCaps evaluation/test set and the MusicCaps evaluation set.

## Dataset Source

The audio is downloaded from the 16,000 Hz version of VGGSound hosted on HuggingFace. The caption file used in this pipeline is `VGGSound_AudioSetCaps_caption.filtered.csv`

## Workflow

1. Prepare a local directory for storing the downloaded `.wav` files.
2. Run a small test job to verify the pipeline.
3. Launch the full preprocessing and manifest generation run.
4. Validate the generated Lhotse manifest.
5. (Optional) Remove the downloaded HuggingFace cache after preprocessing to save disk space.

## 1. Download and Preprocess the Dataset

The preprocessing script downloads the 16,000 Hz sample rate version from Hugging Face, stores the `.wav` files locally at `/path/to/audio`, and then generates the Lhotse manifests.

### Test Run

Use a small test run for debugging purposes:

    python generate_manifest.py /path/to/audio VGGSound_AudioSetCaps_caption.filtered.csv /output/jsonl/gz/path --split train --limit 100

### Full Run

Once the test run succeeds, launch the full run:

    python generate_manifest.py /path/to/audio VGGSound_AudioSetCaps_caption.filtered.csv /output/jsonl/gz/path --split train

## 2. Expected Output

After preprocessing, the audio files will be stored locally under `/path/to/audio`, and the generated Lhotse manifest files will be written to `/output/jsonl/gz/path`

## 3. Validate the Output

After manifest generation, validate the output manifest(s):

    lhotse validate /output/jsonl/gz/path

You can also inspect basic statistics with:

    lhotse cut describe /output/jsonl/gz/path

## 4. Optional Cleanup

After downloading and preprocessing, you may want to delete the downloaded Hugging Face dataset cache to save disk space:

    cd /path/to/huggingface/dataset/cache
    rm -rf txya900619___vggsound-16k/

## Caption Reference

The captions are provided in the following work:

> Bai, Jisheng, et al. *AudioSetCaps: An Enriched Audio-Caption Dataset Using Automated Generation Pipeline with Large Audio and Language Models.* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2025.

## Notes

- This version is pre-filtered to remove overlap with common captioning evaluation benchmarks.
- It is recommended to run the small test job first before launching the full preprocessing pipeline.
- After preprocessing, you may safely remove the Hugging Face cache if you no longer need the downloaded source files.
- Make sure you have sufficient disk space for both the local audio files and the generated manifest files.