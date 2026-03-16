# YouTube-8M Preparing Script

This guide describes how to download the YouTube-8M audio dataset from Google Drive, extract the archives efficiently, and generate Lhotse manifests for training use.

We pre-filtered the YouTube-8M dataset to remove overlap with common evaluation benchmarks, including the AudioCaps evaluation/test set, the VggSound test set, and the MusicCaps evaluation set.

## Dataset Source

The dataset is hosted on Google Drive:

<https://drive.google.com/drive/folders/1ZKyRZw3AhS3HkWivgMqtMODB0TkVPNk5?usp=sharing>

Because this dataset is large (around 600 GB), a good practice is to first make a copy to your own Google Drive. This helps avoid overusing the shared link and usually makes the download workflow more stable.

## Recommended Workflow

1. Make a copy of the shared dataset folder into your own Google Drive.
2. Install and configure `rclone`.
3. Download the copied folder from your own Google Drive to your local machine or remote server.
4. Extract all `.tar.gz` archives.
5. Generate Lhotse manifests.
6. Validate the manifests and inspect basic statistics.

## 1. Install `rclone`

It is highly recommended to use `rclone` for downloading from Google Drive.

You can install it with either of the following methods:

    conda install conda-forge::rclone

or

    sudo -v ; curl https://rclone.org/install.sh | sudo bash

## 2. Configure `rclone` Access to Google Drive

Follow the official `rclone` Google Drive setup guide:

<https://rclone.org/drive/>

If you are working on a remote server, also refer to the remote setup documentation:

<https://rclone.org/remote_setup/>

After setup, make sure your Google Drive remote is accessible. In the examples below, we assume your configured remote is named `mydrive`. Replace it with your actual remote name if needed.

## 3. Download the Dataset with `rclone`

Once the folder has been copied into your own Google Drive, download it with `rclone`:

    rclone copy mydrive:/path/to/copied/YouTube8M /path/to/YouTube8M --progress

You may also add additional `rclone` options depending on your environment, for example increasing transfer concurrency:

    rclone copy mydrive:/path/to/copied/YouTube8M /path/to/YouTube8M \
      --progress \
      --transfers 8 \
      --checkers 16

After downloading, the dataset directory should contain many `.tar.gz` files distributed across batch folders.

## 4. Extract All `.tar.gz` Files

Because there are more than 1000 `.tar.gz` files, it is highly recommended to use `pv` and `pigz` for progress monitoring and faster decompression.

Install them with:

    conda install pv pigz

Then run:

    find /path/to/YouTube8M -type f -name "*.tar.gz" | while read f; do
      echo "Extracting $f"
      pv "$f" | pigz -dc | tar -xf - -C "$(dirname "$f")"
    done

This extracts each `.tar.gz` into its parent directory.
Otherwise, you may also use the normal extraction command:

    find folder -type f -name "*.tar.gz" | while read f; do
        d=$(dirname "$f")
        tar -xzf "$f" -C "$d"
    done


## 5. Expected Folder Structure After Extraction

After extraction, the folder structure should look like this:

    YouTube8M/
    ├── batch1/           (110 categories)
    │   ├── Terra_(comics)/
    │   │   └── *.mp3 files
    │   ├── Walking/
    │   ├── Walkman/
    │   ├── Wall/
    │   └── ... (106 more categories)
    │
    ├── batch2/           (72 categories)
    ├── batch3/           (174 categories)
    ├── batch4/           (240 categories)
    ├── batch5/           (190 categories)
    ├── batch6/           (340 categories)
    ├── batch7/           (196 categories)
    ├── batch8/           (243 categories)
    ├── batch9/           (247 categories)
    ├── batch10/          (294 categories)
    └── batch11/          (305 categories)

Each category directory contains `.mp3` files.

## 6. Generate Lhotse Manifests

After extraction, download the caption `.csv` file and generate the Lhotse manifests with your `generate_manifest.py` script.

Download the caption file:

    wget https://github.com/RayTzeng/CaptionStew-data-prep/releases/download/v1.0.0/YouTube-8M_AudioSetCaps_caption.filtered.csv

### Test Run

Use a small test run for debugging purposes:

    python generate_manifest.py \
      /path/to/YouTube8M \
      YouTube-8M_AudioSetCaps_caption.filtered.csv \
      /output/jsonl/gz/path \
      --limit 100

### Full Run

Once the test run succeeds, launch the full manifest generation:

    python generate_manifest.py \
      /path/to/YouTube8M \
      YouTube-8M_AudioSetCaps_caption.filtered.csv \
      /output/jsonl/gz/path

## 7. Validate the Output

After manifest generation, validate the output manifest(s):

    lhotse validate /output/jsonl/gz/path

You can also inspect basic statistics with:

    lhotse cut describe /output/jsonl/gz/path

If your script writes multiple manifest files, replace `/output/jsonl/gz/path` with the specific manifest file you want to validate or inspect.

## Caption Reference

The caption file and the YouTube-8M downloading link provided in this folder is based on:

>Bai, Jisheng, et al. *AudioSetCaps: An Enriched Audio-Caption Dataset Using Automated Generation Pipeline with Large Audio and Language Models.* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2025.


## Notes

- The full dataset is large, so make sure you have sufficient disk space before downloading and extraction.
- Extraction can take a long time because the dataset contains a very large number of archives.
- Running a small manifest-generation test first is strongly recommended before launching the full pipeline.
- Using your own Google Drive copy is recommended for reliability and to avoid overusing the original shared folder.