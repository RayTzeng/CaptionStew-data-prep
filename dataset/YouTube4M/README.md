# YouTube-4M Preparing Script

This guide describes how to download the YouTube-4M audio (the YouTube-8M part of AudioSetCaps) from Google Drive and extract the archives efficiently, so that it can be linked to the CaptionStew captions (`dataset_splits/YouTube4M.jsonl.gz`, 4.01M clips with AudioSetCaps captions).

The CaptionStew captions are pre-filtered to remove overlap with common evaluation benchmarks, including the AudioCaps evaluation/test set, the VggSound test set, and the MusicCaps evaluation set.

## Dataset Source

The dataset is hosted on Google Drive. The link is shared on request: open an issue in this repository using the
[**YouTube-4M access request**](https://github.com/RayTzeng/CaptionStew-data-prep/issues/new?template=youtube4m-access.yml)
template and provide the email address (Google account) the folder should be shared with, your affiliation, and your
intended use. The link will be sent to that email address.

Issues are public, so your email address will be visible to anyone who can see the repository.

Because this dataset is large (around 600 GB), a good practice is to first make a copy to your own Google Drive. This helps avoid overusing the shared link and usually makes the download workflow more stable.

## Recommended Workflow

1. Request the Google Drive link (see above) and make a copy of the shared dataset folder into your own Google Drive.
2. Install and configure `rclone`.
3. Download the copied folder from your own Google Drive to your local machine or remote server.
4. Extract all `.tar.gz` archives.
5. Link the captions to the audio.
6. Validate the linked manifests.

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

    rclone copy mydrive:/path/to/copied/YouTube4M "${audio_root}/youtube4m" --progress

You may also add additional `rclone` options depending on your environment, for example increasing transfer concurrency:

    rclone copy mydrive:/path/to/copied/YouTube4M "${audio_root}/youtube4m" \
      --progress \
      --transfers 8 \
      --checkers 16

If you prefer not to copy the folder, `rclone` can also read the shared folder directly by its ID, the last part of the folder link you received (you need to be signed in to Google Drive with the remote):

    rclone copy mydrive: "${audio_root}/youtube4m" \
      --drive-root-folder-id <folder_id> \
      --progress

After downloading, the dataset directory should contain many `.tar.gz` files distributed across batch folders.

## 4. Extract All `.tar.gz` Files

Because there are more than 1000 `.tar.gz` files, it is highly recommended to use `pv` and `pigz` for progress monitoring and faster decompression.

Install them with:

    conda install pv pigz

Then run:

    find "${audio_root}/youtube4m" -type f -name "*.tar.gz" | while read f; do
      echo "Extracting $f"
      pv "$f" | pigz -dc | tar -xf - -C "$(dirname "$f")" && rm "$f"
    done

This extracts each `.tar.gz` into its parent directory and deletes the archive afterwards (drop `&& rm "$f"` to keep the archives).
Otherwise, you may also use the normal extraction command:

    find "${audio_root}/youtube4m" -type f -name "*.tar.gz" | while read f; do
        d=$(dirname "$f")
        tar -xzf "$f" -C "$d" && rm "$f"
    done

## 5. Expected Folder Structure After Extraction

After extraction, the folder structure should look like this:

    ${audio_root}/youtube4m/
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

Each category directory contains `.mp3` files named `<youtube_id>_<start>_<duration>.mp3`.

## 6. Link the Captions

These commands link the per-dataset caption file(s) this source serves (Outcome B in the top-level README). To build CaptionStew 10M (Outcome A), prepare every audio source first and then link `captions/10M/shard_*.jsonl.gz` as described in the [top-level README](../../README.md#outcome-a-captionstew-10m).

    python ../../link_captions.py /path/to/captions/dataset_splits/YouTube4M.jsonl.gz \
      --root youtube4m="${audio_root}/youtube4m" \
      --output-dir /path/to/manifests

The script reports how many clips were found and lists missing ones in `/path/to/manifests/missing_audio.txt`.

## 7. Validate the Output

    python ../../validate.py /path/to/manifests/dataset_splits/YouTube4M.jsonl.gz

## Caption Reference

The captions and the YouTube-8M audio provided in this folder are based on:

>Bai, Jisheng, et al. *AudioSetCaps: An Enriched Audio-Caption Dataset Using Automated Generation Pipeline with Large Audio and Language Models.* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2025.


## Notes

- The full dataset is large, so make sure you have sufficient disk space before downloading and extraction.
- Extraction can take a long time because the dataset contains a very large number of archives.
- Using your own Google Drive copy is recommended for reliability and to avoid overusing the original shared folder.
- About 10k clips appear in two category folders. CaptionStew keeps one cut per clip (with the captions of both entries), pointing to the first folder it was listed under.
