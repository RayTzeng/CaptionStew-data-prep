
# ParaSpeechCaps Preparing Script

This guide describes how to prepare the ParaSpeechCaps dataset by downloading each source dataset, applying the required preprocessing steps, and generating Lhotse manifests for training use.

ParaSpeechCaps contains speaking-style captions paired with audio from VoxCeleb, EARS, Expresso, and Emilia.

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

## Recommended Workflow

1. Download and prepare each source dataset.
2. Run a small test job to verify the pipeline.
3. Generate Lhotse manifests.
4. Validate the generated manifests and inspect basic statistics.

## 1. Download and Prepare Each Source Dataset

### 1.1 VoxCeleb

Request access to the VoxCeleb dataset here:

<https://mm.kaist.ac.kr/datasets/voxceleb/>

Download the audio files for both VoxCeleb1 and VoxCeleb2.

#### VoxCeleb1

A minimal preparation script looks like:

    mkdir -p "${voxceleb1_root}"
    cd "${voxceleb1_root}"
    mkdir -p dev test

    cd dev
    wget "link to VoxCeleb1 dev zip parts"
    cat vox1_dev_wav* > vox1_dev_wav.zip
    unzip vox1_dev_wav.zip

    cd ../test
    wget "link to VoxCeleb1 test zip"
    unzip vox1_test_wav.zip

After extraction, the directory structure should look like:

    ${voxceleb1_root}/
    ├── dev/
    │   └── wav/
    └── test/
        └── wav/

#### VoxCeleb2

A minimal preparation script looks like:

    mkdir -p "${voxceleb2_root}"
    cd "${voxceleb2_root}"
    mkdir -p dev test

    cd dev
    wget "link to VoxCeleb2 dev zip parts"
    cat vox2_dev_aac* > vox2_aac.zip
    unzip vox2_aac.zip

    cd ../test
    wget "link to VoxCeleb2 test zip"
    unzip vox2_test_aac.zip

After extraction, the directory structure should look like:

    ${voxceleb2_root}/
    ├── dev/
    │   └── aac/
    └── test/
        └── aac/

Convert the `.m4a` files in VoxCeleb2 to `.wav` files using the following script. This creates a copy of each audio file with a `.wav` extension in the same directory:

    ./preprocessing/convert_m4a_to_wav.py "${voxceleb2_root}"

### 1.2 Expresso

Download the Expresso dataset and extract it:

    wget https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar
    tar -xvf expresso.tar

After extraction, the directory structure should look like:

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

    python ./audio_preprocessing/apply_expresso_vad.py "${expresso_root}"

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

Create the target directory and download the English subset archives:

    mkdir -p "${emilia_root}"
    cd "${emilia_root}"

    repo="amphion/Emilia-Dataset"
    rev="fc71e07e8572f5f3be1dbd02ed3172a4d298f152"

    for i in $(seq 1 113); do
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

    for i in $(seq 1 113); do
      tag=$(printf "B%05d" "$i")
      tar -xzf "EN_${tag}.tar.gz"
    done

After extraction, the directory structure should look like:

    ${emilia_root}/
    └── EN/
        ├── EN_B00001/
        ├── EN_B00002/
        ├── ...
        └── EN_B00113/

## 2. Generate Lhotse Manifests

After all source datasets are prepared, generate the ParaSpeechCaps manifests.

### Test Run

Use a small test run for debugging purposes:

    python generate_manifest.py \
      --voxceleb1_path "${voxceleb1_root}" \
      --voxceleb2_path "${voxceleb2_root}" \
      --ears_path "${ears_root}" \
      --expresso_path "${expresso_root}" \
      --emilia_path "${emilia_root}" \
      --output_dir /path/to/store/manifests \
      --num_workers 8 \
      --no-find \
      --limit 100

### Full Run

Once the test run succeeds, launch the full run:

    python generate_manifest.py \
      --voxceleb1_path "${voxceleb1_root}" \
      --voxceleb2_path "${voxceleb2_root}" \
      --ears_path "${ears_root}" \
      --expresso_path "${expresso_root}" \
      --emilia_path "${emilia_root}" \
      --output_dir /path/to/store/manifests \
      --num_workers 8 \
      --no-find

The process will generate:

    ParaSpeechCaps_train.jsonl.gz
    ParaSpeechCaps_dev.jsonl.gz
    ParaSpeechCaps_test.jsonl.gz

## 3. Validate the Output

After manifest generation, validate the generated manifests:

    lhotse validate /path/to/store/manifests/ParaSpeechCaps_train.jsonl.gz
    lhotse validate /path/to/store/manifests/ParaSpeechCaps_dev.jsonl.gz
    lhotse validate /path/to/store/manifests/ParaSpeechCaps_test.jsonl.gz

You can also inspect basic statistics with:

    lhotse cut describe /path/to/store/manifests/ParaSpeechCaps_train.jsonl.gz
    lhotse cut describe /path/to/store/manifests/ParaSpeechCaps_dev.jsonl.gz
    lhotse cut describe /path/to/store/manifests/ParaSpeechCaps_test.jsonl.gz

## Caption Reference

The captions are provided in the following work, and some of the preprocessing scripts are derived from their repository:

> Diwan, Anuj, et al. *Scaling Rich Style-Prompted Text-to-Speech Datasets.* Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025.

## Notes

- Make sure all source datasets are downloaded and organized under the expected directory structure before running manifest generation.
- VoxCeleb requires access approval before download.
- Emilia requires access to the Hugging Face repository and a working `hf download` setup.
- It is strongly recommended to run the small test job first before launching the full preprocessing pipeline.
- The `--no-find` flag should be kept if your pipeline expects the source datasets to already be fully prepared in the specified directories.

