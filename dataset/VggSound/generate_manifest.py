#!/usr/bin/env python3
"""
Generate Lhotse manifest for VggSound dataset with AudioSetCaps captions.

This script downloads VggSound dataset from HuggingFace, saves audio files locally,
and creates a Lhotse manifest with captions from the filtered CSV file.

Usage:
    # Process all files (downloads from HuggingFace)
    python generate_manifest.py \
        /path/to/save/audio \
        VGGSound_AudioSetCaps_caption.filtered.csv \
        output.jsonl.gz

    # Test mode (process only 100 files from test split)
    python generate_manifest.py \
        /path/to/save/audio \
        VGGSound_AudioSetCaps_caption.filtered.csv \
        output.jsonl.gz \
        --split test \
        --limit 100

    # Process with custom number of workers
    python generate_manifest.py \
        /path/to/save/audio \
        VGGSound_AudioSetCaps_caption.filtered.csv \
        output.jsonl.gz \
        --workers 16
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch
import torchaudio
from datasets import load_dataset

from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment


def load_captions(csv_path: str) -> Dict[str, str]:
    """Load captions from CSV into a dictionary."""
    print(f"Loading captions from {csv_path}...")
    captions = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            captions[row['id']] = row['caption']
    print(f"Loaded {len(captions)} captions")
    return captions


def process_and_save_audio(args: Tuple) -> Tuple[Optional[Recording], Optional[SupervisionSegment]]:
    """
    Download, save audio file, and create Recording and SupervisionSegment.

    Args:
        args: Tuple of (item, audio_base_path, split, captions)
            - item: HuggingFace dataset item
            - audio_base_path: Base path to save audio files
            - split: Dataset split (train/test)
            - captions: Dictionary of captions keyed by ID

    Returns:
        Tuple of (Recording, SupervisionSegment) or (None, None) if no caption
    """
    item, audio_base_path, split, captions = args

    try:
        # Extract audio ID from path
        # Remove file extension
        uid = str(item["audio"]['path'].split(".")[0])

        # Remove trailing "-0" suffix if present (channel indicator in HF dataset)
        if uid.endswith("-0"):
            uid = uid[:-2]

        # Check if we have a caption for this audio
        if uid not in captions:
            return None, None

        # Prepare audio tensor
        wav = torch.from_numpy(item["audio"]["array"]).float().unsqueeze(0)  # [1, T]

        # Create split directory
        wav_root = audio_base_path / split
        wav_root.mkdir(parents=True, exist_ok=True)

        # Save audio file
        wav_path = wav_root / f"{uid}.wav"
        torchaudio.save(wav_path.as_posix(), wav, 16000)

        # Create Recording using lhotse
        recording = Recording.from_file(
            path=wav_path.as_posix(),
            recording_id=uid
        )

        # Create SupervisionSegment with caption
        supervision = SupervisionSegment(
            id=uid,
            recording_id=uid,
            start=0.0,
            duration=recording.duration,
            channel=0,
            custom={'caption': [captions[uid]]}
        )

        return recording, supervision

    except Exception as e:
        # Suppress individual errors to avoid cluttering output
        return None, None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate Lhotse manifest for VggSound dataset with captions (downloads from HuggingFace)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process both train and test splits
  python generate_manifest.py /path/to/save/audio captions.csv output.jsonl.gz

  # Test mode: process only 100 files from test split
  python generate_manifest.py /path/to/save/audio captions.csv output.jsonl.gz --split test --limit 100

  # Process with 16 workers
  python generate_manifest.py /path/to/save/audio captions.csv output.jsonl.gz --workers 16

  # Combine options
  python generate_manifest.py /path/to/save/audio captions.csv output.jsonl.gz --split train --limit 1000 --workers 8
        """
    )
    parser.add_argument('audio_root', type=str,
                        help='Path to save downloaded audio files')
    parser.add_argument('caption_csv', type=str,
                        help='Path to the caption CSV file')
    parser.add_argument('output_file', type=str,
                        help='Path to the output JSONL.gz file')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'test'],
                        help='Process only specific split (default: process both train and test)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to process per split (e.g., --limit 100 for testing)')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers (default: CPU count, max 32)')
    parser.add_argument('--show-samples', type=int, default=1,
                        help='Number of sample cuts to display at the end (default: 1)')

    args = parser.parse_args()

    # Determine which splits to process
    if args.split:
        splits = [args.split]
        print("=" * 60)
        print(f"Processing only {args.split} split")
        if args.limit:
            print(f"LIMIT MODE: Processing only {args.limit} files per split")
        print("=" * 60)
    else:
        splits = ['train', 'test']
        print("=" * 60)
        print(f"Processing both train and test splits")
        if args.limit:
            print(f"LIMIT MODE: Processing only {args.limit} files per split")
        print("=" * 60)

    # Load captions
    captions = load_captions(args.caption_csv)

    # Convert audio_root to Path
    audio_base_path = Path(args.audio_root)

    # Determine number of workers
    if args.workers:
        num_workers = min(args.workers, cpu_count())
    else:
        num_workers = min(cpu_count(), 32)  # Limit to 32 workers max

    print(f"Using {num_workers} workers for parallel processing")

    # Process all splits
    all_recordings = []
    all_supervisions = []
    total_processed = 0
    total_skipped = 0

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        # Load dataset from HuggingFace
        print(f"Loading {split} split from HuggingFace (txya900619/vggsound-16k)...")
        ds = load_dataset("txya900619/vggsound-16k", split=split)

        print(f"Downloaded {len(ds)} items from {split} split")

        # Apply limit if specified
        if args.limit and len(ds) > args.limit:
            print(f"Limiting to first {args.limit} items for testing")
            ds = ds.select(range(args.limit))

        # Prepare arguments for multiprocessing
        args_list = [(item, audio_base_path, split, captions) for item in ds]

        # Process items in parallel
        processed_count = 0
        skipped_count = 0

        with Pool(num_workers) as pool:
            for recording, supervision in tqdm(
                pool.imap_unordered(process_and_save_audio, args_list, chunksize=10),
                total=len(args_list),
                desc=f"Processing {split} split"
            ):
                if recording is not None and supervision is not None:
                    all_recordings.append(recording)
                    all_supervisions.append(supervision)
                    processed_count += 1
                else:
                    skipped_count += 1

        print(f"\n{split} split complete:")
        print(f"  Processed: {processed_count}")
        print(f"  Skipped (no caption): {skipped_count}")

        total_processed += processed_count
        total_skipped += skipped_count

    print(f"\n{'='*60}")
    print(f"Creating RecordingSet and SupervisionSet...")
    recording_set = RecordingSet.from_recordings(all_recordings)
    supervision_set = SupervisionSet.from_segments(all_supervisions)

    print(f"Creating CutSet...")
    cuts = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set
    )

    print(f"Writing output to {args.output_file}...")
    cuts.to_file(args.output_file)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (no caption): {total_skipped}")
    print(f"Audio files saved to: {audio_base_path}")
    print(f"Manifest written to: {args.output_file}")
    print(f"{'='*60}")

    # Print samples
    if args.show_samples > 0:
        print(f"\nShowing {args.show_samples} sample cut(s):")
        import json
        for i, cut in enumerate(cuts.subset(first=args.show_samples)):
            print(f"\n--- Sample {i+1} ---")
            print(json.dumps(cut.to_dict(), indent=2))


if __name__ == "__main__":
    main()
