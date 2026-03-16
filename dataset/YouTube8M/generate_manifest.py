#!/usr/bin/env python3
"""
Generate Lhotse manifest for YouTube8M dataset using lhotse package.
Processes ~4M audio files from batch directories and creates JSONL manifest.

Usage:
    # Full processing (4M files)
    python generate_manifest.py /path/to/audio /path/to/captions.csv /path/to/output.jsonl.gz

    # Test mode (100 files)
    python generate_manifest.py /path/to/audio /path/to/captions.csv /path/to/output.jsonl.gz --limit 100

    # Custom limit (1000 files)
    python generate_manifest.py /path/to/audio /path/to/captions.csv /path/to/output.jsonl.gz --limit 1000

    # Custom number of workers
    python generate_manifest.py /path/to/audio /path/to/captions.csv /path/to/output.jsonl.gz --workers 16
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.audio import Recording, AudioSource
from lhotse.supervision import SupervisionSegment
from lhotse.cut import MonoCut

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

def collect_audio_files(audio_root: str, max_files: Optional[int] = None) -> list:
    """Collect all audio file paths from the corpus."""
    limit_msg = f" (limit: {max_files})" if max_files else ""
    print(f"Collecting audio files from {audio_root}{limit_msg}...")
    audio_files = []

    corpus_path = Path(audio_root)
    for batch_dir in sorted(corpus_path.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith('batch'):
            continue
        if max_files and len(audio_files) >= max_files:
            break

        for category_dir in batch_dir.iterdir():
            if not category_dir.is_dir():
                continue
            if max_files and len(audio_files) >= max_files:
                break

            for audio_file in category_dir.glob('*.mp3'):
                audio_files.append(str(audio_file))
                if max_files and len(audio_files) >= max_files:
                    break

    print(f"Found {len(audio_files)} audio files")
    return audio_files

def process_audio_file(args):
    """Process a single audio file and return Recording and SupervisionSegment."""
    audio_path, captions = args

    # Extract ID from filename (remove .mp3 extension)
    audio_id = Path(audio_path).stem

    # Check if we have a caption for this audio
    if audio_id not in captions:
        return None, None

    try:
        # Create Recording using lhotse
        recording = Recording.from_file(
            path=audio_path,
            recording_id=audio_id
        )

        # Create SupervisionSegment with caption
        supervision = SupervisionSegment(
            id=audio_id,
            recording_id=audio_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            custom={'caption': [captions[audio_id]]}
        )

        return recording, supervision

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate Lhotse manifest for YouTube8M dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full processing
  python generate_manifest.py /path/to/audio /path/to/captions.csv /path/to/output.jsonl.gz

  # Test mode (100 files)
  python generate_manifest.py /path/to/audio /path/to/captions.csv /path/to/output.jsonl.gz --limit 100

  # Custom limit (1000 files)
  python generate_manifest.py /path/to/audio /path/to/captions.csv /path/to/output.jsonl.gz --limit 1000
        """
    )
    parser.add_argument('audio_root', type=str,
                        help='Path to the audio root directory containing batch folders')
    parser.add_argument('caption_csv', type=str,
                        help='Path to the caption CSV file')
    parser.add_argument('output_file', type=str,
                        help='Path to the output JSONL.gz file')
    parser.add_argument('--limit', type=int, metavar='N', default=None,
                        help='Limit number of files to process (e.g., --limit 100 for testing)')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers (default: CPU count, max 32)')
    parser.add_argument('--show-samples', type=int, default=1,
                        help='Number of sample cuts to display at the end (default: 1)')

    args = parser.parse_args()

    # Determine file limit
    max_files = None
    if args.limit:
        max_files = args.limit
        print("=" * 60)
        print(f"LIMIT MODE: Processing only {max_files} files")
        print("=" * 60)

    # Load captions
    captions = load_captions(args.caption_csv)

    # Collect all audio files
    audio_files = collect_audio_files(args.audio_root, max_files=max_files)

    # Prepare arguments for multiprocessing
    args_list = [(audio_path, captions) for audio_path in audio_files]

    # Process files in parallel
    if args.workers:
        num_workers = min(args.workers, cpu_count())
    else:
        num_workers = min(cpu_count(), 32)  # Limit to 32 workers max
    print(f"Processing files with {num_workers} workers...")

    recordings = []
    supervisions = []
    processed_count = 0
    error_count = 0

    with Pool(num_workers) as pool:
        for recording, supervision in tqdm(
            pool.imap_unordered(process_audio_file, args_list, chunksize=100),
            total=len(args_list),
            desc="Processing"
        ):
            if recording is not None and supervision is not None:
                recordings.append(recording)
                supervisions.append(supervision)
                processed_count += 1
            else:
                error_count += 1

            # Periodic status update
            if (processed_count + error_count) % 10000 == 0:
                print(f"\nProcessed: {processed_count}, Errors: {error_count}")

    print(f"\nCreating RecordingSet and SupervisionSet...")
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    print(f"Creating CutSet...")
    cuts = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set
    )

    print(f"Writing output to {args.output_file}...")
    cuts.to_file(args.output_file)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total errors: {error_count}")
    print(f"Output written to: {args.output_file}")
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
