#!/usr/bin/env python3
"""
Generate Lhotse manifest for WavCaps dataset (BBC_Sound_Effects and SoundBible subsets).

This script processes WavCaps audio files and their corresponding captions from JSON files,
creating a Lhotse CutSet manifest in JSONL.gz format. It supports parallel processing
and can filter by specific subsets.

Usage:
    python generate_manifest.py <root_folder> <output_file> [options]

Example:
    python generate_manifest.py /path/to/WavCaps_root output.jsonl.gz --subsets BBC_Sound_Effects SoundBible --limit 100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from lhotse import CutSet, Recording, MonoCut, SupervisionSegment
from lhotse.audio import AudioSource

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_captions(json_file: Path) -> Dict[str, dict]:
    """
    Load captions from a WavCaps JSON file.

    Args:
        json_file: Path to the JSON file (e.g., bbc_final.json or sb_final.json)

    Returns:
        Dictionary mapping audio ID to caption entry
    """
    logger.info(f"Loading captions from {json_file}")

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract the data list
    caption_list = data.get('data', [])

    # Create a mapping from ID to entry
    caption_dict = {}
    for entry in caption_list:
        audio_id = entry.get('id')
        if audio_id:
            caption_dict[audio_id] = entry

    logger.info(f"Loaded {len(caption_dict)} captions from {json_file.name}")
    return caption_dict


def process_audio_file(
    audio_path: Path,
    caption_data: dict,
    subset_name: str
) -> Optional[MonoCut]:
    """
    Process a single audio file and create a MonoCut with caption.

    Args:
        audio_path: Path to the audio file
        caption_data: Caption entry from JSON
        subset_name: Name of the subset (BBC_Sound_Effects or SoundBible)

    Returns:
        MonoCut object or None if processing fails
    """
    try:
        # Create recording from audio file
        recording = Recording.from_file(audio_path)

        # Create ID with subset prefix
        # BBC_Sound_Effects -> BBCSoundEffects_{id}
        # SoundBible -> SoundBible_{id}
        if subset_name == 'BBC_Sound_Effects':
            prefixed_id = f"BBCSoundEffects_{recording.id}"
        elif subset_name == 'SoundBible':
            prefixed_id = f"SoundBible_{recording.id}"
        else:
            prefixed_id = recording.id

        # Extract caption and metadata
        caption = caption_data.get('caption', '')
        if not caption:
            logger.warning(f"No caption found for {audio_path.name}")
            return None

        # Create supervision with caption
        supervision = SupervisionSegment(
            id=prefixed_id,
            recording_id=prefixed_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            custom={
                'caption': [caption],
            }
        )

        # Create new Recording with prefixed ID
        recording = Recording(
            id=prefixed_id,
            sources=recording.sources,
            sampling_rate=recording.sampling_rate,
            num_samples=recording.num_samples,
            duration=recording.duration,
            channel_ids=recording.channel_ids
        )

        # Create MonoCut
        cut = MonoCut(
            id=prefixed_id,
            start=0,
            duration=recording.duration,
            channel=0,
            recording=recording,
            supervisions=[supervision]
        )

        return cut

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return None


def process_batch(
    audio_files: List[Path],
    caption_dict: Dict[str, dict],
    subset_name: str
) -> List[MonoCut]:
    """
    Process a batch of audio files (for multiprocessing).

    Args:
        audio_files: List of audio file paths
        caption_dict: Dictionary of captions
        subset_name: Name of the subset

    Returns:
        List of MonoCut objects
    """
    cuts = []
    for audio_path in audio_files:
        # Extract ID from filename (remove .flac extension)
        audio_id = audio_path.stem

        # Get caption data
        caption_data = caption_dict.get(audio_id)
        if not caption_data:
            logger.warning(f"No caption found for ID {audio_id}")
            continue

        # Process the file
        cut = process_audio_file(audio_path, caption_data, subset_name)
        if cut:
            cuts.append(cut)

    return cuts


def process_subset(
    root_folder: Path,
    subset_name: str,
    workers: int = 16,
    limit: Optional[int] = None
) -> CutSet:
    """
    Process a single subset (BBC_Sound_Effects or SoundBible).

    Args:
        root_folder: Root folder containing WavCaps data
        subset_name: Name of the subset to process
        workers: Number of parallel workers
        limit: Optional limit on number of files to process

    Returns:
        CutSet containing all processed cuts
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing subset: {subset_name}")
    logger.info(f"{'='*60}")

    # Define paths
    audio_dir = root_folder / "WavCaps" / "Audio" / subset_name

    # Map subset names to their JSON filenames
    json_filename_map = {
        'BBC_Sound_Effects': 'bbc_final.json',
        'SoundBible': 'sb_final.json'
    }
    json_file = root_folder / "WavCaps" / "json_files" / subset_name / json_filename_map[subset_name]

    # Validate paths
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"JSON file: {json_file}")

    # Load captions
    caption_dict = load_captions(json_file)

    # Get all audio files
    audio_files = sorted(audio_dir.glob("*.flac"))
    if limit:
        audio_files = audio_files[:limit]

    logger.info(f"Found {len(audio_files)} audio files to process")

    # Split files into batches for multiprocessing
    batch_size = max(1, len(audio_files) // workers)
    batches = [audio_files[i:i + batch_size] for i in range(0, len(audio_files), batch_size)]

    logger.info(f"Processing with {workers} workers...")

    # Process batches in parallel
    process_func = partial(process_batch, caption_dict=caption_dict, subset_name=subset_name)

    all_cuts = []
    with Pool(workers) as pool:
        for cuts in tqdm(pool.imap(process_func, batches), total=len(batches), desc=f"Processing {subset_name}"):
            all_cuts.extend(cuts)

    logger.info(f"Successfully processed {len(all_cuts)} cuts from {subset_name}")

    return CutSet.from_cuts(all_cuts)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Lhotse manifest for WavCaps dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process both BBC and SoundBible subsets
  python generate_manifest.py /path/to/root output.jsonl.gz --subsets BBC_Sound_Effects SoundBible

  # Process only BBC_Sound_Effects with 100 files limit
  python generate_manifest.py /path/to/root bbc.jsonl.gz --subsets BBC_Sound_Effects --limit 100

  # Process with custom worker count
  python generate_manifest.py /path/to/root output.jsonl.gz --workers 32
        """
    )

    parser.add_argument(
        'root_folder',
        type=Path,
        help='Root folder containing WavCaps subdirectory (with Audio and json_files folders)'
    )
    parser.add_argument(
        'output_file',
        type=Path,
        help='Output JSONL.gz file path for the manifest'
    )
    parser.add_argument(
        '--subsets',
        nargs='+',
        choices=['BBC_Sound_Effects', 'SoundBible'],
        default=['BBC_Sound_Effects', 'SoundBible'],
        help='Which subsets to process (default: both)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of files to process per subset (for testing)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of parallel workers (default: 16)'
    )
    parser.add_argument(
        '--show-samples',
        type=int,
        default=1,
        help='Number of sample cuts to display (default: 1)'
    )

    args = parser.parse_args()

    # Validate root folder
    if not args.root_folder.exists():
        raise FileNotFoundError(f"Root folder not found: {args.root_folder}")

    logger.info(f"Root folder: {args.root_folder}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Subsets: {', '.join(args.subsets)}")
    logger.info(f"Workers: {args.workers}")
    if args.limit:
        logger.info(f"Limit: {args.limit} files per subset")

    # Process each subset
    all_cutsets = []
    for subset_name in args.subsets:
        cutset = process_subset(
            args.root_folder,
            subset_name,
            workers=args.workers,
            limit=args.limit
        )
        all_cutsets.append(cutset)

    # Combine all cutsets
    if len(all_cutsets) > 1:
        logger.info(f"\nCombining {len(all_cutsets)} subsets...")
        combined_cutset = CutSet.from_cuts(
            cut for cutset in all_cutsets for cut in cutset
        )
    else:
        combined_cutset = all_cutsets[0]

    # Save manifest
    logger.info(f"\nSaving manifest to {args.output_file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_cutset.to_file(args.output_file)

    # Display statistics
    logger.info(f"\n{'='*60}")
    logger.info("Manifest Statistics")
    logger.info(f"{'='*60}")
    logger.info(f"Total cuts: {len(combined_cutset)}")

    # Calculate total duration
    total_duration = sum(cut.duration for cut in combined_cutset)
    logger.info(f"Total duration: {total_duration / 3600:.2f} hours")

    # Count by subset
    subset_counts = {}
    for cut in combined_cutset:
        subset = cut.supervisions[0].custom.get('subset', 'unknown')
        subset_counts[subset] = subset_counts.get(subset, 0) + 1

    for subset, count in sorted(subset_counts.items()):
        logger.info(f"  {subset}: {count} cuts")

    # Show sample cuts
    if args.show_samples > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"Sample Cuts (showing {args.show_samples})")
        logger.info(f"{'='*60}")

        for i, cut in enumerate(list(combined_cutset)[:args.show_samples]):
            supervision = cut.supervisions[0]
            logger.info(f"\nCut {i+1}:")
            logger.info(f"  ID: {cut.id}")
            logger.info(f"  Subset: {supervision.custom.get('subset')}")
            logger.info(f"  Duration: {cut.duration:.2f}s")
            logger.info(f"  Caption: {supervision.custom.get('caption', ['N/A'])[0]}")
            logger.info(f"  Recording: {cut.recording.sources[0].source}")

    logger.info(f"\n{'='*60}")
    logger.info("Done!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
