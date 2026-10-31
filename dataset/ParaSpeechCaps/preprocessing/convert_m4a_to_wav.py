#!/usr/bin/env python3
"""
Convert all .m4a files to .wav format using multithreading.

With --manifest, only the VoxCeleb2 utterances referenced by a CaptionStew
manifest (e.g. dataset_splits/ParaSpeechCaps.jsonl.gz) are converted.

Usage:
    python convert_m4a_to_wav.py <search_directory> [--workers N]
    python convert_m4a_to_wav.py <voxceleb2_root> --manifest /path/to/captions/dataset_splits/ParaSpeechCaps.jsonl.gz
"""

import argparse
import gzip
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


def convert_m4a_to_wav(m4a_file: Path) -> tuple[Path, bool, str]:
    """
    Convert a single M4A file to WAV format.

    Args:
        m4a_file: Path to the M4A file

    Returns:
        Tuple of (m4a_file, success, error_message)
    """
    wav_file = m4a_file.with_suffix('.wav')

    try:
        result = subprocess.run(
            [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-nostdin',
                '-y',
                '-i', str(m4a_file),
                str(wav_file)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return (m4a_file, True, "")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        return (m4a_file, False, error_msg)
    except Exception as e:
        return (m4a_file, False, str(e))


def find_m4a_files(search_dir: Path) -> List[Path]:
    """
    Find all .m4a files in the directory and its subdirectories.

    Args:
        search_dir: Directory to search

    Returns:
        List of Path objects pointing to M4A files
    """
    return list(search_dir.rglob('*.m4a'))


def find_referenced_m4a_files(voxceleb2_root: Path, manifest: Path) -> List[Path]:
    """
    Find the M4A files referenced by "voxceleb2/..." keys in a CaptionStew manifest.

    Args:
        voxceleb2_root: Root directory of VoxCeleb2 (containing dev/aac/...)
        manifest: CaptionStew manifest (.jsonl.gz)

    Returns:
        List of Path objects pointing to existing M4A files
    """
    files = []
    with gzip.open(manifest, 'rt', encoding='utf-8') as f:
        for line in f:
            if '"voxceleb2/' not in line:
                continue
            key = json.loads(line)['recording']['sources'][0]['source']
            m4a_file = voxceleb2_root / (key[len('voxceleb2/'):] + '.m4a')
            if m4a_file.exists():
                files.append(m4a_file)
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Convert M4A files to WAV format using multithreading'
    )
    parser.add_argument(
        'search_dir',
        type=Path,
        help='Directory to search for M4A files'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of worker threads (default: 16)'
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=None,
        help='CaptionStew manifest; only convert the VoxCeleb2 utterances it references'
    )

    args = parser.parse_args()

    # Validate search directory
    if not args.search_dir.exists():
        print(f"Error: Directory '{args.search_dir}' does not exist")
        return 1

    if not args.search_dir.is_dir():
        print(f"Error: '{args.search_dir}' is not a directory")
        return 1

    # Find all M4A files
    if args.manifest:
        print(f"Collecting M4A files referenced by {args.manifest}...")
        m4a_files = find_referenced_m4a_files(args.search_dir, args.manifest)
    else:
        print(f"Searching for M4A files in {args.search_dir}...")
        m4a_files = find_m4a_files(args.search_dir)

    if not m4a_files:
        print("No M4A files found")
        return 0

    print(f"Found {len(m4a_files)} M4A files")
    print(f"Converting with {args.workers} workers...")

    # Convert files using ThreadPoolExecutor
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(convert_m4a_to_wav, m4a_file): m4a_file
            for m4a_file in m4a_files
        }

        # Process results as they complete
        for future in as_completed(futures):
            m4a_file, success, error_msg = future.result()

            if success:
                success_count += 1
                if success_count % 100 == 0:
                    print(f"Progress: {success_count}/{len(m4a_files)} completed")
            else:
                error_count += 1
                print(f"Error converting {m4a_file}: {error_msg}")

    # Print summary
    print(f"\nConversion complete!")
    print(f"Successfully converted: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {len(m4a_files)}")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
