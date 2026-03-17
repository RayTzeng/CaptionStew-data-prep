#!/usr/bin/env python3
"""
Generate Lhotse manifests for ParaSpeechCaps dataset.

This script:
1. Discovers all source audio files from local dataset roots
2. Derives unique utterance IDs from file paths
3. Loads ParaSpeechCaps splits from Hugging Face
4. Matches each dataset entry to its local audio file
5. Exports Lhotse manifests for train/dev/test

Usage:
    # Full processing
    python generate_manifest.py \
        --voxceleb1_path /path/to/voxceleb1 \
        --voxceleb2_path /path/to/voxceleb2 \
        --ears_path /path/to/EARS \
        --expresso_path /path/to/expresso \
        --emilia_path /path/to/emilia \
        --output_dir /path/to/output \
        --num_workers 16

    # Test mode (process only 100 entries per split)
    python generate_manifest.py \
        --voxceleb1_path /path/to/voxceleb1 \
        --voxceleb2_path /path/to/voxceleb2 \
        --ears_path /path/to/EARS \
        --expresso_path /path/to/expresso \
        --emilia_path /path/to/emilia \
        --output_dir /path/to/output \
        --limit 100
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from datasets import load_dataset
from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment


class AudioIndexer:
    """Index local audio files and derive utterance IDs."""

    def __init__(self, num_workers: int = 16, use_find: bool = True):
        self.num_workers = num_workers
        self.use_find = use_find and sys.platform.startswith('linux')  # Only use find on Linux
        self.audio_index: Dict[str, Path] = {}
        self.duplicate_ids: Dict[str, List[Path]] = defaultdict(list)
        self.stats: Dict[str, int] = defaultdict(int)

    def index_all_sources(
        self,
        voxceleb1_path: Optional[Path],
        voxceleb2_path: Optional[Path],
        ears_path: Optional[Path],
        expresso_path: Optional[Path],
        emilia_path: Optional[Path]
    ) -> None:
        """Index all audio files from all source datasets."""
        print("="*80)
        print("STEP 1: Indexing local audio files")
        print("="*80)

        sources = [
            ("voxceleb1", voxceleb1_path, [".wav"]),
            ("voxceleb2", voxceleb2_path, [".wav"]),
            ("EARS", ears_path, [".wav"]),
            ("expresso", expresso_path, [".wav"]),
            ("emilia", emilia_path, [".mp3"])
        ]

        for source_name, source_path, extensions in sources:
            if source_path and source_path.exists():
                print(f"\nIndexing {source_name} from {source_path}...")
                self._index_source(source_name, source_path, extensions)
                print(f"  Indexed {self.stats[f'{source_name}_files']} files")
            else:
                print(f"\nSkipping {source_name} (path not provided or doesn't exist)")

        print(f"\n{'='*80}")
        print(f"Total indexed files: {len(self.audio_index)}")
        print(f"Duplicate IDs detected: {len(self.duplicate_ids)}")
        if self.duplicate_ids:
            print(f"WARNING: {len(self.duplicate_ids)} duplicate IDs found!")
            for uid, paths in list(self.duplicate_ids.items())[:5]:
                print(f"  {uid}: {len(paths)} occurrences")
            if len(self.duplicate_ids) > 5:
                print(f"  ... and {len(self.duplicate_ids) - 5} more")
        print("="*80)

    def _index_source(self, source_name: str, source_path: Path, extensions: List[str]) -> None:
        """Index audio files from a single source dataset using fast file discovery."""
        # Choose fastest method based on platform
        if self.use_find:
            audio_files = self._fast_find_files(source_path, extensions)
        else:
            audio_files = self._walk_files(source_path, extensions)

        print(f"  Found {len(audio_files)} audio files, processing...")

        # Process files with progress bar
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for audio_path in audio_files:
                future = executor.submit(self._derive_id_from_path, source_name, audio_path, source_path)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {source_name}"):
                uid, audio_path = future.result()
                if uid:
                    if uid in self.audio_index:
                        # Duplicate ID detected
                        self.duplicate_ids[uid].append(audio_path)
                        if len(self.duplicate_ids[uid]) == 1:
                            # First duplicate - add the original path too
                            self.duplicate_ids[uid].insert(0, self.audio_index[uid])
                    else:
                        self.audio_index[uid] = audio_path
                        self.stats[f'{source_name}_files'] += 1

    def _fast_find_files(self, source_path: Path, extensions: List[str]) -> List[Path]:
        """Use system 'find' command for fastest file discovery (Linux only)."""
        print(f"  Scanning directory tree with 'find' command...")
        audio_files = []

        # Build find command with multiple -name conditions
        find_cmd = ['find', str(source_path), '-type', 'f', '(']
        for i, ext in enumerate(extensions):
            if i > 0:
                find_cmd.append('-o')
            find_cmd.extend(['-name', f'*{ext}'])
        find_cmd.append(')')

        try:
            result = subprocess.run(
                find_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            audio_files = [Path(line) for line in result.stdout.strip().split('\n') if line]
        except subprocess.CalledProcessError as e:
            print(f"  Warning: 'find' command failed, falling back to os.walk")
            audio_files = self._walk_files(source_path, extensions)

        return audio_files

    def _walk_files(self, source_path: Path, extensions: List[str]) -> List[Path]:
        """Use os.walk for file discovery (faster than rglob) with progress tracking."""
        print(f"  Scanning directory tree with os.walk...")
        audio_files = []
        extensions_set = set(extensions)

        # Count directories first for progress bar
        dir_count = sum(1 for _ in os.walk(source_path))

        # Now walk with progress
        with tqdm(total=dir_count, desc=f"  Scanning dirs", unit="dir") as pbar:
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    if any(file.endswith(ext) for ext in extensions_set):
                        audio_files.append(Path(root) / file)
                pbar.update(1)
                # Update description with current file count
                if len(audio_files) % 1000 == 0 and len(audio_files) > 0:
                    pbar.set_postfix(files=len(audio_files))

        return audio_files

    def _derive_id_from_path(
        self,
        source_name: str,
        audio_path: Path,
        root_path: Path
    ) -> Tuple[Optional[str], Path]:
        """Derive utterance ID from local audio path based on source."""
        try:
            if source_name in ["voxceleb1", "voxceleb2"]:
                return self._derive_voxceleb_id(audio_path, root_path), audio_path
            elif source_name == "EARS":
                return self._derive_ears_id(audio_path, root_path), audio_path
            elif source_name == "expresso":
                return self._derive_expresso_id(audio_path, root_path), audio_path
            elif source_name == "emilia":
                return self._derive_emilia_id(audio_path), audio_path
        except Exception as e:
            print(f"\nError deriving ID from {audio_path}: {e}")
            return None, audio_path

        return None, audio_path

    @staticmethod
    def _derive_voxceleb_id(audio_path: Path, root_path: Path) -> str:
        """
        Derive VoxCeleb ID from path.
        Pattern: {voxceleb_version}/{split}/{audio_type}/{speaker_id}/{conversation_id}/{audio_id}_voicefixer.wav
        Output: {voxceleb_version}_{split}_{speaker_id}_{conversation_id}_{audio_id}
        """
        relative = audio_path.relative_to(root_path)
        parts = relative.parts

        # Strip _voicefixer suffix from filename
        filename_stem = audio_path.stem
        if filename_stem.endswith("_voicefixer"):
            audio_id = filename_stem[:-11]  # Remove "_voicefixer"
        else:
            audio_id = filename_stem

        # Determine voxceleb version from root path
        voxceleb_version = root_path.name.lower()
        if "voxceleb1" in voxceleb_version:
            version = "voxceleb1"
        elif "voxceleb2" in voxceleb_version:
            version = "voxceleb2"
        else:
            version = root_path.name

        # Expected structure: split/audio_type/speaker_id/conversation_id/audio_file
        if len(parts) >= 5:
            split = parts[-5]
            speaker_id = parts[-3]
            conversation_id = parts[-2]
            return f"{version}_{split}_{speaker_id}_{conversation_id}_{audio_id}"
        else:
            raise ValueError(f"Unexpected VoxCeleb path structure: {relative}")

    @staticmethod
    def _derive_ears_id(audio_path: Path, root_path: Path) -> str:
        """
        Derive EARS ID from path.
        Pattern: {speaker_id}/{emotion_type_name_id}.wav
        Output: EARS_{speaker_id}_{emotion_type_name_id}
        """
        relative = audio_path.relative_to(root_path)
        parts = relative.parts

        if len(parts) >= 2:
            speaker_id = parts[-2]
            emotion_id = audio_path.stem
            return f"EARS_{speaker_id}_{emotion_id}"
        else:
            raise ValueError(f"Unexpected EARS path structure: {relative}")

    @staticmethod
    def _derive_expresso_id(audio_path: Path, root_path: Path) -> str:
        """
        Derive expresso ID from path.
        Pattern: {recording_type}/.../recording_name.wav
        Output: expresso_{recording_type}_{recording_name}
        """
        relative = audio_path.relative_to(root_path)
        parts = relative.parts

        if len(parts) >= 1:
            recording_type = parts[-4]
            recording_name = audio_path.stem
            return f"expresso_{recording_type}_{recording_name}"
        else:
            raise ValueError(f"Unexpected expresso path structure: {relative}")

    @staticmethod
    def _derive_emilia_id(audio_path: Path) -> str:
        """
        Derive Emilia ID from path.
        Pattern: .../EN/EN_B00030/EN_B00030_S01984/mp3/EN_B00030_S01984_W000037.mp3
        Output: Emilia_{path.stem}
        """
        return f"Emilia_{audio_path.stem}"


class HFDatasetMatcher:
    """Load HuggingFace dataset and match entries to local audio files."""

    def __init__(self, audio_index: Dict[str, Path], hf_repo: str = "ajd12342/paraspeechcaps", limit_per_split: Optional[int] = None):
        self.audio_index = audio_index
        self.hf_repo = hf_repo
        self.limit_per_split = limit_per_split
        self.stats: Dict[str, int] = defaultdict(int)
        self.unmatched_entries: List[Dict[str, Any]] = []

    def load_and_match_splits(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load HF splits and match to local files."""
        print("\n" + "="*80)
        print("STEP 2: Loading HuggingFace dataset and matching to local files")
        if self.limit_per_split:
            print(f"TEST MODE: Processing only {self.limit_per_split} entries per split")
        print("="*80)

        splits_data = {}

        # Load each split
        for split_name in ["train_base", "train_scaled", "dev", "holdout"]:
            print(f"\nLoading {split_name} split from HuggingFace...")
            dataset = load_dataset(self.hf_repo, split=split_name)
            print(f"  Loaded {len(dataset)} entries")

            # Apply limit if specified
            if self.limit_per_split and len(dataset) > self.limit_per_split:
                print(f"  Limiting to first {self.limit_per_split} entries for testing")
                dataset = dataset.select(range(self.limit_per_split))

            # Match entries to local files
            matched_entries = self._match_split(dataset, split_name)
            splits_data[split_name] = matched_entries

            print(f"  Matched: {self.stats[f'{split_name}_matched']}")
            print(f"  Unmatched: {self.stats[f'{split_name}_unmatched']}")

        print(f"\n{'='*80}")
        print("Matching summary:")
        total_matched = sum(self.stats[f'{s}_matched'] for s in ["train_base", "train_scaled", "dev", "holdout"])
        total_unmatched = sum(self.stats[f'{s}_unmatched'] for s in ["train_base", "train_scaled", "dev", "holdout"])
        print(f"  Total matched: {total_matched}")
        print(f"  Total unmatched: {total_unmatched}")
        if self.unmatched_entries:
            print(f"\nFirst 5 unmatched entries:")
            for entry in self.unmatched_entries[:5]:
                print(f"  {entry['source']}: {entry['relative_audio_path']}")
        print("="*80)

        return splits_data

    def _match_split(self, dataset: Any, split_name: str) -> List[Dict[str, Any]]:
        """Match entries in a split to local audio files."""
        matched_entries = []

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for entry in dataset:
                future = executor.submit(self._match_entry, entry)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Matching {split_name}"):
                result = future.result()
                if result:
                    matched_entries.append(result)
                    self.stats[f'{split_name}_matched'] += 1
                else:
                    self.stats[f'{split_name}_unmatched'] += 1

        return matched_entries

    def _match_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Match a single HF entry to local audio file."""
        try:
            # Derive utterance ID from HF entry
            uid = self._derive_id_from_hf_entry(entry)

            # Look up in audio index
            if uid in self.audio_index:
                return {
                    'uid': uid,
                    'audio_path': self.audio_index[uid],
                    'text_description': entry.get('text_description', []),
                    'transcription': entry.get('transcription', ''),
                    'source': entry.get('source', ''),
                    'metadata': entry
                }
            else:
                # Track unmatched entry
                self.unmatched_entries.append({
                    'source': entry.get('source', 'unknown'),
                    'relative_audio_path': entry.get('relative_audio_path', 'unknown'),
                    'derived_uid': uid
                })
                return None
        except Exception as e:
            self.unmatched_entries.append({
                'source': entry.get('source', 'unknown'),
                'relative_audio_path': entry.get('relative_audio_path', 'unknown'),
                'error': str(e)
            })
            return None

    def _derive_id_from_hf_entry(self, entry: Dict[str, Any]) -> str:
        """Derive utterance ID from HF entry based on source."""
        source = entry.get('source', '').lower()
        relative_path = entry.get('relative_audio_path', '')

        # Convert to Path for easier manipulation
        path = Path(relative_path)

        if source == "voxceleb":
            return self._derive_id_from_voxceleb_path(path)
        elif source in ["ears", "ear"]:
            return self._derive_id_from_ears_path(path)
        elif source == "expresso":
            return self._derive_id_from_expresso_path(path)
        elif source == "emilia":
            return self._derive_id_from_emilia_path(path)
        else:
            raise ValueError(f"Unknown source: {source}")

    @staticmethod
    def _derive_id_from_voxceleb_path(path: Path) -> str:
        """Derive ID from VoxCeleb relative path in HF dataset."""
        parts = path.parts

        # Strip _voicefixer suffix
        filename_stem = path.stem
        if filename_stem.endswith("_voicefixer"):
            audio_id = filename_stem[:-11]
        else:
            audio_id = filename_stem

        # Determine version from path
        if parts[0].startswith("voxceleb1"):
            version = "voxceleb1"
            # Structure: voxceleb1/split/audio_type/speaker_id/conversation_id/audio_file
            split = parts[-5]
            speaker_id = parts[-3]
            conversation_id = parts[-2]
        elif parts[0].startswith("voxceleb2"):
            version = "voxceleb2"
            # Structure: voxceleb2/split/audio_type/speaker_id/conversation_id/audio_file
            split = parts[-5]
            speaker_id = parts[-3]
            conversation_id = parts[-2]
        else:
            raise ValueError(f"Cannot determine VoxCeleb version from path: {path}")

        return f"{version}_{split}_{speaker_id}_{conversation_id}_{audio_id}"

    @staticmethod
    def _derive_id_from_ears_path(path: Path) -> str:
        """Derive ID from EARS relative path in HF dataset."""
        parts = path.parts
        # Expected: EARS/speaker_id/emotion_file.wav
        if len(parts) >= 2:
            speaker_id = parts[-2]
            emotion_id = path.stem
            return f"EARS_{speaker_id}_{emotion_id}"
        else:
            raise ValueError(f"Unexpected EARS path structure: {path}")

    @staticmethod
    def _derive_id_from_expresso_path(path: Path) -> str:
        """Derive ID from expresso relative path in HF dataset."""
        parts = path.parts
        # Expected: expresso/recording_type/.../recording_name.wav
        if len(parts) >= 2:
            recording_type = parts[-4]
            recording_name = path.stem
            return f"expresso_{recording_type}_{recording_name}"
        else:
            raise ValueError(f"Unexpected expresso path structure: {path}")

    @staticmethod
    def _derive_id_from_emilia_path(path: Path) -> str:
        """Derive ID from Emilia relative path in HF dataset."""
        return f"Emilia_{path.stem}"


class LhotseManifestGenerator:
    """Generate Lhotse manifests from matched entries."""

    def __init__(self, num_workers: int = 16):
        self.num_workers = num_workers
        self.stats: Dict[str, int] = defaultdict(int)

    def generate_manifests(
        self,
        splits_data: Dict[str, List[Dict[str, Any]]],
        output_dir: Path
    ) -> None:
        """Generate Lhotse manifests for all splits."""
        print("\n" + "="*80)
        print("STEP 3: Generating Lhotse manifests")
        print("="*80)

        # Combine train_base and train_scaled into train
        train_entries = splits_data.get("train_base", []) + splits_data.get("train_scaled", [])
        dev_entries = splits_data.get("dev", [])
        test_entries = splits_data.get("holdout", [])

        manifest_configs = [
            ("train", train_entries, "ParaSpeechCaps_train.jsonl.gz"),
            ("dev", dev_entries, "ParaSpeechCaps_dev.jsonl.gz"),
            ("test", test_entries, "ParaSpeechCaps_test.jsonl.gz")
        ]

        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, entries, filename in manifest_configs:
            if entries:
                print(f"\nGenerating {split_name} manifest ({len(entries)} entries)...")
                output_path = output_dir / filename
                self._generate_split_manifest(entries, output_path, split_name)
                print(f"  Written to {output_path}")
                print(f"  Successfully created: {self.stats[f'{split_name}_success']} cuts")
                print(f"  Failed: {self.stats[f'{split_name}_failed']} cuts")
            else:
                print(f"\nSkipping {split_name} (no entries)")

        print(f"\n{'='*80}")
        print("Manifest generation complete!")
        print("="*80)

    def _generate_split_manifest(
        self,
        entries: List[Dict[str, Any]],
        output_path: Path,
        split_name: str
    ) -> None:
        """Generate manifest for a single split."""
        recordings = []
        supervisions = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for entry in entries:
                future = executor.submit(self._create_lhotse_objects, entry)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Creating {split_name}"):
                result = future.result()
                if result:
                    recording, supervision = result
                    recordings.append(recording)
                    supervisions.append(supervision)
                    self.stats[f'{split_name}_success'] += 1
                else:
                    self.stats[f'{split_name}_failed'] += 1

        # Create CutSet and write to file
        if recordings and supervisions:
            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)
            cuts = CutSet.from_manifests(
                recordings=recording_set,
                supervisions=supervision_set
            )
            cuts.to_file(str(output_path))

    def _create_lhotse_objects(
        self,
        entry: Dict[str, Any]
    ) -> Optional[Tuple[Recording, SupervisionSegment]]:
        """Create Lhotse Recording and SupervisionSegment from matched entry."""
        try:
            uid = entry['uid']
            audio_path = entry['audio_path']

            # Create Recording
            recording = Recording.from_file(
                path=str(audio_path),
                recording_id=uid
            )

            # Get text descriptions (may be a list)
            text_descriptions = entry.get('text_description', [])
            if not isinstance(text_descriptions, list):
                text_descriptions = [text_descriptions] if text_descriptions else []

            # Create SupervisionSegment
            supervision = SupervisionSegment(
                id=uid,
                recording_id=uid,
                start=0.0,
                duration=recording.duration,
                channel=0,
                custom={
                    'caption': text_descriptions,
                }
            )

            return recording, supervision

        except Exception as e:
            print(f"\nError creating Lhotse objects for {entry.get('uid', 'unknown')}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate Lhotse manifests for ParaSpeechCaps dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_manifest.py \\
    --voxceleb1_path /path/to/voxceleb1 \\
    --voxceleb2_path /path/to/voxceleb2 \\
    --ears_path /path/to/EARS \\
    --expresso_path /path/to/expresso \\
    --emilia_path /path/to/emilia \\
    --output_dir /path/to/output \\
    --num_workers 16
        """
    )

    parser.add_argument('--voxceleb1_path', type=str, required=True,
                        help='Path to VoxCeleb1 dataset root')
    parser.add_argument('--voxceleb2_path', type=str, required=True,
                        help='Path to VoxCeleb2 dataset root')
    parser.add_argument('--ears_path', type=str, required=True,
                        help='Path to EARS dataset root')
    parser.add_argument('--expresso_path', type=str, required=True,
                        help='Path to expresso dataset root')
    parser.add_argument('--emilia_path', type=str, required=True,
                        help='Path to Emilia dataset root')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to write output manifests')
    parser.add_argument('--num_workers', type=int, default=min(32, os.cpu_count() or 16),
                        help='Number of parallel workers (default: min(32, CPU count))')
    parser.add_argument('--hf_repo', type=str, default='ajd12342/paraspeechcaps',
                        help='HuggingFace repository for ParaSpeechCaps (default: ajd12342/paraspeechcaps)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of entries to process per split for testing (e.g., --limit 100)')
    parser.add_argument('--no-find', action='store_true',
                        help='Disable fast "find" command and use Python os.walk instead')

    args = parser.parse_args()

    # Convert paths to Path objects
    voxceleb1_path = Path(args.voxceleb1_path) if args.voxceleb1_path else None
    voxceleb2_path = Path(args.voxceleb2_path) if args.voxceleb2_path else None
    ears_path = Path(args.ears_path) if args.ears_path else None
    expresso_path = Path(args.expresso_path) if args.expresso_path else None
    emilia_path = Path(args.emilia_path) if args.emilia_path else None
    output_dir = Path(args.output_dir)

    print("="*80)
    print("ParaSpeechCaps Lhotse Manifest Generator")
    print("="*80)
    print(f"Configuration:")
    print(f"  VoxCeleb1 path: {voxceleb1_path}")
    print(f"  VoxCeleb2 path: {voxceleb2_path}")
    print(f"  EARS path: {ears_path}")
    print(f"  expresso path: {expresso_path}")
    print(f"  Emilia path: {emilia_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of workers: {args.num_workers}")
    print(f"  HuggingFace repo: {args.hf_repo}")
    print("="*80)

    # Step 1: Index local audio files
    indexer = AudioIndexer(num_workers=args.num_workers, use_find=not args.no_find)
    indexer.index_all_sources(
        voxceleb1_path=voxceleb1_path,
        voxceleb2_path=voxceleb2_path,
        ears_path=ears_path,
        expresso_path=expresso_path,
        emilia_path=emilia_path
    )

    # Step 2: Load HF dataset and match to local files
    matcher = HFDatasetMatcher(
        audio_index=indexer.audio_index,
        hf_repo=args.hf_repo,
        limit_per_split=args.limit
    )
    splits_data = matcher.load_and_match_splits()

    # Step 3: Generate Lhotse manifests
    generator = LhotseManifestGenerator(num_workers=args.num_workers)
    generator.generate_manifests(splits_data, output_dir)

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\nIndexed files per source:")
    for source in ["voxceleb1", "voxceleb2", "EARS", "expresso", "emilia"]:
        count = indexer.stats.get(f'{source}_files', 0)
        print(f"  {source}: {count}")
    print(f"  Total: {len(indexer.audio_index)}")

    print("\nHuggingFace matching:")
    for split in ["train_base", "train_scaled", "dev", "holdout"]:
        matched = matcher.stats.get(f'{split}_matched', 0)
        unmatched = matcher.stats.get(f'{split}_unmatched', 0)
        print(f"  {split}: {matched} matched, {unmatched} unmatched")

    print("\nGenerated manifests:")
    for split in ["train", "dev", "test"]:
        success = generator.stats.get(f'{split}_success', 0)
        failed = generator.stats.get(f'{split}_failed', 0)
        print(f"  {split}: {success} cuts created, {failed} failed")

    print("\nOutput files:")
    for filename in ["ParaSpeechCaps_train.jsonl.gz", "ParaSpeechCaps_dev.jsonl.gz", "ParaSpeechCaps_test.jsonl.gz"]:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (not created)")

    print("="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
