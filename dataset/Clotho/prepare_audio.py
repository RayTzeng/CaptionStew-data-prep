#!/usr/bin/env python3
"""
Extract the Clotho v2.1 development audio into the layout used by CaptionStew.

CaptionStew refers to Clotho clips by their file name with spaces replaced by
underscores, e.g. "Forest, light breeze, bird song.wav" becomes
"clotho/development/Forest,_light_breeze,_bird_song". This script extracts
`clotho_audio_development.7z` and moves every file to

    ${clotho_root}/development/<file name with spaces replaced by _>.wav

Usage:
    python prepare_audio.py /path/to/clotho_audio_development.7z "${audio_root}/clotho"
"""

import argparse
import shutil
import subprocess
from pathlib import Path

EXPECTED_FILES = 3839
# Two pairs of development files map to the same name once spaces are replaced. CaptionStew uses
# one file of each pair (identified by its captions); the other one is not referenced.
PREFERRED_ON_CLASH = {"Car_Keys.wav": "Car_Keys.wav", "cow_bells.wav": "cow bells.wav"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, help="Path to clotho_audio_development.7z")
    parser.add_argument("clotho_root", type=Path, help="Output folder (the 'clotho' source root)")
    parser.add_argument("--keep-archive", action="store_true", help="Do not delete the .7z after extraction")
    args = parser.parse_args()

    if shutil.which("7z") is None:
        raise SystemExit("7z not found. Install p7zip (e.g. `conda install -c conda-forge p7zip` or `apt install p7zip-full`).")

    tmp = args.clotho_root / "_extract"
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {args.archive} ...")
    subprocess.run(["7z", "x", "-y", f"-o{tmp}", str(args.archive)], check=True, stdout=subprocess.DEVNULL)

    out_dir = args.clotho_root / "development"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_target = {}
    for wav in sorted(tmp.rglob("*.wav")):
        by_target.setdefault(wav.name.replace(" ", "_"), []).append(wav)

    moved = skipped = 0
    for target, wavs in by_target.items():
        chosen = wavs[0]
        if len(wavs) > 1:
            preferred = PREFERRED_ON_CLASH.get(target)
            chosen = next((w for w in wavs if w.name == preferred), wavs[0])
            skipped += len(wavs) - 1
            print(f"Name clash for {target}: using '{chosen.name}', skipping {[w.name for w in wavs if w != chosen]}")
        shutil.move(str(chosen), out_dir / target)
        moved += 1
    shutil.rmtree(tmp)
    if not args.keep_archive:
        args.archive.unlink()

    print(f"Moved {moved} files to {out_dir} ({skipped} unreferenced duplicates skipped)")
    if moved + skipped != EXPECTED_FILES:
        print(f"WARNING: expected {EXPECTED_FILES} files in the Clotho v2.1 development split")


if __name__ == "__main__":
    main()
