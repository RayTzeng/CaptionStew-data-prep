#!/usr/bin/env bash
# Download WavCaps audio from Hugging Face (cvssp/WavCaps) into the CaptionStew layout:
#
#   ${wavcaps_root}/<Subset>/<id>.flac
#
# Usage:
#   ./prepare_audio.sh <wavcaps_root> [Subset ...]
#
# Subsets: SoundBible BBC_Sound_Effects AudioSet_SL FreeSound (default: all four).
# Each subset is downloaded as a split zip archive, extracted with 7z (no merging needed),
# and the archive is deleted afterwards. Finished subsets are marked with <Subset>/.done.
# Set WAVCAPS_DOWNLOAD_DIR to download the archives somewhere else (default: <wavcaps_root>/_download).

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <wavcaps_root> [SoundBible BBC_Sound_Effects AudioSet_SL FreeSound]" >&2
  exit 1
fi
command -v 7z >/dev/null || { echo "7z not found (install p7zip)" >&2; exit 1; }
command -v hf >/dev/null || { echo "hf not found (pip install -U huggingface_hub)" >&2; exit 1; }

root="$1"
shift
subsets=("$@")
if [ ${#subsets[@]} -eq 0 ]; then
  subsets=(SoundBible BBC_Sound_Effects AudioSet_SL FreeSound)
fi
download_dir="${WAVCAPS_DOWNLOAD_DIR:-${root}/_download}"

for subset in "${subsets[@]}"; do
  case "${subset}" in
    SoundBible|BBC_Sound_Effects|AudioSet_SL|FreeSound) ;;
    *) echo "Unknown subset: ${subset}" >&2; exit 1 ;;
  esac
  if [ -f "${root}/${subset}/.done" ]; then
    echo "[${subset}] already prepared, skipping"
    continue
  fi

  echo "[${subset}] downloading archive..."
  hf download cvssp/WavCaps --repo-type dataset --include "Zip_files/${subset}/*" --local-dir "${download_dir}"

  echo "[${subset}] extracting..."
  mkdir -p "${root}/${subset}"
  # `e` drops the archive's internal folders, leaving <id>.flac directly in the subset folder.
  7z e -y -bd -o"${root}/${subset}" "${download_dir}/Zip_files/${subset}/${subset}.zip" "*.flac" -r > /dev/null

  n=$(find "${root}/${subset}" -maxdepth 1 -name "*.flac" | wc -l)
  echo "[${subset}] ${n} files in ${root}/${subset}"
  rm -rf "${download_dir}/Zip_files/${subset}"
  touch "${root}/${subset}/.done"
done

if [ -z "${WAVCAPS_DOWNLOAD_DIR:-}" ] && [ -d "${download_dir}" ] && [ -z "$(find "${download_dir}" -type f -not -path "*/.cache/*")" ]; then
  rm -rf "${download_dir}"
fi
