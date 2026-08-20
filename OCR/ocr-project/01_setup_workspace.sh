#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# STEP 1 of 2 — build an OCR-D workspace with TWO image fileGrps:
#
#   OCR-D-IMG      original 144 dpi scans, untouched, for provenance
#   OCR-D-IMG-360  2.5x Lanczos upscale, 360 dpi — the grid the workflow uses
#
# Usage:  ./01_setup_workspace.sh ./originals
# Then:   ./02_run_ocr.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SRC="${1:?usage: $0 <dir-with-master-tifs>}"
WORK="$PWD/ocrd-work"
IMG="ocrd/all:latest"

OCRD() {
  docker run --rm --platform linux/amd64 \
    -u "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp/mpl \
    -v "$WORK:/data" -w /data "$IMG" "$@"
}

command -v magick >/dev/null 2>&1 && IM=magick || IM=convert

if [ -e "$WORK" ]; then
  echo "ERROR: $WORK already exists. Remove it first — re-running ocrd" >&2
  echo "       workspace add over an existing METS duplicates file IDs." >&2
  exit 1
fi

mkdir -p "$WORK/OCR-D-IMG" "$WORK/OCR-D-IMG-360"

# ---------------------------------------------------------------------------
# Copy + derive. Filenames get sanitised (spaces -> underscores) because they
# end up inside METS IDs, and spaces there cause trouble downstream.
# ---------------------------------------------------------------------------
n=0
declare -a PAGEIDS=()
declare -a NAMES=()

for f in "$SRC"/*.tif "$SRC"/*.TIF; do
  [ -e "$f" ] || continue
  n=$((n+1))
  pageid=$(printf "PHYS_%04d" "$n")
  name=$(basename "${f%.*}" | tr ' ' '_' | tr -cd '[:alnum:]_-')

  cp "$f" "$WORK/OCR-D-IMG/${name}.tif"

  "$IM" "$f" -colorspace Gray -filter Lanczos -resize 250% \
        -density 360 -units PixelsPerInch -compress LZW \
        "$WORK/OCR-D-IMG-360/${name}.tif"

  PAGEIDS+=("$pageid")
  NAMES+=("$name")
  echo "prepared $pageid  $name"
done

[ "$n" -gt 0 ] || { echo "ERROR: no .tif files found in $SRC" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Init METS and register both groups. The SAME -g pageid is used for both, so
# METS knows the two images are two representations of one physical page.
# ---------------------------------------------------------------------------
cd "$WORK"
OCRD ocrd workspace init
OCRD ocrd workspace set-id "SdK_StGallen_StFiden"

for i in "${!PAGEIDS[@]}"; do
  pageid="${PAGEIDS[$i]}"
  name="${NAMES[$i]}"

  OCRD ocrd workspace add \
    -G OCR-D-IMG     -g "$pageid" -m image/tiff \
    -i "OCR-D-IMG_${pageid}"     "OCR-D-IMG/${name}.tif"

  OCRD ocrd workspace add \
    -G OCR-D-IMG-360 -g "$pageid" -m image/tiff \
    -i "OCR-D-IMG-360_${pageid}" "OCR-D-IMG-360/${name}.tif"
done

echo
OCRD ocrd workspace list-group
echo
echo "Workspace ready at $WORK"
echo "Originals preserved in OCR-D-IMG; workflow will run on OCR-D-IMG-360."
