#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# STEP 2 of 2 — run OCR on the upscaled fileGrp.
#
# Every step below descends from OCR-D-IMG-360. OCR-D-IMG is never touched.
#
# Usage:  ./02_run_ocr.sh            # all pages
#         ./02_run_ocr.sh PHYS_0006  # single page, for testing
# ---------------------------------------------------------------------------
set -euo pipefail

WORK="$PWD/ocrd-work"
IMG="ocrd/all:latest"
TESSMODEL="$PWD/tessdata/deu.traineddata"
PAGE="${1:-}"
PAGEARG=""
[ -n "$PAGE" ] && PAGEARG="-g $PAGE"

[ -f "$WORK/mets.xml" ] || { echo "No workspace — run 01_setup_workspace.sh first" >&2; exit 1; }

OCRD() {
  docker run --rm --platform linux/amd64 \
    -u "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp/mpl \
    -v "$WORK:/data" \
    -v "$TESSMODEL:/usr/local/share/tessdata/deu.traineddata:ro" \
    -w /data "$IMG" "$@"
}

cd "$WORK"

# --- models into the writable bind mount -----------------------------------
#OCRD ocrd resmgr download --location cwd ocrd-tesserocr-recognize deu.traineddata
#OCRD ocrd resmgr download --location cwd ocrd-tesserocr-recognize osd.traineddata

# --- the chain: note -I OCR-D-IMG-360 on the FIRST step only ---------------
# Everything after inherits that coordinate grid. Do not point any step back
# at OCR-D-IMG or coordinates will be off by exactly the 2.5x scale factor.
OCRD ocrd process --overwrite $PAGEARG \
  "olena-binarize           -I OCR-D-IMG-360   -O OCR-D-BIN      -P impl sauvola-ms-split -P k 0.34" \
  "cis-ocropy-denoise       -I OCR-D-BIN       -O OCR-D-DEN      -P level-of-operation page -P noise_maxsize 2.0" \
  "cis-ocropy-deskew        -I OCR-D-DEN       -O OCR-D-DSK      -P level-of-operation page -P maxskew 2.0" \
  "tesserocr-segment-region -I OCR-D-DSK       -O OCR-D-SEG-REG  -P find_tables false -P sparse_text false -P padding 4" \
  "segment-repair           -I OCR-D-SEG-REG   -O OCR-D-SEG-REP  -P plausibilize true" \
  "cis-ocropy-clip          -I OCR-D-SEG-REP   -O OCR-D-CLIP     -P level-of-operation region" \
  "cis-ocropy-segment       -I OCR-D-CLIP      -O OCR-D-SEG-LINE -P level-of-operation region -P spread 2.4" \
  "cis-ocropy-dewarp        -I OCR-D-SEG-LINE  -O OCR-D-DEW" \
  "tesserocr-recognize      -I OCR-D-DEW       -O OCR-D-OCR      -P model deu -P segmentation_level word -P textequiv_level word"

# --- export ----------------------------------------------------------------
OCRD ocrd-fileformat-transform --overwrite -I OCR-D-OCR -O OCR-D-ALTO -P from-to "page alto"
OCRD ocrd-fileformat-transform --overwrite -I OCR-D-OCR -O OCR-D-TXT  -P from-to "page text"

echo
echo "PAGE-XML: $WORK/OCR-D-OCR"
echo "ALTO:     $WORK/OCR-D-ALTO"
echo "text:     $WORK/OCR-D-TXT"
