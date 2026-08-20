# OCR pipeline — *Die Kunstdenkmäler der Schweiz*, St. Gallen volume

OCR workflow for scanned pages of *Die Kunstdenkmäler der Schweiz: Die Stadt
St. Gallen*, producing PAGE-XML, ALTO, and plain text from grayscale TIFFs.

Built on [OCR-D](https://ocr-d.de/) running in Docker.

---

## Source material

| Property | Value |
|---|---|
| Format | TIFF, LZW-compressed, 8-bit grayscale |
| Resolution | 144 dpi (~1021 × 1446 px per page) |
| Typography | 20th-century Antiqua (Bembo-like), German |
| Page features | Running header, body text, figure captions, footnotes, halftone photographic plates |
| Complications | Italics, small caps, superscript footnote markers, old-style figures, `ffi`/`ffn` ligatures, letterspaced headers |

---

## Requirements

- Docker (the `ocrd/all:latest` image)
- ImageMagick 7 (`magick`)
- Python 3 (for text extraction)
- `deu.traineddata` from [tessdata_best](https://github.com/tesseract-ocr/tessdata_best)

On Apple Silicon, `ocrd/all` runs under x86 emulation (`--platform linux/amd64`).
Enabling Rosetta in Docker Desktop is strongly recommended. TensorFlow-based
processors (`eynollah`, `calamari`) do **not** work under emulation — this
pipeline avoids them entirely.

---

## Directory layout

```
ocr-project/
├── originals/               Master TIFs, 16 pages. Read-only, never in METS.
├── tessdata/                deu.traineddata (+ eng, equ, osd), mounted into the container
│
├── 01_setup_workspace.sh    Stage 1 — build the workspace
├── 02_run_ocr.sh            Stage 2 — the OCR-D chain
├── extract.py               Stage 3 — PAGE-XML to plaintext/
├── 03_postprocess.py        Stage 4 — machinery only
├── 03_rules.py              Stage 4 — every editable rule
├── 04_assemble.py           Stage 5 — the pages as one text
├── crop.py                  Cut a line or word out of the page image to check a reading
│
├── plaintext/               16 files, raw extraction            (Stage 3)
├── corrected/               16 files, page-parallel             (Stage 4)
├── reflowed/                16 files, hyphenation closed        (Stage 4)
├── assembled/               full.txt, reading.txt, body.txt     (Stage 5)
├── review/                  report.md, diff.txt, crops/         (Stage 4)
│
├── PIPELINE.md              This file
├── PIPELINE.md.pdf          The same, exported
├── Read-me-ocr              Short orientation for the repository
├── ocrd.log, test.log       Console logs from Stage 2
├── __pycache__/             03_rules and 03_postprocess are imported as modules
└── ocrd-work/               The OCR-D workspace
    ├── mets.xml             The source of truth for what exists
    ├── OCR-D-IMG/           144 dpi originals, registered for provenance
    ├── OCR-D-IMG-360/       Upscaled derivatives — the grid the workflow runs on
    ├── OCR-D-BIN/           ┐
    ├── OCR-D-DEN/           │
    ├── OCR-D-DSK/           │
    ├── OCR-D-SEG-REG/       ├ one fileGrp per processor, in chain order;
    ├── OCR-D-SEG-REP/       │ see Stage 2
    ├── OCR-D-CLIP/          │
    ├── OCR-D-SEG-LINE/      │
    ├── OCR-D-DEW/           ┘
    ├── OCR-D-OCR/           PAGE-XML: regions, lines, words, coordinates, confidences
    ├── OCR-D-ALTO/          ALTO export
    ├── OCR-D-TXT/           Empty — see the note under Output
    ├── ocrd.log, test.log   Processor logs (larger than the copies in the root)
    └── ?/                   Stray. A container process wrote an xmlresolver cache
                             to an unset $HOME. Not in METS, safe to delete.
```

METS references files by paths relative to the workspace directory, so
everything registered as a fileGrp must live under `ocrd-work/`.

# Difference between corrected/ and reflowed/
Both hold the same corrected text; they differ in what they preserve. corrected/ mirrors the page for proofreading, reflowed/ normalizes for machine reading.
Same 16 pages, same corrections, same filenames — three differences:
Line breaks -- CORRECTED one line per line of the page  -- REFLOWED one line per flow of text (paragraph / caption / footnote block)
Hyphenation -- CORRECTED left as printed: viel- / leicht   --   REFLOWED closed up: vielleicht
Small caps  -- CORRECTED flattened to capitals, as deu read them: DIE STADT -- REFLOWED recased: Die Stadt St. Gallen

Everything upstream is identical: the same noise removal, column splits, character rules, gazetteer names and footnote markers feed both. reflowed/ is corrected/ plus the last two passes of Stage 4.

Which to use:
- corrected/ for proofreading — line n of the file is line n of the scan, so you can check it against the page, and crop.py 16 --line 21 cuts out exactly that line. review/diff.txt is also computed against it.
- reflowed/ for NER and any downstream text processing. Both differences matter to a tagger: Not-/kersegg is two tokens and neither is a name, ERWIN POESCHEL reads as an acronym rather than a person, and an entity spanning a line break (PETER ANTONI / MOOSBRUGGER) is only contiguous once the flow is joined.

---

## Stage 0 — Upscaling (before OCR-D)

The source is 144 dpi, giving an x-height of roughly 10 px. Tesseract performs
best at 20+ px. Each page is therefore upscaled 250% (Lanczos) to ~360 dpi
before the workspace is built:

```
magick in.tif -colorspace Gray -filter Lanczos -resize 250% \
       -density 360 -units PixelsPerInch -compress LZW out.tif
```

**This must happen outside OCR-D.** PAGE-XML coordinates are absolute pixel
offsets into the page image, and every derived image must stay registered to
that same grid. Rescaling mid-workflow silently invalidates every region and
line polygon written before it. `ocrd-preprocess-image` explicitly forbids
size-changing operations for this reason.

Upscaling adds no information — it helps only because the recognizer's internal
normalization works better above its minimum glyph size. Testing at 250 / 320 /
400% showed negligible differences; 400% was measurably worse (interpolation
noise). 250% is used.

---

## Stage 1 — Workspace setup (`01_setup_workspace.sh`)

Creates `ocrd-work/` with two image fileGrps:

| fileGrp | Contents |
|---|---|
| `OCR-D-IMG` | Originals at 144 dpi — preserved for provenance, not processed |
| `OCR-D-IMG-360` | Upscaled derivatives — **the grid the workflow runs on** |

Both are registered against the same physical page IDs (`PHYS_0001` …), so METS
records them as two representations of one page.

The script refuses to run against an existing workspace, since re-registering
files duplicates METS IDs.

---

## Stage 2 — OCR chain (`02_run_ocr.sh`)

Nine processors, each consuming the previous one's output:

| # | Processor | Output group | Purpose |
|---|---|---|---|
| 1 | `ocrd-olena-binarize` | `OCR-D-BIN` | Sauvola multiscale (`sauvola-ms-split`, k=0.34). The multiscale variant copes better with halftone plates than plain Sauvola. |
| 2 | `ocrd-cis-ocropy-denoise` | `OCR-D-DEN` | Removes dither speckle from the binarized plates. |
| 3 | `ocrd-cis-ocropy-deskew` | `OCR-D-DSK` | Page-level deskew. Reports 0.0° on all pages — these are flatbed scans. |
| 4 | `ocrd-tesserocr-segment-region` | `OCR-D-SEG-REG` | Layout analysis. Classifies `FLOWING_TEXT`, `CAPTION_TEXT`, `FLOWING_IMAGE` — this is what keeps the photographic plates out of the text. |
| 5 | `ocrd-segment-repair` | `OCR-D-SEG-REP` | Drops implausible/nested regions. |
| 6 | `ocrd-cis-ocropy-clip` | `OCR-D-CLIP` | Clips regions to their own content. |
| 7 | `ocrd-cis-ocropy-segment` | `OCR-D-SEG-LINE` | Line segmentation within each region. |
| 8 | `ocrd-cis-ocropy-dewarp` | `OCR-D-DEW` | Line dewarping. |
| 9 | `ocrd-tesserocr-recognize` | `OCR-D-OCR` | Recognition, model `deu`, word-level segmentation and text. |

Then `ocrd-fileformat-transform` exports `OCR-D-ALTO`.

### Critical parameter

Step 9 **must** use `segmentation_level word` together with
`textequiv_level word`. Setting `segmentation_level none` while requesting
word-level text produces empty `TextEquiv` elements with no error — the run
appears to succeed and yields blank output.

---

## Stage 3 — Text extraction (`extract.py`)

Run from `ocrd-work/`. Walks each `TextLine` in the PAGE-XML, collects its
direct `Word` children, and writes one line of text per line of the page into
`../plaintext/`.

Must read only direct `Word` children: PAGE-XML stores the same text
redundantly at word, line, and region level, so a naive subtree walk duplicates
every line.

---

## Stage 4 — Post-processing (`03_postprocess.py`)

Produces two texts, because proofreading and machine reading want opposite
things, plus `review/report.md` listing every change:

- `corrected/` — page-parallel, line breaks and hyphenation exactly as they
  fall on the page, so it can be checked against the scan.
- `reflowed/` — line-end hyphenation closed up, each flow of text run together.

All tunable rules live in `03_rules.py`; the main script is only machinery.

**Why not OCR-D.** Its post-correction processors are `ocrd-cor-asv-ann-process`
and `ocrd-keraslm-rate`. Both require a trained Keras `.h5` model, none ships in
`ocrd/all`, and both are TensorFlow-based — which this pipeline already avoids
under x86 emulation. The published cor-asv models target 19th-century Fraktur,
not this typography. `ocrd-cis-align` needs a second engine to vote against and
`ocrd-dinglehopper` only measures against ground truth.

**Why it reads PAGE-XML, not `plaintext/`.** The XML carries per-word `conf` and
polygons. Those detect the two defects no character rule can reach: spurious
text regions inside halftone plates (20 junk lines on page 202) and
side-by-side captions merged into one full-width region.

| Pass | What it does |
|---|---|
| Noise | Drops lines by confidence, letter count, width and ImageRegion overlap. Citation, date and footnote-entry shapes are protected — they score like noise but are real. |
| Stray words | Removes lone low-confidence punctuation inside good lines (`202 . DIE STADT` → `202 DIE STADT`). |
| Column split | Finds the column boundary from the lines that show it clearly, then applies that x to the whole region. |
| Character rules | Narrow, context-anchored substitutions: `ı`→`1` in numeric positions, `T`→`†` before a year, `X`→`×` between figures. |
| Gazetteer | Small-caps proper names, keyed on variants actually observed. |
| Footnote markers | See below. |
| Hyphenation | See below. Writes `reflowed/`; `corrected/` is unaffected. |
| Small-caps recasing | See below. Writes `reflowed/`; `corrected/` is unaffected. |

### Footnote markers

Markers are fused into the preceding word (`geplündert‘.`) and word-level is the
finest segmentation available, so there is no glyph geometry to exploit —
detection is textual, and some markers are lost by the OCR entirely.

Three things make this safe:

- **Brackets are tracked across line breaks.** Counting openers against closers
  is circular, because a marker that *is* a `)` inflates the closer count. Per
  line is not enough either: `(UB III,` ends one line and `S.365).` opens the
  next.
- **Only running text is consulted.** Garbled figure legends survive the noise
  filter deliberately, but their stray brackets mimic markers. Body lines are
  identified by sitting in a region with other body lines — not by confidence
  (a line of small-caps names scores 0.64) and not by word count (`wurden
  (Abb. 193)*.` is three words and genuine).
- **Numbering is applied only where it is forced.** Candidates must ascend, and
  a marker that survived as a literal digit pins its own value; where the
  feasible interval collapses to one number the assignment is proven. Equal
  counts prove nothing — page 194 balanced at 12 against 12 while having
  invented two markers and lost three.

The footnote block is located by baseline pitch (body 59–72 px, footnotes
42–56), which numerals cannot skew the way glyph height can, and anchored to
the last body line — plate captions and the printer's signature line sit below
the block, so a bottom-up scan overshoots.

Of 16 pages: 4 fully resolved, 4 partially, 8 need manual work. Unresolved
markers are left exactly as the OCR read them and listed in the report.

### Line-end hyphenation

`corrected/` keeps the break, `reflowed/` closes it. This is what NER needs:
`Ulrich von Not-` / `kersegg` is two tokens to a tagger, and neither is a name.

76 line-end hyphens occur across these 16 pages. In 70 the continuation starts
with a lowercase letter, where closing up is unambiguous. The remaining six are
not decidable from the text, and every possible reading occurs among them:

| On the page | Reading | Result |
|---|---|---|
| `RITT-` / `MEYER` | a name split by hyphenation | `RITTMEYER` |
| `THIEME-` / `BECKER` | a genuine compound | `THIEME-BECKER` |
| `St.-` / `Thecla-Altar` | a genuine compound | `St.-Thecla-Altar` |
| `von 1954-` / `Kultusgeräte.` | no continuation at all — the next line opens a new section | left broken, reported |

So a capital is resolved only from `HYPHEN_JOIN` and `HYPHEN_COMPOUND` in
`03_rules.py`. Anything unlisted stays broken and is reported, which is the
policy the footnote markers follow too.

Two details that each cost a bug:

- **A candidate must be a line end in the source.** After a join the line can
  end in a hyphen again: page 2's `Glaser-, Schlosser-, Stein-` + `hauer- und
  Schreinerwerk` gives `Steinhauer-`, whose hyphen is suspended, not broken.
  Closing that one too yields `Steinhauer-und`.
- **A continuation belongs to the region, and can cross the page turn.** Page 6's
  footnote ends `Andreas Moos-` and continues `brugger.` in page 7's footnote
  block, with a figure caption sitting between them on page 6. The continuation
  is therefore the next page's first paragraph *of the same stratum*, never
  simply the next line. Page 4 does the same in body text: `sich be-` +
  `dienende`.

Finding that stratum needs more than the footnote block's pitch boundary, which
answers a different question — where body text stops, not where footnotes start:

- page 5 closes a paragraph with `wurde (SKL I, S.71).`, four words that no
  measure calls body text although the paragraph plainly continues;
- pages 2, 11 and 15 keep body text and footnotes in one region, so the region
  change that marks the block elsewhere is missing;
- page 11 overshoots by two lines, because the first footnote line's own pitch
  is measured across the white space that separates the block.

So the block is taken from the first positive evidence instead: an entry opening
at footnote word height, or a region that itself starts at the boundary — page
7's block opens mid-word with `brugger.` and carries no entry number at all.

`--reflow lines` completes the words but keeps the page line breaks, for a
proofreading text without hyphenation.

### Small-caps recasing

The volume sets proper names in small caps and `deu` flattens them to capitals.
`corrected/` keeps that, since it is what stands on the page; `reflowed/`
restores the case, because to a tagger `ERWIN POESCHEL` is an acronym and not a
person. Any all-capital token of two letters or more is recased, which also
settles the running header and the section headings (`DAS KLOSTER NOTKERSEGG` →
`Das Kloster Notkersegg`). 101 distinct tokens, 177 occurrences.

Single letters are left alone, so initials keep their capital — `D. F. RITTMEYER`
→ `D. F. Rittmeyer` — and each part of a hyphenated name is recased on its own:
`THIEME-BECKER` → `Thieme-Becker`, `LANDOLT-TH.` → `Landolt-Th.`.

Three kinds survive untouched, listed in `03_rules.py`:

| Kind | Examples | Rule |
|---|---|---|
| Sigla and marks | `UB III`, `SKL IV`, `HBLS`, `MAGZ`, `SLM`, `ZAK`, the silver mark `J CS` | `CAPS_KEEP` — the capitals are the reading, not a font |
| Roman numerals | `Vad. I, S.322`, `XLV`, `MDCCCXVIII` (1818) | `ROMAN`, matched strictly so a name of the same letters is not mistaken for one |
| Citation tokens | `S.69`, `S.99-103` | unchanged by recasing already |

German function words are lowercased inside a run of capitals but never where
the run opens, since that word carries the sentence: `DIE KIRCHE ZUM HERZEN JESU
IN ST. FIDEN` → `Die Kirche zum Herzen Jesu in St. Fiden`, `GESCHICHTE UND
BAUGESCHICHTE` → `Geschichte und Baugeschichte`. See `CAPS_LOWER`.

Recasing runs after the hyphen joins and the gazetteer, both of which are keyed
on the flattened capitals.

One consequence to be aware of: the lapidary inscription on page 207 is set in
capitals on the page and gets recased along with the names (`CRUCEM HANC
PANCRATIUS…` → `Crucem Hanc Pancratius…`). Add those words to `CAPS_KEEP` if the
inscription should stand as printed.

### Checking a reading

`crop.py` cuts a line or word out of the 360 dpi image:

```
python3 crop.py 4 --word Nörzuıı --open
```

This is how `NÖTZLI` and `LUTZ` were settled; both then moved from `UNCERTAIN`
into `GAZETTEER`.

---

## Stage 5 — Assembling one text (`04_assemble.py`)

Turns the sixteen page files into a continuous text, three ways:

| Output | What it is |
|---|---|
| `assembled/full.txt` | `reflowed/` concatenated in page order, verbatim — byte-identical to `cat reflowed/*.txt` |
| `assembled/reading.txt` | Running text: page furniture removed, captions and footnotes moved to the end |
| `assembled/body.txt` | The same running text, without them — 12 paragraphs, 25 kB against 33 kB |

`body.txt` is `reading.txt` cut off before the apparatus; the prose in the two is
identical, so anything measured on one holds for the other.

The script reads its text from `reflowed/` — what you proofread there is what
comes out — but takes the *structure* from the PAGE-XML, importing
`03_postprocess` as a library and reusing its stratum labels. Paragraph *i* of
page *N* is line *i* of `reflowed/PHYS_N.txt`, and the alignment is checked
before anything is written. Nothing in Stage 4 is modified.

### Why the structure cannot come from the text

- **A chapter title and the running header are both set in capitals.** Position
  separates them: the title is centred on the measure (offsets of +0.016 and
  0.000 of the page width here), the header sits in the outer margin and
  alternates with recto and verso (+0.15 odd, −0.12 even). The page number rides
  in the header, so removing the header removes it too.
- **A caption is not reliably "a line beginning with Abb."** Page 205 carries two
  captions side by side and the column splitter left the second one's `Abb.` in
  the first column, so it opens `196. St. Fiden.`.

19 paragraphs are removed as furniture: 16 running headers, 2 chapter titles,
and the printer's signature on page 193 (`13 - Kunstdenkmäler XXXVII St.G. 11.`),
which survives every statistical filter because it is a perfectly clean line.
Each removal is listed on stdout. `--keep-titles` keeps the two chapter titles
in the reading text.

### Joining what the page grid broke

A paragraph that does not close a sentence is continued by the next one of its
kind, which is what makes the text continuous rather than page-shaped:

- **Body**, over the page turn: page 196 ends `…sich bedienende` and page 197
  opens `Disposition wirkt beinahe…`. 19 body paragraphs become 12.
- **Footnotes**, where a block does not open with an entry number: page 199
  begins `Vgl. SKL IV, S.318`, which is still footnote 1 of page 198. 15 blocks
  become 12 entries, each labelled with the printed folio it starts on.

A chapter title stops the join in both directions, kept or dropped — it is a
real break in the text, not an artefact of the page.

Captions are never joined to each other: each belongs to its own plate.
Three of them do arrive as one entry, on pages 202, 205 and 206, because Stage 4
groups a caption by region and the column splitter leaves both halves in one.
The script reports those three rather than guessing a split point in text that
has already been rewritten; separating them means grouping by column as well as
region in Stage 4.

---

## Output

| Group | Format | Use |
|---|---|---|
| `OCR-D-OCR` | PAGE-XML | Full structure: regions, lines, words, coordinates |
| `OCR-D-ALTO` | ALTO XML | Interchange format, retains word coordinates |
| `plaintext/` | UTF-8 text | Raw extraction |
| `corrected/` | UTF-8 text | Post-processed, for proofreading |
| `reflowed/` | UTF-8 text | Hyphenation closed, one flow per line — for NER |
| `assembled/full.txt` | UTF-8 text | The whole volume, verbatim concatenation |
| `assembled/reading.txt` | UTF-8 text | The whole volume as running text, apparatus at the end |
| `assembled/body.txt` | UTF-8 text | Running text only, no captions or footnotes |

`OCR-D-TXT` is **empty** — every file contains only newlines. The text export
ran at a level where nothing had been written. Use `plaintext/` or `corrected/`;
regenerating that fileGrp is unfinished business.

---

## Accuracy

Measured on a representative body-text page: roughly **0.5% CER**
(~4 errors in ~700 characters).

Handled well: ligatures, umlauts, italics, letterspaced headers, region/caption
separation.

Known systematic errors — inherent to the model and typography, not fixable by
parameter tuning:

| Error | Cause |
|---|---|
| Superscript footnote markers → punctuation (`¹`→`"`, `1`→`!`, `2`→`*`) | Superscript digits are ~5 px in the source; `deu` has effectively no superscript training data |
| Old-style `1` → `ı` (dotless i) | Old-style figures set `1` at x-height with no ascender |
| Small caps flattened to ALL CAPS; proper names unreliable (`KNONAU`→`Knonavu`) | No small-caps model available |
| `f` → `I` in abbreviations (`ZfSKG`→`ZISKG`) | Letterform ambiguity at small sizes |
| Occasional `h`→`c` (`Siehe`→`Siche`) | Low x-height |
| French loanwords lose accents (`à`→`ä`) | German-only model |

These are consistent enough to correct with targeted substitution rules during
post-processing. Broad character-class substitutions should be avoided — they
corrupt legitimate occurrences.

Better results on old-style figures would require a Calamari model trained on
historical prints, which needs TensorFlow and therefore an x86_64 machine.

---

## Known non-fatal warnings

`segment "…" image … has not been cropped properly (…)` — logged at ERROR level
by step 8. `ocrd-cis-ocropy-dewarp` reflows line images by a few pixels without
updating their `Coords`; OCR-D falls back to cropping by coordinates.
Recognition is unaffected. Since these are flatbed scans with no curvature, the
dewarp step could be removed altogether.

## Operational notes

- **METS is the source of truth, not the filesystem.** Never delete a fileGrp
  directory with `rm` — use `ocrd workspace remove-group -f -r <NAME>`, or the
  METS entry is orphaned and the next run fails with "already in METS".
- **Regenerating beats overwriting.** For results you intend to keep, remove the
  group and re-run rather than relying on `--overwrite`.
- **Pin paths before changing directory.** `$PWD` inside a shell function is
  evaluated at call time; if the script has already `cd`-ed, bind mounts will
  resolve to the wrong location and Docker will silently create empty
  directories.
