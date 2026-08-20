#!/usr/bin/env python3
"""Stage 5 - assemble the pages into one continuous text.

Three outputs, from the same material:

  assembled/full.txt      reflowed/ concatenated in page order, verbatim
  assembled/reading.txt   running text with the page furniture removed and the
                          captions and footnotes moved to the end
  assembled/body.txt      the same running text, without them

`full.txt` is the plain answer: exactly what `cat reflowed/*.txt` gives, so
nothing is interpreted and nothing can be lost. `reading.txt` and `body.txt`
are the edited ones - identical prose, differing only in whether the apparatus
is carried along - and every decision they make is reported on stdout.

The text is read from reflowed/, which is authoritative - what you proofread
there is what comes out here. The structure is derived from the PAGE-XML by
importing 03_postprocess and using its own stratum labels, because the page
furniture cannot be identified from the text alone:

  - The chapter title and the running header are both set in capitals. What
    separates them is position: the title is centred on the measure, the header
    sits in the outer margin, alternating with recto and verso.
  - A caption is not reliably "a line starting with Abb.". Page 13 carries two
    captions side by side; the column splitter left the second one's "Abb." in
    the first column, so it opens "196. St. Fiden.".

Paragraph i of page N is line i of reflowed/PHYS_N.txt, and the script checks
that alignment before it writes anything.

Usage:
  python3 04_assemble.py                 # from the project root
  python3 04_assemble.py --keep-titles   # keep chapter titles in the reading text
"""

import argparse
import importlib
import os
import re
import sys

P = importlib.import_module("03_postprocess")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# A chapter title is centred on the measure; in this volume its centre lands
# within 0.016 of the page width of the centre, while the running header sits
# at +0.15 (recto) or -0.12 (verso).
TITLE_CENTER = 0.03

# The printer's signature at the foot of the first sheet of a gathering ("13 -
# Kunstdenkmäler XXXVII St.G. 11."). Binding furniture, not text, and it
# survives every statistical filter because it is a perfectly clean line.
SIGNATURE = re.compile(r"^\d{1,3} - Kunstdenkmäler\b")

# The printed folio, read out of the running header ("194 DIE STADT ST. GALLEN",
# "ST. FIDEN 193") rather than assumed from the file number.
FOLIO = re.compile(r"\b(\d{2,4})\b")

# A paragraph that does not end a sentence is continued by the next one of its
# kind - across the page turn, and across the column break that interrupts a
# footnote. Trailing quotes, brackets and footnote markers are looked past.
SENTENCE_END = ".!?:;"
TRAILING = "»)\"'’‘ ⁰¹²³⁴⁵⁶⁷⁸⁹*"

DROPPED = ("header", "title", "signature")


def all_caps(text):
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 2 and "".join(letters).isupper()


def is_title(lines, width):
    return all(all_caps(l.text)
               and abs((l.x0 + l.x1) / 2 - width / 2) <= TITLE_CENTER * width
               for l in lines)


def open_sentence(text):
    t = text.rstrip()
    while t and t[-1] in TRAILING:
        t = t[:-1].rstrip()
    return bool(t) and t[-1] not in SENTENCE_END


def grouped(lines):
    """03_postprocess.paragraphs(), but keeping the Line objects.

    Mirrors that grouping deliberately - same function for the labels, same
    key - so the paragraph count matches reflowed/ line for line, while the
    geometry needed for the furniture tests stays reachable.
    """
    out = []
    for line, stratum in zip(lines, P.strata(lines)):
        if not line.text.strip():
            continue
        if out and out[-1][0] == stratum and out[-1][1] == line.region:
            out[-1][2].append(line)
        else:
            out.append([stratum, line.region, [line]])
    return out


def relabel(stratum, lines, width):
    """Refine 03_postprocess's labels with the two furniture tests."""
    if stratum == "header":
        return "header"
    if SIGNATURE.match(lines[0].text):
        return "signature"
    if stratum in ("body", "caption") and is_title(lines, width):
        return "title"
    return stratum


def read_page(path, text_dir):
    """Labelled paragraphs for one page, text taken from reflowed/."""
    page = P.Page(path)
    log = []
    P.drop_noise(page, log)
    P.drop_stray_words(page, log)
    P.split_columns(page, log)
    P.drop_noise(page, log)
    lines = P.apply_text_rules(page, log)
    P.fix_markers(lines, log, page.num)

    paras = grouped(lines)
    src = os.path.join(text_dir, "PHYS_%s.txt" % page.num)
    if not os.path.exists(src):
        sys.exit("missing %s - run 03_postprocess.py first" % src)
    texts = open(src).read().splitlines()
    if len(texts) != len(paras):
        sys.exit("%s has %d lines but the page holds %d paragraphs.\n"
                 "Was it written with --reflow lines? This script needs the "
                 "default one-paragraph-per-line form." % (
                     src, len(texts), len(paras)))

    out = []
    for (stratum, _, ls), text in zip(paras, texts):
        # Stage 4 groups by region, and the column splitter leaves both halves
        # of a side-by-side caption in one. Flag it rather than guess a split
        # point in text that has already been rewritten.
        cols = {l.id[-3:] for l in ls if l.id.endswith(("_c1", "_c2"))}
        out.append((relabel(stratum, ls, page.width), text, len(cols) > 1))
    return page.num, out


def folio(paras, fallback):
    for stratum, text, _ in paras:
        if stratum == "header":
            m = FOLIO.search(text)
            if m:
                return m.group(1)
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="ocrd-work")
    ap.add_argument("--group", default="OCR-D-OCR")
    ap.add_argument("--reflowed", default="reflowed")
    ap.add_argument("--out", default="assembled")
    ap.add_argument("--keep-titles", action="store_true",
                    help="keep chapter titles in the reading text instead of "
                         "dropping them with the rest of the furniture")
    args = ap.parse_args()

    src = os.path.join(args.workspace, args.group)
    files = sorted(f for f in os.listdir(src) if f.endswith(".xml"))
    if not files:
        sys.exit("no PAGE-XML in %s" % src)
    os.makedirs(args.out, exist_ok=True)

    pages = [read_page(os.path.join(src, f), args.reflowed) for f in files]

    # ---- output 1: verbatim concatenation --------------------------------
    full = "".join(open(os.path.join(args.reflowed, "PHYS_%s.txt" % num)).read()
                   for num, _ in pages)
    with open(os.path.join(args.out, "full.txt"), "w") as fh:
        fh.write(full)

    # ---- output 2: reading text ------------------------------------------
    body, captions, notes = [], [], []
    counts, dropped, merged = {}, [], []
    # May the next body paragraph continue the last one? A running header sits
    # between every pair of pages and must not stop that - the whole point is
    # to carry a sentence over the page turn. A chapter title is the opposite:
    # a real break in the text, so it stops the join whether it is kept or
    # dropped, and it never absorbs the paragraph that follows it.
    joinable = False
    for num, paras in pages:
        page_no = folio(paras, num)
        for stratum, text, two_columns in paras:
            counts[stratum] = counts.get(stratum, 0) + 1
            if stratum in ("title", "signature"):
                joinable = False
            if stratum == "title" and args.keep_titles:
                body.append(text)
            elif stratum in DROPPED:
                dropped.append((page_no, stratum, text))
            elif stratum == "caption":
                captions.append((page_no, text))
                if two_columns:
                    merged.append((page_no, text))
            elif stratum == "footnote":
                # A footnote block that does not open with an entry number is
                # the tail of the previous entry, carried over the page turn or
                # the column break: page 199 opens "Vgl. SKL IV, S.318", which
                # is still footnote 1 of page 198.
                if notes and (open_sentence(notes[-1][1])
                              or not P.ENTRY.match(text)):
                    notes[-1] = (notes[-1][0], notes[-1][1] + " " + text)
                else:
                    notes.append((page_no, text))
            else:
                if body and joinable and open_sentence(body[-1]):
                    body[-1] += " " + text
                else:
                    body.append(text)
                joinable = True

    with open(os.path.join(args.out, "body.txt"), "w") as fh:
        fh.write("\n\n".join(body) + "\n")

    out = list(body)
    for title, items in (("CAPTIONS", captions), ("FOOTNOTES", notes)):
        if not items:
            continue
        out.append("=== %s ===" % title)
        out += ["[p. %s] %s" % (p, t) for p, t in items]
    with open(os.path.join(args.out, "reading.txt"), "w") as fh:
        fh.write("\n\n".join(out) + "\n")

    # ---- report ----------------------------------------------------------
    print("pages: %d" % len(pages))
    print("paragraphs: " + "  ".join("%s=%d" % kv for kv in sorted(counts.items())))
    print("\nbody.txt:    %d paragraphs of running text" % len(body))
    print("reading.txt: the same, plus %d captions and %d footnotes at the end"
          % (len(captions), len(notes)))
    print("removed as page furniture (%d):" % len(dropped))
    for page_no, stratum, text in dropped:
        print("  p.%-4s %-9s %s" % (page_no, stratum, text[:64]))
    if merged:
        print("\ntwo captions in one entry (%d) - Stage 4 groups a caption by "
              "region,\nand the column splitter leaves both halves in one:"
              % len(merged))
        for page_no, text in merged:
            print("  p.%-4s %s" % (page_no, text[:70]))
    print("\nwrote %s/: full.txt (%d chars), reading.txt, body.txt"
          % (args.out, len(full)))


if __name__ == "__main__":
    main()
