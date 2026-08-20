#!/usr/bin/env python3
"""Crop a line or word from the 360 dpi page image, for checking a reading.

The report flags items whose correct reading cannot be settled from the text
alone - uncertain proper names, unresolved footnote markers. This pulls the
actual glyphs out of the image so you can look at them.

Usage:
  python3 crop.py 4 Noerzuii            # every line on page 4 matching a string
  python3 crop.py 4 --word Fol.         # tighten to the matching word
  python3 crop.py 2 --line 6            # by 1-based line number instead
  python3 crop.py 12 LurTz --open       # and open it in the default viewer

Writes PNGs to review/crops/ and prints the paths.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from PIL import Image

PAD = 12


def bbox(points):
    pts = [tuple(map(int, p.split(","))) for p in points.split()]
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page", help="page number, e.g. 4 or 0004")
    ap.add_argument("pattern", nargs="?", help="text to look for on the page")
    ap.add_argument("--word", metavar="TEXT", help="crop just the matching word")
    ap.add_argument("--line", type=int, help="crop this 1-based line instead")
    ap.add_argument("--workspace", default="ocrd-work")
    ap.add_argument("--group", default="OCR-D-OCR")
    ap.add_argument("--images", default="OCR-D-IMG-360")
    ap.add_argument("--out", default="review/crops")
    ap.add_argument("--open", action="store_true", help="open the crops")
    args = ap.parse_args()

    if not (args.pattern or args.word or args.line):
        sys.exit("give a pattern, --word or --line")

    num = args.page.zfill(4)
    xml = os.path.join(args.workspace, args.group,
                       "%s_PHYS_%s.xml" % (args.group, num))
    if not os.path.exists(xml):
        sys.exit("no such page: %s" % xml)

    # METS records one image per page; match on the page number in its name.
    imgs = sorted(glob.glob(os.path.join(args.workspace, args.images, "*")))
    hits = [p for p in imgs if re.search(r"[_-]0*%d\." % int(num), p)]
    if not hits:
        sys.exit("no image for page %s in %s/%s" % (num, args.workspace, args.images))
    img = Image.open(hits[0])

    root = ET.parse(xml).getroot()
    uri = re.match(r"\{(.*)\}", root.tag).group(1)
    q = lambda t: "{%s}%s" % (uri, t)

    os.makedirs(args.out, exist_ok=True)
    made = []
    for n, tl in enumerate(root.iter(q("TextLine")), 1):
        words = tl.findall(q("Word"))
        texts = []
        for w in words:
            te = w.find(q("TextEquiv"))
            u = te.find(q("Unicode")) if te is not None else None
            texts.append(u.text if u is not None and u.text else "")
        joined = " ".join(t for t in texts if t)

        if args.line is not None:
            if n != args.line:
                continue
        elif args.pattern and args.pattern not in joined:
            continue
        elif not args.pattern and args.word and args.word not in joined:
            continue

        targets = [("line%03d" % n, bbox(tl.find(q("Coords")).get("points")))]
        if args.word:
            targets = [("line%03d_%s" % (n, re.sub(r"\W+", "", t)[:16]),
                        bbox(w.find(q("Coords")).get("points")))
                       for w, t in zip(words, texts) if args.word in t] or targets

        for label, (x0, y0, x1, y1) in targets:
            box = (max(0, x0 - PAD), max(0, y0 - PAD),
                   min(img.width, x1 + PAD), min(img.height, y1 + PAD))
            path = os.path.join(args.out, "p%s_%s.png" % (num, label))
            img.crop(box).save(path)
            made.append(path)
            print("%s   %s" % (path, joined[:70]))

    if not made:
        sys.exit("nothing matched on page %s" % num)
    if args.open:
        subprocess.run(["open"] + made, check=False)


if __name__ == "__main__":
    main()
