#!/usr/bin/env python3
"""Stage 4 - post-processing: PAGE-XML -> corrected page-parallel plain text.

Reads OCR-D-OCR/*.xml (not the flat text), because the XML carries per-word
confidence and polygons. Those are what make plate noise and merged two-column
captions detectable at all; no character rule can reach them.

Two texts come out of it, because proofreading and machine reading want
opposite things. corrected/ keeps the line breaks exactly as they stand on the
page, hyphenation included, so it can be checked against the scan. reflowed/
closes up the line-end hyphenation, restores the case of the small-caps names
and runs each flow of text together - which is what an NER model needs. To a
tagger, "Ulrich von Not-/kersegg" is two tokens and neither is a name, and
"ERWIN POESCHEL" is an acronym rather than a person.

Output:
  corrected/PHYS_NNNN.txt   corrected text, page-parallel, one file per page
  reflowed/PHYS_NNNN.txt    same text, hyphenation closed, one flow per line
  review/report.md          every change, grouped by page and kind
  review/diff.txt           unified diff against plaintext/

Usage:
  python3 03_postprocess.py                  # from the project root
  python3 03_postprocess.py --dry-run        # report only, write nothing
  python3 03_postprocess.py --reflow lines   # de-hyphenate, keep line breaks
"""

import argparse
import bisect
import collections
import difflib
import importlib
import os
import re
import statistics
import sys
import xml.etree.ElementTree as ET

R = importlib.import_module("03_rules")

PAGE_RE = re.compile(r"(\d{4})")


# --------------------------------------------------------------------------
# PAGE-XML model
# --------------------------------------------------------------------------

def bbox(points):
    pts = [tuple(map(int, p.split(","))) for p in points.split()]
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    return min(xs), min(ys), max(xs), max(ys)


class Word:
    __slots__ = ("text", "conf", "x0", "y0", "x1", "y1")

    def __init__(self, text, conf, box):
        self.text = text
        self.conf = conf
        self.x0, self.y0, self.x1, self.y1 = box

    @property
    def height(self):
        return self.y1 - self.y0


class Line:
    def __init__(self, region, ident, box, words):
        self.region = region
        self.id = ident
        self.x0, self.y0, self.x1, self.y1 = box
        self.words = words
        self.dropped = None          # reason string once filtered out
        self.text = " ".join(w.text for w in words if w.text)

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def conf(self):
        c = [w.conf for w in self.words if w.conf is not None]
        return sum(c) / len(c) if c else None

    @property
    def med_word_h(self):
        h = sorted(w.height for w in self.words)
        return h[len(h) // 2] if h else 0

    @property
    def letters(self):
        return sum(ch.isalpha() for ch in self.text)


class Page:
    def __init__(self, path):
        self.path = path
        self.num = PAGE_RE.search(os.path.basename(path)).group(1)
        root = ET.parse(path).getroot()
        uri = re.match(r"\{(.*)\}", root.tag).group(1)
        self.q = lambda t: "{%s}%s" % (uri, t)
        q = self.q

        pg = root.find(q("Page"))
        self.width = int(pg.get("imageWidth"))
        self.height = int(pg.get("imageHeight"))
        self.images = [bbox(i.find(q("Coords")).get("points"))
                       for i in root.iter(q("ImageRegion"))]

        self.regions = []            # [(region_id, [Line, ...])]
        for tr in root.iter(q("TextRegion")):
            lines = []
            for tl in tr.findall(q("TextLine")):
                words = []
                for w in tl.findall(q("Word")):
                    te = w.find(q("TextEquiv"))
                    if te is None:
                        continue
                    u = te.find(q("Unicode"))
                    if u is None or not u.text or not u.text.strip():
                        continue
                    conf = te.get("conf")
                    words.append(Word(u.text.strip(),
                                      float(conf) if conf else None,
                                      bbox(w.find(q("Coords")).get("points"))))
                lines.append(Line(tr.get("id"), tl.get("id"),
                                  bbox(tl.find(q("Coords")).get("points")),
                                  words))
            if lines:
                self.regions.append((tr.get("id"), lines))

    def lines(self):
        for _, ls in self.regions:
            for l in ls:
                yield l

    def kept(self):
        return [l for l in self.lines() if not l.dropped]

    def in_image(self, line):
        cx, cy = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
        return any(a <= cx <= c and b <= cy <= d for a, b, c, d in self.images)

    @property
    def column_width(self):
        """Median width of confident body lines - the page's own scale."""
        w = [l.width for l in self.lines()
             if l.med_word_h >= R.BODY_MIN_WORD_H
             and (l.conf or 0) >= 0.85 and l.letters >= 10]
        if not w:
            w = [l.width for l in self.lines() if l.letters >= 10]
        return statistics.median(w) if w else self.width * 0.7


# --------------------------------------------------------------------------
# Stage A - noise removal
# --------------------------------------------------------------------------

def drop_noise(page, log):
    col = page.column_width
    for line in page.lines():
        if line.dropped:                  # second pass over the split halves
            continue
        t = line.text.strip()
        conf = line.conf
        protected = (bool(R.CITATION_KEEP.match(t))
                     or bool(R.DATE_KEEP.match(t))
                     or bool(R.FOOTNOTE_KEEP.match(t)
                             and line.letters >= R.FOOTNOTE_KEEP_LETTERS))
        reason = None
        if not t:
            reason = "empty"
        elif protected:
            reason = None
        elif (conf is not None and conf < R.CONF_DROP
              and line.letters < R.CONF_LETTERS):
            reason = "conf %.2f < %.2f" % (conf, R.CONF_DROP)
        elif line.letters == 0 and not protected:
            reason = "no letters"
        elif (line.letters < R.STUB_LETTERS
              and line.width < R.STUB_WIDTH * col and not protected):
            reason = "stub (%d letters, %d px < %.0f px)" % (
                line.letters, line.width, R.STUB_WIDTH * col)
        elif page.in_image(line) and not protected and (
                (conf or 0) < R.IMG_CONF or line.letters < R.IMG_LETTERS):
            reason = "inside ImageRegion (conf %.2f, %d letters)" % (
                conf or 0, line.letters)
        if reason:
            line.dropped = reason
            log.append(("noise", reason, t, "%r  [%s]" % (t, line.region)))


def drop_stray_words(page, log):
    """Remove lone low-confidence punctuation inside otherwise good lines."""
    for line in page.kept():
        if len(line.words) < 3:
            continue
        keep = []
        for w in line.words:
            if (R.WORD_DROP_RE.match(w.text)
                    and w.conf is not None and w.conf < R.WORD_DROP_CONF):
                log.append(("stray word", line.region, line.text,
                            "removed %r (conf %.2f)" % (w.text, w.conf)))
                continue
            keep.append(w)
        if len(keep) != len(line.words):
            line.words = keep
            line.text = " ".join(w.text for w in keep if w.text)


# --------------------------------------------------------------------------
# Stage B - split merged side-by-side captions
# --------------------------------------------------------------------------

def split_columns(page, log):
    for idx, (rid, lines) in enumerate(page.regions):
        live = [l for l in lines if not l.dropped]
        if not live or len(live) > R.COL_MAX_LINES:
            continue
        if max(l.width for l in live) < R.COL_MIN_WIDTH * page.width:
            continue

        # The boundary belongs to the region, not to each line: a left column
        # whose text runs long leaves only a narrow gap on that line (97 px on
        # page 10). Locate the boundary on the lines that show it clearly, then
        # apply the same x to every line in the region.
        seen = []
        for line in live:
            for a, b in zip(line.words, line.words[1:]):
                if b.x0 - a.x1 >= R.COL_MIN_GAP:
                    seen.append((a.x1 + b.x0) / 2)
        if not seen:
            continue
        at = statistics.median(seen)

        left, right = [], []
        for line in live:
            lw = [w for w in line.words if w.x0 < at]
            rw = [w for w in line.words if w.x0 >= at]
            if not lw or not rw:
                left.append(line)
                continue
            a = Line(rid, line.id + "_c1", (line.x0, line.y0, lw[-1].x1, line.y1), lw)
            b = Line(rid, line.id + "_c2", (rw[0].x0, line.y0, line.x1, line.y1), rw)
            left.append(a)
            right.append(b)
            log.append(("column split", rid, line.text,
                        "split at x=%d -> %r | %r" % (at, a.text, b.text)))
        if right:
            page.regions[idx] = (rid, left + right)


# --------------------------------------------------------------------------
# Stage C - character rules and the small-caps gazetteer
# --------------------------------------------------------------------------

CHAR_RULES = [(re.compile(p), r, lbl) for p, r, lbl in R.CHAR_RULES]


def apply_text_rules(page, log):
    out = []
    for line in page.kept():
        text = line.text
        before = text
        for rx, repl, label in CHAR_RULES:
            new = rx.sub(repl, text)
            if new != text:
                log.append(("rule", label, text, new))
                text = new

        parts = text.split(" ")
        for i, tok in enumerate(parts):
            # Strip the line-break hyphen and any surviving footnote marker, so
            # "Ritt-" and "MOSBRUGGER?" both resolve and keep their trailing
            # character. Markers are renumbered later, after this pass.
            core = tok.strip(".,;:()«»\"'’‘‚*®°!?|}][-")
            if core in R.GAZETTEER:
                parts[i] = tok.replace(core, R.GAZETTEER[core], 1)
                log.append(("name", core, text, R.GAZETTEER[core]))
            elif core in R.UNCERTAIN:
                log.append(("UNCERTAIN", core, text, R.UNCERTAIN[core]))
        text = " ".join(parts)

        if text != before:
            line.text = text
        out.append(line)
    return out


# --------------------------------------------------------------------------
# Stage D - footnote block and reference markers
# --------------------------------------------------------------------------

# Entries open with a bare number and a space ("7 Stiftsarch., Bd. 396").
# Requiring no period is what separates them from continuation lines that begin
# with a figure ("18. Jahrhundert (Aarau 1914), S. 59." is not footnote 18).
ENTRY = re.compile(r"^(\d{1,2}) ")


def footnote_block(lines):
    """Index of the first footnote line, and the entry numbers found.

    Classifies by baseline pitch, not glyph height: footnotes are set smaller
    and leaded tighter, and pitch cannot be skewed by a line full of numerals.
    The block is the bottom-anchored run of tight-pitch lines, which is also
    what keeps figure noise from posing as footnote 1 (see page 3, where
    "1... 1776/78 0 10" sits inside a floor plan).
    """
    if len(lines) < 2:
        return len(lines), []
    pitch = [0] + [lines[i].y0 - lines[i - 1].y0 for i in range(1, len(lines))]

    # Body pitch from consecutive full-measure line pairs only.
    body = [pitch[i] for i in range(1, len(lines))
            if len(lines[i].words) >= 8 and len(lines[i - 1].words) >= 8
            and lines[i].med_word_h >= R.BODY_MIN_WORD_H
            and lines[i - 1].med_word_h >= R.BODY_MIN_WORD_H]
    if not body:
        return len(lines), []
    thr = R.BODY_PITCH_FRAC * statistics.median(body)

    # The block is not reliably bottom-anchored: on page 1 a plate caption and
    # the printer's signature line ("13 - Kunstdenkmaeler XXXVII") sit below it.
    # So take the line after the last body line rather than scanning upward.
    # Height alone would misjudge numeral-heavy footnotes and pitch alone would
    # misjudge one-word continuations ("vorgesehen."); require all three.
    start = len(lines)
    for i, line in enumerate(lines):
        if (len(line.words) >= R.BODY_MIN_WORDS
                and line.med_word_h >= R.BODY_MIN_WORD_H
                and pitch[i] >= thr):
            start = i + 1
    if start >= len(lines):
        return len(lines), []

    # Entries ascend from 1, but the sequence may have holes where a line was
    # too garbled to survive. Tolerate a single gap; a jump larger than that
    # means we have left the block (page 1's "13 - Kunstdenkmaeler" signature).
    last = 0
    for j in range(start, len(lines)):
        m = ENTRY.match(lines[j].text)
        if not m:
            continue
        n = int(m.group(1))
        if last < n <= last + 2:
            last = n
        elif n > last + 2:
            break
    return start, list(range(1, last + 1))


CLOSERS = {")": "(", "»": "«", "]": "["}


def real_closers(text):
    """Positions of closing brackets that actually close an opener.

    Counting openers against closers page-wide is circular: a marker that is
    itself a ")" inflates the closer count, which then licenses treating
    genuine parentheses as markers. On page 2 that turned
    "(UB III, S. 360)." into "(UB III, S. 360¹." Tracking depth per line
    instead asks the only question that matters - does this bracket close
    something?
    """
    stack, legit = {}, set()
    for i, c in enumerate(text):
        if c in CLOSERS.values():
            stack.setdefault(c, []).append(i)
        elif c in CLOSERS and stack.get(CLOSERS[c]):
            stack[CLOSERS[c]].pop()
            legit.add(i)
    return legit


def find_markers(text):
    """Word-final marker clusters. Returns [(start, end, cluster)]."""
    legit = real_closers(text)
    found = []
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        if R.ENUMERATOR.match(tok):
            continue
        core = tok.rstrip(".,;:")
        tail = tok[len(core):]
        i, cluster = len(core), ""
        while i > 0:
            c = core[i - 1]
            if c in R.MARKER_STRONG:
                cluster, i = c + cluster, i - 1
            elif c in CLOSERS and (m.start() + i - 1) not in legit:
                cluster, i = c + cluster, i - 1
            elif c == "?" and tail:
                cluster, i = c + cluster, i - 1
            elif (c.isdigit() and i >= 2
                  and (core[i - 2].islower() or core[i - 2] in R.MARKER_DIGIT_AFTER)):
                cluster, i = c + cluster, i - 1
            else:
                break
        if not cluster or i == 0:
            continue
        prev = core[i - 1]
        if prev.isalpha() or prev.isdigit() or prev in "»)":
            found.append((m.start() + i, m.start() + len(core), cluster))
    return found


def render(n):
    if R.MARKER_STYLE == "unicode":
        return str(n).translate(R.SUPERSCRIPT)
    return "[%d]" % n


def fix_markers(lines, log, page_num):
    body_end, entries = footnote_block(lines)
    body = lines[1:body_end]                      # line 0 is the running header

    # Brackets nest across line breaks in running text - "(UB III," ends one
    # line and "S.365)." opens the next - so depth has to be tracked over the
    # whole body. Scanning line by line made that closer look unbalanced and
    # invented a marker on page 1.
    starts, parts, pos = [], [], 0
    for line in body:
        starts.append(pos)
        parts.append(line.text)
        pos += len(line.text) + 1
    full = "\n".join(parts)

    # Regions that actually carry running text. Stray figure legends survive
    # the noise filter and their brackets mimic markers, but they sit alone.
    per_region = collections.Counter(
        l.region for l in body if l.med_word_h >= R.BODY_MIN_WORD_H)
    prose = {r for r, n in per_region.items() if n >= R.MARKER_MIN_REGION_LINES}
    if not prose:
        prose = set(per_region)

    # Untrusted lines stay in `full` so bracket depth keeps tracking, but
    # contribute no candidates of their own.
    cands = []
    for start, end, cluster in find_markers(full):
        i = bisect.bisect_right(starts, start) - 1
        line = body[i]
        if line.region not in prose or (line.conf or 0) < R.MARKER_MIN_CONF:
            log.append(("  ignored", cluster, line.text,
                        "not running text (region %s, conf %.2f): %r" % (
                            line.region, line.conf or 0, line.text[:40])))
            continue
        cands.append((line, start - starts[i], end - starts[i], cluster))

    if not entries:
        log.append(("footnotes", "no footnotes",
                    "0 entries / %d candidates" % len(cands),
                    "left unchanged - review below"))
        for line, start, end, cluster in cands:
            log.append(("  candidate", cluster, line.text,
                        "…%s" % line.text[max(0, start - 30):end]))
        return

    numbers, note = assign(cands, len(entries))
    settled = sum(1 for n in numbers if n)
    status = ("applied" if settled == len(cands) == len(entries)
              else "partial" if settled else "MISMATCH")
    log.append(("footnotes", status,
                "%d entries / %d candidates, %d settled" % (
                    len(entries), len(cands), settled),
                note))

    by_line = {}
    for (line, start, end, cluster), n in zip(cands, numbers):
        if n is None:
            log.append(("  candidate", cluster, line.text,
                        "unresolved …%s" % line.text[max(0, start - 30):end]))
            continue
        by_line.setdefault(id(line), []).append((start, end, n, cluster))
    for line in body:
        edits = by_line.get(id(line))
        if not edits:
            continue
        text = line.text
        for start, end, n, cluster in sorted(edits, reverse=True):
            text = text[:start] + render(n) + text[end:]
        log.append(("marker", "p%s" % page_num, line.text, text))
        line.text = text


def assign(cands, total):
    """Map candidates to footnote numbers, but only where the map is forced.

    Candidates are in reading order, so their numbers must strictly increase.
    A candidate that survived OCR as a literal digit pins its own value. With
    those constraints the feasible number for each position is an interval;
    where the interval collapses to one value, the assignment is proven and
    safe to apply. Where it does not - because a marker was lost and the gap
    could sit in more than one place - the candidate is left untouched.

    This replaces gating on equal counts, which is not proof of anything:
    page 2 balanced at 12 candidates against 12 entries while having invented
    two markers and lost three.
    """
    n = len(cands)
    if n == 0 or n > total:
        return [None] * n, ("more candidates than footnotes"
                            if n > total else "no candidates")

    pin = [int(c) if c.isdigit() and 1 <= int(c) <= total else None
           for _, _, _, c in cands]

    lo, hi = [0] * n, [0] * n
    prev = 0
    for i in range(n):
        prev = max(prev + 1, pin[i] or 0)
        lo[i] = prev
    nxt = total + 1
    for i in range(n - 1, -1, -1):
        nxt = min(nxt - 1, pin[i] or total)
        hi[i] = nxt

    if any(lo[i] > hi[i] for i in range(n)):
        return [None] * n, "digit evidence contradicts reading order"

    numbers = [lo[i] if lo[i] == hi[i] else None for i in range(n)]
    missing = total - n
    note = ""
    if missing:
        note = "%d marker(s) lost by OCR; %d position(s) still forced" % (
            missing, sum(1 for x in numbers if x))
    return numbers, note


# --------------------------------------------------------------------------
# Stage E - reading text: close up line-end hyphenation
# --------------------------------------------------------------------------

def footnote_start(lines, body_end, first):
    """Index of the first footnote line.

    `footnote_block` reports where body text stops, which is not the same as
    where footnotes begin, and the difference cuts both ways. Page 5 closes its
    last paragraph with "wurde (SKL I, S.71).", four words that no measure
    calls body text although the paragraph plainly continues. Pages 2 and 15
    put body text and footnotes in a single region, so the region change that
    marks the block elsewhere is absent.

    So move forward from the boundary to the first positive evidence of a
    footnote instead: a line opening an entry, or a region that itself starts
    here (page 7's block opens mid-word with "brugger.", carrying no entry
    number at all).

    And backwards, because the boundary can also land too late. The first
    footnote line's own pitch is measured across the white space that separates
    the block, so a long numeral-heavy one reads as body text, and its
    continuation can too - page 11 overshoots by two lines that way. What
    cannot be faked is the way the block opens: entry number 1, at footnote
    size. Word height is the guard that separates it from an enumerated body
    line, and ENTRY already refuses "1. Vier Glocken" by requiring no period.
    """
    out = len(lines)
    for j in range(body_end, len(lines)):
        if ENTRY.match(lines[j].text) or first[lines[j].region] >= body_end:
            out = j
            break
    for j, line in enumerate(lines[:out]):
        m = ENTRY.match(line.text)
        if j and m and m.group(1) == "1" and line.med_word_h < R.BODY_MIN_WORD_H:
            return j
    return out


def strata(lines):
    """Label every line header / body / caption / footnote.

    Reuses the discriminators already established for markers rather than
    inventing new ones: the footnote block boundary from baseline pitch, and
    `prose` regions - a body line keeps company with many other body lines,
    while a stray caption or legend sits alone in its region.
    """
    body_end, _ = footnote_block(lines)
    per_region = collections.Counter(
        l.region for l in lines[1:body_end] if l.med_word_h >= R.BODY_MIN_WORD_H)
    prose = {r for r, n in per_region.items() if n >= R.MARKER_MIN_REGION_LINES}
    first = {}
    for i, line in enumerate(lines):
        first.setdefault(line.region, i)
    foot = footnote_start(lines, body_end, first)

    out = []
    for i, line in enumerate(lines):
        if i == 0:
            st = "header"
        elif R.CAPTION_OPEN.match(lines[first[line.region]].text):
            st = "caption"          # tested first: a caption can sit below the
        elif i >= foot:             # footnote block (page 6) and must never be
            st = "footnote"         # taken for the continuation of a word
        else:
            st = "body" if line.region in prose else "caption"
        out.append(st)
    return out


def paragraphs(lines):
    """Consecutive lines grouped by (stratum, region) - one flow of text each.

    Region is the unit that matters: on page 6 a caption region sits between a
    footnote and its continuation, so "next line" is not "next line of this
    text". Joining across that boundary would weld a caption onto a word.
    """
    out = []
    for line, st in zip(lines, strata(lines)):
        if not line.text.strip():
            continue
        if out and out[-1][0] == st and out[-1][1] == line.region:
            out[-1][2].append(line.text)
        else:
            out.append([st, line.region, [line.text]])
    return out


def hyphen_action(head, tail):
    """'join' (close up), 'compound' (keep the hyphen) or 'open' (leave it)."""
    if tail[:1].islower():
        return "join"
    if (head, tail) in R.HYPHEN_JOIN:
        return "join"
    if (head, tail) in R.HYPHEN_COMPOUND:
        return "compound"
    return "open"


def fragments(line, nxt):
    """The two halves of a candidate break: end of `line`, start of `nxt`."""
    return (line[:-1].rsplit(" ", 1)[-1].strip(R.HYPHEN_STRIP),
            nxt.split(" ", 1)[0].strip(R.HYPHEN_STRIP))


def close_hyphens(texts, note):
    """Complete words broken by a line-end hyphen, within one paragraph.

    Only the *first* token of the continuation moves up, so the line structure
    survives: `--reflow lines` keeps it, `--reflow paragraph` joins afterwards.

    A candidate must be a line end in the source. That matters because a
    completed line can end in a hyphen again - page 2's "Glaser-, Schlosser-,
    Stein-" + "hauer- und Schreinerwerk" yields "...Steinhauer-", whose hyphen
    is suspended, not broken. Joining that one would produce "Steinhauer-und".
    Testing `texts[i]` rather than the accumulated line is what avoids it.
    """
    out, pull = [], None
    for i, raw in enumerate(texts):
        t = raw.strip()
        if pull is not None:
            tok, _, rest = t.partition(" ")
            out[-1] += pull + tok            # finish the word where it started
            pull, t = None, rest.strip()
            if not t:                        # continuation held nothing else
                continue
        out.append(t)
        if i + 1 == len(texts) or not t.endswith("-"):
            continue
        head, tail = fragments(t, texts[i + 1])
        act = hyphen_action(head, tail)
        note(act, head, tail, t)
        if act == "open":
            continue
        out[-1] = t[:-1]
        pull = "" if act == "join" else "-"
    return out


def stitch(pages):
    """Complete a word broken across a page turn.

    Two occur in this volume: page 4's body ends "sich be-" and continues
    "dienende Disposition" at the top of page 5, and page 6's footnote ends
    "Andreas Moos-" and continues "brugger." in page 7's footnote block. The
    continuation is therefore the next page's first paragraph *of the same
    stratum* - on page 6 a caption sits between the two - and only the word
    itself moves, not the rest of the paragraph.
    """
    for i, page in enumerate(pages[:-1]):
        for stratum, _, lines in page["paras"]:
            if not lines or not lines[-1].endswith("-"):
                continue
            nxt = next((p for p in pages[i + 1]["paras"] if p[0] == stratum), None)
            note = page["note"]
            if nxt is None or not nxt[2]:
                note("open", *fragments(lines[-1], ""), lines[-1])
                continue
            head, tail = fragments(lines[-1], nxt[2][0])
            act = hyphen_action(head, tail)
            note(act, head, tail, lines[-1],
                 "across the page turn %s->%s" % (page["num"], pages[i + 1]["num"]))
            if act == "open":
                continue
            tok, _, rest = nxt[2][0].partition(" ")
            lines[-1] = lines[-1][:-1] + ("" if act == "join" else "-") + tok
            if rest.strip():
                nxt[2][0] = rest.strip()
            else:
                nxt[2].pop(0)


def recase(text, note):
    """Restore case to small-caps names: ERWIN -> Erwin.

    Single letters are left alone, so initials keep their capital ("D. F.
    RITTMEYER" -> "D. F. Rittmeyer"), and each part of a hyphenated name is
    recased on its own ("THIEME-BECKER" -> "Thieme-Becker", "LANDOLT-TH." ->
    "Landolt-Th."). Tokens already carrying a lowercase letter are not
    candidates at all.
    """
    parts = text.split(" ")
    run = False                        # inside a run of capitals?
    for i, tok in enumerate(parts):
        core = tok.strip(R.HYPHEN_STRIP)
        if len(core) < 2 or not core.isupper() or not any(c.isalpha() for c in core):
            run = False
            continue
        if core in R.CAPS_KEEP or R.ROMAN.match(core):
            run = True                 # belongs to the run, but reads as it is
            continue
        if run and core in R.CAPS_LOWER:
            new = core.lower()
        else:
            new = "-".join(p.capitalize() for p in core.split("-"))
        run = True
        if new != core:
            parts[i] = tok.replace(core, new, 1)
            note(core, new)
    return " ".join(parts)


def noter(log):
    """Report callback for the two hyphen stages."""
    def note(act, head, tail, line, where=""):
        shown = "%s-%s" % (head, tail)
        if act == "open":
            log.append(("  hyphen open", shown, line,
                        "left broken%s - resolve in 03_rules.py "
                        "(HYPHEN_JOIN / HYPHEN_COMPOUND)"
                        % (", " + where if where else "")))
        else:
            joined = head + ("" if act == "join" else "-") + tail
            log.append(("hyphen", shown, line,
                        "%s%s" % (joined, " " + where if where else "")))
    return note


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def process(path):
    page = Page(path)
    log = []
    drop_noise(page, log)
    drop_stray_words(page, log)
    split_columns(page, log)
    drop_noise(page, log)                 # re-check the split halves
    lines = apply_text_rules(page, log)
    fix_markers(lines, log, page.num)
    return (page, [l.text for l in lines if l.text.strip()], log,
            paragraphs(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="ocrd-work")
    ap.add_argument("--group", default="OCR-D-OCR")
    ap.add_argument("--out", default="corrected")
    ap.add_argument("--reflowed", default="reflowed")
    ap.add_argument("--review", default="review")
    ap.add_argument("--compare", default="plaintext")
    ap.add_argument("--reflow", choices=("paragraph", "lines"),
                    default="paragraph",
                    help="paragraph: one flow of text per line (for NER). "
                         "lines: keep the page line breaks.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = os.path.join(args.workspace, args.group)
    files = sorted(f for f in os.listdir(src) if f.endswith(".xml"))
    if not files:
        sys.exit("no PAGE-XML in %s" % src)

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        os.makedirs(args.reflowed, exist_ok=True)
        os.makedirs(args.review, exist_ok=True)

    # Pass 1 - per page. Hyphenation is closed inside each paragraph here; the
    # breaks that run over a page turn need every page in hand, so they wait.
    pages = []
    for name in files:
        page, lines, log, paras = process(os.path.join(src, name))
        note = noter(log)
        for para in paras:
            para[2] = close_hyphens(para[2], note)
        pages.append({"num": page.num, "lines": lines, "log": log,
                      "paras": paras, "note": note})
    stitch(pages)

    # Recasing runs last: HYPHEN_JOIN and the gazetteer are both keyed on the
    # flattened capitals, so it has to see them before they are undone. Logged
    # by distinct token rather than per occurrence - 297 of them would bury the
    # rest of the report.
    for entry in pages:
        seen = collections.Counter()
        for para in entry["paras"]:
            para[2] = [recase(l, lambda a, b: seen.update([(a, b)]))
                       for l in para[2]]
        for (was, now), n in sorted(seen.items()):
            entry["log"].append(("caps", was, "",
                                 now + ("  (%d×)" % n if n > 1 else "")))

    # Pass 2 - write.
    report = ["# Post-processing report", ""]
    diffs = []
    totals = {}

    for entry in pages:
        lines, log = entry["lines"], entry["log"]
        report.append("## Page %s  (%d lines)" % (entry["num"], len(lines)))
        report.append("")
        kinds = {}
        for kind, a, b, c in log:
            kinds.setdefault(kind, []).append((a, b, c))
            totals[kind] = totals.get(kind, 0) + 1
        for kind in ("noise", "stray word", "column split", "rule", "name",
                     "UNCERTAIN", "marker", "footnotes", "hyphen", "caps",
                     "  candidate", "  ignored", "  hyphen open"):
            if kind not in kinds:
                continue
            report.append("**%s** (%d)" % (kind.strip(), len(kinds[kind])))
            report.append("")
            for a, b, c in kinds[kind]:
                report.append("- `%s` — %s" % (a, c if c else b))
            report.append("")

        if not args.dry_run:
            with open(os.path.join(args.out, "PHYS_%s.txt" % entry["num"]), "w") as fh:
                fh.write("\n".join(lines) + "\n")
            glue = " " if args.reflow == "paragraph" else "\n"
            with open(os.path.join(args.reflowed,
                                   "PHYS_%s.txt" % entry["num"]), "w") as fh:
                fh.write("\n".join(glue.join(ls)
                                   for _, _, ls in entry["paras"] if ls) + "\n")

        old = os.path.join(args.compare, "OCR-D-OCR_PHYS_%s.txt" % entry["num"])
        if os.path.exists(old):
            before = open(old).read().splitlines()
            diffs += list(difflib.unified_diff(
                before, lines, "plaintext/%s" % entry["num"],
                "corrected/%s" % entry["num"], lineterm="", n=0))

    report.insert(1, "")
    report.insert(1, "  ".join("%s=%d" % (k.strip(), v)
                               for k, v in sorted(totals.items())))
    report.insert(1, "")
    report.insert(1, "**Totals**")

    if not args.dry_run:
        with open(os.path.join(args.review, "report.md"), "w") as fh:
            fh.write("\n".join(report) + "\n")
        with open(os.path.join(args.review, "diff.txt"), "w") as fh:
            fh.write("\n".join(diffs) + "\n")

    print("\n".join("%-14s %d" % (k.strip(), v) for k, v in sorted(totals.items())))
    print("\npages: %d   %s" % (len(files),
          "dry run, nothing written" if args.dry_run
          else "wrote %s/, %s/ and %s/" % (args.out, args.reflowed,
                                           args.review)))


if __name__ == "__main__":
    main()
