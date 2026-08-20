"""Editable correction rules for the St. Fiden post-processing pass.

Everything a human should tune lives here. `03_postprocess.py` only implements
the machinery. Rules are deliberately narrow: broad character-class
substitutions corrupt legitimate text (see PIPELINE.md, "Accuracy").
"""

import re

# ---------------------------------------------------------------------------
# Noise-line detection
#
# Thresholds are fractions of the page's own median body-line width, so they
# survive a change of scan resolution. Every dropped line is listed in the
# report — audit that list rather than trusting these numbers.
# ---------------------------------------------------------------------------

CONF_DROP      = 0.55   # mean word confidence below this ...
CONF_LETTERS   = 12     # ... and fewer letters than this: noise. A long line
                        # is real text however badly it was read - page 16 lost
                        # "THEODOSIUS Ernst in Lindau (1642)" at conf 0.50.
                        # Keeping garbled text beats deleting real text; the
                        # proofreader can see it, a silent deletion they cannot.
STUB_LETTERS   = 4      # fewer alphabetic chars than this ...
STUB_WIDTH     = 0.35   # ... and narrower than this fraction of the column
IMG_CONF       = 0.85   # inside an ImageRegion, demand at least this conf
IMG_LETTERS    = 6      # ... and this many letters

# Short citation continuations ("1954/55, S. 184.") look like noise by every
# statistical measure but are real footnote text. Protect them explicitly.
CITATION_KEEP = re.compile(
    r"^[\(\[]?[\dIVXLC/,.\s–—-]*\b(?:S|Bd|Nr|Fol|Abb|Taf|Anm|Cap|Fasz)\."
)

# A footnote entry opening with heavy abbreviation ("5 a. a. O., S. 119.")
# scores low on every statistical measure - page 3 lost one at conf 0.38, which
# punched a hole in the block. Require a few letters so figure noise such as
# "1... 1776/78 0 10" is not protected along with it.
FOOTNOTE_KEEP = re.compile(r"^\d{1,2}[ .]\S")
FOOTNOTE_KEEP_LETTERS = 3

# Bare dates carry almost no letters and sit on their own line in captions
# ("Um 1728." under a figure), so the stub rule deletes them. Requiring three
# or four digits keeps single-digit plate noise unprotected.
DATE_KEEP = re.compile(
    r"^(?:Um|Von|Nach|Vor|Ende|Anfang|Mitte)?\s*\d{3,4}(?:\s*[/-]\s*\d{1,4})?\.?$"
)

# Single low-confidence punctuation words inside an otherwise good line
# ("202 . DIE STADT ST. GALLEN" -> "202 DIE STADT ST. GALLEN").
# Dashes are excluded on purpose: this text uses a spaced dash as parenthetical
# punctuation ("in untergeordneten Einzelheiten - den Triglyphen"), and "*" can
# be a surviving footnote marker. Removing either destroys real content.
WORD_DROP_CONF = 0.40
WORD_DROP_RE   = re.compile(r"^[.,;:|_]{1,2}$")

# ---------------------------------------------------------------------------
# Two-column region splitting
#
# `ocrd-tesserocr-segment-region` merges side-by-side figure captions into one
# full-width region, so its lines run across both columns. Split on a wide
# internal word gap.
# ---------------------------------------------------------------------------

COL_MIN_WIDTH  = 0.60   # region must span this fraction of the page
COL_MAX_LINES  = 4      # captions are short; body text must never be split
COL_MIN_GAP    = 120    # px gap (at 360 dpi) that separates the columns

# ---------------------------------------------------------------------------
# Stratum detection by median word height (px at 360 dpi)
#
# Measured: running header 25-26, footnotes/captions 31-33, body 37-39.
# ---------------------------------------------------------------------------

BODY_MIN_WORD_H = 35    # at or above this: body text
FOOT_MAX_WORD_H = 34    # at or below this: footnote or caption

# Word height is contaminated by numerals, which are set at full cap height:
# a digit-heavy footnote line ("18. Jahrhundert (Aarau 1914), S. 59.") measures
# as tall as body text. Baseline pitch is immune to that - measured 59-72 px for
# body against 42-56 px for footnotes, with a wide gap at the block boundary.
BODY_PITCH_FRAC = 0.85  # pitch at or above this fraction of body pitch: body
BODY_MIN_WORDS  = 5     # ... and this many words, or the median means nothing

# ---------------------------------------------------------------------------
# Character-level rules, applied in order, to whole lines.
#
# (pattern, replacement, label). Keep each one anchored to a context that
# cannot occur in correct German — that is what makes them safe.
# ---------------------------------------------------------------------------

CHAR_RULES = [
    # -- old-style figures: dotless i standing for 1 -------------------------
    (r"^ı(?=[ .,])",            "1",      "footnote marker i->1"),
    (r"(?<=[ (])ı(?=[.,)] )",   "1",      "enumerator i->1"),
    (r"(?<=\d)ı\b",             "1",      "trailing i->1 (9i -> 91)"),
    (r"ı(?=\d)",                "1",      "leading i->1 (i2 -> 12)"),
    (r"\bıo0\b",                "100",    "io0 -> 100"),
    (r"(?<=\bFol\. )ı(?=\d)",   "1",      "Fol. i6 -> 16"),
    (r"(?<=\bNr\. )ı\b",        "1",      "Nr. i -> 1"),
    (r"(?<=\bH\. )ı(?=\d)",     "1",      "H. i2 -> 12"),

    # -- misread digits in citations ----------------------------------------
    (r"\bS\.go\b",                   "S.90",   "S.go -> S.90"),
    (r"\bBd\.g\b",                   "Bd.9",   "Bd.g -> Bd.9"),
    (r"\bTab\.\s*1I\b",              "Tab. II","Tab.1I -> Tab. II"),
    (r"\bBerung\b",                  "ßerung", "B -> sharp s (Vergroe-Berung)"),
    (r"([,(]\s*)5\.(\s*\d)",         r"\1S.\2","5. -> S. in citation"),

    # -- symbols ------------------------------------------------------------
    (r"\b[Tt] (?=1[6-9]\d\d\b)",     "† ", "T -> dagger before year"),
    (r"\((?:T|t) (1[6-9]\d\d)\)",    r"(† \1)", "(T year) -> (dagger year)"),
    (r"(?<=\d) X (?=\d)",            " × ", "X -> multiplication sign"),
    (r"\b1’Iconographie\b",     "l’Iconographie", "1' -> l' (French)"),
    (r"\bSign\. IT,",                "Sign. II,", "IT -> II"),
    (r"(?<=\d) \.(?=[a-z])",         " ",      "stray period before word"),
    (r"\bE, (?=[A-Z]{2,})",          "E. ",    "E, -> E. before name"),

    # -- letterform confusions ---------------------------------------------
    (r"\bSiche\b",                   "Siehe",  "Siche -> Siehe"),
    (r"\bAbb\. (\d):(\d\d)\b",        r"Abb. \1\2", "Abb. 1:86 -> Abb. 186"),
    (r"\bZISKG\b",                   "ZfSKG",  "ZISKG -> ZfSKG"),
]

# ---------------------------------------------------------------------------
# Small-caps proper names.
#
# The volume sets names in small caps, which `deu` flattens unreliably. Only
# variants actually observed in this run are listed; each maps to the reading
# confirmed from context elsewhere in the same text.
# ---------------------------------------------------------------------------

GAZETTEER = {
    # Antoni Dick, painter
    "Dıck": "DICK", "DıcK": "DICK", "Dick": "DICK",
    "AnTonı": "ANTONI", "ANTonı": "ANTONI",
    "ANnToNnI": "ANTONI", "ANnTonI": "ANTONI", "AnTonI": "ANTONI",
    # Franz Anton Dürr, sculptor
    "DüÜrrs": "DÜRRS",
    # Rittmeyer
    "Rırrmeyer": "RITTMEYER", "Rıtt": "RITT", "Rırrt": "RITT",
    # Heinrich Dumeisen, goldsmith
    "Dumzısen": "DUMEISEN", "Dumeısens": "DUMEISENS",
    # misc.
    "Carı": "CARL", "Gysı": "GYSI", "JoacHım": "JOACHIM",
    "Icnaz": "IGNAZ", "IGNn": "IGN",
    "Künsrtie": "KÜNSTLE",
    "ZEcKEL": "ZECKEL", "PoEsCHEL": "POESCHEL", "PAuL": "PAUL",
    "FRAnz": "FRANZ", "ERwInN": "ERWIN", "BENnz": "BENZ",
    "AnT": "ANT", "MicHAEL": "MICHAEL",
    "TuEgopDosius": "THEODOSIUS",
    "NAEr": "NAEF",                      # NAEF spelled correctly on page 16
    "Knonavu": "KNONAU",                 # documented in PIPELINE.md
    "MOSBRUGGER": "MOOSBRUGGER",         # spelled in full on page 6
    # Roman numeral III, misread with a digit or an l
    "I1I": "III", "IlI": "III",
    # Both read off the scan with crop.py
    "Nörzuıı": "NÖTZLI", "LurTz": "LUTZ",
    "TH1EME-BECKER": "THIEME-BECKER", "THJEME-BEcKEr": "THIEME-BECKER",
    "TH1EME": "THIEME",
    # Joh. = Johann. Keys are period-free because lookup strips trailing
    # punctuation; the period in the token is preserved by the substitution.
    # NOTE: "Jos." is a correct abbreviation for Joseph (Jos. Ant. Seethaler)
    # and must NOT be mapped here.
    "Joy": "JOH", "Jon": "JOH", "JoH": "JOH",
}

# Names that are wrong but whose correct reading cannot be inferred from the
# text alone. Reported for manual adjudication; never changed automatically.
# Use `crop.py` to look at the actual glyphs, then move the entry into
# GAZETTEER above - that is how NÖTZLI and LUTZ were settled.
UNCERTAIN = {}

# ---------------------------------------------------------------------------
# Footnote reference markers
#
# Superscript digits are ~5 px in the source and `deu` has no superscript
# training data, so they surface as punctuation fused onto the preceding word
# ("gepluendert'." for "gepluendert^1."). Word-level segmentation is the finest
# available, so there is no glyph geometry to exploit: detection is textual.
# ---------------------------------------------------------------------------

# Characters that are never legitimate word-final in this text.
MARKER_STRONG = "‘’‚'*®°!|}]"

# Characters that are legitimate unless unbalanced across the whole page.
MARKER_PAIRED = {")": "(", "»": "«", "\"": "\"", "]": "["}

# A digit is a marker only when glued to the end of a word or a closing quote.
MARKER_DIGIT_AFTER = "»)"

# "a)" "b)" "c)" enumerate list items; never footnote markers.
ENUMERATOR = re.compile(r"^[a-z]\)[.,;:]?$")

# Marker inference needs trustworthy text. Garbled figure legends survive the
# noise filter on purpose (deleting real text is worse than keeping it), but
# their stray brackets look exactly like markers - "L__}) Erweiterung 1953 | nn]"
# invented two on page 3 and broke the digit anchors. Keep such lines in the
# output, just do not read markers off them.
#
# Filter on the stratum, not on confidence: body lines carrying several
# small-caps names score as low as 0.64 ("ANTONI DICK aus Isny!") while their
# markers are perfectly genuine. Glyph height separates body from figure
# legend without punishing hard-to-read body text.
#
# Word count cannot do it either: "wurden (Abb. 193)*." is a legitimate
# paragraph-final line of three words carrying a real marker. What separates
# them is company - body lines sit in a region alongside many other body lines,
# a stray legend sits alone.
MARKER_MIN_CONF = 0.45
MARKER_MIN_REGION_LINES = 3

# 'unicode' -> gepluendert¹.   'bracket' -> gepluendert[1].
MARKER_STYLE = "unicode"

SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# ---------------------------------------------------------------------------
# Line-end hyphenation (Stage E, the reflowed/ output)
#
# A hyphen at a line end is normally a syllable break to be closed up
# ("viel-" + "leicht" -> "vielleicht"). That reading is safe as long as the
# continuation starts with a lowercase letter, which covers 70 of the 76
# line-end hyphens in this volume.
#
# Where the continuation starts with a capital or a digit the readings are
# indistinguishable from the text alone, and all three occur here:
#
#   RITT- / MEYER            a name split by hyphenation  -> RITTMEYER
#   THIEME- / BECKER         a genuine compound           -> THIEME-BECKER
#   von 1954- / Kultus...    not a continuation at all; the next line opens a
#                            new section and the hyphen ends an unfinished date
#
# So capitals are resolved from these two tables only. Anything unlisted is
# left broken and reported — the same policy the footnote markers follow.
# Keys are punctuation-free, as in GAZETTEER: "St.-" keys as "St".
# ---------------------------------------------------------------------------

HYPHEN_JOIN = {                      # close up; the word was split
    ("RITT", "MEYER"),               # D. F. RITTMEYER, cf. GAZETTEER
}

HYPHEN_COMPOUND = {                  # keep the hyphen; the word contains one
    ("St", "Thecla-Altar"),
    ("THIEME", "BECKER"),            # THIEME-BECKER, cf. GAZETTEER
}

# Punctuation stripped before a table lookup. Superscript digits are included
# because markers have already been rendered by the time Stage E runs.
HYPHEN_STRIP = ".,;:()«»\"'’‘‚*®°!?|}][-⁰¹²³⁴⁵⁶⁷⁸⁹"

# Figure captions open with the plate reference throughout the volume ("Abb.
# 188. St. Fiden. Kirche. Teilstück der Deckenstukkatur."). They sit below the
# footnote block on some pages, so position alone mislabels them, and a caption
# must never be treated as the continuation of a broken word.
CAPTION_OPEN = re.compile(r"^Abb\.")

# ---------------------------------------------------------------------------
# Small-caps recasing (Stage E, the reflowed/ output)
#
# The volume sets proper names in small caps and `deu` flattens them to
# capitals. corrected/ keeps that, because it is what stands on the page. For
# NER it is actively wrong: a tagger reads ERWIN POESCHEL as an acronym, not as
# a person. So reflowed/ restores the case - ERWIN -> Erwin.
#
# Applied to any all-capital token of two letters or more, which also fixes the
# running header and the section headings ("DAS KLOSTER NOTKERSEGG" -> "Das
# Kloster Notkersegg"). Three kinds have to survive it untouched.
# ---------------------------------------------------------------------------

# Bibliographic sigla, and a silversmith's mark ("Meistermarke J CS"). Here the
# capitals are the reading itself, not a flattened font.
CAPS_KEEP = {"UB", "SKL", "HBLS", "MAGZ", "SLM", "ZAK", "CS"}

# Roman numerals: "UB III", "Vad. I, S.322", "MDCCCXVIII" for 1818. Matched
# strictly rather than by letter set, so a name spelled from the same letters
# is not mistaken for a numeral.
ROMAN = re.compile(r"^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})"
                   r"(?:IX|IV|V?I{0,3})$")

# German function words, lowercased inside a run of capitals but never where
# the run opens - that word carries the sentence: "DIE KIRCHE ZUM HERZEN JESU
# IN ST. FIDEN" -> "Die Kirche zum Herzen Jesu in St. Fiden", and "GESCHICHTE
# UND BAUGESCHICHTE" -> "Geschichte und Baugeschichte". It also gives the
# nobiliary particle its correct form in "HANS VON HORNSTEIN".
CAPS_LOWER = {"DER", "DIE", "DAS", "DES", "DEM", "DEN", "UND", "ODER", "ALS",
              "IN", "IM", "AN", "AM", "AUF", "AUS", "BEI", "FÜR", "MIT",
              "NACH", "VON", "VOM", "VOR", "ZU", "ZUM", "ZUR", "ÜBER"}
