"""Coarse label canonicalization across HIPE subsets.

The HIPE de/fr/en coarse-NE subsets do NOT share one label ontology:

  hipe2020  (de, fr):        pers, loc, org, prod, time
  letemps   (fr):             pers, loc, org
  newseye   (de, fr):         PER, LOC, ORG, HumanProd      <- different case!
  ajmc      (de, fr, en):     pers, loc, date, work, object, scope
  topres19th(en):             LOC, BUILDING, STREET
  sonar     (de):              pers, loc, org

Same-concept types that only differ by case/abbreviation (pers/PER, loc/LOC,
org/ORG) are folded together. Genuinely different concepts (hipe2020's
"prod" vs newseye's "HumanProd", ajmc's classics-commentary types) are kept
as distinct classes rather than guessed into one another. See README.md for
the reasoning and how to narrow this if you want a cleaner single-domain
label set instead of the full union.
"""
from __future__ import annotations

import json
from pathlib import Path

_ALIASES = {
    "pers": "PER",
    "per": "PER",
    "loc": "LOC",
    "org": "ORG",
    "prod": "PROD",
    "time": "TIME",
    "date": "DATE",
    "work": "WORK",
    "object": "OBJECT",
    "scope": "SCOPE",
    "humanprod": "HUMANPROD",
    "building": "BUILDING",
    "street": "STREET",
}


def canonicalize_tag(raw_tag: str) -> str:
    """'B-pers' -> 'B-PER', 'I-HumanProd' -> 'I-HUMANPROD', '_'/'O' -> 'O'."""
    if raw_tag in ("_", "", "O"):
        return "O"
    prefix, _, body = raw_tag.partition("-")
    if prefix not in ("B", "I") or not body:
        return "O"
    canon_body = _ALIASES.get(body.lower(), body.upper())
    return f"{prefix}-{canon_body}"


def build_label_list(tag_sequences) -> list[str]:
    """Collect the sorted, canonicalized label set from an iterable of raw
    per-token tag sequences (e.g. all training sentences)."""
    seen = {"O"}
    for tags in tag_sequences:
        for t in tags:
            seen.add(canonicalize_tag(t))
    # Deterministic order: O first, then B-/I- pairs grouped by type name.
    types = sorted({t[2:] for t in seen if t != "O"})
    labels = ["O"]
    for t in types:
        labels += [f"B-{t}", f"I-{t}"]
    return labels


def save_label_map(labels: list[str], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"labels": labels}, indent=2, ensure_ascii=False), encoding="utf-8")


def load_label_map(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["labels"]
