"""Reader for HIPE-2022 IOB TSV files.

Format (see https://github.com/hipe-eval/HIPE-2022-data):
  - one header line: TOKEN  NE-COARSE-LIT  NE-COARSE-METO  ...  MISC
  - comment lines starting with '#' (document metadata)
  - empty lines mark document boundaries
  - MISC carries pipe-separated flags; EndOfSentence marks sentence ends
    (there is no separate sentence-boundary column)
  - unannotated columns are '_'

We only need TOKEN, NE-COARSE-LIT and the EndOfSentence flag in MISC for a
coarse, flat BIO scheme. Everything else (fine type, components, nesting,
linking) is deliberately dropped here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sentence:
    tokens: list[str]
    tags: list[str]  # raw NE-COARSE-LIT values, e.g. "B-pers", "O", "_"
    document_id: str | None = None


def read_hipe_tsv(path: str | Path) -> list[Sentence]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    header = None
    col_idx: dict[str, int] = {}
    sentences: list[Sentence] = []
    cur_tokens: list[str] = []
    cur_tags: list[str] = []
    cur_doc_id: str | None = None

    def flush():
        nonlocal cur_tokens, cur_tags
        if cur_tokens:
            sentences.append(Sentence(cur_tokens, cur_tags, cur_doc_id))
        cur_tokens, cur_tags = [], []

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if header is None:
            if line.startswith("TOKEN"):
                header = line.split("\t")
                col_idx = {name: i for i, name in enumerate(header)}
            continue  # skip anything before the header (shouldn't happen)

        if line == "":
            # empty line = document boundary: always ends the current sentence too
            flush()
            continue

        if line.startswith("#"):
            if "document_id" in line:
                # "# hipe2022:document_id = NZZ-1798-01-20-a-p0001"
                cur_doc_id = line.split("=", 1)[-1].strip()
            continue

        fields = line.split("\t")
        token = fields[col_idx["TOKEN"]]
        tag = fields[col_idx["NE-COARSE-LIT"]]
        misc = fields[col_idx["MISC"]] if "MISC" in col_idx and len(fields) > col_idx["MISC"] else "_"

        cur_tokens.append(token)
        cur_tags.append(tag)

        if "EndOfSentence" in misc:
            flush()

    flush()
    return sentences


def iter_hipe_files(hipe_root: str | Path, dataset: str, lang: str, split: str) -> Path | None:
    """Locate the TSV for a given (dataset, lang, split), or None if absent.

    Not every dataset/language/split combination exists (e.g. hipe2020 has no
    English subset, some datasets ship no released test labels).
    """
    hipe_root = Path(hipe_root)
    pattern = f"HIPE-2022-*-{dataset}-{split}-{lang}.tsv"
    matches = sorted((hipe_root / dataset / lang).glob(pattern)) if (hipe_root / dataset / lang).is_dir() else []
    return matches[0] if matches else None
