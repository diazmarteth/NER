#!/usr/bin/env python
"""Parse the configured HIPE TSVs into cached JSONL + a shared label map.

Usage:
    python scripts/build_dataset.py --config configs/default.yaml

Writes, under data.cache_dir:
    train.jsonl, dev.jsonl, test.jsonl   (each line: {"tokens": [...], "ner_tags": [...]})
    label_map.json                        (canonical label list, derived from train only)
    manifest.json                         (per dataset/lang/split token+sentence counts)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hipe_ner.label_scheme import build_label_list, canonicalize_tag, save_label_map
from hipe_ner.tsv_io import iter_hipe_files, read_hipe_tsv


def load_split(hipe_root, datasets_cfg, split):
    sentences = []
    manifest = []
    for entry in datasets_cfg:
        ds, lang = entry["dataset"], entry["lang"]
        path = iter_hipe_files(hipe_root, ds, lang, split)
        if path is None:
            manifest.append({"dataset": ds, "lang": lang, "split": split, "found": False})
            continue
        sents = read_hipe_tsv(path)
        for s in sents:
            s.tags = [canonicalize_tag(t) for t in s.tags]
        sentences.extend(sents)
        manifest.append({
            "dataset": ds, "lang": lang, "split": split, "found": True,
            "sentences": len(sents), "tokens": sum(len(s.tokens) for s in sents),
        })
    return sentences, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(args.config).read_text())["data"]
    hipe_root = project_root / cfg["hipe_root"]
    cache_dir = project_root / cfg["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    full_manifest = []
    train_sentences = None
    for split in ("train", "dev", "test"):
        sentences, manifest = load_split(hipe_root, cfg["datasets"], split)
        full_manifest.extend(manifest)
        if split == "train":
            train_sentences = sentences
        out_path = cache_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for s in sentences:
                f.write(json.dumps({"tokens": s.tokens, "ner_tags": s.tags}, ensure_ascii=False) + "\n")
        print(f"[{split}] {len(sentences)} sentences -> {out_path}")

    labels = build_label_list(s.tags for s in train_sentences)
    save_label_map(labels, cache_dir / "label_map.json")
    print(f"label set ({len(labels)}): {labels}")

    label_counts = Counter(t for s in train_sentences for t in s.tags)
    (cache_dir / "manifest.json").write_text(
        json.dumps({"files": full_manifest, "train_label_counts": label_counts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    missing = [m for m in full_manifest if not m["found"]]
    if missing:
        print(f"NOTE: {len(missing)} dataset/lang/split combinations not found (expected — not every "
              f"dataset has every language or a released test split): {missing}")


if __name__ == "__main__":
    main()
