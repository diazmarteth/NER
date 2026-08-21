#!/usr/bin/env python
"""Evaluate a saved checkpoint on the cached dev/test split with seqeval.

    python -m hipe_ner.evaluate --checkpoint checkpoints/stacked-hmbert-hipe/best --split test

For an official-style comparison against published HIPE numbers, feed this
same checkpoint's predictions through the CLEF-HIPE-scorer instead (it scores
NERC-Coarse with fuzzy boundary matching, which seqeval does not do) — see
README.md.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hipe_ner.dataset import build_tokenized_dataset
from hipe_ner.model import StackedHistBertForTokenClassification
from hipe_ner.train import evaluate, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = Path(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = StackedHistBertForTokenClassification.from_pretrained(ckpt, map_location=device).to(device)

    import json
    labels = json.loads((ckpt / "label_map.json").read_text())["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    ds = build_tokenized_dataset(
        Path(args.cache_dir) / f"{args.split}.jsonl", tokenizer, label2id, args.max_length
    )
    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator)

    metrics = evaluate(model, loader, device, id2label)
    print(f"{args.split} loss {metrics['loss']:.4f} f1 {metrics['f1']:.4f}")
    print(metrics["report"])


if __name__ == "__main__":
    main()
