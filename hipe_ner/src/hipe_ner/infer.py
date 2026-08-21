#!/usr/bin/env python
"""Apply a trained checkpoint to raw text — in particular the assembled
plaintext coming out of the OCR-D pipeline (OCR/ocr-project/assembled/body.txt).

    python -m hipe_ner.infer --checkpoint checkpoints/stacked-hmbert-hipe/best \
        --input ../OCR/ocr-project/assembled/body.txt --output ner_outputs/hipe_entities.csv

Output CSV columns match ner_outputs/entities.csv (from the flair/GLiNER
notebook workflow) so predictions are directly comparable:
    sentence_id, text, label, score, start, end, sentence
Written utf-8-sig, per this project's existing convention (plain utf-8 CSVs
get their umlauts garbled in Excel).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from segtok.segmenter import split_multi
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hipe_ner.model import StackedHistBertForTokenClassification
from hipe_ner.train import pick_device


def decode_bio_spans(tokens: list[str], tags: list[str], offsets: list[tuple[int, int]]):
    """BIO tags + character offsets (relative to the sentence) -> entity spans."""
    spans = []
    cur_type, cur_start, cur_end = None, None, None
    for tag, (start, end) in zip(tags, offsets):
        if tag.startswith("B-"):
            if cur_type is not None:
                spans.append((cur_type, cur_start, cur_end))
            cur_type, cur_start, cur_end = tag[2:], start, end
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur_end = end
        else:
            if cur_type is not None:
                spans.append((cur_type, cur_start, cur_end))
            cur_type, cur_start, cur_end = None, None, None
    if cur_type is not None:
        spans.append((cur_type, cur_start, cur_end))
    return spans


@torch.no_grad()
def predict_sentence(sentence: str, model, tokenizer, id2label, device, max_length):
    enc = tokenizer(sentence, return_offsets_mapping=True, truncation=True, max_length=max_length, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits[0]
    probs = logits.softmax(-1)
    pred_ids = probs.argmax(-1).tolist()
    scores = probs.max(-1).values.tolist()

    tags, kept_offsets, tok_scores = [], [], []
    for pid, score, (start, end) in zip(pred_ids, scores, offsets):
        if start == end:  # special tokens ([CLS]/[SEP]/padding) have empty offsets
            continue
        tags.append(id2label[pid])
        kept_offsets.append((start, end))
        tok_scores.append(score)

    spans = decode_bio_spans([], tags, kept_offsets)
    # attach a score per span: mean of its tokens' confidence
    results = []
    for etype, start, end in spans:
        span_scores = [s for s, (a, b) in zip(tok_scores, kept_offsets) if a >= start and b <= end]
        results.append({
            "text": sentence[start:end], "label": etype,
            "score": round(sum(span_scores) / max(len(span_scores), 1), 4),
            "start": start, "end": end,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="plaintext file, e.g. assembled/body.txt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = Path(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = StackedHistBertForTokenClassification.from_pretrained(ckpt, map_location=device).to(device).eval()
    labels = json.loads((ckpt / "label_map.json").read_text())["labels"]
    id2label = {i: l for i, l in enumerate(labels)}

    text = Path(args.input).read_text(encoding="utf-8")
    sentences = [s for s in split_multi(text) if s.strip()]

    rows = []
    for sid, sentence in enumerate(sentences, start=1):
        for ent in predict_sentence(sentence, model, tokenizer, id2label, device, args.max_length):
            rows.append({
                "sentence_id": sid, "text": ent["text"], "label": ent["label"], "score": ent["score"],
                "start": ent["start"], "end": ent["end"], "sentence": sentence,
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence_id", "text", "label", "score", "start", "end", "sentence"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} entity mentions over {len(sentences)} sentences -> {out_path}")


if __name__ == "__main__":
    main()
