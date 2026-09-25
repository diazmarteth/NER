#!/usr/bin/env python
"""Fine-tune the stacked hmBERT model on the cached HIPE coarse-NE data.

    python -m hipe_ner.train --config configs/default.yaml

Run scripts/build_dataset.py first to produce data/cache/{train,dev,test}.jsonl
and label_map.json. This script is meant to be run where there's a real GPU
(e.g. ETH Euler) for the full config; on a laptop, drop epochs/dataset size
for a smoke test — see scripts/smoke_test.py for a pre-built tiny one.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hipe_ner.dataset import build_tokenized_dataset
from hipe_ner.label_scheme import load_label_map
from hipe_ner.model import StackedHistBertConfig, StackedHistBertForTokenClassification


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides:
        key, value = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
        node[parts[-1]] = value
    return cfg


@torch.no_grad()
def evaluate(model, loader, device, id2label) -> dict:
    model.eval()
    all_true, all_pred = [], []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        n_batches += 1
        preds = out.logits.argmax(-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        for pred_seq, label_seq in zip(preds, labels):
            true_tags, pred_tags = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                true_tags.append(id2label[l])
                pred_tags.append(id2label[p])
            all_true.append(true_tags)
            all_pred.append(pred_tags)
    return {
        "loss": total_loss / max(n_batches, 1),
        "f1": f1_score(all_true, all_pred),
        "report": classification_report(all_true, all_pred, digits=3, zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing resume checkpoint and start over")
    ap.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    cfg = load_config(args.config, args.overrides)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["training"]

    set_seed(tcfg["seed"])
    device = pick_device(tcfg["device"])
    print(f"device: {device}")

    cache_dir = project_root / dcfg["cache_dir"]
    labels = load_label_map(cache_dir / "label_map.json")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    print(f"{len(labels)} labels: {labels}")

    tokenizer = AutoTokenizer.from_pretrained(mcfg["base_model"])
    train_ds = build_tokenized_dataset(cache_dir / "train.jsonl", tokenizer, label2id, mcfg["max_length"])
    dev_ds = build_tokenized_dataset(cache_dir / "dev.jsonl", tokenizer, label2id, mcfg["max_length"])

    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_ds, batch_size=tcfg["eval_batch_size"], shuffle=False, collate_fn=collator)

    model_cfg = StackedHistBertConfig(
        base_model=mcfg["base_model"],
        num_labels=len(labels),
        num_extra_layers=mcfg["num_extra_layers"],
        num_heads=mcfg["num_heads"],
        ffn_size=mcfg["ffn_size"],
        dropout=mcfg["dropout"],
        classifier_dropout=mcfg["classifier_dropout"],
    )
    model = StackedHistBertForTokenClassification(model_cfg).to(device)

    freeze_epochs = tcfg["freeze_base_epochs"]
    model.set_encoder_trainable(freeze_epochs == 0)

    head_params = list(model.classifier.parameters())
    if model.noise_adapter is not None:
        head_params += list(model.noise_adapter.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": tcfg["lr_base"]},
        {"params": head_params, "lr": tcfg["lr_head"]},
    ], weight_decay=tcfg["weight_decay"])

    total_steps = len(train_loader) * tcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * tcfg["warmup_ratio"]), num_training_steps=total_steps
    )

    out_dir = project_root / tcfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "resume_state.pt"
    best_f1 = -1.0
    step = 0
    start_epoch = 0

    if resume_path.exists() and not args.fresh:
        # weights_only=False: this checkpoint holds optimizer/scheduler state,
        # not just tensors, and we wrote it ourselves, so it's trusted.
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_f1 = state["best_f1"]
        step = state["step"]
        model.set_encoder_trainable(start_epoch >= freeze_epochs)
        print(f"resumed from {resume_path}: epoch {start_epoch}/{tcfg['epochs']}, "
              f"step {step}, best dev f1 so far {best_f1:.4f}")
    elif resume_path.exists() and args.fresh:
        print(f"ignoring existing {resume_path} (--fresh passed), starting over")

    for epoch in range(start_epoch, tcfg["epochs"]):
        if epoch == freeze_epochs:
            print("unfreezing hmBERT encoder")
            model.set_encoder_trainable(True)

        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % tcfg["log_every"] == 0:
                print(f"epoch {epoch} step {step} loss {out.loss.item():.4f}")

        print(f"epoch {epoch} done in {time.time() - t0:.1f}s")
        if tcfg["eval_every_epoch"]:
            metrics = evaluate(model, dev_loader, device, id2label)
            print(f"epoch {epoch} dev loss {metrics['loss']:.4f} f1 {metrics['f1']:.4f}")
            print(metrics["report"])
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                model.save_pretrained(out_dir / "best")
                tokenizer.save_pretrained(out_dir / "best")
                (out_dir / "best" / "label_map.json").write_text(json.dumps({"labels": labels}, indent=2))
                print(f"new best dev f1 {best_f1:.4f} -> saved to {out_dir / 'best'}")

        torch.save({
            "epoch": epoch,
            "step": step,
            "best_f1": best_f1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, resume_path)
        print(f"resume checkpoint saved (epoch {epoch}) -> {resume_path}")

    model.save_pretrained(out_dir / "last")
    tokenizer.save_pretrained(out_dir / "last")
    (out_dir / "last" / "label_map.json").write_text(json.dumps({"labels": labels}, indent=2))
    print(f"training done. best dev f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
