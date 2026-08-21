#!/usr/bin/env python
"""Fast sanity check, NOT a real training run: loads real hmBERT weights,
builds the stacked model, and runs a handful of optimizer steps over a small
slice of the real cached training data to confirm shapes line up and loss
decreases -- on whatever device is available (this is meant to be cheap
enough for a laptop; full training belongs on a GPU, see jobs/euler_train.sbatch).
"""
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hipe_ner.dataset import build_tokenized_dataset
from hipe_ner.label_scheme import load_label_map
from hipe_ner.model import StackedHistBertConfig, StackedHistBertForTokenClassification
from hipe_ner.train import pick_device, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((PROJECT_ROOT / "configs/default.yaml").read_text())
    mcfg = cfg["model"]
    set_seed(0)
    device = pick_device("auto")
    print(f"device: {device}")

    cache_dir = PROJECT_ROOT / cfg["data"]["cache_dir"]
    labels = load_label_map(cache_dir / "label_map.json")
    label2id = {l: i for i, l in enumerate(labels)}
    print(f"{len(labels)} labels")

    tokenizer = AutoTokenizer.from_pretrained(mcfg["base_model"])
    t0 = time.time()
    ds = build_tokenized_dataset(cache_dir / "train.jsonl", tokenizer, label2id, mcfg["max_length"])
    ds = ds.select(range(64))  # tiny slice, just to exercise the pipeline
    print(f"tokenized 64 examples in {time.time() - t0:.1f}s")

    model_cfg = StackedHistBertConfig(
        base_model=mcfg["base_model"], num_labels=len(labels), num_extra_layers=mcfg["num_extra_layers"],
        num_heads=mcfg["num_heads"], ffn_size=mcfg["ffn_size"], dropout=mcfg["dropout"],
        classifier_dropout=mcfg["classifier_dropout"],
    )
    t0 = time.time()
    model = StackedHistBertForTokenClassification(model_cfg).to(device)
    print(f"built model with real hmBERT weights in {time.time() - t0:.1f}s, "
          f"{sum(p.numel() for p in model.parameters()):,} params "
          f"({sum(p.numel() for p in model.noise_adapter.parameters()):,} in the noise-adaptation stack)")

    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    losses = []
    enc_grad = adapter_grad = False
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        if step == 0:
            # gradient sanity, checked before zero_grad() clears .grad: both
            # the pretrained encoder and the new stack should receive
            # gradients (encoder is trainable here since we never called
            # set_encoder_trainable(False)).
            enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
            adapter_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.noise_adapter.parameters())
        optimizer.step()
        optimizer.zero_grad()
        losses.append(out.loss.item())
        print(f"step {step} loss {out.loss.item():.4f} logits {tuple(out.logits.shape)}")
    print(f"{len(losses)} steps in {time.time() - t0:.1f}s on {device}")
    print(f"loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")
    print(f"encoder received gradients: {enc_grad}, noise-adaptation stack received gradients: {adapter_grad}")
    assert enc_grad and adapter_grad, "gradient flow broken somewhere in the stack"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
