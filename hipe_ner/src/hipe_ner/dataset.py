"""Tokenization and subword-label alignment for the cached HIPE JSONL files."""
from __future__ import annotations

from pathlib import Path

from datasets import Dataset


def load_jsonl_dataset(path: str | Path) -> Dataset:
    return Dataset.from_json(str(path))


def tokenize_and_align(examples, tokenizer, label2id: dict, max_length: int):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word_id:
                label_ids.append(label2id[tags[wid]])
            else:
                # subsequent subwords of the same word: don't score them
                label_ids.append(-100)
            prev_word_id = wid
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def build_tokenized_dataset(jsonl_path, tokenizer, label2id: dict, max_length: int) -> Dataset:
    ds = load_jsonl_dataset(jsonl_path)
    return ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer, label2id, max_length),
        batched=True,
        remove_columns=ds.column_names,
        desc=f"tokenizing {Path(jsonl_path).name}",
    )
