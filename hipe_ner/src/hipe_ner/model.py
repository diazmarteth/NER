"""Stacked-BERT architecture: hmBERT encoder + a noise-adaptation Transformer
stack + a token-classification head.

    tokens -> hmBERT (dbmdz/bert-base-historic-multilingual-cased)
           -> [N extra Transformer encoder layers]   ("noise-adaptation stack")
           -> dropout -> linear -> per-token coarse NE logits

hmBERT was pretrained on clean(ish) digitized newspaper/book text, not on
the specific OCR-error distribution of any one downstream corpus. The extra
layers sit after the pretrained encoder and are fine-tuned only on the task,
so they're free to learn corrections for whatever spelling variation and
character-level noise show up in the *labeled* NER data (and, by extension,
generalize somewhat to similar noise in your own OCR-D output). The stack is
residual (input added back after the layers, then layer-normed) so early
training with a randomly initialized stack doesn't have to overwrite hmBERT's
good pretrained representations — that's also why `freeze_base_epochs` exists
in training: give the stack a head start before hmBERT itself starts moving.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput


@dataclass
class StackedHistBertConfig:
    base_model: str = "dbmdz/bert-base-historic-multilingual-cased"
    num_labels: int = 2
    num_extra_layers: int = 2
    num_heads: int = 8
    ffn_size: int = 1024
    dropout: float = 0.1
    classifier_dropout: float = 0.1


class NoiseAdaptationStack(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, ffn_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ffn_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # nn.TransformerEncoderLayer's padding mask convention is the inverse
        # of HF's attention_mask: True means "ignore this position".
        padding_mask = attention_mask == 0
        x = hidden_states
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return self.final_norm(x + hidden_states)


class StackedHistBertForTokenClassification(nn.Module):
    def __init__(self, cfg: StackedHistBertConfig):
        super().__init__()
        self.cfg = cfg
        self.hf_config = AutoConfig.from_pretrained(cfg.base_model)
        self.encoder = AutoModel.from_pretrained(cfg.base_model)
        hidden_size = self.hf_config.hidden_size

        self.noise_adapter = (
            NoiseAdaptationStack(hidden_size, cfg.num_extra_layers, cfg.num_heads, cfg.ffn_size, cfg.dropout)
            if cfg.num_extra_layers > 0
            else None
        )
        self.dropout = nn.Dropout(cfg.classifier_dropout)
        self.classifier = nn.Linear(hidden_size, cfg.num_labels)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        hidden = outputs.last_hidden_state
        if self.noise_adapter is not None:
            hidden = self.noise_adapter(hidden, attention_mask)
        logits = self.classifier(self.dropout(hidden))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.cfg.num_labels), labels.view(-1), ignore_index=-100
            )
        return TokenClassifierOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        (save_dir / "stacked_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def from_pretrained(cls, save_dir: str | Path, map_location=None) -> "StackedHistBertForTokenClassification":
        save_dir = Path(save_dir)
        cfg = StackedHistBertConfig(**json.loads((save_dir / "stacked_config.json").read_text()))
        model = cls(cfg)
        state_dict = torch.load(save_dir / "model.pt", map_location=map_location)
        model.load_state_dict(state_dict)
        return model
