# hipe_ner — stacked hmBERT NER for HIPE-style historical NE recognition

Fine-tunes a stacked Transformer architecture for named entity recognition on
historical, OCR'd text, following the CLEF-HIPE shared-task conventions, and
applies it to this project's own OCR-D pipeline output
(`OCR/ocr-project/assembled/body.txt`).

## Architecture

```
tokens → hmBERT (dbmdz/bert-base-historic-multilingual-cased)
       → [N extra Transformer encoder layers]   "noise-adaptation stack"
       → dropout → linear → per-token coarse NE tag
```

- **Base**: `dbmdz/bert-base-historic-multilingual-cased` ("hmBERT") — BERT-base
  pretrained specifically on digitized historical newspapers/books in German,
  French, English (plus Finnish/Swedish), which is why it's the right
  multilingual base for de/fr/en historical text rather than plain
  multilingual BERT.
- **Noise-adaptation stack**: `model.num_extra_layers` (default 2) additional
  `nn.TransformerEncoderLayer`s on top of hmBERT's last hidden state, applied
  residually (`LayerNorm(x + adapter(x))`). hmBERT's own pretraining wasn't
  tuned to any one corpus's specific OCR-error distribution; these layers are
  fine-tuned end-to-end on the labeled NER data and are free to absorb
  whatever spelling variation/character noise shows up there. Set
  `num_extra_layers: 0` to get a plain hmBERT+linear-head baseline for
  comparison.
- Training (`src/hipe_ner/train.py`) freezes hmBERT for
  `training.freeze_base_epochs` epochs so the randomly-initialized stack gets
  a head start before its gradients start moving the pretrained encoder.
- Coarse, flat BIO tagging only (no fine subtype, no entity components, no
  nesting, no metonymic/literal sense, no entity linking) — this was a
  deliberate simplification versus the full HIPE tag scheme, both for
  trainability and because your own corpus has no gold labels to sanity-check
  the fancier outputs against.

## The label-ontology problem (read this before training)

The HIPE de/fr/en subsets do **not** share one coarse label set:

| dataset | languages | coarse types |
|---|---|---|
| hipe2020 | de, fr | pers, loc, org, prod, time |
| letemps | fr | pers, loc, org |
| newseye | de, fr | PER, LOC, ORG, HumanProd |
| sonar | de | pers, loc, org *(dev/test only — no train split, it's an out-of-domain eval set)* |
| ajmc | de, fr, en | pers, loc, date, work, object, scope *(classics-commentary domain)* |
| topres19th | en | LOC, BUILDING, STREET *(place-name domain only, no PER/ORG at all)* |

`src/hipe_ner/label_scheme.py` canonicalizes case/abbreviation variants of the
same concept (`pers`/`PER` → `PER`, `loc`/`LOC` → `LOC`, etc.) but does **not**
guess equivalences between genuinely different ontologies (e.g. hipe2020's
`prod` and newseye's `HumanProd` are kept as separate classes). The default
config (`configs/default.yaml`) takes the **union** of every de/fr/en coarse
corpus, giving a 24-type (+`O`) joint label space. Real consequence: a
sentence from `topres19th` has no PER/ORG annotations at all, so the model is
trained to treat any person/org mentions in it as `O` — that's a real
label-noise source, not a bug, and something to keep in mind when reading
per-type dev/test scores.

If you'd rather have a cleaner, single-domain 5-way scheme (closer to what
you'd actually want for the art-history corpus: `PER/LOC/ORG/PROD/TIME`), trim
`data.datasets` in the config to `hipe2020(de,fr) + letemps(fr) + sonar(de,
eval-only)` and drop English fine-tuning entirely — hmBERT's own multilingual
pretraining gives you some zero-shot cross-lingual transfer to English
without needing an English-specific ontology mismatch. Both are legitimate
choices; the default just maximizes multilingual coverage per your original
ask.

## Setup

```bash
cd hipe_ner
conda activate ner                 # existing project env; has torch/transformers/flair/gliner already
pip install -e .                    # adds datasets, seqeval, segtok, pyyaml if missing
scripts/download_hipe_data.sh       # clones HIPE-2022-data (~200MB) into data/raw/, gitignored
python scripts/build_dataset.py --config configs/default.yaml
```

`build_dataset.py` parses every configured TSV, canonicalizes tags, and
writes `data/cache/{train,dev,test}.jsonl` + `label_map.json` +
`manifest.json` (per-corpus sentence/token counts and which dataset/lang/split
combinations were actually found — several are expected to be missing, e.g.
hipe2020 has no English subset).

Already verified working on this machine: `scripts/build_dataset.py` against
the real cloned data (60,159 / 7,679 / 14,400 train/dev/test sentences, 25
canonical labels), and `scripts/smoke_test.py` (real hmBERT weights, real
tokenized batches, a few optimizer steps) — loss dropped from 3.39 to 0.28
over 8 steps on this Mac's MPS backend, with confirmed gradient flow through
both hmBERT and the noise-adaptation stack. That is *not* a trained model —
it's proof the architecture and data pipeline are wired correctly before you
spend GPU-hours on a real run.

## Training (do this on a GPU, not this laptop)

hmBERT-base (110M params) + the extra layers (~8M for the default 2-layer
stack) fine-tuned over ~90k sentences for several epochs is a multi-hour job
even on MPS, and MPS lacks some of the memory/throughput headroom a real GPU
gives you for this size of run. `jobs/euler_train.sbatch` is a skeleton for
ETH Euler:

```bash
sbatch jobs/euler_train.sbatch
```

Adjust the `module load` line for whatever software stack is current on Euler
when you run this (`module avail python`, `module avail cuda`), and check
`--gpus`/`--gres`/`--time` against your allocation. It runs
`build_dataset.py` then `hipe_ner.train`, writing the best-dev-F1 checkpoint
to `checkpoints/stacked-hmbert-hipe/best/`.

Key knobs in `configs/default.yaml`: `num_extra_layers`, `freeze_base_epochs`,
`lr_base` vs `lr_head` (separate learning rates — the pretrained encoder
needs a much smaller LR than the freshly initialized stack+classifier),
`batch_size`, `max_length`.

## Evaluation

```bash
python -m hipe_ner.evaluate --checkpoint checkpoints/stacked-hmbert-hipe/best --split test
```

Reports seqeval precision/recall/F1 per type. Note seqeval does exact-boundary
matching; the official CLEF-HIPE-scorer (https://github.com/hipe-eval/HIPE-scorer)
uses fuzzy boundary matching and reports the numbers comparable to the shared
task leaderboard — worth running separately if you want an apples-to-apples
comparison to published HIPE results.

## Applying it to your own OCR-D output

```bash
python -m hipe_ner.infer \
    --checkpoint checkpoints/stacked-hmbert-hipe/best \
    --input ../OCR/ocr-project/assembled/body.txt \
    --output ../ner_outputs/hipe_entities.csv
```

Segments the raw text into sentences (`segtok`, same library the existing
`ner_workflow.ipynb` uses), tags each sentence, and writes a CSV with the same
columns as `ner_outputs/entities.csv` (`sentence_id, text, label, score,
start, end, sentence`), utf-8-sig encoded, so predictions line up directly
next to the existing flair/GLiNER output for comparison. This path is
smoke-tested against the real `body.txt` already (ran cleanly end-to-end on
an untrained model — 3,323 raw candidate spans over 335 sentences — to
confirm the sentence segmentation → tagging → BIO decoding → CSV pipeline is
correct; the *predictions* themselves are meaningless until you train).

## Extensions worth trying later

- **CRF on top of the classifier** for globally-consistent BIO decoding
  (`torchcrf` or similar) instead of independent per-token softmax.
- **Character-level features** (a small char-CNN or char-embedding merged in
  before the noise-adaptation stack) — more directly targets OCR
  character-substitution noise than stacking more Transformer layers alone.
- **Official HIPE-scorer** integration for leaderboard-comparable metrics.
- **Synthetic OCR-noise augmentation** (character substitution/deletion on
  clean HIPE text) if you want the noise-adaptation stack to see noise
  patterns closer to your own scanner/Tesseract output rather than only
  whatever noise happens to already be in the HIPE gold text.
