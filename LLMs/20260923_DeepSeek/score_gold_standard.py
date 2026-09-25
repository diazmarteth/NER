"""
score_gold_standard.py — Entity + relationship scoring for the ROCOCO NER/RE
pipeline (rococo_ner_pipeline_20260923_relation_vocab_demo.py) against a
hand-annotated gold standard.

Why strict AND relaxed matching (see "NER Annotation Guidelines.md", the
span-boundary discussion): the pipeline extracts a deduplicated list of
entity-name STRINGS per category, with no character offsets and no
prompt-enforced span convention (article/preposition handling, adjective
dropping, compound splitting). Scoring purely on exact string equality
conflates two different failure classes — the model choosing the wrong
CATEGORY for a correctly-identified entity, vs. the model and the gold
annotator drawing the entity's boundary differently while agreeing on what
it is and what category it belongs to. Reporting both numbers keeps those
apart:
  - strict  : casefolded, whitespace-normalized exact string match
  - relaxed : strict + strips one leading article/preposition + quote marks

Gold JSON schema (one file per document — building a converter from your
annotation tool's native export, e.g. INCEpTION UIMA CAS XMI, into this
format is a separate, later step; this format is the scoring contract):

{
  "doc_id": "St Fiden_p1-5",
  "entities": {
    "persons": ["Johannes Murer", ...],
    "places": [...],
    "physical_human_made_thing": [...],
    "visual_item": [...],
    "iconographic_subject": [...],
    "dates": [...],
    "groups": [...]
  },
  "relationships": [
    {"subject": "Johannes Murer", "subject_type": "person",
     "relation_type": "created", "object": "Hochaltar",
     "object_type": "physical_human_made_thing"},
    ...
  ]
}

Prediction JSON: the pipeline's own result_*.json output — same "entities"
and "relationships" shape; extra per-relationship fields it adds (valid,
invalid_reason, coverage, db_schema_match, cidoc_property, evidence, page)
are ignored here, since Section 8.2 of the guideline scopes gold/scored
relationships to the (subject, subject_type, relation_type, object,
object_type) triple only.

Usage:
    # Single document pair
    python score_gold_standard.py --gold gold/st_fiden.gold.json --pred results/result_St_Fiden.json

    # Batch: every gold/<doc>.gold.json matched against results/result_<doc>.json
    python score_gold_standard.py --gold-dir gold/ --pred-dir results/ --report report.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ENTITY_CATEGORIES = [
    "persons", "places", "physical_human_made_thing",
    "visual_item", "iconographic_subject", "dates", "groups",
]

# Category key -> the subject_type/object_type string relationships use for it.
CATEGORY_TO_RELATION_TYPE = {
    "persons": "person",
    "places": "place",
    "physical_human_made_thing": "physical_human_made_thing",
    "visual_item": "visual_item",
    "iconographic_subject": "iconographic_subject",
    "dates": "date",
    "groups": "group",
}

_ARTICLES_PREPOSITIONS = {
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einer", "eines", "einem", "einen",
    "im", "in", "an", "am", "auf", "von", "zu", "zur", "zum", "vom",
}
_QUOTE_CHARS = "\"'„“”»«"


def normalize_strict(s: str) -> str:
    """Whitespace-normalized, casefolded exact-string basis. Deliberately does
    NOT strip articles/prepositions or punctuation — a strict-match miss there
    is exactly the span-boundary signal this scorer is meant to surface."""
    return re.sub(r"\s+", " ", (s or "")).strip().casefold()


def normalize_relaxed(s: str) -> str:
    """Tolerant of span-boundary convention differences: also strips one
    leading article/preposition token and surrounding quote/punctuation, so
    e.g. model output "der Hochaltar" still matches gold "Hochaltar"."""
    s = normalize_strict(s).strip(_QUOTE_CHARS + " ")
    tokens = s.split(" ")
    if tokens and tokens[0] in _ARTICLES_PREPOSITIONS:
        tokens = tokens[1:]
    return " ".join(tokens).strip()


def _prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Entity scoring — per-category SET comparison (the pipeline dedupes entity
# mentions into a flat string list with no offsets, so this is set-based
# scoring, not span/offset scoring).
# ---------------------------------------------------------------------------
def score_entities(gold: Dict, pred: Dict, normalize) -> Dict:
    report = {}
    tp_total = fp_total = fn_total = 0
    for cat in ENTITY_CATEGORIES:
        gold_set = {normalize(s) for s in gold.get(cat, []) if normalize(s)}
        pred_set = {normalize(s) for s in pred.get(cat, []) if normalize(s)}
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        entry = {"tp": tp, "fp": fp, "fn": fn, **_prf1(tp, fp, fn),
                 "false_positives": sorted(pred_set - gold_set),
                 "false_negatives": sorted(gold_set - pred_set)}
        report[cat] = entry
        tp_total += tp
        fp_total += fp
        fn_total += fn

    macro_f1 = (sum(v["f1"] for v in report.values()) / len(report)) if report else 0.0
    report["_overall"] = {"tp": tp_total, "fp": fp_total, "fn": fn_total,
                           **_prf1(tp_total, fp_total, fn_total), "macro_f1": macro_f1}
    return report


# ---------------------------------------------------------------------------
# Relationship scoring — triple-level SET comparison on
# (subject, subject_type, relation_type, object, object_type).
# ---------------------------------------------------------------------------
def _triple_key(r: Dict, normalize) -> Tuple[str, str, str, str, str]:
    return (
        normalize(r.get("subject", "")),
        (r.get("subject_type") or "").strip().lower(),
        (r.get("relation_type") or "").strip().lower(),
        normalize(r.get("object", "")),
        (r.get("object_type") or "").strip().lower(),
    )


def _link_key(r: Dict, normalize) -> Tuple[str, str, str, str]:
    """Subject/object identity + type, WITHOUT relation_type — used to separate
    entity-linking failures (wrong/missing subject or object) from relation-
    classification failures (right entities, wrong relation_type)."""
    return (
        normalize(r.get("subject", "")),
        (r.get("subject_type") or "").strip().lower(),
        normalize(r.get("object", "")),
        (r.get("object_type") or "").strip().lower(),
    )


def score_relationships(gold_rels: List[Dict], pred_rels: List[Dict], normalize) -> Dict:
    gold_by_type: Dict[str, list] = {}
    pred_by_type: Dict[str, list] = {}
    for r in gold_rels:
        gold_by_type.setdefault((r.get("relation_type") or "").strip().lower(), []).append(r)
    for r in pred_rels:
        pred_by_type.setdefault((r.get("relation_type") or "").strip().lower(), []).append(r)

    report = {}
    tp_total = fp_total = fn_total = 0
    for rel_type in sorted(set(gold_by_type) | set(pred_by_type)):
        gold_keys = {_triple_key(r, normalize) for r in gold_by_type.get(rel_type, [])}
        pred_keys = {_triple_key(r, normalize) for r in pred_by_type.get(rel_type, [])}
        tp = len(gold_keys & pred_keys)
        fp = len(pred_keys - gold_keys)
        fn = len(gold_keys - pred_keys)
        report[rel_type] = {"tp": tp, "fp": fp, "fn": fn, **_prf1(tp, fp, fn)}
        tp_total += tp
        fp_total += fp
        fn_total += fn

    macro_f1 = (sum(v["f1"] for v in report.values()) / len(report)) if report else 0.0
    report["_overall"] = {"tp": tp_total, "fp": fp_total, "fn": fn_total,
                           **_prf1(tp_total, fp_total, fn_total), "macro_f1": macro_f1}
    return report


def relation_type_confusions(gold_rels: List[Dict], pred_rels: List[Dict], normalize) -> List[Dict]:
    """Diagnostic, not a P/R/F1 metric: for every gold triple whose subject
    and object the model DID find (right entities, right types), report what
    relation_type the model assigned instead, if different from gold. This is
    the breakdown the evaluation plan calls for — separating "found the right
    entities but classified the relation wrong" from "missed the entities
    entirely", which the strict/relaxed triple F1 above cannot distinguish on
    its own."""
    pred_by_link: Dict[tuple, list] = {}
    for r in pred_rels:
        pred_by_link.setdefault(_link_key(r, normalize), []).append(
            (r.get("relation_type") or "").strip().lower()
        )

    confusions = []
    for r in gold_rels:
        link = _link_key(r, normalize)
        gold_rt = (r.get("relation_type") or "").strip().lower()
        if link in pred_by_link and gold_rt not in pred_by_link[link]:
            confusions.append({
                "subject": r.get("subject"), "object": r.get("object"),
                "gold_relation_type": gold_rt,
                "predicted_relation_type(s)": pred_by_link[link],
            })
    return confusions


# ---------------------------------------------------------------------------
# I/O and aggregation across multiple documents
# ---------------------------------------------------------------------------
def _load(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _merge_counts(agg: Dict, doc_report: Dict, keys: List[str]) -> None:
    for key in keys:
        entry = doc_report.get(key)
        if entry is None:
            continue
        target = agg.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
        target["tp"] += entry["tp"]
        target["fp"] += entry["fp"]
        target["fn"] += entry["fn"]


def score_document(gold: Dict, pred: Dict) -> Dict:
    return {
        "entities_strict": score_entities(gold.get("entities", {}), pred.get("entities", {}), normalize_strict),
        "entities_relaxed": score_entities(gold.get("entities", {}), pred.get("entities", {}), normalize_relaxed),
        "relationships_strict": score_relationships(gold.get("relationships", []), pred.get("relationships", []), normalize_strict),
        "relationships_relaxed": score_relationships(gold.get("relationships", []), pred.get("relationships", []), normalize_relaxed),
        "relation_type_confusions": relation_type_confusions(gold.get("relationships", []), pred.get("relationships", []), normalize_relaxed),
    }


def aggregate(per_doc_reports: Dict[str, Dict]) -> Dict:
    """Micro-aggregate tp/fp/fn across all documents, per category / relation_type,
    for each of the four (entities|relationships) x (strict|relaxed) views."""
    agg = {
        "entities_strict": {}, "entities_relaxed": {},
        "relationships_strict": {}, "relationships_relaxed": {},
    }
    for report in per_doc_reports.values():
        _merge_counts(agg["entities_strict"], report["entities_strict"], ENTITY_CATEGORIES + ["_overall"])
        _merge_counts(agg["entities_relaxed"], report["entities_relaxed"], ENTITY_CATEGORIES + ["_overall"])
        rel_types = sorted({k for r in (report["relationships_strict"], report["relationships_relaxed"]) for k in r})
        _merge_counts(agg["relationships_strict"], report["relationships_strict"], rel_types)
        _merge_counts(agg["relationships_relaxed"], report["relationships_relaxed"], rel_types)

    for view in agg.values():
        for key, counts in view.items():
            counts.update(_prf1(counts["tp"], counts["fp"], counts["fn"]))
    return agg


def _print_view(title: str, view: Dict) -> None:
    print(f"\n--- {title} ---")
    overall = view.get("_overall", {})
    print(f"  micro P={overall.get('precision', 0):.3f} R={overall.get('recall', 0):.3f} "
          f"F1={overall.get('f1', 0):.3f}  (tp={overall.get('tp', 0)} fp={overall.get('fp', 0)} fn={overall.get('fn', 0)})")
    for key, counts in sorted(view.items()):
        if key == "_overall":
            continue
        print(f"    {key:28s} P={counts['precision']:.3f} R={counts['recall']:.3f} F1={counts['f1']:.3f} "
              f"(tp={counts['tp']} fp={counts['fp']} fn={counts['fn']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, help="Single gold JSON file")
    parser.add_argument("--pred", type=Path, help="Single prediction JSON file (pipeline result_*.json)")
    parser.add_argument("--gold-dir", type=Path, help="Directory of <doc_id>.gold.json files")
    parser.add_argument("--pred-dir", type=Path, help="Directory of result_<doc_id>.json files")
    parser.add_argument("--report", type=Path, help="Write the full per-document + aggregate report as JSON here")
    args = parser.parse_args()

    pairs: Dict[str, Tuple[Path, Path]] = {}
    if args.gold and args.pred:
        pairs[args.gold.stem] = (args.gold, args.pred)
    elif args.gold_dir and args.pred_dir:
        for gold_path in sorted(args.gold_dir.glob("*.gold.json")):
            doc_id = gold_path.name[: -len(".gold.json")]
            pred_path = args.pred_dir / f"result_{doc_id}.json"
            if not pred_path.exists():
                print(f"⚠️  No prediction file for '{doc_id}' — expected {pred_path}, skipping", file=sys.stderr)
                continue
            pairs[doc_id] = (gold_path, pred_path)
    else:
        parser.error("Pass either --gold/--pred for one document, or --gold-dir/--pred-dir for a batch")

    if not pairs:
        parser.error("No gold/prediction pairs found")

    per_doc_reports = {}
    for doc_id, (gold_path, pred_path) in pairs.items():
        gold = _load(gold_path)
        pred = _load(pred_path)
        per_doc_reports[doc_id] = score_document(gold, pred)
        print(f"\n=== {doc_id} ===")
        for view_key, title in [("entities_strict", "Entities (strict)"),
                                 ("entities_relaxed", "Entities (relaxed)"),
                                 ("relationships_strict", "Relationships (strict)"),
                                 ("relationships_relaxed", "Relationships (relaxed)")]:
            _print_view(title, per_doc_reports[doc_id][view_key])
        confusions = per_doc_reports[doc_id]["relation_type_confusions"]
        if confusions:
            print(f"\n  Relation-type confusions (entities linked correctly, wrong relation_type): {len(confusions)}")
            for c in confusions[:10]:
                print(f"    {c['subject']} -> {c['object']}: gold={c['gold_relation_type']!r} "
                      f"predicted={c['predicted_relation_type(s)']}")

    if len(pairs) > 1:
        agg = aggregate(per_doc_reports)
        print("\n\n########## AGGREGATE ACROSS ALL DOCUMENTS ##########")
        for view_key, title in [("entities_strict", "Entities (strict)"),
                                 ("entities_relaxed", "Entities (relaxed)"),
                                 ("relationships_strict", "Relationships (strict)"),
                                 ("relationships_relaxed", "Relationships (relaxed)")]:
            _print_view(title, agg[view_key])
    else:
        agg = None

    if args.report:
        out = {"per_document": per_doc_reports, "aggregate": agg}
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nFull report written to: {args.report}")


if __name__ == "__main__":
    main()
