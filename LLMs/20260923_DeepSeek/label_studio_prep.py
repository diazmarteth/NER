"""
label_studio_prep.py — Convert a rococo_ner_pipeline result_*.json into a
Label Studio import package: one task per scanned PDF page, each with the
corrected text, the matching page image, and the pipeline's own extracted
entities/relationships pre-loaded as editable predictions.

Why per-page tasks with an image: the pipeline runs OCR + LLM grammar
correction per page, so page boundaries ([[Page N]] markers) are already
in extracted_text/raw_text. Showing the scanned page image next to the
corrected text lets annotators catch OCR/correction errors instead of
labeling blind. Predictions are pre-filled from the pipeline's own
`entities`/`relationships` output so annotators correct spans instead of
labeling from scratch — they still edit/confirm each one, so the result
stays an independent gold standard for score_gold_standard.py.

Span offsets: the pipeline outputs deduplicated entity name STRINGS with
no character offsets (see score_gold_standard.py's docstring), so this
script locates each string in the page text via literal substring search.
An entity can match more than once per page — every occurrence is
labeled. A relationship is pre-filled as a Label Studio "relation" only
when both its subject and object were found as spans on the SAME page
(relations are intra-task in Label Studio); cross-page relationships are
skipped and reported so you know to add them by hand.

Usage:
    python label_studio_prep.py "results/result_St Fiden_3.json"
    python label_studio_prep.py "results/result_St Fiden_3.json" \\
        --pdf-dir scanned_documents --out-dir label_studio_export --dpi 150

Output (under --out-dir, default label_studio_export/):
    label_config.xml         Label Studio labeling interface (import once
                              per project, in Settings > Labeling Interface)
    tasks.json                Import file: Settings > Import
    images/<doc_id>/pNNN.png  Rendered page images tasks.json points at

Serving the images: Label Studio needs LOCAL FILE SERVING enabled to show
them. Before starting Label Studio:

    export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/absolute/path/to/label_studio_export/images
    label-studio start

tasks.json references images as "/data/local-files/?d=<doc_id>/pNNN.png",
resolved relative to that document root.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pdf2image import convert_from_path

SCRIPT_DIR = Path(__file__).resolve().parent

# Entity-extraction category key -> Label Studio label name.
CATEGORY_TO_LABEL = {
    "persons": "PER",
    "places": "PLACE",
    "physical_human_made_thing": "OBJ",
    "visual_item": "VIS",
    "iconographic_subject": "ICO",
    "dates": "DATE",
    "groups": "GROUP",
}

LABEL_COLORS = {
    "PER": "#e6194B",
    "PLACE": "#3cb44b",
    "OBJ": "#4363d8",
    "VIS": "#f58231",
    "ICO": "#911eb4",
    "DATE": "#42d4f4",
    "GROUP": "#f032e6",
}

# relation_type vocabulary from rococo_ner_pipeline_20260923_relation_vocab_demo.py's
# extract_relationships_llm() prompt / _RELATION_TAXONOMY.
RELATION_TYPES = [
    "created", "attributed_to", "has_geographic_epithet", "has_patron",
    "happened_in_date", "has_lifespan", "was_restored_in", "has_current_location",
    "has_original_location", "worked_in", "born_in", "died_in", "influenced_by",
    "collaborated_with", "married_to", "student_of", "teacher_of", "owned_by",
    "depicts", "decorated_with", "carries_inscription", "part_of",
]

PAGE_MARKER_RE = re.compile(r"\[\[\s*page\s+(\d+)\s*\]\]", re.IGNORECASE)


def split_pages(text: str) -> dict:
    """Split text on [[Page N]] / [[PAGE N]] markers -> {page_num: page_text}."""
    matches = list(PAGE_MARKER_RE.finditer(text))
    if not matches:
        return {1: text.strip()}
    pages = {}
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pages[page_num] = text[start:end].strip()
    return pages


def find_spans(page_text: str, entity: str) -> list:
    entity = entity.strip()
    if not entity:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(entity), page_text)]


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def build_tasks(result: dict, pdf_path: Path, images_dir: Path, dpi: int) -> list:
    doc_id = sanitize(pdf_path.stem)
    extracted_pages = split_pages(result["extracted_text"])
    raw_pages = split_pages(result["raw_text"])
    entities = result.get("entities", {})
    relationships = result.get("relationships", [])

    print(f"Rendering {pdf_path.name} at {dpi} DPI ({len(extracted_pages)} pages expected)...")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    doc_images_dir = images_dir / doc_id
    doc_images_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    total_skipped_relations = 0

    for page_num in sorted(extracted_pages):
        page_text = extracted_pages[page_num]
        raw_text = raw_pages.get(page_num, "")

        img_idx = page_num - 1
        image_rel = None
        if 0 <= img_idx < len(images):
            img_path = doc_images_dir / f"p{page_num:03d}.png"
            images[img_idx].save(img_path)
            image_rel = f"{doc_id}/p{page_num:03d}.png"

        span_ids_by_entity = {}
        ls_results = []
        span_counter = 0

        for category, label in CATEGORY_TO_LABEL.items():
            for entity in entities.get(category, []) or []:
                for (s, e) in find_spans(page_text, entity):
                    span_id = f"p{page_num}_e{span_counter}"
                    span_counter += 1
                    ls_results.append({
                        "id": span_id,
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {"start": s, "end": e, "text": page_text[s:e], "labels": [label]},
                    })
                    span_ids_by_entity.setdefault(entity.strip().lower(), []).append(span_id)

        skipped_here = 0
        for rel in relationships:
            if rel.get("page") != page_num:
                continue
            subj_ids = span_ids_by_entity.get((rel.get("subject") or "").strip().lower())
            obj_ids = span_ids_by_entity.get((rel.get("object") or "").strip().lower())
            if not subj_ids or not obj_ids:
                skipped_here += 1
                continue
            ls_results.append({
                "from_id": subj_ids[0],
                "to_id": obj_ids[0],
                "type": "relation",
                "labels": [rel.get("relation_type", "")],
                "direction": "right",
            })
        total_skipped_relations += skipped_here
        if skipped_here:
            print(f"   page {page_num}: {skipped_here} relationship(s) skipped "
                  f"(subject/object not both found as a span on this page)")

        task = {
            "data": {
                "doc_id": doc_id,
                "page": page_num,
                "text": page_text,
                "raw_text": raw_text,
            },
            "predictions": [{
                "model_version": "rococo-deepseek-pipeline",
                "result": ls_results,
            }],
        }
        if image_rel:
            task["data"]["image"] = f"/data/local-files/?d={image_rel}"
        tasks.append(task)

    if total_skipped_relations:
        print(f"Total relationships skipped (cross-page or entity not spanned): {total_skipped_relations}")
    return tasks


def write_label_config(out_path: Path) -> None:
    labels_xml = "\n".join(
        f'    <Label value="{name}" background="{color}"/>'
        for name, color in LABEL_COLORS.items()
    )
    relations_xml = "\n".join(f'    <Relation value="{r}"/>' for r in RELATION_TYPES)

    config = f"""<View>
  <Header value="$doc_id — page $page"/>
  <View style="display: flex;">
    <View style="flex: 1; padding-right: 10px;">
      <Image name="page_image" value="$image" zoom="true" zoomControl="true" rotateControl="true" width="100%"/>
    </View>
    <View style="flex: 1; max-height: 800px; overflow-y: auto;">
      <Text name="text" value="$text"/>
    </View>
  </View>

  <Labels name="label" toName="text">
{labels_xml}
  </Labels>

  <Relations>
{relations_xml}
  </Relations>

  <Collapse>
    <Panel value="Raw OCR text (uncorrected) — reference only">
      <Text name="raw_text_display" value="$raw_text"/>
    </Panel>
  </Collapse>
</View>
"""
    out_path.write_text(config, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_json", help="Path to a rococo_ner_pipeline result_*.json file")
    parser.add_argument("--pdf-dir", default="scanned_documents",
                         help="Directory containing the source PDF (default: scanned_documents)")
    parser.add_argument("--out-dir", default="label_studio_export",
                         help="Output directory for tasks.json / label_config.xml / images (default: label_studio_export)")
    parser.add_argument("--dpi", type=int, default=150,
                         help="DPI for rendered page images shown in Label Studio (default: 150; the pipeline's own OCR uses 300)")
    args = parser.parse_args()

    result_path = Path(args.result_json)
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    pdf_name = Path(result["source"]["file_path"]).name
    pdf_path = Path(args.pdf_dir) / pdf_name
    if not pdf_path.exists():
        print(f"ERROR: source PDF not found at {pdf_path} "
              f"(result's source.file_path was a temp path: {result['source']['file_path']}). "
              f"Pass --pdf-dir pointing at the folder containing '{pdf_name}'.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(result, pdf_path, images_dir, args.dpi)

    tasks_path = out_dir / "tasks.json"
    with open(tasks_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    config_path = out_dir / "label_config.xml"
    write_label_config(config_path)

    n_entities = sum(len(p["value"]["labels"]) for t in tasks for p in t["predictions"][0]["result"] if p.get("type") == "labels")
    n_relations = sum(1 for t in tasks for p in t["predictions"][0]["result"] if p.get("type") == "relation")
    print(f"\nWrote {len(tasks)} tasks ({n_entities} entity spans, {n_relations} relations) to {tasks_path}")
    print(f"Wrote labeling config to {config_path}")
    print(f"Wrote page images under {images_dir}")
    print(f"\nBefore starting Label Studio:\n"
          f"  export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true\n"
          f"  export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={images_dir.resolve()}\n"
          f"  label-studio start")


if __name__ == "__main__":
    main()
