"""
Rococo NER Pipeline - SECURE Web Interface (Local Ollama Edition)
With PAGE REFERENCES for every entity and relationship.

Setup:
1. Install Ollama: https://ollama.com/download
2. Pull a model:   ollama pull qwen2.5:14b
3. pip install openai gradio pytesseract pillow pdf2image
4. (Optional) Set OLLAMA_BASE_URL if Ollama isn't on localhost:11434
"""

import gradio as gr
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from rococo_ner_pipeline_20260923 import RococoNERPipeline

# ── Colour palette ───────────────────────────────────────────────────────────
CREAM, TEAL, TEAL_DK = "#F5F0E8", "#4ECDB4", "#35B89E"
BLACK, GREY, BORDER  = "#111111", "#888888", "#D9D4CC"

CUSTOM_CSS = f"""
body, .gradio-container {{
    background-color: {CREAM} !important;
    font-family: 'Georgia', 'Times New Roman', serif;
    color: {BLACK};
}}
.gradio-container h1 {{ font-size: 1.9rem; font-weight: 900; color: {BLACK};
    letter-spacing: -0.5px; margin-bottom: 0.2rem; }}
.gradio-container h2 {{ font-size: 1.35rem; font-weight: 700; color: {BLACK};
    border-bottom: 2px solid {TEAL}; padding-bottom: 4px; margin-top: 1.2rem; }}
.gradio-container h3 {{ font-size: 1.05rem; font-weight: 700; color: {BLACK}; }}
.institution-tag {{ font-size: 0.78rem; font-weight: 600; color: {GREY};
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1rem; }}
.gr-panel, .gr-box, .gr-form, .gradio-container .block {{
    background-color: #FDFAF4 !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
}}
label span, .gr-form label {{ font-weight: 700 !important; color: {BLACK} !important;
    font-size: 0.88rem !important; letter-spacing: 0.03em; }}
.gr-button-primary, button.primary {{ background-color: {TEAL} !important;
    color: {BLACK} !important; border: none !important; border-radius: 4px !important;
    font-weight: 700 !important; letter-spacing: 0.05em; }}
.gr-button-primary:hover, button.primary:hover {{ background-color: {TEAL_DK} !important; }}
.gr-button, button.secondary {{ background-color: {BLACK} !important;
    color: #FFFFFF !important; border: none !important; border-radius: 4px !important;
    font-weight: 700 !important; }}
.gr-tab-nav button {{ font-weight: 700 !important; color: {GREY} !important;
    border-bottom: 3px solid transparent !important; background: transparent !important; }}
.gr-tab-nav button.selected {{ color: {BLACK} !important;
    border-bottom-color: {TEAL} !important; }}
input, textarea, .gr-text-input, .gr-textbox textarea {{ background-color: #FFFFFF !important;
    border: 1px solid {BORDER} !important; border-radius: 4px !important; color: {BLACK} !important; }}
input:focus, textarea:focus {{ border-color: {TEAL} !important; outline: none !important;
    box-shadow: 0 0 0 2px {TEAL}33 !important; }}
.footer-strip {{ font-size: 0.78rem; color: {GREY};
    border-top: 1px solid {BORDER}; margin-top: 1.5rem; padding-top: 0.6rem; }}
.badge {{ display: inline-block; background: {TEAL}; color: {BLACK};
    font-weight: 700; font-size: 0.75rem; padding: 2px 8px; border-radius: 3px; letter-spacing: 0.06em; }}
.page-badge {{ display: inline-block; background: {BLACK}; color: #fff;
    font-size: 0.70rem; padding: 1px 6px; border-radius: 3px; margin-left: 4px;
    font-weight: 700; letter-spacing: 0.04em; }}
"""

HEADER_MD = """
# **Rococo Text Analysis Pipeline**
### **OCR + Named Entity Recognition for Historical Documents**

<p class="institution-tag">Chair for History and Theory of Architecture · D-ARCH ETH · Local Ollama Edition</p>

**Powered by:** Ollama (local) · Qwen2.5 14B Instruct · Tesseract OCR

**Security:** Fully local inference — no data leaves this machine · 50 MB max &nbsp;|&nbsp;
**Validation:** ✅ valid &nbsp;|&nbsp; ⚠️ flagged &nbsp;|&nbsp; 📖 page refs &nbsp;|&nbsp;
**Formats:** JPG · PNG · TIFF · BMP · PDF &nbsp;|&nbsp;
**Languages:** English · German · French
"""

HELP_MD = """
## **Local Ollama Edition — with Page References**

Every extracted entity and every relationship is now tagged with the **page
number(s)** of the source document where it was found, plus a short
**evidence snippet** copied verbatim from the OCR-cleaned text. This lets
you verify each result against the original.

### **How page references work**
1. PDFs are OCR'd page by page — each page's text is kept separately.
2. The LLM is asked to return an `evidence` snippet (5–15 words, verbatim)
   for every entity and every relationship.
3. Each snippet is matched back against the per-page text to pin down
   the page number. If the exact snippet can't be located (OCR drift, minor
   LLM edits), we fall back to the set of pages the LLM chunk spanned —
   so the field is still informative, just less precise.
4. For single-image uploads, all results are tagged page 1.

### **Relationship Validation**
**✅** = type combination is valid  
**⚠️** = flagged (unexpected type combination — review manually)  
Flagged relationships appear as dashed red edges in the graph.

### **Tips**
- 300 DPI scans give the best OCR accuracy
- Select the correct language for better results
- Keep files under 50 MB
"""

FOOTER_MD = """
---
<div class="footer-strip">
<strong>Backend:</strong> Ollama (local) + Qwen2.5 14B &nbsp;|&nbsp;
<strong>Rate limit:</strong> 10/hour (protects shared local GPU) &nbsp;|&nbsp;
<strong>Chair for History and Theory of Architecture · DARCH ETH</strong>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers for the new entity format (dict with name + pages + evidence)
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_pages(pages):
    """[1,3,4] -> 'p. 1, 3–4'. Empty list -> ''."""
    if not pages:
        return ""
    pages = sorted(set(int(p) for p in pages))
    # collapse ranges
    groups, start = [], pages[0]
    prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        groups.append((start, prev))
        start = prev = p
    groups.append((start, prev))
    parts = [f"{a}" if a == b else f"{a}–{b}" for a, b in groups]
    return "p. " + ", ".join(parts)


def _entity_name(e):
    if isinstance(e, dict):
        return e.get("name", "")
    return str(e)


def _entity_pages(e):
    if isinstance(e, dict):
        return e.get("pages", []) or []
    return []


def _entity_evidence(e):
    if isinstance(e, dict):
        return e.get("evidence", "") or ""
    return ""


class SecureRococoUI:
    """Secure web interface with rate limiting — local Ollama edition, with page refs."""

    def __init__(self):
        self.pipeline = RococoNERPipeline(
            output_dir="results",
            use_fast_model=False,
            max_text_length=10000,
        )

        self.usage_tracker = defaultdict(list)
        self.MAX_REQUESTS_PER_HOUR = 10
        self.MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

        print("Secure local Ollama pipeline initialised (with page references)")
        print(f"Rate limit: {self.MAX_REQUESTS_PER_HOUR} requests/hour per user")

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def check_rate_limit(self, request: gr.Request):
        client_ip = request.client.host if request else "unknown"
        now = datetime.now()
        self.usage_tracker[client_ip] = [
            t for t in self.usage_tracker[client_ip]
            if now - t < timedelta(hours=1)
        ]
        current = len(self.usage_tracker[client_ip])
        remaining = self.MAX_REQUESTS_PER_HOUR - current
        if current >= self.MAX_REQUESTS_PER_HOUR:
            return False, (f"❌ Rate limit exceeded {current}/{self.MAX_REQUESTS_PER_HOUR} "
                           f"requests used this hour.")
        self.usage_tracker[client_ip].append(now)
        return True, f"✅ Request allowed ({remaining - 1} remaining this hour)"

    # ------------------------------------------------------------------
    # Processing handlers
    # ------------------------------------------------------------------
    def process_file(self, file, language_choice, request: gr.Request):
        allowed, message = self.check_rate_limit(request)
        if not allowed:
            return message, "", "", "", None, ""

        if file is None:
            return "❌ Please upload a file first.", "", "", "", None, ""

        try:
            if os.path.getsize(file.name) > self.MAX_FILE_SIZE:
                return (f"❌ File too large. Max: {self.MAX_FILE_SIZE // 1024 // 1024} MB",
                        "", "", "", None, "")
        except Exception:
            pass

        try:
            lang_code = self._lang_code(language_choice)
            result = self.pipeline.process_document(file_path=file.name, language=lang_code)

            if result.get("status") == "failed":
                err = result.get("error", "Unknown pipeline error")
                return f"❌ Pipeline error: {err}", "", "", "", None, ""

            summary       = self._format_summary(result)
            entities_json = json.dumps(result.get("entities", {}), indent=2, ensure_ascii=False)
            graph_html    = self._generate_graph_html(
                result.get("relationships", []),
                title=f"Relationships — {Path(file.name).name}",
            )
            output_path = self.pipeline.save_results(result, f"result_{Path(file.name).stem}.json")
            raw_text    = result.get("raw_text", "")
            extracted   = result.get("extracted_text", "")

            return summary, entities_json, raw_text, extracted, str(output_path), graph_html

        except Exception as e:
            import traceback
            return (f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```",
                    "", "", "", None, "")

    def process_batch(self, files, language_choice, request: gr.Request):
        if not files:
            return "❌ Please upload files first.", "", "", "", None, ""

        num_files = len(files)
        client_ip = request.client.host if request else "unknown"
        now = datetime.now()
        self.usage_tracker[client_ip] = [
            t for t in self.usage_tracker[client_ip]
            if now - t < timedelta(hours=1)
        ]
        remaining = self.MAX_REQUESTS_PER_HOUR - len(self.usage_tracker[client_ip])
        if num_files > remaining:
            return (f"❌ Batch too large! You have {remaining} requests remaining "
                    f"but are trying to process {num_files} files.",
                    "", "", "", None, "")

        try:
            lang_code = self._lang_code(language_choice)
            all_results, summaries = [], []

            for file in files:
                self.usage_tracker[client_ip].append(now)
                if os.path.getsize(file.name) > self.MAX_FILE_SIZE:
                    summaries.append(f"❌ {Path(file.name).name}: File too large (skipped)")
                    continue

                result = self.pipeline.process_document(file_path=file.name, language=lang_code)
                all_results.append(result)
                self.pipeline.save_results(result, f"result_{Path(file.name).stem}.json")

                fname = Path(file.name).name
                status = "✅" if result["status"] == "success" else "❌"
                entity_count = sum(result["statistics"].values()) if result["status"] == "success" else 0
                entry = f"{status} **{fname}**: {entity_count} entities found"

                if result["status"] == "success":
                    rels = result.get("relationships", [])
                    if rels:
                        entry += f"\n\n **Relationships ({len(rels)}):**\n\n"
                        entry += self._format_relationships(rels)

                summaries.append(entry)

            batch_summary = {
                "timestamp": datetime.now().isoformat(),
                "total_documents": len(all_results),
                "successful": sum(1 for r in all_results if r["status"] == "success"),
                "failed": sum(1 for r in all_results if r["status"] == "failed"),
                "results": all_results,
            }
            batch_path = self.pipeline.save_results(batch_summary, "batch_summary.json")

            summary_text = (
                f"## **Batch Processing Complete**\n"
                f"**Total:** {len(all_results)} | "
                f"**OK:** {batch_summary['successful']} | "
                f"**Failed:** {batch_summary['failed']}\n\n"
                "### **Individual Results:**\n" + "\n".join(summaries)
            )
            batch_json = json.dumps(batch_summary, indent=2, ensure_ascii=False)

            all_rels = [r for res in all_results if res.get("status") == "success"
                        for r in res.get("relationships", [])]
            graph_html = self._generate_graph_html(all_rels, title=f"Combined — {len(all_results)} docs")

            raw_parts, proc_parts = [], []
            for r in all_results:
                if r.get("status") == "success":
                    h = f"=== {Path(r['source']['file_path']).name} ==="
                    if r.get("raw_text"):
                        raw_parts.append(f"{h}\n\n{r['raw_text']}")
                    if r.get("extracted_text"):
                        proc_parts.append(f"{h}\n\n{r['extracted_text']}")

            return (summary_text, batch_json,
                    "\n\n\n".join(raw_parts), "\n\n\n".join(proc_parts),
                    str(batch_path), graph_html)

        except Exception as e:
            return f"❌ Error: {str(e)}", "", "", "", None, ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _lang_code(choice: str) -> str:
        return {"English + German": "deu+eng", "English": "eng",
                "German": "deu", "French": "fra"}.get(choice, "deu+eng")

    def _is_valid_relationship(self, r):
        # Relationships produced by the pipeline already carry the validation
        # verdict; trust it when present. Otherwise defer to the single source
        # of truth — the pipeline's semantic validator — to avoid rule drift.
        if "valid" in r:
            return r["valid"], r.get("invalid_reason", "")
        return self.pipeline._validate_relationship(r)

    # ------------------------------------------------------------------
    # Relationship table — now shows page(s) and evidence
    # ------------------------------------------------------------------
    def _format_relationships(self, relationships):
        if not relationships:
            return ""
        icons = {"person":"👤","place":"📍","physical_human_made_thing":"🎨",
                 "visual_item":"✨","date":"📅"}
        table  = "| | Page | Subject | Relation | Object | Evidence |\n"
        table += "|--|------|---------|----------|--------|----------|\n"
        for r in relationships:
            valid, reason = self._is_valid_relationship(r)
            flag = "⚠️" if not valid else "✅"
            si = icons.get(r.get("subject_type", ""), "•")
            oi = icons.get(r.get("object_type", ""), "•")
            obj_c = f"{oi} {r.get('object','')}"
            if not valid:
                obj_c += f" *(⚠️ {reason})*"
            page_str = _fmt_pages(r.get("pages", [])) or "—"
            # Trim evidence for table display; escape pipes
            evi = (r.get("evidence", "") or "").replace("|", "\\|").replace("\n", " ")
            if len(evi) > 80:
                evi = evi[:77] + "…"
            table += (f"| {flag} | {page_str} | {si} {r.get('subject','')} | "
                      f"➜ {r.get('relation','')} | {obj_c} | *{evi}* |\n")
        flagged = sum(1 for r in relationships if not self._is_valid_relationship(r)[0])
        if flagged:
            table += f"\n> ⚠️ **{flagged} relationship(s) flagged** as semantically unexpected.\n"
        return table

    # ------------------------------------------------------------------
    # Relationship graph — tooltip now includes page refs
    # ------------------------------------------------------------------
    def _generate_graph_html(self, relationships, title="Entity Relationship Graph"):
        if not relationships:
            return ("<p style='color:#888;font-style:italic;padding:12px;'>"
                    "No relationships to display.</p>")

        type_colors = {"person":"#4A90D9","place":"#27AE60",
                       "physical_human_made_thing":"#E67E22",
                       "visual_item":"#9B59B6","date":"#E74C3C"}
        type_icons  = {"person":"👤","place":"📍","physical_human_made_thing":"🎨",
                       "visual_item":"✨","date":"📅"}

        node_map, nid = {}, 0
        for r in relationships:
            for name, ntype in [(r.get("subject",""), r.get("subject_type","")),
                                (r.get("object",""),  r.get("object_type",""))]:
                if name and name not in node_map:
                    node_map[name] = {"id": nid, "type": ntype}
                    nid += 1

        def js(s):
            return (s or "").replace("\\","\\\\").replace('"','\\"') \
                             .replace("\n"," ").replace("\r","")

        nodes_js = [
            f'{{id:{info["id"]},label:"{type_icons.get(info["type"],"")} {js(label)}",'
            f'color:{{background:"{type_colors.get(info["type"],"#95A5A6")}",'
            f'border:"{type_colors.get(info["type"],"#95A5A6")}"}},'
            f'font:{{color:"#fff",size:13}},shape:"box",'
            f'title:"{js(info["type"])}: {js(label)}"}}'
            for label, info in node_map.items()
        ]
        edges_js = []
        for r in relationships:
            subj, obj, rel = r.get("subject",""), r.get("object",""), r.get("relation","")
            if subj in node_map and obj in node_map:
                valid, reason = self._is_valid_relationship(r)
                page_str = _fmt_pages(r.get("pages", []))
                label_bits = []
                if page_str:
                    label_bits.append(f"[{page_str}]")
                label_bits.append(rel)
                elabel = " ".join(label_bits)
                if not valid:
                    elabel = "⚠️ " + elabel

                if not valid:
                    ec, ed = '"color":{color:"#e74c3c"}', "dashes:true,"
                    tooltip = f"⚠️ Invalid: {reason}"
                    if page_str:
                        tooltip += f" | {page_str}"
                    if r.get("evidence"):
                        tooltip += f" | “{r['evidence']}”"
                    et = js(tooltip)
                else:
                    ec, ed = '"color":{color:"#bbb"}', ""
                    tooltip_parts = []
                    if page_str:
                        tooltip_parts.append(page_str)
                    if r.get("evidence"):
                        tooltip_parts.append(f"“{r['evidence']}”")
                    et = js(" | ".join(tooltip_parts) if tooltip_parts else r.get("evidence",""))

                edges_js.append(
                    f'{{from:{node_map[subj]["id"]},to:{node_map[obj]["id"]},'
                    f'label:"{js(elabel)}",{ed}arrows:"to",'
                    f'font:{{size:11,color:"#444",align:"middle"}},{ec},'
                    f'title:"{et}"}}'
                )

        legend = "".join(
            f'<span style="display:inline-flex;align-items:center;margin:2px 8px 2px 0;">'
            f'<span style="width:11px;height:11px;border-radius:2px;background:{c};'
            f'display:inline-block;margin-right:4px;"></span>'
            f'{type_icons.get(t,"")} {t.replace("_"," ").title()}</span>'
            for t, c in type_colors.items()
        )
        legend += ('<span style="display:inline-flex;align-items:center;margin:2px 8px 2px 0;">'
                   '<span style="width:22px;height:2px;border-top:2px dashed #e74c3c;'
                   'display:inline-block;margin-right:4px;"></span>⚠️ Invalid</span>')
        legend += ('<span style="display:inline-flex;align-items:center;margin:2px 8px 2px 0;">'
                   '📖 Page refs shown on edges</span>')

        safe_title = js(title)
        nodes_str  = "[" + ",".join(nodes_js) + "]"
        edges_str  = "[" + ",".join(edges_js) + "]"

        inner = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:sans-serif;background:#F5F0E8}}
#header{{background:#111;color:#fff;padding:8px 14px;font-size:13px;font-weight:bold}}
#accent{{height:3px;background:#4ECDB4}}
#legend{{padding:6px 14px;background:#FDFAF4;border-bottom:1px solid #D9D4CC;font-size:11px;flex-wrap:wrap}}
#graph{{width:100%;height:440px}}
#footer{{padding:4px 14px;background:#FDFAF4;border-top:1px solid #D9D4CC;font-size:10px;color:#888}}
</style></head><body>
<div id="header">🔗 {safe_title}</div>
<div id="accent"></div>
<div id="legend">{legend}</div>
<div id="graph"></div>
<div id="footer">💡 Drag · Scroll to zoom · Hover edges for page & evidence · ⚠️ dashed red = invalid</div>
<script>
var nodes=new vis.DataSet({nodes_str});
var edges=new vis.DataSet({edges_str});
new vis.Network(document.getElementById("graph"),{{nodes:nodes,edges:edges}},{{
  layout:{{improvedLayout:true}},
  physics:{{solver:"forceAtlas2Based",forceAtlas2Based:{{gravitationalConstant:-60,springLength:130,springConstant:0.08}},stabilization:{{iterations:200}}}},
  interaction:{{hover:true,tooltipDelay:150}},
  edges:{{smooth:{{type:"curvedCW",roundness:0.15}}}}
}});
</script></body></html>"""

        srcdoc = inner.replace("&","&amp;").replace('"','&quot;')
        return (f'<iframe srcdoc="{srcdoc}" '
                f'style="width:100%;height:520px;border:1px solid #D9D4CC;border-radius:8px;" '
                f'sandbox="allow-scripts" referrerpolicy="no-referrer"></iframe>')

    # ------------------------------------------------------------------
    # Summary — lists entities with their page numbers
    # ------------------------------------------------------------------
    def _format_entity_list(self, items, max_shown=15):
        """
        Render a list of entity dicts like:
            Hochaltar (p. 2, 4), Kanzel (p. 3), …
        """
        if not items:
            return ""
        parts = []
        for it in items[:max_shown]:
            name = _entity_name(it)
            pages = _entity_pages(it)
            if pages:
                parts.append(f"{name} *({_fmt_pages(pages)})*")
            else:
                parts.append(name)
        line = ", ".join(parts)
        if len(items) > max_shown:
            line += f" … and {len(items) - max_shown} more"
        return line

    def _format_summary(self, result):
        if result["status"] == "failed":
            return f"## ❌ Processing Failed\n\n{result.get('error','Unknown error')}"

        source   = result.get("source", {})
        entities = result.get("entities", {})
        stats    = result.get("statistics", {})

        summary = (
            f"## ✅ **Extraction Complete**\n"
            f"**File:** {Path(source['file_path']).name} \n"
            f"**Type:** {source['file_type'].upper()} \n"
            f"**OCR Confidence:** {source['ocr_confidence']:.1f}%\n"
        )
        if source["file_type"] == "pdf":
            summary += f"**Pages:** {source.get('num_pages','?')}\n"

        summary += (f"\n**Total Entities:** {sum(stats.values())}\n\n"
                    f"### **Entity Breakdown** (page references shown in parentheses)\n\n")

        for icon, label, key in [
            ("👤","Persons","persons"), ("📍","Places","places"),
            ("🎨","Physical Human-Made Things","physical_human_made_thing"),
            ("✨","Visual Items","visual_item"), ("📅","Dates","dates"),
        ]:
            items = entities.get(key, [])
            if items:
                summary += f"**{icon} {label} ({len(items)}):**\n"
                summary += self._format_entity_list(items) + "\n\n"

        pre = result.get("preprocessing", {})
        if pre:
            raw_len, clean_len = pre.get("raw_length", 0), pre.get("cleaned_length", 0)
            summary += ("---\n\n### 🧹 **Text Preprocessing**\n\n"
                        f"**Characters removed:** {raw_len - clean_len} "
                        f"({raw_len} → {clean_len})\n\n")
            if pre.get("correction_notes"):
                summary += f"**Correction notes:** {pre['correction_notes']}\n\n"

        rels = result.get("relationships", [])
        if rels:
            flagged = sum(1 for r in rels if not self._is_valid_relationship(r)[0])
            valid   = len(rels) - flagged
            summary += f"---\n\n### 🔗 **Relationships ({len(rels)} found"
            if flagged:
                summary += f" · ✅ {valid} valid · ⚠️ {flagged} flagged"
            summary += ")**\n\n" + self._format_relationships(rels) + "\n"

        return summary

    # ------------------------------------------------------------------
    # Gradio interface
    # ------------------------------------------------------------------
    def create_interface(self):
        with gr.Blocks(title="Rococo NER Pipeline — DARCH ETH", css=CUSTOM_CSS) as demo:
            gr.Markdown(HEADER_MD)

            with gr.Tabs():
                with gr.Tab("📄 Single Document"):
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(
                                label="Upload Document",
                                file_types=[".jpg",".jpeg",".png",".tiff",".tif",".bmp",".pdf"],
                            )
                            language_single = gr.Radio(
                                choices=["English + German","English","German","French"],
                                value="English + German", label="OCR Language",
                            )
                            process_btn = gr.Button("🔍 Process Document", variant="primary")
                        with gr.Column():
                            summary_output = gr.Markdown(label="Summary")

                    with gr.Row():
                        entities_output = gr.Textbox(
                            label="Extracted Entities (JSON) — incl. page numbers & evidence",
                            lines=15, max_lines=20
                        )
                        download_output = gr.File(label="Download Full Results")
                    with gr.Row():
                        raw_text_output  = gr.Textbox(label="Raw OCR Text", lines=12, max_lines=20,
                                                      info="Unprocessed Tesseract output")
                        proc_text_output = gr.Textbox(label="Cleaned & Corrected Text",
                                                      lines=12, max_lines=20,
                                                      info="After cleaning + Llama grammar correction")
                    graph_output = gr.HTML(label="Relationship Graph (page numbers on edges)")

                    process_btn.click(
                        fn=self.process_file,
                        inputs=[file_input, language_single],
                        outputs=[summary_output, entities_output, raw_text_output,
                                 proc_text_output, download_output, graph_output],
                    )

                with gr.Tab("📚 Batch Processing"):
                    gr.Markdown("Upload multiple documents — each file counts as one request toward your hourly limit.")
                    with gr.Row():
                        with gr.Column():
                            files_input = gr.File(
                                label="Upload Multiple Documents",
                                file_count="multiple",
                                file_types=[".jpg",".jpeg",".png",".tiff",".tif",".bmp",".pdf"],
                            )
                            language_batch = gr.Radio(
                                choices=["English + German","English","German","French"],
                                value="English + German", label="OCR Language",
                            )
                            batch_btn = gr.Button("🔍 Process Batch", variant="primary")
                        with gr.Column():
                            batch_summary = gr.Markdown(label="Batch Summary")

                    with gr.Row():
                        batch_json     = gr.Textbox(label="Batch Results (JSON)",
                                                    lines=15, max_lines=20)
                        batch_download = gr.File(label="Download Batch Summary")
                    with gr.Row():
                        batch_raw  = gr.Textbox(label="Raw OCR Texts", lines=12, max_lines=20)
                        batch_proc = gr.Textbox(label="Cleaned & Corrected Texts",
                                                lines=12, max_lines=20)
                    batch_graph = gr.HTML(label="Combined Relationship Graph")

                    batch_btn.click(
                        fn=self.process_batch,
                        inputs=[files_input, language_batch],
                        outputs=[batch_summary, batch_json, batch_raw,
                                 batch_proc, batch_download, batch_graph],
                    )

                with gr.Tab("ℹ️ Help & Info"):
                    gr.Markdown(HELP_MD)

            gr.Markdown(FOOTER_MD)

        return demo


def main():
    Path("results").mkdir(exist_ok=True)
    print("Launching Rococo NER Pipeline (Local Ollama Edition — with Page Refs)...")
    print("Make sure Ollama is running (`ollama serve`) with the required model pulled.")
    ui = SecureRococoUI()
    demo = ui.create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
