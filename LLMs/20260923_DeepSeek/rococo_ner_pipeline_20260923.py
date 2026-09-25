"""
Rococo Text Analysis Pipeline — Local Ollama Edition
OCR + NER for extracting entities from scanned historical documents

Drop-in replacement for rococo_ner_pipeline.py that uses:
  - Ollama (local, self-hosted inference) instead of Groq/Anthropic
  - Qwen2.5-Instruct (strong multilingual EN/DE, reliable JSON output)

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model:   ollama pull qwen2.5:14b        (16-24 GB VRAM)
                      ollama pull qwen2.5:7b         (8-12 GB VRAM)
                      ollama pull qwen2.5:32b        (24 GB+ VRAM, higher accuracy)
  3. pip install openai pytesseract pillow pdf2image
     (the `openai` package is used only as a client — no OpenAI API key or
     network call leaves your machine; it talks to Ollama's local,
     OpenAI-compatible endpoint at http://localhost:11434/v1)
  4. (Optional) Set OLLAMA_BASE_URL if Ollama runs on a different host/port
     than the default http://localhost:11434/v1

Model options (all run 100% locally via Ollama):
  - qwen2.5:7b   fastest, lowest VRAM — default "fast" tier
  - qwen2.5:14b  best accuracy/speed balance — default "accurate" tier
  - qwen2.5:32b  highest accuracy, needs 24 GB+ VRAM
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from openai import OpenAI

# PDF support (optional — graceful fallback if poppler not installed)
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ---------------------------------------------------------------------------
# Model registry  (all run locally via Ollama — no API cost, no rate limit)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "fast": {
        "model_id": "qwen2.5-7b-ctx16k",
        "label":    "Qwen2.5 7B Instruct, 16K context (fast, bilingual EN/DE)",
    },
    "accurate": {
        "model_id": "qwen2.5-14b-ctx16k",
        "label":    "Qwen2.5 14B Instruct, 16K context (higher accuracy)",
    },
}
# NOTE: these model IDs assume you've built context-extended variants on the
# Ollama host via a Modelfile (`PARAMETER num_ctx 16384`), e.g.:
#   ollama create qwen2.5-14b-ctx16k -f Modelfile.qwen14b
# Ollama's default num_ctx (2048) is too small for this pipeline's prompts —
# without this, long chunks truncate mid-generation and produce invalid JSON.

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class RococoNERPipeline:
    """Pipeline for extracting entities from scanned Rococo texts — local Ollama edition"""

    def __init__(self, ollama_base_url: str = None, output_dir: str = "results",
                 use_fast_model: bool = True, max_text_length: int = 10000):
        """
        Initialise the pipeline.

        Args:
            ollama_base_url:  Ollama's OpenAI-compatible endpoint (falls back to
                               OLLAMA_BASE_URL env var, then localhost default).
            output_dir:       Directory to save JSON results.
            use_fast_model:   True → qwen2.5:7b (default, lower VRAM).
                              False → qwen2.5:14b (higher accuracy).
            max_text_length:  Max characters sent to LLM per call.
        """
        base_url = (ollama_base_url or os.getenv("OLLAMA_BASE_URL")
                    or DEFAULT_OLLAMA_BASE_URL)

        # Ollama's local server ignores the API key, but the OpenAI client
        # requires a non-empty string to be passed.
        self.client = OpenAI(base_url=base_url, api_key="ollama")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_text_length = max_text_length

        cfg = MODEL_CONFIGS["fast"] if use_fast_model else MODEL_CONFIGS["accurate"]
        self.model = cfg["model_id"]

        print(f"✅ Local Ollama pipeline initialised")
        print(f"   Endpoint: {base_url}")
        print(f"   Model: {self.model} ({cfg['label']})")
        print(f"   Max text length: {max_text_length} chars")
        print(f"   Cost: FREE (fully local inference)")
        print(f"   Note: make sure the model is pulled — `ollama pull {self.model}`")

    # ------------------------------------------------------------------
    # Internal: call the local Ollama chat endpoint
    # ------------------------------------------------------------------
    def _chat(self, prompt: str, max_tokens: int = 4000, json_object: bool = False) -> str:
        """Send a single-turn prompt and return the assistant's text.

        json_object=True asks Ollama to constrain output to a valid JSON
        object — meaningfully improves reliability for local models like
        Qwen2.5. Only use it where the expected output is a JSON *object*
        (not a top-level JSON array).
        """
        kwargs = {}
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,   # deterministic — important for structured JSON
            **kwargs,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # OCR helpers  (identical to original)
    # ------------------------------------------------------------------
    def extract_text_from_pdf(self, pdf_path: str, language: str = "deu+eng") -> Dict:
        """Extract text from PDF using OCR (Tesseract)."""
        if not PDF_SUPPORT:
            raise ImportError("Install: pip install pdf2image")

        print(f"Processing PDF: {pdf_path}")
        try:
            images = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            return {"text": "", "confidence": 0, "pdf_path": pdf_path,
                    "language": language, "error": str(e)}

        print(f"  Found {len(images)} page(s)")
        all_text, all_confidences = [], []

        for i, img in enumerate(images, 1):
            print(f"  Processing page {i}/{len(images)}...")
            img = img.convert("L")
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.filter(ImageFilter.SHARPEN)

            page_text = pytesseract.image_to_string(img, lang=language)
            # Inline page marker, carried through cleaning/correction/chunking so
            # downstream relationship extraction can report a page number.
            all_text.append(f"[[PAGE {i}]]\n{page_text}")
            data = pytesseract.image_to_data(img, lang=language,
                                              output_type=pytesseract.Output.DICT)
            all_confidences.extend(int(c) for c in data["conf"] if int(c) > 0)

        combined = "\n\n".join(all_text)
        avg_conf  = sum(all_confidences) / len(all_confidences) if all_confidences else 0

        return {
            "text": combined.strip(), "confidence": avg_conf,
            "pdf_path": pdf_path, "language": language, "num_pages": len(images),
        }

    def preprocess_image(self, image_path: str) -> Image.Image:
        img = Image.open(image_path).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        return img.filter(ImageFilter.SHARPEN)

    def extract_text_ocr(self, image_path: str, language: str = "deu+eng") -> Dict:
        print(f"Processing OCR for: {image_path}")
        img  = self.preprocess_image(image_path)
        text = pytesseract.image_to_string(img, lang=language)
        text = f"[[PAGE 1]]\n{text}"
        data = pytesseract.image_to_data(img, lang=language,
                                          output_type=pytesseract.Output.DICT)
        confs    = [int(c) for c in data["conf"] if int(c) > 0]
        avg_conf = sum(confs) / len(confs) if confs else 0
        return {"text": text.strip(), "confidence": avg_conf,
                "image_path": image_path, "language": language}

    # ------------------------------------------------------------------
    # Text cleaning  (identical to original)
    # ------------------------------------------------------------------
    def strip_footnotes(self, text: str) -> tuple:
        lines = text.split("\n")
        main_lines, fn_lines, in_footnotes = [], [], False

        SEP = re.compile(
            r"^[\s_\-\*\=\u2014\u2013]{4,}$"
            r"|^\s*[\u00b9\u00b2\u00b3\u2070-\u2079]"
        )
        FN_ENTRY = re.compile(
            r"^\s*(\d{1,3}[\.]\)\s|\[\d{1,3}\]\s|[¹²³⁴⁵⁶⁷⁸⁹⁰]\s|\*+\s)"
        )

        for line in lines:
            if SEP.match(line):
                in_footnotes = True
                fn_lines.append(line)
                continue
            if in_footnotes:
                fn_lines.append(line)
            else:
                if FN_ENTRY.match(line):
                    fn_lines.append(line)
                else:
                    main_lines.append(line)

        cleaned = "\n".join(main_lines)
        cleaned = re.sub(r"[\u00b9\u00b2\u00b3\u2070-\u2079]+", "", cleaned)
        cleaned = re.sub(r"\^\d+", "", cleaned)
        cleaned = re.sub(r"(?<=\w)\s*\[\d+\]", "", cleaned)

        footnotes = "\n".join(fn_lines).strip()
        if footnotes:
            fn_count = len(re.findall(
                r"^\s*(\d{1,3}[\.]\)|[\[]\d+\]|[¹²³⁴⁵⁶⁷⁸⁹⁰]|\*+)",
                footnotes, re.MULTILINE))
            print(f"   🗒️  Footnote stripping: removed ~{fn_count} entries ({len(footnotes)} chars)")

        return cleaned, footnotes

    def normalize_small_caps(self, text: str) -> str:
        """
        Fix names that OCR renders from small-caps typography as erratic mixed case.
        Small caps in printed books OCR as ALL CAPS or erratic MiXeD cAsE.
        Also fixes dotless-i (ı → i) used by OCR for small-caps I.
        Detects and title-cases these patterns, leaving normal prose untouched.
        """
        # Step 1: Replace dotless-i (U+0131) — OCR artefact for small-caps I
        text = text.replace("\u0131", "i")

        def fix_token(token: str) -> str:
            if len(token) < 2 or not any(c.isalpha() for c in token):
                return token
            alpha = [c for c in token if c.isalpha()]
            n = len(alpha)
            upper_count = sum(1 for c in alpha if c.isupper())
            uppers_after_first = sum(1 for c in alpha[1:] if c.isupper())

            # ALL CAPS (2+ alpha chars) -> Title case
            if upper_count == n and n >= 2:
                return token[0].upper() + token[1:].lower()
            # Mostly caps: ≥ 60% uppercase and word is ≥ 3 chars (catches "ULi", "CHr")
            if n >= 3 and upper_count / n >= 0.6:
                return token[0].upper() + token[1:].lower()
            # Erratic interior caps: 2+ uppercase after first char (catches "MuRer", "JoHAnn")
            if uppers_after_first >= 2:
                return token[0].upper() + token[1:].lower()
            return token

        # Match Unicode letters including umlauts
        result = re.sub(r"[A-Za-z\u00C0-\u024F]+", lambda m: fix_token(m.group()), text)
        return result

    def clean_ocr_text(self, text: str) -> str:
        text, _ = self.strip_footnotes(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\\n", "\n").replace("\\t", " ")
        text = re.sub(r"[^\S\n\t ]+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?m)^[^a-zA-ZaeouAOUss\u00C0-\u024F]{0,3}$\n?", "", text)
        text = text.replace("\u017f", "s")
        # Normalize small-caps OCR artefacts (must come after basic cleaning)
        text = self.normalize_small_caps(text)
        return text.strip()

    # ------------------------------------------------------------------
    # Grammar correction  (uses local Ollama/Qwen2.5)
    # ------------------------------------------------------------------
    def correct_german_text(self, text: str):
        """Use the local Qwen2.5 model via Ollama to correct OCR-induced errors in German/mixed text."""
        print("   Running German grammar/spelling correction...")

        # llama-3.3-70b-versatile has 32K token context — safe to send much more.
        # 1 token ≈ 4 chars; 20000 chars ≈ 5000 tokens, leaving plenty for output.
        MAX_CORRECTION_CHARS = 20000
        excerpt = text[:MAX_CORRECTION_CHARS]

        prompt = f"""You are an expert in German art history, church history, and historical German academic writing.
The text below was extracted by OCR from a scanned German academic art history publication about Swiss/German churches and monuments (Kunstdenkmäler series).

CRITICAL ISSUE — SMALL CAPS TYPOGRAPHY:
Academic art history books typeset persons' names in SMALL CAPS. OCR cannot render small caps correctly and produces one of these artefacts:
  - ALL UPPERCASE:      "JOHANNES MURER", "MICHEL VON SAROY"
  - Erratic mixed case: "MicHEL von Saroy", "JoHAnNnEs MuRer", "MıcHAEL"
  - Partially garbled:  "Jon. GRUBENMANN", "MicHEL", "MıcHEL"
You MUST recognise these as person names and convert them to normal Title Case:
  "JOHANNES MURER" → "Johannes Murer"
  "MicHEL von Saroy" → "Michel von Saroy"
  "Jon. GRUBENMANN" → "Jon. Grubenmann"

OTHER OCR ERRORS TO FIX:
- Broken words: "Jo hann" → "Johann", "Werk mei ster" → "Werkmeister"
- Merged words: "vondem" → "von dem", "imJahr" → "im Jahr"
- Wrong chars in names: "0" (zero) → "O", "1" → "l", "rn" → "m", "li" → "h", "ı" → "i"
- Umlaut errors: "Wiirzburg" → "Würzburg", "Munchen" → "München", "Sal menswiler" → "Salmannsweiler"
- Long-s artefacts: "ſ" → "s"
- Stray OCR noise: isolated letters on lines, "ı" (dotless i), random symbols

DOMAIN KNOWLEDGE — names likely to appear in Swiss/German church history texts:
Persons: Johannes Murer, Michel von Saroy (also: Michel von Safoy), Heinrich von Greifensee,
  Johannes Wyss, Ulrich Tanner, Johannes Grubenmann, Hans Ulrich Grubenmann,
  Joh. Georg Müller, Joh. Chr. Kunkler, Ferdinand Stadler, Emil Rittmeyer,
  Joseph Anton Feuchtmayer, Joh. Georg Dirr, Karl Ulrich Rheiner,
  Ambros Schlatter, C. Gonzenbacher, Otmar Scheitlin, Otmar Bomer
Places: St. Gallen, Konstanz, Salem (Salmannsweiler), Überlingen, Ravensburg, Bischofszell,
  Bermatingen, Lindau, Teufen, Rorschach, Mosnang, Wien, Florenz, Zürich, Landsberg

PAGE MARKERS: The text may contain lines like "[[PAGE 3]]" — these are structural
page-boundary markers inserted by the pipeline, not part of the original document.
Copy them through UNCHANGED, in their original position, exactly as "[[PAGE N]]".
Never translate, reword, remove, or treat them as OCR noise or a sentence.

Your tasks:
1. Convert ALL small-caps person names to normal Title Case
2. Reconstruct broken/split words
3. Fix umlaut and character errors
4. Remove OCR noise (stray symbols, dotless-i artefacts)
5. Do NOT translate — keep original German
6. Do NOT add or remove sentences — only correct what OCR garbled
7. Leave any "[[PAGE N]]" markers exactly as they are

Return ONLY a valid JSON object — no markdown fences, no commentary:
{{
  "corrected_text": "the full corrected text here",
  "correction_notes": "list the main small-caps names fixed and other corrections made"
}}

Text to correct:
{excerpt}"""

        try:
            raw = self._chat(prompt, max_tokens=6000, json_object=True)
            raw = self._strip_json_fences(raw)
            parsed = json.loads(raw)

            corrected = parsed.get("corrected_text", text)
            if not isinstance(corrected, str) or len(corrected) < 10:
                corrected = text
            notes = parsed.get("correction_notes", "No notes provided.")
            if not isinstance(notes, str):
                notes = "No notes provided."

            corrected = corrected.replace("\\n", "\n").replace("\\t", " ")
            corrected = corrected.replace("\r\n", "\n").replace("\r", "\n")

            # Append any text beyond the corrected portion unchanged
            if len(text) > MAX_CORRECTION_CHARS:
                corrected += "\n\n" + text[MAX_CORRECTION_CHARS:]
                notes += " (Note: only the first portion was grammar-checked due to length.)"

            return corrected, notes

        except Exception as e:
            print(f"   Grammar correction failed: {e} — using cleaned text as-is.")
            return text, f"Grammar correction skipped: {e}"

    # ------------------------------------------------------------------
    # Preprocessing orchestrator  (identical logic, uses new correct_german_text)
    # ------------------------------------------------------------------
    def preprocess_text(self, raw_text: str, language: str = "deu+eng") -> Dict:
        print("Preprocessing text...")
        cleaned     = self.clean_ocr_text(raw_text)
        raw_len     = len(raw_text)
        cleaned_len = len(cleaned)
        print(f"   Cleaning: {raw_len} → {cleaned_len} chars ({raw_len - cleaned_len} removed)")

        if "deu" in language.lower():
            corrected, notes = self.correct_german_text(cleaned)
        else:
            corrected = cleaned
            notes = "Grammar correction not applied (no German in language setting)."

        return {
            "cleaned_text":     cleaned,
            "corrected_text":   corrected,
            "correction_notes": notes,
            "raw_length":       raw_len,
            "cleaned_length":   cleaned_len,
        }

    # ------------------------------------------------------------------
    # Entity extraction  (uses local Ollama/Qwen2.5)
    # ------------------------------------------------------------------
    def _call_entity_extraction(self, chunk: str, chunk_idx: int, total: int) -> Dict:
        label = f"chunk {chunk_idx+1}/{total}" if total > 1 else "full text"
        print(f"   Extracting entities from {label} ({len(chunk)} chars)...")

        prompt = f"""You are a specialist in German and Swiss art history and church monuments (Kunstdenkmäler), trained to extract named entities from academic German texts — including texts with residual OCR errors.

CRITICAL — SMALL CAPS: Academic art history books print person names in SMALL CAPS. OCR produces ALL CAPS or erratic MiXeD cAsE for these. Recognise them as person names and output in correct Title Case:
  Examples from this domain: "JOHANNES MURER" → "Johannes Murer", "MicHEL von Saroy" → "Michel von Saroy",
  "Jon. GRUBENMANN" → "Jon. Grubenmann", "MıcHAEL" → "Michael"

Ignore any "[[PAGE N]]" markers in the text below — these are structural page-boundary
markers inserted by the pipeline, not part of the original document. Never extract them
as entities.

Text to analyse:
{chunk}

Extract and categorise EVERY entity — do not skip anything. The Categories are based on CIDOC-CRM Classes and their scope notes.
Categories:

1. PER – Person (E21) — Real persons who live or are assumed to have lived. Legendary figures who may have existed (e.g. Ulysses, King Arthur) are PER only if the text refers to them as historical figures. If it is unclear whether two names refer to the same person, annotate each mention as it stands.
 All named individuals including:
   - Architects and master builders (Werkmeister, Baumeister, Oberbaumeister): e.g. Johannes Murer, Michel von Saroy, Johannes Grubenmann, Hans Ulrich Grubenmann, Joh. Georg Müller, Joh. Chr. Kunkler, Ferdinand Stadler, Magnus Hetzer, Linhart Ränftler
   - Painters, sculptors, woodcarvers: e.g. Michael Lang (Maler Michael), Joseph Anton Feuchtmayer, Joh. Georg Dirr, Karl Ulrich Rheiner
   - Patrons, clergy, city officials, donors: e.g. Abt Heinrich III., Bischof von Konstanz, Otmar Bomer, Johannes Wyss
   - Art historians, authors cited: e.g. Wegelin, Vadian, Keßler, Rütiner, H. Rott
   - Bell casters, craftsmen: e.g. Ulrich Schnabelburg, Karl Rosenlächer, Schalch
   Include abbreviated names ("Jon.", "J. C.", "H.") as separate entries if they appear.
 Are not PER: saints, biblical, mythological and allegorical figures in depictions (→ ICO, see 4.3); workshops and institutions. 
  
2. PLACE – Place (E53) — Extents in natural space, in particular on the Earth's surface, independent of time and matter. Places describe where things or events are located. They are usually determined by reference to immobile objects (buildings, cities, mountains, rivers), may have fuzzy boundaries, and can be defined relative to a physical thing, such as a room within a church. All named locations, including:
   - Cities and towns: St. Gallen, Konstanz, Salem, Überlingen, Ravensburg, Bischofszell, Wien, Zürich, Lindau, Teufen, Rorschach, Mosnang, Landsberg, Florenz, Schaffhausen
   - Churches and religious buildings: St. Laurenzen, St. Mangen, Münster, Stiftskirche, Kloster Salem
   - Specific spaces within buildings: Sakristei, Chor, Langhaus, Empore, Kapelle, Turm
   - Archives, museums: Historisches Museum, Stiftsarchiv (StA), Stiftsbibliothek
A building or space is PLACE only when it locates something else; as a product of human activity it is OBJ (see 4.2).
Is Not PLACE: nationality adjectives and style labels ("Konstanzer Werkstatt", "Bodensee-Gotik").

3. OBJ – Physical human-made thing (E24 / E22) — TDiscrete, identifiable human-made items documented as single units and characterised by relative stability. A building is OBJ when it is cited not as a place but as the product of human activity. It can be a building, when its cited not as places in a relationship but as product of human activitiy.
It can include, for example: artworks, everyday utensils, textile products, and architectural elements, for example:
   - Altars: Hochaltar, Seitenaltar, Marienaltar, Sebastianaltar, Jakobsaltar, Annaaltar, Mauritiusaltar
   - Furnishings: Kanzel (pulpit), Orgel (organ), Taufstein (baptismal font), Gestühl, Empore
   - Artworks: Glasgemälde, Flügelaufsatz, Skulpturen, Figuren, Kreuzigungsgruppe, Reliquiare
   - Architectural elements: Arkaden, Pfeiler, Fenster, Maßwerk, Turm, Portal, Gewölbe, Glocke
   - Documents: Jahrzeitbuch, Ablaßbrief, Urkunde
  Some terms also appear under PLACE (Empore, Turm) or VIS (Maßwerk). 


4. VIS – Visual item (E36) — The intellectual or conceptual aspect of recognisable marks, images and other visual works: the underlying prototype, not an individual physical embodiment. A logo stays the same logo on any number of publications, even if size, orientation and colour change; the same holds for images reproduced many times. Visual items are therefore independent of their physical support. VIS links physical things that carry the same visual qualities (symbols, marks, images).
It can also cover decorative and ornament forms such as:
• Foliage and flowers: Laubwerk, Blattwerk, Blattornament, Ranken, Rankenornament, Blumen, Blumengebinden, Blumenzweigen, Blumenfeldern, Blumendekor in der Vase, stilisierter Blumendekor in einem Topf, Tulpenblüten, Nelken, Rose, Rosetten, Palmette, Lebensbaum
• Architectural ornament and profiles: Maßwerk, Fischblasen, Wirbelrosetten, Akanthuskonsolen, Karniesprofilierung, Kehlen, Rundstäbe, Schweifungen, Kapitell, Halbsäule, Lisene, Fries, Bogenfries, Ornamentbogen
• Baroque and Rococo ornament: Hochbarockornamenten, Rocaille, Rocaillerahmen, Muschelwerk, Bandelwerk, Rollwerk, Ohrmuschelornament, Kartusche, Lambrequin, Volute, Groteske, Arabeske, Chinoiserie, Zirkelschlagornamentik
• Surface and applied decoration: Vergoldung, Eisengitter, Marmorierung, Maserierung, Intarsien, Draperie, Feston, Girlande, Medaillon, Bordüre, Laufender Hund
• Inscriptions, monograms, signs: Inschriften, Sinnspruch, Monogramme IHS, Monogramm für Maria und Josef, Jesussymbole, Mariasymbole, Herz Jesu
• Decorative animal and fruit motifs: Hirschen, Pferd, Vögel, Früchteschalen, Apfel, Birne, Ackerfrüchte

5.  ICO – Iconographic subject — for depicted scenes, figures, allegories. An Iconography can include visual items (a scene built from motifs).
It can include for example:
• Biblical scenes and cycles: Kreuzigung, Christi Kreuzestod, Verkündigung, Verkündigung der Maria, Himmelfahrt, Couronnement de Marie, Szene von Geburt, Szene von Tod, Bilder aus dem Neuen Testament, Bilder des Guten Hirten
• Holy figures and groups: Vier Evangelisten, Heilige Familie, Josef und Maria mit Kind, Heilige Barbara, Heiligen Margareta, Heiligen Ottilie, Heiligenbrustbildern, weibliche Heilige in einer Landschaft
• Allegories and symbols: Allegorie, Tugenden/Laster, Darstellung der Jahreszeiten, Darstellung der Winterjahreszeit, Symbole für Christi Geburt, Marterwerkzeugen der Kreuzigung
• Landscape, architecture and everyday life: Landschaftsbilder, Häusern in Landschaft, Gebäude am See, Arkadenarchitektur, Szenen aus dem Dorfleben,Alpabfahrt, Scheibenschiessen, Fechten, Reiten, Jagen und Fischen

6. DATE – Date / time expression (E52, E49) — All temporal references (be exhaustive):
   - Years: 1225, 1413, 1418, 1504, 1577, 1764, 1851
   - Ranges: 1851–1853, 1730–1745
   - Centuries: 12. Jahrhundert, 14. Jahrhundert, Ende 15. Jahrhundert
   - Qualifiers that change the meaning are included: um 1500, ca. 1520, vor 1520, nach 1530
   - Named events with dates: Stadtbrand von 1314, Reformation 1526.
 Excluded: plain prepositions and articles (im \[14. Jahrhundert\]); style periods (Gotik, Barock); relative expressions without an anchor ("drei Jahre später"). Events without a date ("nach der Reformation")
 
7. GROUP – Group (E74) – Any gatherings or organisations of human individuals or groups that act collectively or in a similar way due to any form of unifying relationship. It exhibits organisational characteristics typified by a set of ideas or beliefs held in common, or actions performed together related, for example, to communication, creation of artefacts, common purposes.It can also include nationality, married couples and families.
 

Return ONLY valid JSON — no markdown fences, no commentary:
{{
  "persons": [],
  "places": [],
  "physical_human_made_thing": [],
  "visual_item": [],
  "dates": [],
}}

Rules:
- Be EXHAUSTIVE — extract every name, date, place, and object mentioned
- Output ALL person names in correct Title Case (fix small-caps OCR artefacts)
- Preserve German spelling with umlauts
- Each entity o
nce only (deduplicate)"""

        raw = self._chat(prompt, max_tokens=4000, json_object=True)
        raw = self._strip_json_fences(raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            fixed = self._repair_truncated_json(raw)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  JSON parse failed for {label}: {e}")
                raise

    def _merge_entity_results(self, results: List[Dict]) -> Dict:
        LIST_KEYS = ("persons", "places", "physical_human_made_thing", "visual_item",
                     "dates")
        merged: Dict = {k: [] for k in LIST_KEYS}
        seen: Dict[str, set] = {k: set() for k in LIST_KEYS}

        for r in results:
            for k in LIST_KEYS:
                for item in r.get(k, []):
                    norm = item.strip().lower()
                    if norm and norm not in seen[k]:
                        seen[k].add(norm)
                        merged[k].append(item.strip())

        return merged

    def extract_entities_llm(self, text: str, language: str = "auto") -> Dict:
        print("Extracting entities with LLM...")
        CHUNK_SIZE, OVERLAP = 8000, 500

        chunks = [c for _, c in self._chunk_text(text, CHUNK_SIZE, OVERLAP)]
        if len(chunks) > 1:
            print(f"   📄 Text split into {len(chunks)} chunks")

        results = []
        for i, chunk in enumerate(chunks):
            try:
                results.append(self._call_entity_extraction(chunk, i, len(chunks)))
            except Exception as e:
                import traceback
                print(f"   ⚠️  Entity extraction failed for chunk {i+1}: {e}")
                print(traceback.format_exc())

        if not results:
            return {
                "persons": [], "places": [], "physical_human_made_thing": [],
                "visual_item": [], "dates": [], "error": "All chunks failed",
            }

        return self._merge_entity_results(results)

    # ------------------------------------------------------------------
    # Relationship extraction  (uses local Ollama/Qwen2.5)
    # ------------------------------------------------------------------
    def extract_relationships_llm(self, text: str, entities: Dict) -> List[Dict]:
        print("Extracting relationships with LLM...")
        CHUNK_SIZE, OVERLAP = 8000, 500

        page_index = self._build_page_index(text)
        chunks = self._chunk_text(text, CHUNK_SIZE, OVERLAP)
        if len(chunks) > 1:
            print(f"   Relationship extraction across {len(chunks)} chunks")

        entity_summary = json.dumps(
            {k: v for k, v in entities.items() if isinstance(v, list)},
            ensure_ascii=False,
        )

        all_relationships = []

        for chunk_idx, (start_offset, chunk) in enumerate(chunks):
            start_page = self._page_at_offset(page_index, start_offset)
            prompt = f"""You are a specialist in German Baroque and Rococo art history. Extract ALL relationships between entities from this academic German art history text.

This excerpt begins on page {start_page}. The text below may contain markers like
"[[PAGE N]]" marking where a new scanned page starts — these are structural markers
inserted by the pipeline, NOT part of the original document content. Use them only to
determine which page each relationship's evidence comes from; never extract them as
entities or evidence. If a relationship's evidence appears before the first "[[PAGE N]]"
marker in this excerpt, its page is {start_page}.

Text:
{chunk}

Known entities already extracted:
{entity_summary}

A "relationship" is any factual connection between two entities stated or clearly implied
in the text.

For each relationship, classify it with a canonical, VERBAL ENGLISH relation_type in
lower_snake_case — NOT the raw German/English phrase from the text. Pick the closest match
from this vocabulary, or coin a new lower_snake_case verb-phrase in the same style if
nothing fits well:

| relation_type          | subject_type →                                   | object_type                                    | German/English triggers |
|-------------------------|---------------------------------------------------|-------------------------------------------------|--------------------------|
| created                 | person                                             | physical_human_made_thing, place, visual_item    | "erbaut von", "geschaffen von", "gemalt von", "modelliert von", "entworfen von" |
| has_author               | physical_human_made_thing, visual_item, place      | person                                           | inverse of "created" — the work's creator |
| attributed_to            | physical_human_made_thing, visual_item             | person                                           | "zugeschrieben" |
| has_patron                | physical_human_made_thing, place                    | person                                           | "gestiftet von", "in Auftrag gegeben von", "finanziert von", "beauftragt von" |
| happened_in_date         | physical_human_made_thing, place, person            | date                                             | "erbaut 1745", "vollendet", "geweiht", "datiert", "entstanden" |
| has_lifespan              | person                                              | date                                             | a person's birth/death year(s) or life dates |
| was_restored_in          | physical_human_made_thing, place                    | date                                             | "restauriert", "renoviert", "erneuert" |
| has_current_location     | physical_human_made_thing, visual_item              | place                                            | "befindet sich in", "steht in" |
| has_original_location    | physical_human_made_thing, visual_item              | place                                            | former/original location, before it moved |
| worked_in                | person                                              | place                                            | "wurde tätig in", "arbeitete in" |
| born_in                  | person                                              | place                                            | birthplace |
| died_in                  | person                                              | place                                            | place of death |
| influenced_by            | person, physical_human_made_thing, place            | person, physical_human_made_thing, visual_item   | "beeinflusst von", "im Stil von", "geprägt durch" |
| collaborated_with        | person                                              | person                                           | "arbeitete zusammen mit", "assistierte" |
| married_to               | person                                              | person                                           | "heiratete" |
| student_of               | person                                              | person                                           | "war Schüler von" |
| teacher_of               | person                                              | person                                           | "war Lehrer von" |
| owned_by                 | physical_human_made_thing                           | person                                           | "gehört", "im Besitz von" |
| depicts                  | physical_human_made_thing, visual_item              | visual_item                                      | "zeigt", "steht für", "illustriert" |
| decorated_with           | physical_human_made_thing, place                    | visual_item                                      | "verziert mit", "geschmückt mit" |
| part_of                  | physical_human_made_thing, place                    | physical_human_made_thing, place                 | subject is a component of object |

For each relationship produce one JSON object:
{{
  "subject":       "entity name (corrected spelling)",
  "subject_type":  "person|place|physical_human_made_thing|visual_item|date",
  "relation_type": "canonical lower_snake_case relation from the table above (or a coined one in the same style)",
  "object":        "entity name (corrected spelling)",
  "object_type":   "person|place|physical_human_made_thing|visual_item|date",
  "evidence":      "the exact short phrase from the source text supporting this relationship, in its original language",
  "page":          12
}}

Rules:
- Extract EVERY relationship — a typical paragraph has 3–8 relationships
- For "X was built by Y in Z": produce TWO triples: (X, has_author, Y) and (X, happened_in_date, Z)
- Use corrected name spellings even if OCR garbled them in the text
- Include relationships involving entities NOT in the known list if they appear in the text
- "page" must be an integer, taken from the nearest preceding "[[PAGE N]]" marker (or {start_page} if no marker precedes the evidence in this excerpt)
- Return ONLY a JSON array, no markdown, no commentary. If no relationships: return []"""

            try:
                raw = self._chat(prompt, max_tokens=4000)
                raw = self._strip_json_fences(raw)

                try:
                    chunk_rels = json.loads(raw)
                except json.JSONDecodeError:
                    fixed = self._repair_truncated_json(raw)
                    chunk_rels = json.loads(fixed)

                if isinstance(chunk_rels, list):
                    all_relationships.extend(chunk_rels)

            except Exception as e:
                print(f"   Relationship extraction failed for chunk {chunk_idx+1}: {e}")

        # Deduplicate and validate
        seen_rels, deduped = set(), []
        for r in all_relationships:
            key = (r.get("subject", "").lower(), r.get("relation_type", ""), r.get("object", "").lower())
            if key not in seen_rels:
                seen_rels.add(key)
                valid, reason = self._validate_relationship(r)
                r["valid"]          = valid
                r["invalid_reason"] = reason if not valid else ""
                deduped.append(r)

        flagged = sum(1 for r in deduped if not r["valid"])
        if flagged:
            print(f"   ⚠️  {flagged} relationship(s) flagged as semantically unexpected")

        return deduped

    # ------------------------------------------------------------------
    # Relationship semantic validator
    # ------------------------------------------------------------------
    # Canonical relation_type vocabulary the LLM is prompted to use (see the
    # relation_type table in extract_relationships_llm) — direct lookup, since
    # relation is now a controlled English label rather than a free phrase.
    _RELATION_TAXONOMY = {
        "created":                {"allowed_subjects": {"person"},                                                    "allowed_objects": {"physical_human_made_thing", "place", "visual_item"}},
        "has_author":             {"allowed_subjects": {"physical_human_made_thing", "visual_item", "place"},         "allowed_objects": {"person"}},
        "attributed_to":          {"allowed_subjects": {"physical_human_made_thing", "visual_item"},                  "allowed_objects": {"person"}},
        "has_patron":             {"allowed_subjects": {"physical_human_made_thing", "place"},                        "allowed_objects": {"person"}},
        "happened_in_date":       {"allowed_subjects": {"physical_human_made_thing", "place", "person"},              "allowed_objects": {"date"}},
        "has_lifespan":           {"allowed_subjects": {"person"},                                                    "allowed_objects": {"date"}},
        "was_restored_in":        {"allowed_subjects": {"physical_human_made_thing", "place"},                        "allowed_objects": {"date"}},
        "has_current_location":   {"allowed_subjects": {"physical_human_made_thing", "visual_item"},                  "allowed_objects": {"place"}},
        "has_original_location":  {"allowed_subjects": {"physical_human_made_thing", "visual_item"},                  "allowed_objects": {"place"}},
        "worked_in":              {"allowed_subjects": {"person"},                                                    "allowed_objects": {"place"}},
        "born_in":                {"allowed_subjects": {"person"},                                                    "allowed_objects": {"place"}},
        "died_in":                {"allowed_subjects": {"person"},                                                    "allowed_objects": {"place"}},
        "influenced_by":          {"allowed_subjects": {"person", "physical_human_made_thing", "place"},              "allowed_objects": {"person", "physical_human_made_thing", "visual_item"}},
        "collaborated_with":      {"allowed_subjects": {"person"},                                                    "allowed_objects": {"person"}},
        "married_to":             {"allowed_subjects": {"person"},                                                    "allowed_objects": {"person"}},
        "student_of":             {"allowed_subjects": {"person"},                                                    "allowed_objects": {"person"}},
        "teacher_of":             {"allowed_subjects": {"person"},                                                    "allowed_objects": {"person"}},
        "owned_by":               {"allowed_subjects": {"physical_human_made_thing"},                                 "allowed_objects": {"person"}},
        "depicts":                {"allowed_subjects": {"physical_human_made_thing", "visual_item"},                  "allowed_objects": {"visual_item"}},
        "decorated_with":         {"allowed_subjects": {"physical_human_made_thing", "place"},                        "allowed_objects": {"visual_item"}},
        "part_of":                {"allowed_subjects": {"physical_human_made_thing", "place"},                        "allowed_objects": {"physical_human_made_thing", "place"}},
    }

    def _classify_relation(self, relation_type: str) -> dict | None:
        return self._RELATION_TAXONOMY.get((relation_type or "").strip().lower())

    def _validate_relationship(self, r: Dict) -> tuple:
        relation_type = r.get("relation_type", "")
        subject_type  = r.get("subject_type", "").lower().strip()
        object_type   = r.get("object_type",  "").lower().strip()

        cat = self._classify_relation(relation_type)
        if cat is None:
            # Not in the controlled vocabulary (LLM coined its own label) — skip validation.
            return True, ""

        reasons = []
        if subject_type not in cat["allowed_subjects"]:
            reasons.append(
                f"subject type '{subject_type}' unexpected for relation_type '{relation_type}' "
                f"(expected: {', '.join(sorted(cat['allowed_subjects']))})"
            )
        if object_type not in cat["allowed_objects"]:
            reasons.append(
                f"object type '{object_type}' unexpected for relation_type '{relation_type}' "
                f"(expected: {', '.join(sorted(cat['allowed_objects']))})"
            )
        return (False, "; ".join(reasons)) if reasons else (True, "")

    # ------------------------------------------------------------------
    # Main document pipeline
    # ------------------------------------------------------------------
    def process_document(self, file_path: str, language: str = "deu+eng") -> Dict:
        try:
            file_ext = Path(file_path).suffix.lower()

            # Step 1: OCR
            if file_ext == ".pdf":
                if not PDF_SUPPORT:
                    return {"status": "failed",
                            "error": "pdf2image/poppler not installed. Add poppler-utils to packages.txt",
                            "file_path": file_path}
                ocr_result = self.extract_text_from_pdf(file_path, language)
            else:
                ocr_result = self.extract_text_ocr(file_path, language)

            if not ocr_result.get("text", "").strip():
                return {"status": "failed", "error": "No text extracted from document",
                        "file_path": file_path}

            # Step 2: Preprocessing
            preprocessing  = self.preprocess_text(ocr_result["text"], language)
            processed_text = preprocessing["corrected_text"]

            # Step 3: NER
            entities = self.extract_entities_llm(processed_text)

            # Step 3b: Relationships
            relationships = self.extract_relationships_llm(processed_text, entities)

            # Step 4: Assemble result
            result = {
                "status":    "success",
                "timestamp": datetime.now().isoformat(),
                "source": {
                    "file_path":      file_path,
                    "file_type":      "pdf" if file_ext == ".pdf" else "image",
                    "ocr_confidence": ocr_result.get("confidence", 0),
                },
                "raw_text":       ocr_result["text"],
                "extracted_text": processed_text,
                "preprocessing": {
                    "raw_length":       preprocessing["raw_length"],
                    "cleaned_length":   preprocessing["cleaned_length"],
                    "correction_notes": preprocessing["correction_notes"],
                },
                "entities":      entities,
                "relationships": relationships,
                "statistics": {
                    "persons":    len(entities.get("persons", [])),
                    "places":     len(entities.get("places", [])),
                    "physical_human_made_thing": len(entities.get("physical_human_made_thing", [])),
                    "visual_item": len(entities.get("visual_item", [])),
                    "dates":      len(entities.get("dates", [])),
                },
            }

            if file_ext == ".pdf" and "num_pages" in ocr_result:
                result["source"]["num_pages"] = ocr_result["num_pages"]

            return result

        except Exception as e:
            import traceback
            return {
                "status": "failed",
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "file_path": file_path,
            }

    def save_results(self, result: Dict, output_filename: str = None):
        if output_filename is None:
            output_filename = f"rococo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
        return output_path

    def process_batch(self, file_dir: str, language: str = "deu+eng") -> List[Dict]:
        file_dir   = Path(file_dir)
        extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".pdf"}
        files      = [f for f in file_dir.iterdir() if f.suffix.lower() in extensions]
        print(f"Found {len(files)} files to process")

        all_results = []
        for i, file in enumerate(files, 1):
            print(f"\n--- Processing {i}/{len(files)}: {file.name} ---")
            result = self.process_document(str(file), language)
            all_results.append(result)
            self.save_results(result, f"result_{file.stem}.json")

        summary = {
            "timestamp":       datetime.now().isoformat(),
            "total_documents": len(all_results),
            "successful":      sum(1 for r in all_results if r["status"] == "success"),
            "failed":          sum(1 for r in all_results if r["status"] == "failed"),
            "results":         all_results,
        }
        self.save_results(summary, "batch_summary.json")
        return all_results

    def print_summary(self, result: Dict):
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        if result["status"] == "failed":
            print(f"Status: FAILED — {result.get('error', 'Unknown error')}")
            return

        print(f"Source: {result['source']['file_path']}")
        print(f"Type:   {result['source']['file_type'].upper()}")
        if result["source"]["file_type"] == "pdf":
            print(f"Pages:  {result['source'].get('num_pages', '?')}")
        print(f"OCR Confidence: {result['source']['ocr_confidence']:.2f}%")
        print("\n--- Extracted Text Preview ---")
        preview = result["extracted_text"]
        print((preview[:300] + "...") if len(preview) > 300 else preview)

        print("\n--- Entities Found ---")
        for label, key in [
            ("Persons",                    "persons"),
            ("Places",                     "places"),
            ("Physical Human-Made Things", "physical_human_made_thing"),
            ("Visual Items",               "visual_item"),
            ("Dates",                      "dates"),
        ]:
            items = result["entities"].get(key, [])
            if items:
                print(f"\n{label} ({len(items)}):")
                for item in items:
                    print(f"  • {item}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Internal utility
    # ------------------------------------------------------------------
    _PAGE_MARKER_RE = re.compile(r"\[\[PAGE (\d+)\]\]")

    @classmethod
    def _build_page_index(cls, text: str) -> List[Tuple[int, int]]:
        """(char_offset, page_number) for every "[[PAGE N]]" marker in text, in order."""
        return [(m.start(), int(m.group(1))) for m in cls._PAGE_MARKER_RE.finditer(text)]

    @staticmethod
    def _page_at_offset(page_index: List[Tuple[int, int]], offset: int) -> int:
        """Page number in effect at a character offset — the last marker at/before it,
        or page 1 if the offset precedes every marker (e.g. no markers at all)."""
        page = 1
        for marker_offset, page_num in page_index:
            if marker_offset <= offset:
                page = page_num
            else:
                break
        return page

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, str]]:
        """Split text into (start_offset, chunk_text) pairs, overlapping by `overlap` chars."""
        if len(text) <= chunk_size:
            return [(0, text)]
        chunks, start = [], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append((start, text[start:end]))
            if end == len(text):
                break
            start = end - overlap
        return chunks

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """Remove markdown code fences that open-source models sometimes add."""
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            # parts[1] is the fenced block content
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()
        return text

    @staticmethod
    def _repair_truncated_json(raw: str) -> str:
        """Best-effort repair of JSON truncated mid-generation (e.g. hitting
        max_tokens or a too-small context window): closes an unterminated
        string literal before closing any open brackets/braces."""
        fixed = raw.rstrip().rstrip(",")
        if len(re.findall(r'(?<!\\)"', fixed)) % 2 == 1:
            fixed += '"'
        fixed += "]" * max(0, fixed.count("[") - fixed.count("]"))
        fixed += "}" * max(0, fixed.count("{") - fixed.count("}"))
        return fixed


def main():
    """Example usage."""
    pipeline = RococoNERPipeline(
        output_dir="results",
        use_fast_model=True,      # False → qwen2.5:14b (higher accuracy)
        max_text_length=10000,
    )

    # Single document
    # result = pipeline.process_document("sample_rococo_page.jpg", language="deu+eng")
    # pipeline.print_summary(result)

    # Batch
    results = pipeline.process_batch("scanned_documents/test_batch_jpg", language="deu+eng")
    print(f"Processed {len(results)} documents")


if __name__ == "__main__":
    main()
