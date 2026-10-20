"""FLORA-Fundamentals — Handbook PDF reader.

Reads handbook/textbook PDFs and extracts structured flow chemistry rules.
Supports two model backends:

  Hybrid Claude (default)
    Scan:    Claude Haiku 4.5  — cheap relevance scoring
    Extract: Claude Sonnet 4   — detailed rule extraction

  Hybrid GPT
    Scan:    GPT-4o-mini        — cheap relevance scoring
    Extract: GPT-4o             — detailed rule extraction

Two-pass mode (default):
  Pass 1 — Scan model scores relevance 0-10.
            Chunks below threshold are skipped.
  Pass 2 — Extract model reads relevant chunks in full.

This saves ~60-70% cost on large handbooks.
"""

import base64
import json
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from flora_fundamentals.schemas import FlowRule, HandbookIndex

logger = logging.getLogger("flora.fundamentals.reader")

DPI = 150
RELEVANCE_THRESHOLD = 4

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_PROFILES = {
    "Hybrid Claude": {
        "provider":      "anthropic",
        "scan_model":    "claude-haiku-4-5-20251001",
        "extract_model": "claude-sonnet-4-20250514",
        "description":   "Haiku scans, Sonnet 4 extracts — best chemistry accuracy",
    },
    "Hybrid GPT": {
        "provider":      "openai",
        "scan_model":    "gpt-4o-mini",
        "extract_model": "gpt-4o",
        "description":   "GPT-4o-mini scans, GPT-4o extracts — strong table/figure reading",
    },
}

DEFAULT_PROFILE = "Hybrid Claude"


# ---------------------------------------------------------------------------
# Prompts  (shared across providers)
# ---------------------------------------------------------------------------

SCAN_SYSTEM = """\
You are a flow chemistry expert quickly scanning a handbook page set. \
Your ONLY job is to score how relevant these pages are for extracting \
actionable flow chemistry design rules.

Relevant content includes:
- Reactor design guidelines (tubing ID, length, volume, material selection)
- Mixing rules (T-mixer, static mixer, micromixer, Dean vortices)
- Heat transfer and temperature control in flow
- Pressure drop calculations, BPR usage
- Photochemistry in flow (light penetration, LED placement, optical path)
- Solvent compatibility with tubing materials
- Safety rules (exothermic reactions, gas handling, clogging prevention)
- Scale-up guidelines (numbering up, longer operation, flow rate ranges)
- Residence time distribution, Taylor dispersion, back-mixing
- Formulas: Hagen-Poiseuille, Reynolds number, Damköhler number

NOT relevant:
- History of flow chemistry, biographies, general introductions
- Unrelated chemistry (biology, analytical, non-flow synthesis)
- References, appendices with no content, table of contents, index pages
- Marketing material, equipment vendor comparisons without technical data

Return ONLY this JSON:
{"relevance_score": <0-10>, "reason": "<one sentence>", "topics_found": ["<topic1>", ...]}

0 = completely irrelevant, 10 = dense with actionable rules."""

SCAN_USER = (
    'Scan these pages ({page_range}) from "{handbook_title}". '
    "Score relevance 0-10 for flow chemistry rule extraction."
)

EXTRACTION_SYSTEM = """\
You are a flow chemistry expert reading a handbook or textbook. Your task is \
to extract ACTIONABLE RULES that a flow chemistry process designer needs to know.

For every rule you find, structure it as:
{
  "category": "<mixing|heat_transfer|photochemistry|pressure|materials|scale_up|\
safety|residence_time|mass_transfer|reactor_design|solvent|catalyst|general>",
  "condition": "WHEN this rule applies (be specific)",
  "recommendation": "WHAT to do (be specific, include numbers)",
  "reasoning": "WHY — the chemistry or physics explanation",
  "quantitative": "Any thresholds, formulas, or numerical ranges mentioned",
  "exceptions": "When this rule does NOT apply (if mentioned)",
  "severity": "hard_rule | guideline | tip"
}

EXTRACTION RULES:
- Extract rules from TEXT, TABLES, FIGURES, and FORMULAS. Read everything.
- A "rule" is any statement that tells a chemist WHAT TO DO or WHAT TO AVOID \
in flow chemistry.
- Do NOT extract vague statements. Only extract actionable, specific rules.
- Include quantitative details: temperatures, pressures, concentrations, \
Reynolds numbers, dimensions, formulas.
- If a figure contains a graph, read the axes, trends, and key data points.
- If a table contains compatibility data, extract each row as a separate rule.

Return a JSON ARRAY of rules. Return ONLY the JSON array."""

EXTRACTION_USER = (
    "Read pages {page_range} from \"{handbook_title}\".\n\n"
    "Extract ALL actionable flow chemistry rules from the text, tables, figures, "
    "and formulas visible in these pages.\n\n"
    "Return a JSON array of rule objects."
)


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict | list:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for sc, ec in [("[", "]"), ("{", "}")]:
            s, e = text.find(sc), text.rfind(ec)
            if s != -1 and e != -1:
                return json.loads(text[s : e + 1])
        raise


# ---------------------------------------------------------------------------
# Provider-specific API callers
# ---------------------------------------------------------------------------

def _call_anthropic(
    pages_b64: list[str],
    system: str,
    user_text: str,
    model: str,
    max_tokens: int,
) -> str:
    """Call Anthropic API with vision content."""
    import anthropic as _anthropic
    client = _anthropic.Anthropic()

    content = []
    for b64 in pages_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
    content.append({"type": "text", "text": user_text})

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text


def _call_openai(
    pages_b64: list[str],
    system: str,
    user_text: str,
    model: str,
    max_tokens: int,
) -> str:
    """Call OpenAI API with vision content."""
    from openai import OpenAI
    client = OpenAI()

    # OpenAI vision: images go as image_url with data URI
    user_content = []
    for b64 in pages_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })
    user_content.append({"type": "text", "text": user_text})

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content


def _call_model(
    pages_b64: list[str],
    system: str,
    user_text: str,
    model: str,
    provider: str,
    max_tokens: int,
) -> str:
    """Route to correct provider."""
    if provider == "anthropic":
        return _call_anthropic(pages_b64, system, user_text, model, max_tokens)
    elif provider == "openai":
        return _call_openai(pages_b64, system, user_text, model, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class HandbookReader:
    """Read handbook PDFs and extract flow chemistry rules.

    Args:
        output_path: Where to save rules.json.
        model_profile: "Hybrid Claude" or "Hybrid GPT".
    """

    def __init__(
        self,
        output_path: str | Path = "flora_fundamentals/data/rules.json",
        model_profile: str = DEFAULT_PROFILE,
    ):
        self.output_path = Path(output_path)
        profile = MODEL_PROFILES.get(model_profile, MODEL_PROFILES[DEFAULT_PROFILE])
        self.provider      = profile["provider"]
        self.scan_model    = profile["scan_model"]
        self.extract_model = profile["extract_model"]
        self.profile_name  = model_profile
        logger.info(
            f"HandbookReader: profile={model_profile}, "
            f"scan={self.scan_model}, extract={self.extract_model}"
        )

    def _render_chunk(self, doc, start: int, end: int) -> list[str]:
        pages_b64 = []
        for pg_num in range(start, end):
            pix = doc[pg_num].get_pixmap(dpi=DPI)
            pages_b64.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
        return pages_b64

    def _scan_chunk(self, pages_b64: list[str], title: str, page_range: str) -> dict:
        """Pass 1: cheap scan for relevance score."""
        text = _call_model(
            pages_b64,
            system=SCAN_SYSTEM,
            user_text=SCAN_USER.format(page_range=page_range, handbook_title=title),
            model=self.scan_model,
            provider=self.provider,
            max_tokens=150,
        )
        try:
            result = _parse_json(text)
            if isinstance(result, list):
                result = result[0] if result else {}
            return result
        except Exception:
            return {"relevance_score": 5, "reason": "parse failed — including chunk", "topics_found": []}

    def _extract_chunk(self, pages_b64: list[str], title: str, page_range: str) -> list[dict]:
        """Pass 2: detailed rule extraction."""
        text = _call_model(
            pages_b64,
            system=EXTRACTION_SYSTEM,
            user_text=EXTRACTION_USER.format(page_range=page_range, handbook_title=title),
            model=self.extract_model,
            provider=self.provider,
            max_tokens=4096,
        )
        raw = _parse_json(text)
        if isinstance(raw, dict):
            raw = [raw]
        return [r for r in raw if isinstance(r, dict)]

    def read_handbook(
        self,
        pdf_path: str,
        title: str = "",
        chunk_pages: int = 5,
        two_pass: bool = True,
        relevance_threshold: int = RELEVANCE_THRESHOLD,
        progress_callback=None,
    ) -> tuple[HandbookIndex, list[FlowRule]]:
        """Read a handbook PDF and extract flow chemistry rules.

        Args:
            pdf_path: Path to the handbook PDF.
            title: Human-readable title (auto-detected if empty).
            chunk_pages: Pages per extraction call.
            two_pass: Haiku/GPT-mini scan then Sonnet/GPT-4o extract.
            relevance_threshold: Skip chunks scoring below this (0-10).
            progress_callback: fn(float, str).
        """
        pdf_path = Path(pdf_path)
        cb = progress_callback or (lambda p, m: None)

        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)

        if not title:
            first_text = doc[0].get_text("text")[:500]
            title = first_text.split("\n")[0].strip()[:100] or pdf_path.stem

        mode = "two-pass" if two_pass else "single-pass"
        logger.info(
            f"Reading: {pdf_path.name} | {n_pages} pages | "
            f"{mode} | profile={self.profile_name}"
        )

        chunks = list(range(0, n_pages, chunk_pages))
        n_chunks = len(chunks)
        all_rules: list[FlowRule] = []
        rule_counter = 0
        n_skipped = 0
        n_extracted = 0

        for chunk_idx, start_page in enumerate(chunks):
            end_page = min(start_page + chunk_pages, n_pages)
            page_range = f"{start_page + 1}-{end_page}"
            base_progress = chunk_idx / n_chunks

            cb(base_progress, f"{'Scanning' if two_pass else 'Extracting'} pages {page_range}...")

            pages_b64 = self._render_chunk(doc, start_page, end_page)

            # --- Pass 1: relevance scan ---
            if two_pass:
                try:
                    scan = self._scan_chunk(pages_b64, title, page_range)
                    score = int(scan.get("relevance_score", 5))
                    reason = scan.get("reason", "")
                    topics = scan.get("topics_found", [])
                    logger.info(f"  Pages {page_range}: score={score}/10 — {reason}")

                    if score < relevance_threshold:
                        logger.info(f"    SKIP (score {score} < threshold {relevance_threshold})")
                        n_skipped += 1
                        continue

                    logger.info(f"    RELEVANT — topics: {topics}")
                    cb(base_progress + 0.5 / n_chunks, f"Extracting pages {page_range}...")

                except Exception as e:
                    logger.warning(f"  Scan failed pages {page_range}: {e} — extracting anyway")

            # --- Pass 2: extraction ---
            try:
                raw_rules = self._extract_chunk(pages_b64, title, page_range)
                n_extracted += 1

                for r in raw_rules:
                    rule_counter += 1
                    all_rules.append(FlowRule(
                        rule_id=f"{pdf_path.stem}_r{rule_counter:03d}",
                        category=r.get("category", "general"),
                        condition=r.get("condition", ""),
                        recommendation=r.get("recommendation", ""),
                        reasoning=r.get("reasoning", ""),
                        quantitative=r.get("quantitative", ""),
                        exceptions=r.get("exceptions", ""),
                        severity=r.get("severity", "guideline"),
                        source_handbook=pdf_path.name,
                        source_page=page_range,
                    ))

                logger.info(f"    Extracted {len(raw_rules)} rules")

            except Exception as e:
                logger.warning(f"    Extraction failed pages {page_range}: {e}")

        doc.close()

        if two_pass:
            savings = n_skipped / max(n_chunks, 1) * 100
            logger.info(
                f"  Summary: {n_chunks} chunks | {n_skipped} skipped | "
                f"{n_extracted} extracted | {len(all_rules)} rules | "
                f"~{savings:.0f}% cost saved"
            )

        cb(1.0, f"Done — {len(all_rules)} rules ({n_skipped} chunks skipped)")

        return HandbookIndex(
            handbook_id=pdf_path.stem,
            filename=pdf_path.name,
            title=title,
            n_rules_extracted=len(all_rules),
            n_pages=n_pages,
        ), all_rules

    def read_folder(
        self,
        folder: str | Path,
        chunk_pages: int = 5,
        two_pass: bool = True,
        relevance_threshold: int = RELEVANCE_THRESHOLD,
        progress_callback=None,
    ) -> tuple[list[HandbookIndex], list[FlowRule]]:
        """Read all PDFs in a folder."""
        folder = Path(folder)
        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            logger.warning(f"No PDFs found in {folder}")
            return [], []

        all_indices, all_rules = [], []
        for i, pdf in enumerate(pdfs):
            logger.info(f"\n[{i+1}/{len(pdfs)}] {pdf.name}")
            try:
                idx, rules = self.read_handbook(
                    str(pdf), chunk_pages=chunk_pages,
                    two_pass=two_pass, relevance_threshold=relevance_threshold,
                    progress_callback=progress_callback,
                )
                all_indices.append(idx)
                all_rules.extend(rules)
            except Exception as e:
                logger.error(f"  FAILED: {pdf.name} — {e}")

        return all_indices, all_rules

    def save(self, indices: list[HandbookIndex], rules: list[FlowRule]):
        """Save extracted rules to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "handbooks": [h.model_dump() for h in indices],
            "rules": [r.model_dump() for r in rules],
        }
        self.output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"Saved {len(rules)} rules to {self.output_path}")
