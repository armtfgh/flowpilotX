"""FLORA — Flow Literature Oriented Retrieval Agent

5-pass PDF extraction pipeline:
 │
 ▼
Pass 1 — Document Intelligence      ← text-only (no vision needed)
 │
 ▼
Pass 2 — Chemistry Extraction       ← text-only (figures handled in Pass 4)
 │
 ▼
Pass 3 — Quantitative Extraction    ← vision @ 150 DPI (needs table accuracy)
 │
 ▼
Pass 4 — Visual Extraction          ← Haiku classifies, Sonnet extracts
 │
 ▼
Pass 5 — Synthesis + Validation     ← text-only merge
 │
 ▼
Final JSON Record

Cost optimizations:
 - Passes 1-2 use extracted text instead of page images (~90% input savings)
 - Pass 3 renders at 150 DPI instead of 200 (~40% fewer image tokens)
 - Pass 4 classification uses Haiku (~10x cheaper than Sonnet for a trivial task)
 - Pass 4 skips full extraction for figures classified as "other"
 - Prompt caching enabled on system instructions and image blocks
 - Batch API support for folder processing (50% discount)
"""

import json
import base64
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ── Claude models ──────────────────────────────────────────────────────────
MODEL_CLAUDE       = "claude-sonnet-4-20250514"
MODEL_CLAUDE_CHEAP = "claude-haiku-4-5-20251001"   # figure classification

# ── OpenAI models ──────────────────────────────────────────────────────────
MODEL_GPT          = "gpt-4o"           # vision + text extraction
MODEL_GPT_CHEAP    = "gpt-4o-mini"      # figure classification

# ── Active provider (overridden by CLI --provider flag or set_provider()) ──
# "anthropic" or "openai"
PROVIDER = "anthropic"

# Resolved at runtime by _active_models()
MODEL       = MODEL_CLAUDE
MODEL_CHEAP = MODEL_CLAUDE_CHEAP

OUTPUT_DIR = Path("extraction_results")
COST_LOG   = Path("extraction_results") / "_cost_log.json"
VISION_DPI = 150
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# Pricing per 1M tokens (USD)
PRICING = {
    MODEL_CLAUDE:       {"input": 3.00,  "output": 15.00, "cache_read": 0.30,  "cache_write": 3.75},
    MODEL_CLAUDE_CHEAP: {"input": 0.80,  "output": 4.00,  "cache_read": 0.08,  "cache_write": 1.00},
    MODEL_GPT:          {"input": 2.50,  "output": 10.00, "cache_read": 0.00,  "cache_write": 0.00},
    MODEL_GPT_CHEAP:    {"input": 0.15,  "output": 0.60,  "cache_read": 0.00,  "cache_write": 0.00},
}


def set_provider(provider: str) -> None:
    """Switch the active provider. Call before running any extraction.

    Args:
        provider: "anthropic" (default) or "openai"
    """
    global PROVIDER, MODEL, MODEL_CHEAP
    provider = provider.lower().strip()
    if provider not in ("anthropic", "openai"):
        raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'openai'.")
    PROVIDER    = provider
    MODEL       = MODEL_GPT       if provider == "openai" else MODEL_CLAUDE
    MODEL_CHEAP = MODEL_GPT_CHEAP if provider == "openai" else MODEL_CLAUDE_CHEAP
    logger.info(f"Provider set to: {PROVIDER} (main={MODEL}, cheap={MODEL_CHEAP})")

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("flora")

# Lazy clients — instantiated on first use so missing keys don't crash at import
_anthropic_client = None
_openai_client    = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


# ---------------------------------------------------------------------------
# Cost Tracker
# ---------------------------------------------------------------------------


@dataclass
class CostTracker:
    """Tracks token usage and cost across all API calls in a session."""

    calls: list[dict] = field(default_factory=list)

    def record(self, model: str, usage, label: str = "") -> float:
        """Record a single API call's usage. Returns the cost of this call."""
        prices = PRICING.get(model, PRICING[MODEL])
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        cost = (
            (input_tokens / 1_000_000) * prices["input"]
            + (output_tokens / 1_000_000) * prices["output"]
            + (cache_read / 1_000_000) * prices["cache_read"]
            + (cache_write / 1_000_000) * prices["cache_write"]
        )

        entry = {
            "label": label,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "cost_usd": round(cost, 6),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.calls.append(entry)
        return cost

    @property
    def total_cost(self) -> float:
        return sum(c["cost_usd"] for c in self.calls)

    @property
    def total_input(self) -> int:
        return sum(c["input_tokens"] for c in self.calls)

    @property
    def total_output(self) -> int:
        return sum(c["output_tokens"] for c in self.calls)

    def summary(self, label: str = "") -> str:
        """Human-readable cost summary."""
        if label:
            calls = [c for c in self.calls if c["label"].startswith(label)]
        else:
            calls = self.calls
        total = sum(c["cost_usd"] for c in calls)
        inp = sum(c["input_tokens"] for c in calls)
        out = sum(c["output_tokens"] for c in calls)
        cr = sum(c["cache_read_tokens"] for c in calls)
        cw = sum(c["cache_write_tokens"] for c in calls)
        n = len(calls)
        lines = [
            f"{'[' + label + '] ' if label else ''}Cost Summary:",
            f"  API calls:        {n}",
            f"  Input tokens:     {inp:,}",
            f"  Output tokens:    {out:,}",
            f"  Cache read:       {cr:,}",
            f"  Cache write:      {cw:,}",
            f"  Total cost:       ${total:.4f}",
        ]
        return "\n".join(lines)

    def save(self, path: Path = COST_LOG) -> None:
        """Append session costs to a persistent JSON log file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, ValueError):
                existing = []
        existing.extend(self.calls)
        path.write_text(json.dumps(existing, indent=2))


# Global tracker for the current session
tracker = CostTracker()

# ---------------------------------------------------------------------------
# System instructions (cacheable — shared prefix across all calls)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = [
    {
        "type": "text",
        "text": (
            "You are a chemistry data extraction assistant. "
            "Return ONLY valid JSON — no markdown fences, no commentary. "
            "Use null for any field whose value is not explicitly stated in the paper. "
            "NEVER guess or infer values."
        ),
        "cache_control": {"type": "ephemeral"},
    }
]

# ===========================================================================
# PASS PROMPTS
# ===========================================================================

# Pass 1 — Document Intelligence
PASS1_PROMPT = """
Read this chemistry paper and return a document map. JSON:
{
  "paper_type": "<primary_research|review|communication|perspective>",
  "chemistry_class": "",
  "reaction_type": "",
  "process_mode": "<batch|flow|both>",
  "has_batch_baseline": true,
  "has_optimization_table": true,
  "has_scope_table": true,
  "has_reactor_scheme": true,
  "has_kinetic_plots": true,
  "has_mechanistic_scheme": true,
  "sections_present": [],
  "number_of_tables": 0,
  "number_of_figures": 0,
  "key_claim": "",
  "confidence": 1
}
"""

# Pass 2 — Chemistry Extraction
# BACKWARD-COMPATIBLE NOTE: All original fields (photocatalyst, co_catalyst,
# mechanistic_details) are preserved unchanged. The new "catalyst" field is
# ADDITIVE — it captures the primary catalyst for any chemistry type.
# Photochem papers: both "catalyst" and "photocatalyst" will be filled.
# Thermal/other papers: "catalyst" filled, "photocatalyst" null.
PASS2_PROMPT = """
Focus ONLY on the chemistry described in this paper.
Extract from the text. JSON:
{
  "reaction_name": "",
  "reaction_class": "",
  "mechanism_type": "",

  "substrates": {
    "substrate_A": "",
    "substrate_B": "",
    "limiting_reagent": "",
    "stoichiometry": ""
  },

  "catalyst": {
    "name": "",
    "type": "",
    "loading_mol_pct": null,
    "role": ""
  },

  "photocatalyst": {
    "name": "",
    "loading_mol_pct": null,
    "role": "",
    "absorption_max_nm": null
  },

  "co_catalyst": {
    "name": "",
    "loading_mol_pct": null,
    "role": ""
  },

  "additives": [
    {"name": "", "equiv": null, "role": ""}
  ],

  "base": {"name": "", "equiv": null},
  "oxidant": {"name": "", "equiv": null},
  "reductant": {"name": "", "equiv": null},

  "product": "",
  "bond_formed": "",

  "mechanistic_details": {
    "excited_state_lifetime_ns": null,
    "redox_potential_discussed": false,
    "triplet_state_involved": false,
    "key_intermediate": ""
  },

  "scope_summary": ""
}

RULES:
- "catalyst": fill for ANY reaction type. For photocatalysis, copy the photocatalyst
  name here. For thermal reactions, fill with the metal catalyst (e.g. "Pd(OAc)2").
  For enzymatic reactions, fill with the enzyme name. "type" = "photocatalyst" |
  "metal_catalyst" | "organocatalyst" | "enzyme" | "electrocatalyst" | "other".
- "photocatalyst": fill ONLY if the reaction uses a photocatalyst (light-driven).
  Set all subfields to null for non-photochemical reactions.
- All other fields: unchanged from previous format — null for anything not stated.
"""

# Pass 3 — Quantitative Conditions Extraction
PASS3_PROMPT = """
Extract ALL quantitative reaction and process conditions from this paper.
Read BOTH the experimental text AND all tables carefully.

RULES:
- Convert all times to MINUTES, temperatures to CELSIUS.
- For optimization/screening tables: extract the OPTIMAL entry (last entry,
  highlighted row, or entry the authors call "optimal" or "best").
- For flow rates in µL/min: convert to mL/min (divide by 1000).
- null for anything not explicitly stated. Do NOT guess.
- If a value appears in both text and table and they conflict,
  use the table value and note the conflict.

JSON:
{
  "batch_baseline": {
    "temperature_C": null,
    "reaction_time_min": null,
    "solvent": "",
    "concentration_M": null,
    "light_source": "",
    "light_power_W": null,
    "atmosphere": "",
    "yield_percent": null,
    "ee_percent": null,
    "scale_mmol": null,
    "vessel": ""
  },

  "flow_optimized": {
    "temperature_C": null,
    "residence_time_min": null,
    "flow_rate_total_mL_min": null,
    "flow_rate_stream_A_mL_min": null,
    "flow_rate_stream_B_mL_min": null,
    "concentration_M": null,
    "back_pressure_bar": null,
    "atmosphere": "",
    "yield_percent": null,
    "ee_percent": null,
    "scale_mmol": null,
    "throughput_mmol_h": null
  },

  "optimization_table": {
    "variables_screened": [],
    "number_of_entries": null,
    "optimal_entry_number": null,
    "key_findings": ""
  },

  "reactor": {
    "type": "",
    "material": "",
    "volume_mL": null,
    "tubing_diameter_mm": null,
    "channel_depth_mm": null
  },

  "light_source": {
    "type": "",
    "wavelength_nm": null,
    "power_W": null,
    "irradiance_mW_cm2": null,
    "color": "",
    "position_relative_to_reactor": ""
  },

  "pump": {
    "type": "",
    "number_of_inlets": null
  },

  "inline_analytics": [],
  "degassing_method": ""
}
"""

# Pass 4 — Figure type-specific extraction prompts
FIGURE_TYPE_PROMPTS = {
    "reactor_scheme": """
This is a reactor/flow setup schematic. Extract:
{
  "figure_type": "reactor_scheme",
  "components": [],
  "flow_path": "",
  "number_of_inlets": null,
  "mixer_type": "",
  "reactor_geometry": "",
  "light_source_position": "",
  "BPR_present": false,
  "inline_analytics": [],
  "labeled_dimensions": {},
  "key_information": ""
}
""",
    "optimization_plot": """
This is a plot showing reaction optimization (yield/conversion vs a parameter).
Extract:
{
  "figure_type": "optimization_plot",
  "x_axis": "",
  "y_axis": "",
  "x_range": [null, null],
  "y_range": [null, null],
  "data_points": [
    {"x": null, "y": null, "label": ""}
  ],
  "trend": "",
  "optimal_x": null,
  "optimal_y": null,
  "number_of_series": null,
  "series_labels": []
}
""",
    "scope_table": """
This is a reaction scope figure showing substrate examples and yields.
Extract:
{
  "figure_type": "scope_table",
  "total_examples": null,
  "yield_range_percent": [null, null],
  "substrate_classes": [],
  "functional_groups_tolerated": [],
  "notable_failures": [],
  "ee_range_percent": [null, null]
}
""",
    "mechanistic_scheme": """
This is a mechanistic/catalytic cycle figure.
Extract:
{
  "figure_type": "mechanistic_scheme",
  "mechanism_type": "",
  "key_intermediates": [],
  "oxidation_states": {},
  "rate_determining_step": "",
  "key_observations_supporting_mechanism": []
}
""",
    "other": """
Describe this figure briefly.
{
  "figure_type": "other",
  "description": "",
  "relevant_data": {}
}
""",
}

# ===========================================================================
# HELPERS
# ===========================================================================


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from an LLM response, tolerating markdown fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF. Used for passes 1-2."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"--- PAGE {i + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages_text)


def _load_pdf_as_base64_pages(pdf_path: str, dpi: int = VISION_DPI) -> list[str]:
    """Render each PDF page as a base64-encoded PNG for Claude vision input."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        pages.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
    doc.close()
    return pages


def _build_vision_content(
    pages_b64: list[str], prompt: str, *, cache_images: bool = False
) -> list[dict]:
    """Build Claude-format content array: page images + text prompt.

    For OpenAI, this same list is converted inside _call_llm.
    """
    content = []
    for i, b64 in enumerate(pages_b64):
        block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        }
        if cache_images and i == len(pages_b64) - 1:
            block["cache_control"] = {"type": "ephemeral"}
        content.append(block)
    content.append({"type": "text", "text": prompt})
    return content


def _claude_to_openai_content(content: list[dict]) -> list[dict]:
    """Convert Anthropic-format content blocks to OpenAI vision format."""
    oai = []
    for block in content:
        if block.get("type") == "image":
            src = block.get("source", {})
            b64 = src.get("data", "")
            oai.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            })
        elif block.get("type") == "text":
            oai.append({"type": "text", "text": block["text"]})
    return oai


def _call_llm(
    content: list[dict],
    max_tokens: int = 4096,
    model: str | None = None,
    label: str = "",
    cheap: bool = False,
) -> dict:
    """Provider-agnostic LLM call. Routes to Anthropic or OpenAI based on PROVIDER.

    Args:
        content:    Anthropic-format content list (text/image blocks).
        max_tokens: Max output tokens.
        model:      Override model name (uses global MODEL / MODEL_CHEAP if None).
        label:      Cost tracking label.
        cheap:      If True, use the cheap model (Haiku / GPT-4o-mini).
    """
    if model is None:
        model = MODEL_CHEAP if cheap else MODEL

    system_text = (
        "You are a chemistry data extraction assistant. "
        "Return ONLY valid JSON — no markdown fences, no commentary. "
        "Use null for any field whose value is not explicitly stated in the paper. "
        "NEVER guess or infer values."
    )

    if PROVIDER == "openai":
        oai_content = _claude_to_openai_content(content)
        resp = _get_openai().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user",   "content": oai_content},
            ],
        )
        text = resp.choices[0].message.content

        # Track cost from OpenAI usage object
        class _Usage:
            input_tokens  = resp.usage.prompt_tokens
            output_tokens = resp.usage.completion_tokens
            cache_read_input_tokens       = 0
            cache_creation_input_tokens   = 0

        cost = tracker.record(model, _Usage(), label=label)
        logger.debug(f"    [{label}] ${cost:.4f} ({resp.usage.prompt_tokens}in/{resp.usage.completion_tokens}out) [openai]")
        return _parse_json(text)

    else:
        # Anthropic — use cached system instruction for the main model
        if cheap:
            sys_param = system_text   # Haiku: plain string (no cache overhead)
        else:
            sys_param = [{"type": "text", "text": system_text,
                          "cache_control": {"type": "ephemeral"}}]

        resp = _get_anthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=sys_param,
            messages=[{"role": "user", "content": content}],
        )
        cost = tracker.record(model, resp.usage, label=label)
        logger.debug(f"    [{label}] ${cost:.4f} ({resp.usage.input_tokens}in/{resp.usage.output_tokens}out) [anthropic]")
        return _parse_json(resp.content[0].text)


# Keep backward-compatible alias
_call_claude = _call_llm


def _already_extracted(pdf_name: str, output_dir: Path) -> bool:
    """Check if a result file already exists for this PDF."""
    result_path = output_dir / f"{Path(pdf_name).stem}.json"
    return result_path.exists()


def _save_result(record: dict, output_dir: Path) -> Path:
    """Immediately write one paper's extraction result to disk."""
    pdf_stem = Path(record["source_pdf"]).stem
    out_path = output_dir / f"{pdf_stem}.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return out_path


# ===========================================================================
# PASS EXECUTION FUNCTIONS
# ===========================================================================


def run_pass1(pdf_text: str, pdf_name: str = "") -> dict:
    """Pass 1 — Document Intelligence (text-only, no vision needed)."""
    logger.info("  Pass 1: Document Intelligence [text]")
    content = [{"type": "text", "text": pdf_text + "\n\n" + PASS1_PROMPT}]
    return _call_claude(content, max_tokens=1024, label=f"{pdf_name}:pass1")


def run_pass2(pdf_text: str, pdf_name: str = "") -> dict:
    """Pass 2 — Chemistry Extraction (text-only, figures handled in Pass 4)."""
    logger.info("  Pass 2: Chemistry Extraction [text]")
    content = [{"type": "text", "text": pdf_text + "\n\n" + PASS2_PROMPT}]
    return _call_claude(content, max_tokens=2048, label=f"{pdf_name}:pass2")


def run_pass3(pages_b64: list[str], pdf_name: str = "") -> dict:
    """Pass 3 — Quantitative Conditions (vision — needs table accuracy)."""
    logger.info("  Pass 3: Quantitative Conditions Extraction [vision]")
    content = _build_vision_content(pages_b64, PASS3_PROMPT, cache_images=True)
    return _call_claude(content, max_tokens=4096, label=f"{pdf_name}:pass3")


# ---------------------------------------------------------------------------
# Pass 4 — Figure-by-figure visual extraction
# ---------------------------------------------------------------------------


def extract_figures_with_captions(pdf_path: str) -> list[dict]:
    """Extract figures + nearby caption text as paired items."""
    doc = fitz.open(pdf_path)
    figures = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for img in page.get_images(full=True):
            xref = img[0]
            image = doc.extract_image(xref)

            # Skip logos/icons (too small)
            if len(image["image"]) < 15_000:
                continue

            caption = _find_nearby_caption(blocks)

            figures.append(
                {
                    "page": page_num + 1,
                    "data": image["image"],
                    "ext": image["ext"],
                    "caption": caption,
                }
            )

    doc.close()
    return figures


def _find_nearby_caption(blocks) -> str:
    """Heuristic: collect text blocks that look like figure captions."""
    captions = []
    for block in blocks:
        if block["type"] == 0:  # text block
            text = " ".join(
                span["text"]
                for line in block["lines"]
                for span in line["spans"]
            )
            if any(
                text.strip().startswith(p)
                for p in ["Fig", "Figure", "Scheme", "Chart"]
            ):
                captions.append(text[:400])
    return " | ".join(captions)


def _detect_media_type(ext: str) -> str:
    """Map image file extension to MIME type."""
    mapping = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    return mapping.get(ext.lower(), "image/png")


def _image_content_block(b64: str, media_type: str) -> dict:
    """Build an Anthropic-format image block (converted to OpenAI format inside _call_llm)."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64},
    }


def classify_figure(image_data: bytes, ext: str, caption: str) -> str:
    """Step 1 of Pass 4: classify figure using cheap model (Haiku or GPT-4o-mini)."""
    b64 = base64.b64encode(image_data).decode("utf-8")
    classify_prompt = (
        f'Look at this figure from a chemistry paper.\n'
        f'Caption: "{caption}"\n\n'
        f'Classify it as one of:\n'
        f'"reactor_scheme" | "optimization_plot" | "scope_table" | '
        f'"mechanistic_scheme" | "other"\n\n'
        f'Return JSON: {{"figure_type": "..."}}'
    )
    content = [
        _image_content_block(b64, _detect_media_type(ext)),
        {"type": "text", "text": classify_prompt},
    ]
    result = _call_llm(content, max_tokens=64, cheap=True, label="pass4:classify")
    fig_type = result.get("figure_type", "other")
    if fig_type not in FIGURE_TYPE_PROMPTS:
        fig_type = "other"
    return fig_type


def extract_figure(image_data: bytes, ext: str, fig_type: str, caption: str) -> dict:
    """Step 2 of Pass 4: type-specific extraction using main model."""
    b64 = base64.b64encode(image_data).decode("utf-8")
    content = [
        _image_content_block(b64, _detect_media_type(ext)),
        {"type": "text", "text": FIGURE_TYPE_PROMPTS[fig_type]},
    ]
    result = _call_llm(content, max_tokens=1024, label=f"pass4:extract:{fig_type}")
    result["caption"] = caption
    result["figure_type"] = fig_type
    return result


def run_pass4(pdf_path: str) -> list[dict]:
    """Pass 4 — Visual Extraction: Haiku classifies, Sonnet extracts.

    Figures classified as "other" are recorded but NOT sent for full
    extraction (saves one Sonnet call per irrelevant figure).
    """
    logger.info("  Pass 4: Figure-by-Figure Visual Extraction")
    figures = extract_figures_with_captions(pdf_path)
    logger.info(f"    Found {len(figures)} figures above size threshold")

    results = []
    for i, fig in enumerate(figures):
        logger.info(f"    Figure {i + 1}/{len(figures)} (page {fig['page']})")
        try:
            # Step 1: classify with Haiku (cheap)
            fig_type = classify_figure(fig["data"], fig["ext"], fig["caption"])
            logger.info(f"      Classified as: {fig_type}")

            # Step 2: extract with Sonnet — but skip "other" (no useful data)
            if fig_type == "other":
                logger.info("      Skipping extraction (type=other)")
                results.append(
                    {
                        "figure_type": "other",
                        "page": fig["page"],
                        "caption": fig["caption"],
                        "description": "Non-chemistry figure, skipped extraction.",
                    }
                )
            else:
                extracted = extract_figure(
                    fig["data"], fig["ext"], fig_type, fig["caption"]
                )
                extracted["page"] = fig["page"]
                results.append(extracted)

        except Exception as e:
            logger.warning(f"      Failed to process figure {i + 1}: {e}")
            results.append(
                {
                    "figure_type": "error",
                    "page": fig["page"],
                    "caption": fig["caption"],
                    "error": str(e),
                }
            )
    return results


# ---------------------------------------------------------------------------
# Pass 5 — Synthesis and Validation
# ---------------------------------------------------------------------------

PASS5_PROMPT_TEMPLATE = """\
You are a chemistry data curator. Below are extraction results from
multiple passes over the same paper. Your job is to:

1. Merge them into one final, unified JSON record using EXACTLY this schema:
{{
  "chemistry": {{...pass2 fields...}},
  "batch_baseline": {{...from pass3...}},
  "flow_optimized": {{...from pass3...}},
  "optimization_table": {{...from pass3...}},
  "reactor": {{...from pass3...}},
  "light_source": {{...from pass3...}},
  "pump": {{...from pass3...}},
  "figures": [...pass4 results...],
  "translation_logic": {{
    "batch_limitation": "",
    "flow_advantage": "",
    "what_changed": "",
    "what_stayed_same": "",
    "time_reduction_factor": null,
    "yield_change_percent": null,
    "safety_improvement": "",
    "scale_up_discussed": false
  }},
  "process_figures": {{
    "has_reactor_scheme": false,
    "has_flow_diagram": false,
    "has_optimization_table": false,
    "figure_descriptions": []
  }},
  "field_sources": {{}},
  "field_confidence": {{}},
  "conflicts": [],
  "confidence_overall": 0,
  "notes": ""
}}

2. Resolve conflicts: if text says 8 min and table says 10 min,
   use the TABLE value (higher fidelity) and note the conflict.
3. Cross-validate: if a reactor scheme figure shows a BPR but
   pass3 says back_pressure_bar=null, set BPR to true and annotate.
4. Fill derived fields: if flow_rate_total is null but stream A and B
   are given, compute the total. Same for time_reduction_factor, etc.
5. "field_sources" maps field names to "text" | "table" | "figure".
6. "field_confidence" maps field names to 3=certain, 2=probable, 1=inferred.
7. "conflicts" lists any remaining inconsistencies between passes.
8. "confidence_overall" is 1-100 based on how complete the extraction is.

Pass 1 (document map):
{pass1}

Pass 2 (chemistry):
{pass2}

Pass 3 (quantitative):
{pass3}

Pass 4 (figures):
{pass4}

Return the merged JSON record.
"""


def run_pass5(
    pass1: dict, pass2: dict, pass3: dict, pass4_figures: list[dict],
    pdf_name: str = "",
) -> dict:
    """Pass 5 — Synthesis: merge all passes, resolve conflicts, add confidence."""
    logger.info("  Pass 5: Synthesis and Validation [text]")
    prompt = PASS5_PROMPT_TEMPLATE.format(
        pass1=json.dumps(pass1, indent=2),
        pass2=json.dumps(pass2, indent=2),
        pass3=json.dumps(pass3, indent=2),
        pass4=json.dumps(pass4_figures, indent=2),
    )
    content = [{"type": "text", "text": prompt}]
    return _call_claude(content, max_tokens=8192, label=f"{pdf_name}:pass5")


# ===========================================================================
# PIPELINE ORCHESTRATOR
# ===========================================================================


def extract_paper(pdf_path: str, output_dir: Path = OUTPUT_DIR) -> dict | None:
    """Run the full 5-pass pipeline on a single PDF and save result to disk."""
    pdf_name = os.path.basename(pdf_path)

    if _already_extracted(pdf_name, output_dir):
        logger.info(f"Skipping {pdf_name} — already extracted")
        return None

    logger.info(f"Processing: {pdf_name}")
    start = time.time()

    # Extract text once for passes 1-2 (cheap — no vision)
    pdf_text = _extract_pdf_text(pdf_path)
    logger.info(f"  Extracted text: {len(pdf_text)} chars")

    # Render pages at 150 DPI for pass 3 only (vision — table reading)
    pages_b64 = _load_pdf_as_base64_pages(pdf_path)
    logger.info(f"  Rendered {len(pages_b64)} pages at {VISION_DPI} DPI")

    # --- Sequential pass execution ---
    stem = Path(pdf_name).stem
    pass1 = run_pass1(pdf_text, stem)
    pass2 = run_pass2(pdf_text, stem)
    pass3 = run_pass3(pages_b64, stem)
    pass4_figures = run_pass4(pdf_path)
    final_record = run_pass5(pass1, pass2, pass3, pass4_figures, stem)

    # Attach metadata
    final_record["source_pdf"] = pdf_name

    # Save immediately (incremental — protects against crashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _save_result(final_record, output_dir)

    elapsed = time.time() - start
    logger.info(f"  Done in {elapsed:.1f}s -> {out_path}")
    logger.info(f"\n{tracker.summary(stem)}")
    tracker.save()
    return final_record


# ---------------------------------------------------------------------------
# Batch API support — 50% discount for non-urgent folder processing
# ---------------------------------------------------------------------------


def _build_batch_request(
    custom_id: str,
    content: list[dict],
    max_tokens: int,
    model: str | None = None,
) -> dict:
    """Build a single request object for the Anthropic Batch API."""
    return {
        "custom_id": custom_id,
        "params": {
            "model": model or MODEL,
            "max_tokens": max_tokens,
            "system": SYSTEM_INSTRUCTION,
            "messages": [{"role": "user", "content": content}],
        },
    }


def extract_folder_batch(
    folder_path: str, output_dir: Path = OUTPUT_DIR
) -> str:
    """Submit all PDFs as a Batch API job (50% cheaper, async).

    Returns the batch ID. Use `poll_batch()` to check status and
    download results.
    """
    folder = Path(folder_path)
    pdfs = sorted(folder.glob("*.pdf"))
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = []
    for pdf in pdfs:
        pdf_name = pdf.name
        if _already_extracted(pdf_name, output_dir):
            logger.info(f"Skipping {pdf_name} — already extracted")
            continue

        pdf_text = _extract_pdf_text(str(pdf))
        pages_b64 = _load_pdf_as_base64_pages(str(pdf))
        stem = pdf.stem

        # Pass 1 — text
        content1 = [{"type": "text", "text": pdf_text + "\n\n" + PASS1_PROMPT}]
        requests.append(_build_batch_request(f"{stem}__pass1", content1, 1024))

        # Pass 2 — text
        content2 = [{"type": "text", "text": pdf_text + "\n\n" + PASS2_PROMPT}]
        requests.append(_build_batch_request(f"{stem}__pass2", content2, 2048))

        # Pass 3 — vision
        content3 = _build_vision_content(pages_b64, PASS3_PROMPT)
        requests.append(_build_batch_request(f"{stem}__pass3", content3, 4096))

    if not requests:
        logger.info("No new PDFs to process")
        return ""

    logger.info(f"Submitting batch with {len(requests)} requests")
    batch = _get_anthropic().messages.batches.create(requests=requests)
    logger.info(f"Batch submitted: {batch.id}")
    return batch.id


def poll_batch(batch_id: str) -> dict:
    """Check batch status. Returns the batch object."""
    batch = _get_anthropic().messages.batches.retrieve(batch_id)
    logger.info(
        f"Batch {batch_id}: status={batch.processing_status}, "
        f"succeeded={batch.request_counts.succeeded}, "
        f"failed={batch.request_counts.errored}"
    )
    return batch


# ---------------------------------------------------------------------------
# Standard folder processing (synchronous, real-time)
# ---------------------------------------------------------------------------


def extract_folder(folder_path: str, output_dir: Path = OUTPUT_DIR) -> list[dict]:
    """Process all PDFs in a folder. Safe to re-run (skips already-extracted)."""
    folder = Path(folder_path)
    pdfs = sorted(folder.glob("*.pdf"))

    if not pdfs:
        logger.warning(f"No PDFs found in {folder}")
        return []

    logger.info(f"Found {len(pdfs)} PDFs in {folder}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, pdf in enumerate(pdfs):
        logger.info(f"\n[{i + 1}/{len(pdfs)}] {pdf.name}")
        try:
            result = extract_paper(str(pdf), output_dir)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"  FAILED: {pdf.name} — {e}")
            failure = {"source_pdf": pdf.name, "error": str(e), "status": "failed"}
            fail_path = output_dir / f"{pdf.stem}_FAILED.json"
            fail_path.write_text(json.dumps(failure, indent=2))

    logger.info(f"\nExtraction complete: {len(results)} papers processed")
    logger.info(f"\n{tracker.summary()}")
    tracker.save()
    logger.info(f"Cost log saved to {COST_LOG}")
    return results


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def show_cost_report(log_path: Path = COST_LOG) -> None:
    """Print a cost report from the persistent cost log."""
    if not log_path.exists():
        print("No cost data yet. Run an extraction first.")
        return

    calls = json.loads(log_path.read_text())
    total = sum(c["cost_usd"] for c in calls)
    inp = sum(c["input_tokens"] for c in calls)
    out = sum(c["output_tokens"] for c in calls)
    cr = sum(c.get("cache_read_tokens", 0) for c in calls)
    cw = sum(c.get("cache_write_tokens", 0) for c in calls)

    # Group by PDF
    by_pdf = {}
    for c in calls:
        parts = c["label"].split(":")
        pdf = parts[0] if parts[0] else "(unknown)"
        by_pdf.setdefault(pdf, []).append(c)

    print("=" * 60)
    print("FLORA — Cumulative API Cost Report")
    print("=" * 60)
    print(f"  Total API calls:    {len(calls)}")
    print(f"  Input tokens:       {inp:,}")
    print(f"  Output tokens:      {out:,}")
    print(f"  Cache read tokens:  {cr:,}")
    print(f"  Cache write tokens: {cw:,}")
    print(f"  TOTAL COST:         ${total:.4f}")
    print("-" * 60)
    print("  Per paper:")
    for pdf, pdf_calls in sorted(by_pdf.items()):
        pdf_cost = sum(c["cost_usd"] for c in pdf_calls)
        print(f"    {pdf:40s}  ${pdf_cost:.4f}  ({len(pdf_calls)} calls)")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python paper_knowledge_extractor.py <pdf_or_folder> [output_dir]\n"
            "  python paper_knowledge_extractor.py --batch <folder> [output_dir]\n"
            "  python paper_knowledge_extractor.py --poll <batch_id>\n"
            "  python paper_knowledge_extractor.py --cost\n"
            "\n"
            "Provider options (add to any command):\n"
            "  --provider anthropic   Use Claude Sonnet 4 (default)\n"
            "  --provider openai      Use GPT-4o\n"
            "\n"
            "Examples:\n"
            "  python paper_knowledge_extractor.py paper.pdf\n"
            "  python paper_knowledge_extractor.py paper.pdf --provider openai\n"
            "  python paper_knowledge_extractor.py pdfs/ --provider openai\n"
            "\n"
            "Other options:\n"
            "  --batch   Submit folder as Batch API job (Anthropic only, 50% cheaper)\n"
            "  --poll    Check status of a batch job\n"
            "  --cost    Show cumulative API cost report"
        )
        sys.exit(1)

    # Parse --provider flag from anywhere in argv
    args = sys.argv[1:]
    if "--provider" in args:
        idx = args.index("--provider")
        if idx + 1 < len(args):
            set_provider(args[idx + 1])
            args = args[:idx] + args[idx + 2:]   # remove flag and value

    if not args:
        print("Error: no command given after --provider")
        sys.exit(1)

    if args[0] == "--cost":
        show_cost_report()
    elif args[0] == "--batch":
        folder = args[1]
        out = Path(args[2]) if len(args) > 2 else OUTPUT_DIR
        batch_id = extract_folder_batch(folder, out)
        if batch_id:
            print(f"Batch submitted: {batch_id}")
            print(f"Check status:  python {sys.argv[0]} --poll {batch_id}")
    elif args[0] == "--poll":
        poll_batch(args[1])
    else:
        target = args[0]
        out = Path(args[1]) if len(args) > 1 else OUTPUT_DIR

        if os.path.isfile(target) and target.lower().endswith(".pdf"):
            extract_paper(target, out)
        elif os.path.isdir(target):
            extract_folder(target, out)
        else:
            print(f"Error: {target} is not a PDF file or directory")
            sys.exit(1)
