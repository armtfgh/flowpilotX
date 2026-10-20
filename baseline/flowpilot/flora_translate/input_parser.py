"""FLORA-Translate — Input parser: normalizes batch protocol to BatchRecord."""

import json
import logging
import time

import flora_translate.config as cfg
from flora_translate.batch_normalization import enrich_batch_record_dict
from flora_translate.engine.llm_agents import call_model_text
from flora_translate.schemas import BatchRecord

logger = logging.getLogger("flora.input_parser")

PARSE_SYSTEM = (
    "You are a chemistry data extraction assistant. "
    "Given a batch chemistry protocol (free text or structured), extract "
    "the fields into the JSON schema below. Use null for anything not stated. "
    "Return ONLY valid JSON, no markdown fences."
)

PARSE_PROMPT = """\
Extract the batch protocol fields from this input.

JSON schema:
{{
  "reaction_description": "",
  "photocatalyst": null,
  "catalyst_loading_mol_pct": null,
  "base": null,
  "solvent": null,
  "temperature_C": null,
  "reaction_time_h": null,
  "concentration_M": null,
  "scale_mmol": null,
  "yield_pct": null,
  "light_source": null,
  "wavelength_nm": null,
  "additives": [],
  "atmosphere": null
}}

Input:
{input_text}
"""

PARSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reaction_description": {"type": "string"},
        "photocatalyst": {"type": ["string", "null"]},
        "catalyst_loading_mol_pct": {"type": ["number", "null"]},
        "base": {"type": ["string", "null"]},
        "solvent": {"type": ["string", "null"]},
        "temperature_C": {"type": ["number", "null"]},
        "reaction_time_h": {"type": ["number", "null"]},
        "concentration_M": {"type": ["number", "null"]},
        "scale_mmol": {"type": ["number", "null"]},
        "yield_pct": {"type": ["number", "null"]},
        "light_source": {"type": ["string", "null"]},
        "wavelength_nm": {"type": ["number", "null"]},
        "additives": {"type": "array", "items": {"type": "string"}},
        "atmosphere": {"type": ["string", "null"]},
    },
    "required": [
        "reaction_description", "photocatalyst", "catalyst_loading_mol_pct",
        "base", "solvent", "temperature_C", "reaction_time_h",
        "concentration_M", "scale_mmol", "yield_pct", "light_source",
        "wavelength_nm", "additives", "atmosphere",
    ],
    "additionalProperties": False,
}


class InputParser:
    """Normalize user input (free text or JSON) into a BatchRecord."""

    def parse(self, batch_input: str | dict) -> BatchRecord:
        # If already a dict or JSON string representing a dict, try direct parse
        if isinstance(batch_input, dict):
            raw_text = batch_input.get("raw_text") or json.dumps(
                batch_input,
                sort_keys=True,
                default=str,
            )
            normalized = enrich_batch_record_dict(batch_input, raw_text)
            return BatchRecord(**normalized)

        # Try parsing as JSON first
        try:
            data = json.loads(batch_input)
            if isinstance(data, dict):
                normalized = enrich_batch_record_dict(data, data.get("raw_text") or batch_input)
                return BatchRecord(**normalized)
        except (json.JSONDecodeError, ValueError):
            pass

        # Free text → use LLM to extract structured fields
        logger.info("Parsing free-text batch protocol via LLM")
        return self._llm_parse(batch_input)

    def _llm_parse(self, text: str) -> BatchRecord:
        started = time.perf_counter()
        result = call_model_text(
            model=cfg.MODEL_INPUT_PARSER,
            api_name="input_parser",
            max_tokens=2048,
            system=PARSE_SYSTEM,
            user_content=PARSE_PROMPT.format(input_text=text),
            json_schema=PARSE_JSON_SCHEMA,
        )
        logger.debug("Input parser LLM call completed in %.2f ms", (time.perf_counter() - started) * 1000)
        raw = result.text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                raise
            data = json.loads(raw[start : end + 1])
        data["raw_text"] = text
        normalized = enrich_batch_record_dict(data, text)
        return BatchRecord(**normalized)
