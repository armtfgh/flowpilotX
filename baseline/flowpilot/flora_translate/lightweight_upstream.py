"""Lightweight upstream parsing and chemistry-planning path for weak/local models.

This module preserves the main upstream path and adds a smaller chemistry
planning routine that emits a reduced schema, then deterministically rebuilds a
full ChemistryPlan for the rest of FLORA.
"""

from __future__ import annotations

import json
import logging
import re
import time

import flora_translate.config as cfg
from flora_translate.batch_normalization import (
    apply_authoritative_batch_evidence,
    enrich_batch_record_dict,
    normalize_batch_scalar_fields,
)
from flora_translate.chemistry_agent import _parse_json_from_tagged
from flora_translate.component_identity import component_name, component_key, declared_solvent_member, protocol_component_quantity
from flora_translate.engine.llm_agents import call_model_text, infer_provider_for_model
from flora_translate.intake_agent import intake_context_block
from flora_translate.intensification import ensure_intensification_mandate
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    ProcessStage,
    ReagentRole,
    StreamLogic,
)

logger = logging.getLogger("flora.lightweight_upstream")


LOCAL_PARSE_SYSTEM = """\
You are a chemistry protocol extraction assistant for local-model mode.
Return only one valid JSON object. No markdown, no explanation, no notes.
Use null for unknown scalar fields and [] for unknown additive lists.
"""


LOCAL_PARSE_PROMPT = """\
Extract this batch protocol into the exact JSON schema below.

Schema:
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


LOCAL_PARSE_V2_SYSTEM = """\
You are a conservative chemistry protocol extraction assistant for fair
weak-model benchmarking. Return only one valid JSON object. No markdown.
Extract facts from the text; do not infer flow design conditions.
Use null for unknown scalar fields and [] for unknown additive lists.
For concentration_M, use a value only when the protocol explicitly states an M
unit. If the text gives mmol and mL, leave concentration_M null; deterministic
code will calculate it.
"""


LOCAL_PARSE_V2_PROMPT = """\
Extract this batch protocol into the exact JSON schema below.

Schema:
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
  "atmosphere": null,
  "evidence": {{
    "reaction_time": {{"value": null, "unit": null, "evidence_text": ""}},
    "limiting_reagent_amount": {{"value": null, "unit": null, "evidence_text": ""}},
    "reaction_volume": {{"value": null, "unit": null, "evidence_text": ""}},
    "temperature": {{"value": null, "unit": null, "evidence_text": ""}},
    "yield": {{"value": null, "unit": null, "evidence_text": ""}}
  }}
}}

Input:
{input_text}
"""


LIGHTWEIGHT_CHEMISTRY_SYSTEM = """\
You are a chemistry analysis assistant for a batch-to-flow system.
Your job is to produce a compact, machine-readable chemistry plan for weak or
local-model mode. Do not write long explanations. Return only:

<NOTES>
- 3 to 6 concise bullets
</NOTES>
<JSON>
{ ... }
</JSON>

Rules:
- Keep the JSON compact and valid.
- Prefer short lists over long prose.
- If something is uncertain, use conservative defaults instead of inventing.
- incompatible_pairs must be a list of 2-item lists when possible.
- stream_blueprint should describe chemistry-driven stream grouping, not hardware.
- Preserve each reagent's own equivalents/loading and role in reagents. Never
  replace multiple component quantities with one stream-level equivalent.
- stage_blueprint is only needed when the protocol clearly has more than one
  synthetic stage or explicit inter-stage action.
"""


LIGHTWEIGHT_CHEMISTRY_V2_SYSTEM = """\
You are a chemistry analysis assistant for a batch-to-flow system.
This is fair lightweight upstream mode: produce a compact chemistry plan, but do
not inject numeric flow-design answers. Do not propose tau, flow rate, reactor
volume, tubing length, or pressure. Return only:

<NOTES>
- 3 to 6 concise bullets tied to the batch record
</NOTES>
<JSON>
{ ... }
</JSON>

Rules:
- Keep the JSON compact and valid.
- Prefer short lists over long prose.
- Use the batch record facts as authoritative.
- If something is uncertain, use conservative chemistry labels instead of
  inventing details.
- incompatible_pairs must be a list of 2-item lists when possible.
- stream_blueprint should describe chemistry-driven stream grouping, not hardware.
- Preserve each reagent's own equivalents/loading and role in reagents. Never
  replace multiple component quantities with one stream-level equivalent.
- stage_blueprint is only needed for true multi-stage chemistry.
"""


LIGHTWEIGHT_CHEMISTRY_USER_TEMPLATE = """\
Analyze this batch protocol and return a compact chemistry plan.

BATCH RECORD:
{batch_json}

Return JSON with this schema:
{{
  "reaction_name": "",
  "reaction_class": "",
  "mechanism_type": "",
  "bond_formed": "",
  "bond_broken": "",
  "key_intermediate": "",
  "oxygen_sensitive": false,
  "moisture_sensitive": false,
  "temperature_sensitive": false,
  "light_sensitive_reagents": [],
  "reagents": [{{"name": "", "role": "", "equiv_or_loading": ""}}],
  "deoxygenation_required": false,
  "deoxygenation_reasoning": "",
  "quench_required": false,
  "quench_reagent": "",
  "quench_reasoning": "",
  "mixing_order_reasoning": "",
  "retrieval_keywords": [],
  "similar_reaction_classes": [],
  "recommended_wavelength_nm": null,
  "wavelength_reasoning": "",
  "confidence_notes": "",
  "incompatible_pairs": [],
  "stream_blueprint": [
    {{
      "stream_label": "A",
      "reagents": ["species 1", "species 2"],
      "reasoning": "",
      "molar_equiv": 1.0,
      "concentration_M": null
    }}
  ],
  "n_stages": 1,
  "stage_blueprint": [
    {{
      "stage_number": 1,
      "stage_name": "",
      "reaction_type": "",
      "reactor_type": "coil",
      "temperature_C": null,
      "requires_light": false,
      "wavelength_nm": null,
      "batch_time_h": null,
      "feed_streams": [],
      "inlet_from_previous": "",
      "solvent": "",
      "atmosphere": "",
      "oxygen_sensitive": false,
      "moisture_sensitive": false,
      "deoxygenation_required": false,
      "post_stage_action": "",
      "post_stage_reasoning": ""
    }}
  ]
}}

Keep the output compact. Do not include mechanism_steps or detailed species
catalogues. Use stream_blueprint and stage_blueprint only to the level needed
for downstream flow planning.
"""


def is_weak_or_local_model(model: str) -> bool:
    """Heuristic detector for weak/local models that should use lightweight mode."""
    name = (model or "").strip().lower()
    if not name:
        return False
    provider = infer_provider_for_model(name)
    if provider == "ollama":
        return True
    return any(marker in name for marker in cfg.LIGHTWEIGHT_UPSTREAM_WEAK_MODEL_MARKERS)


def is_local_model(model: str) -> bool:
    return infer_provider_for_model(model) == "ollama"


def _extract_json_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise json.JSONDecodeError("empty input", text, 0)
    if "<JSON>" in text:
        start = text.index("<JSON>") + len("<JSON>")
        end = text.index("</JSON>") if "</JSON>" in text else len(text)
        text = text[start:end].strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _parse_batch_record_json(raw: str) -> dict:
    text = _extract_json_text(raw)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("top-level JSON is not an object", text, 0)
    return data


def _normalize_batch_record_data(data: dict, raw_text: str) -> dict:
    normalized = dict(data)
    list_fields = ("additives",)
    for field in list_fields:
        value = normalized.get(field)
        if value is None:
            normalized[field] = []
        elif isinstance(value, str):
            parts = [part.strip() for part in re.split(r"[,\n;/]+", value) if part.strip()]
            normalized[field] = parts
        elif not isinstance(value, list):
            normalized[field] = []

    normalized = normalize_batch_scalar_fields(normalized)

    scalar_fields = (
        "reaction_description",
        "photocatalyst",
        "base",
        "solvent",
        "light_source",
        "atmosphere",
    )
    for field in scalar_fields:
        value = normalized.get(field)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            normalized[field] = str(value)
        elif not isinstance(value, str):
            normalized[field] = str(value)

    if not normalized.get("reaction_description"):
        normalized["reaction_description"] = raw_text[:500]
    normalized["raw_text"] = raw_text
    return enrich_batch_record_dict(normalized, raw_text)


def parse_batch_input(batch_input: str | dict) -> BatchRecord:
    """Route parsing through the main parser unless the selected parser is local."""
    from flora_translate.input_parser import InputParser

    if should_use_lightweight_v2(cfg.MODEL_INPUT_PARSER):
        return EvidenceBackedInputParser().parse(batch_input)
    if not is_local_model(cfg.MODEL_INPUT_PARSER):
        return InputParser().parse(batch_input)
    return LocalInputParser().parse(batch_input)


def should_use_lightweight_upstream(model: str | None = None) -> bool:
    """Decide whether to route chemistry planning through the lightweight path."""
    mode = (getattr(cfg, "LIGHTWEIGHT_UPSTREAM_MODE", "auto") or "auto").lower()
    if mode in {"always", "v2", "fair_v2"}:
        return True
    if mode == "never":
        return False
    selected_model = model or cfg.MODEL_CHEMISTRY_AGENT
    return is_weak_or_local_model(selected_model)


def should_use_lightweight_v2(model: str | None = None) -> bool:
    mode = (getattr(cfg, "LIGHTWEIGHT_UPSTREAM_MODE", "auto") or "auto").lower()
    if mode in {"v2", "fair_v2"}:
        return True
    return False


def analyze_batch_chemistry(batch_record: BatchRecord, intake_package=None, scientific=False) -> ChemistryPlan:
    """Route chemistry analysis through the appropriate upstream path."""
    if should_use_lightweight_upstream(cfg.MODEL_CHEMISTRY_AGENT):
        return LightweightChemistryReasoningAgent().analyze(
            batch_record, intake_package=intake_package, **({"scientific": True} if scientific else {})
        )
    from flora_translate.chemistry_agent import ChemistryReasoningAgent

    plan = ChemistryReasoningAgent().analyze(batch_record, intake_package=intake_package, **({"scientific": True} if scientific else {}))
    setattr(plan, "_upstream_mode", "full")
    return plan


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"true", "yes", "y", "1", "required"}
    return False


def _coerce_float(value) -> float | None:
    if value in (None, "", "N/A", "n/a", "null"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _coerce_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\n;/]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
            elif isinstance(item, dict):
                for key in ("name", "value", "label", "species"):
                    text = item.get(key)
                    if isinstance(text, str) and text.strip():
                        items.append(text.strip())
                        break
        return items
    return []


def _coerce_incompatible_pairs(value) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[;\n]+", value) if part.strip()]
        pairs: list[list[str]] = []
        for part in parts:
            if "|" in part:
                pieces = [piece.strip() for piece in part.split("|") if piece.strip()]
            elif " + " in part:
                pieces = [piece.strip() for piece in part.split(" + ") if piece.strip()]
            elif "," in part:
                pieces = [piece.strip() for piece in part.split(",") if piece.strip()]
            else:
                pieces = [part]
            if pieces:
                pairs.append(pieces[:2])
        return pairs
    if isinstance(value, list):
        pairs = []
        for item in value:
            if isinstance(item, list):
                pair = [str(piece).strip() for piece in item if str(piece).strip()]
                if pair:
                    pairs.append(pair[:2])
            elif isinstance(item, dict):
                candidates = []
                for key in ("species_1", "species_2", "a", "b", "left", "right"):
                    text = item.get(key)
                    if isinstance(text, str) and text.strip():
                        candidates.append(text.strip())
                if not candidates:
                    candidates = _coerce_string_list(item)
                if candidates:
                    pairs.append(candidates[:2])
            elif isinstance(item, str) and item.strip():
                pairs.extend(_coerce_incompatible_pairs(item))
        return pairs
    return []


def _normalize_stream_logic_entry(entry: dict, index: int) -> StreamLogic:
    label = str(entry.get("stream_label") or chr(ord("A") + index)).strip() or chr(ord("A") + index)
    reagents = _coerce_string_list(entry.get("reagents"))
    reasoning = str(entry.get("reasoning") or "").strip()
    molar_equiv = _coerce_float(entry.get("molar_equiv"))
    concentration = _coerce_float(entry.get("concentration_M"))
    return StreamLogic(
        stream_label=label,
        reagents=reagents,
        reasoning=reasoning,
        molar_equiv=molar_equiv if molar_equiv is not None else 1.0,
        concentration_M=concentration,
    )


def _fallback_stream_logic(batch_record: BatchRecord) -> list[StreamLogic]:
    reagents: list[str] = []
    for item in [batch_record.photocatalyst, batch_record.base, *(batch_record.additives or [])]:
        if item and item not in reagents:
            reagents.append(item)
    reasoning = "Lightweight fallback stream grouping; exact split deferred to downstream translation."
    return [StreamLogic(stream_label="A", reagents=reagents, reasoning=reasoning, molar_equiv=1.0)]


def _normalize_stream_logic_list(value, batch_record: BatchRecord) -> list[StreamLogic]:
    entries = value if isinstance(value, list) else []
    streams = [
        _normalize_stream_logic_entry(entry, idx)
        for idx, entry in enumerate(entries)
        if isinstance(entry, dict)
    ]
    if streams:
        return streams
    return _fallback_stream_logic(batch_record)


def _normalize_stage_list(value, fallback_streams: list[StreamLogic], batch_record: BatchRecord) -> list[ProcessStage]:
    if not isinstance(value, list):
        return []
    stages: list[ProcessStage] = []
    for idx, entry in enumerate(value):
        if not isinstance(entry, dict):
            continue
        feed_streams = _normalize_stream_logic_list(entry.get("feed_streams"), batch_record)
        if not feed_streams:
            feed_streams = fallback_streams
        stages.append(
            ProcessStage(
                stage_number=int(_coerce_float(entry.get("stage_number")) or idx + 1),
                stage_name=str(entry.get("stage_name") or "").strip(),
                reaction_type=str(entry.get("reaction_type") or "").strip(),
                reactor_type=str(entry.get("reactor_type") or "coil").strip() or "coil",
                temperature_C=_coerce_float(entry.get("temperature_C")),
                requires_light=_coerce_bool(entry.get("requires_light")),
                wavelength_nm=_coerce_float(entry.get("wavelength_nm")),
                batch_time_h=_coerce_float(entry.get("batch_time_h")),
                feed_streams=feed_streams,
                inlet_from_previous=str(entry.get("inlet_from_previous") or "").strip(),
                solvent=str(entry.get("solvent") or batch_record.solvent or "").strip(),
                atmosphere=str(entry.get("atmosphere") or batch_record.atmosphere or "").strip(),
                oxygen_sensitive=_coerce_bool(entry.get("oxygen_sensitive")),
                moisture_sensitive=_coerce_bool(entry.get("moisture_sensitive")),
                deoxygenation_required=_coerce_bool(entry.get("deoxygenation_required")),
                post_stage_action=str(entry.get("post_stage_action") or "").strip(),
                post_stage_reasoning=str(entry.get("post_stage_reasoning") or "").strip(),
            )
        )
    return stages


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        ordered.append(normalized)
    return ordered


def _build_reagent_roles(
    batch_record: BatchRecord,
    data: dict,
    stream_logic: list[StreamLogic],
    quench_reagent: str,
) -> list[ReagentRole]:
    names: list[str] = []
    for stream in stream_logic:
        names.extend(stream.reagents)
    names.extend(_coerce_string_list(data.get("light_sensitive_reagents")))
    for item in [batch_record.photocatalyst, batch_record.base, batch_record.solvent, *(batch_record.additives or [])]:
        if item:
            names.append(item)
    if quench_reagent:
        names.append(quench_reagent)

    reagent_roles: list[ReagentRole] = []
    supplied = {
        component_key(item["name"]): item for item in data.get("reagents") or []
        if isinstance(item, dict) and item.get("name")
    }
    additives = {item.lower(): item for item in (batch_record.additives or []) if item}
    for name in _dedupe_preserve_order(names):
        name = component_name(name)
        lowered = name.lower()
        item = supplied.get(component_key(name), {})
        quantity = protocol_component_quantity(name, batch_record.raw_text or "") or str(item.get("equiv_or_loading") or "")
        if batch_record.photocatalyst and lowered == batch_record.photocatalyst.lower():
            role = "photocatalyst"
        elif batch_record.base and lowered == batch_record.base.lower():
            role = "base"
        elif batch_record.solvent and lowered == batch_record.solvent.lower():
            role = "solvent"
        elif quench_reagent and lowered == quench_reagent.lower():
            role = "quencher"
        elif lowered in additives:
            role = "additive"
        elif not quantity and declared_solvent_member(name, batch_record.solvent or ""):
            role = "solvent"
        else:
            role = str(item.get("role") or "unknown")
        if role == "photocatalyst" and not quantity and batch_record.catalyst_loading_mol_pct:
            quantity = f"{batch_record.catalyst_loading_mol_pct:g} mol%"
        if component_key(name) not in {component_key(item.name) for item in reagent_roles}:
            reagent_roles.append(ReagentRole(name=name, role=role, equiv_or_loading=quantity, smiles=None, notes=""))
    return reagent_roles


def _default_reaction_class(batch_record: BatchRecord) -> str:
    if batch_record.wavelength_nm or batch_record.light_source or batch_record.photocatalyst:
        return "photoredox"
    return "thermal"


def _default_mechanism_type(batch_record: BatchRecord, reaction_class: str) -> str:
    if reaction_class:
        return reaction_class
    return _default_reaction_class(batch_record)


def _extract_reasoning(text: str) -> str:
    if "<NOTES>" in text and "</NOTES>" in text:
        start = text.index("<NOTES>") + len("<NOTES>")
        end = text.index("</NOTES>")
        return text[start:end].strip()
    if "<JSON>" in text:
        return text[: text.index("<JSON>")].strip()
    return ""


def _build_lightweight_plan(batch_record: BatchRecord, data: dict, reasoning: str) -> ChemistryPlan:
    reaction_class = str(data.get("reaction_class") or _default_reaction_class(batch_record)).strip()
    mechanism_type = str(data.get("mechanism_type") or _default_mechanism_type(batch_record, reaction_class)).strip()
    reaction_name = str(data.get("reaction_name") or batch_record.reaction_description[:120]).strip()
    quench_reagent = str(data.get("quench_reagent") or "").strip()

    stream_logic = _normalize_stream_logic_list(data.get("stream_blueprint"), batch_record)
    stages = _normalize_stage_list(data.get("stage_blueprint"), stream_logic, batch_record)
    n_stages = int(_coerce_float(data.get("n_stages")) or (len(stages) if len(stages) > 1 else 1))
    if n_stages <= 1:
        stages = []
        n_stages = 1
    elif stages and not stream_logic:
        stream_logic = stages[0].feed_streams

    retrieval_keywords = _dedupe_preserve_order(
        _coerce_string_list(data.get("retrieval_keywords"))
        + [reaction_class, mechanism_type]
        + ([batch_record.solvent] if batch_record.solvent else [])
        + ([quench_reagent] if quench_reagent else [])
        + ([batch_record.photocatalyst] if batch_record.photocatalyst else [])
    )
    similar_reaction_classes = _dedupe_preserve_order(
        _coerce_string_list(data.get("similar_reaction_classes")) + ([reaction_class] if reaction_class else [])
    )

    plan = ChemistryPlan(
        reaction_name=reaction_name,
        reaction_class=reaction_class,
        mechanism_type=mechanism_type,
        bond_formed=str(data.get("bond_formed") or "").strip(),
        bond_broken=str(data.get("bond_broken") or "").strip(),
        stages=stages,
        n_stages=n_stages,
        reagents=_build_reagent_roles(batch_record, data, stream_logic, quench_reagent),
        mechanism_steps=[],
        key_intermediate=str(data.get("key_intermediate") or "").strip(),
        excited_state_type="",
        energy_transfer_or_redox="",
        oxygen_sensitive=_coerce_bool(data.get("oxygen_sensitive")),
        moisture_sensitive=_coerce_bool(data.get("moisture_sensitive")),
        temperature_sensitive=_coerce_bool(data.get("temperature_sensitive")),
        light_sensitive_reagents=_coerce_string_list(data.get("light_sensitive_reagents")),
        stream_logic=stream_logic,
        mixing_order_reasoning=str(data.get("mixing_order_reasoning") or "").strip(),
        incompatible_pairs=_coerce_incompatible_pairs(data.get("incompatible_pairs")),
        deoxygenation_required=_coerce_bool(data.get("deoxygenation_required")),
        deoxygenation_reasoning=str(data.get("deoxygenation_reasoning") or "").strip(),
        quench_required=_coerce_bool(data.get("quench_required")),
        quench_reagent=quench_reagent,
        quench_reasoning=str(data.get("quench_reasoning") or "").strip(),
        retrieval_keywords=retrieval_keywords,
        similar_reaction_classes=similar_reaction_classes,
        recommended_wavelength_nm=_coerce_float(data.get("recommended_wavelength_nm")),
        wavelength_reasoning=str(data.get("wavelength_reasoning") or "").strip(),
        confidence_notes=str(data.get("confidence_notes") or "").strip(),
    )
    ensure_intensification_mandate(batch_record, plan)
    setattr(plan, "_reasoning", reasoning)
    setattr(plan, "_upstream_mode", "lightweight")
    return plan


class LocalInputParser:
    """Compact parser path for local/Ollama models."""

    def parse(self, batch_input: str | dict) -> BatchRecord:
        if isinstance(batch_input, dict):
            raw_text = batch_input.get("raw_text") or json.dumps(
                batch_input,
                sort_keys=True,
                default=str,
            )
            normalized = _normalize_batch_record_data(batch_input, raw_text)
            return BatchRecord(**normalized)

        try:
            data = json.loads(batch_input)
            if isinstance(data, dict):
                data["raw_text"] = batch_input
                normalized = _normalize_batch_record_data(data, batch_input)
                return BatchRecord(**normalized)
        except (json.JSONDecodeError, ValueError):
            pass

        logger.info("Parsing batch protocol via local lightweight parser: %s", cfg.MODEL_INPUT_PARSER)
        started = time.perf_counter()
        result = call_model_text(
            model=cfg.MODEL_INPUT_PARSER,
            api_name="lightweight_input_parser",
            max_tokens=cfg.LIGHTWEIGHT_LOCAL_INPUT_PARSER_MAX_TOKENS,
            system=LOCAL_PARSE_SYSTEM,
            user_content=LOCAL_PARSE_PROMPT.format(input_text=batch_input),
        )
        logger.debug(
            "Local lightweight input parser completed in %.2f ms",
            (time.perf_counter() - started) * 1000,
        )
        data = _parse_batch_record_json(result.text)
        normalized = _normalize_batch_record_data(data, batch_input)
        return BatchRecord(**normalized)


class EvidenceBackedInputParser:
    """Fair lightweight parser with deterministic protocol-text arithmetic."""

    def parse(self, batch_input: str | dict) -> BatchRecord:
        raw_text = (
            batch_input.get("raw_text")
            or json.dumps(batch_input, sort_keys=True, default=str)
            if isinstance(batch_input, dict)
            else str(batch_input)
        )
        if isinstance(batch_input, dict):
            data = dict(batch_input)
        else:
            try:
                parsed = json.loads(batch_input)
                data = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                data = self._llm_parse_to_dict(batch_input)

        data.pop("evidence", None)
        normalized = _normalize_batch_record_data(data, raw_text)
        normalized, evidence = apply_authoritative_batch_evidence(normalized, raw_text)
        record = BatchRecord(**normalized)
        setattr(record, "_evidence_report", evidence)
        setattr(record, "_input_parser_mode", "lightweight_v2")
        return record

    def _llm_parse_to_dict(self, text: str) -> dict:
        logger.info("Parsing batch protocol via evidence-backed lightweight parser: %s", cfg.MODEL_INPUT_PARSER)
        started = time.perf_counter()
        result = call_model_text(
            model=cfg.MODEL_INPUT_PARSER,
            api_name="lightweight_v2_input_parser",
            max_tokens=cfg.LIGHTWEIGHT_LOCAL_INPUT_PARSER_MAX_TOKENS,
            system=LOCAL_PARSE_V2_SYSTEM,
            user_content=LOCAL_PARSE_V2_PROMPT.format(input_text=text),
        )
        logger.debug(
            "Evidence-backed lightweight input parser completed in %.2f ms",
            (time.perf_counter() - started) * 1000,
        )
        data = _parse_batch_record_json(result.text)
        data["raw_text"] = text
        return data


class LightweightChemistryReasoningAgent:
    """Compact chemistry-planning path for weak/local models."""

    def analyze(self, batch_record: BatchRecord, intake_package=None, scientific=False) -> ChemistryPlan:
        from flora_translate.scientific_evidence import UPSTREAM_POLICY
        v2 = should_use_lightweight_v2(cfg.MODEL_CHEMISTRY_AGENT)
        logger.info(
            "  %s Chemistry Agent: Analyzing with %s",
            "Lightweight v2" if v2 else "Lightweight",
            cfg.MODEL_CHEMISTRY_AGENT,
        )
        batch_json = json.dumps(batch_record.model_dump(exclude_none=True), indent=2)
        user_prompt = LIGHTWEIGHT_CHEMISTRY_USER_TEMPLATE.format(batch_json=batch_json)
        if intake_package is not None:
            user_prompt += (
                "\n\n"
                + intake_context_block(intake_package)
                + "\n\nUse measured evidence and hard constraints as context. "
                "Treat chemist hypotheses as hypotheses to test, not as facts."
            )

        started = time.perf_counter()
        result = call_model_text(
            model=cfg.MODEL_CHEMISTRY_AGENT,
            api_name="lightweight_v2_chemistry_agent" if v2 else "lightweight_chemistry_agent",
            max_tokens=cfg.LIGHTWEIGHT_CHEMISTRY_MAX_TOKENS,
            system=(LIGHTWEIGHT_CHEMISTRY_V2_SYSTEM if v2 else LIGHTWEIGHT_CHEMISTRY_SYSTEM) + (
                "\n\n" + UPSTREAM_POLICY if scientific else ""),
            user_content=user_prompt,
        )
        logger.debug(
            "Lightweight chemistry agent LLM call completed in %.2f ms",
            (time.perf_counter() - started) * 1000,
        )

        raw_text = result.text
        reasoning = _extract_reasoning(raw_text)
        data = _parse_json_from_tagged(raw_text)
        plan = _build_lightweight_plan(batch_record, data, reasoning)
        if v2:
            setattr(plan, "_upstream_mode", "lightweight_v2")

        logger.info("    Lightweight reaction: %s (%s)", plan.reaction_name, plan.mechanism_type)
        logger.info("    %s upstream mode active", "Lightweight v2" if v2 else "Lightweight")
        logger.info("    Streams: %s", len(plan.stream_logic))
        logger.info("    Retrieval hints: %s", plan.retrieval_keywords[:5])
        return plan
