"""Deterministic recovery of basic batch quantities from protocol text.

This module is intentionally conservative. It only infers values that are
simple, local, and materially important for downstream engineering.
"""

from __future__ import annotations

import re
from typing import Any

_FLOAT_RE = r"(\d+(?:\.\d+)?)"
_EXPLICIT_CONC_RE = re.compile(rf"{_FLOAT_RE}\s*M\b", re.IGNORECASE)
_MMOL_RE = re.compile(rf"{_FLOAT_RE}\s*mmol\b", re.IGNORECASE)
_ML_RE = re.compile(rf"{_FLOAT_RE}\s*mL\b", re.IGNORECASE)
_TIME_RE = re.compile(rf"\bfor\s+{_FLOAT_RE}\s*(min|mins|minute|minutes|h|hr|hrs|hour|hours)\b", re.IGNORECASE)

_CONTAINER_WORDS = (
    "vial",
    "flask",
    "tube",
    "reactor",
    "schlenk",
    "bottle",
    "autoclave",
)
_VOLUME_HINT_WORDS = (
    "des",
    "solvent",
    "mixture",
    "medium",
    "buffer",
    "electrolyte",
    "water",
    "acetonitrile",
    "methanol",
    "ethanol",
    "dcm",
    "dmso",
    "dmf",
    "tbab",
    "eg",
)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(_FLOAT_RE, value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


_BATCH_NUMERIC_FIELDS = (
    "catalyst_loading_mol_pct",
    "temperature_C",
    "reaction_time_h",
    "concentration_M",
    "scale_mmol",
    "yield_pct",
    "wavelength_nm",
)
_BATCH_TEXT_FIELDS = (
    "reaction_description",
    "photocatalyst",
    "base",
    "solvent",
    "light_source",
    "atmosphere",
)


def _numeric_leaf(value: Any, key_hint: str = "") -> float | None:
    """Extract one numeric leaf, converting minute-labelled times to hours."""

    if isinstance(value, dict):
        unit = str(value.get("unit") or value.get("units") or key_hint).lower()
        for key in ("value", "amount", "time", "duration", "yield", "temperature"):
            if key in value:
                number = _coerce_float(value.get(key))
                if number is not None:
                    return number / 60.0 if "min" in unit else number
        return None
    number = _coerce_float(value)
    if number is not None and "min" in key_hint.lower():
        return number / 60.0
    return number


def _structured_numeric(value: Any, field: str) -> float | None:
    """Reduce occasional LLM object/list values to the scalar BatchRecord schema."""

    scalar = _coerce_float(value)
    if scalar is not None:
        return scalar
    if isinstance(value, list):
        values = [
            number
            for index, item in enumerate(value)
            if (number := _numeric_leaf(item, f"item_{index}")) is not None
        ]
        if not values:
            return None
        if field == "reaction_time_h":
            return sum(values)
        if field == "yield_pct":
            return values[-1]
        return values[0]
    if not isinstance(value, dict):
        return None

    normalized = {str(key).strip().lower(): item for key, item in value.items()}
    priorities = {
        "reaction_time_h": (
            "total_h", "total_time_h", "overall_h", "reaction_time_h",
            "total_hours", "total", "value",
        ),
        "yield_pct": (
            "overall_yield_pct", "final_yield_pct", "isolated_yield_pct",
            "yield_pct", "overall", "final", "isolated", "value",
        ),
        "temperature_C": (
            "temperature_c", "setpoint_c", "reaction_temperature_c",
            "step_1_c", "step_1", "value",
        ),
        "concentration_M": (
            "concentration_m", "overall_m", "feed_m", "value",
        ),
        "scale_mmol": ("scale_mmol", "total_mmol", "limiting_mmol", "value"),
        "catalyst_loading_mol_pct": (
            "catalyst_loading_mol_pct", "loading_mol_pct", "mol_pct", "value",
        ),
        "wavelength_nm": ("wavelength_nm", "lambda_nm", "value"),
    }.get(field, (field.lower(), "value"))
    for key in priorities:
        if key not in normalized:
            continue
        number = _numeric_leaf(normalized[key], key)
        if number is not None:
            return number

    stage_values: list[tuple[int, float]] = []
    other_values: list[float] = []
    for key, item in normalized.items():
        number = _numeric_leaf(item, key)
        if number is None:
            continue
        stage_match = re.search(r"(?:step|stage)[_\s-]*(\d+)", key)
        if stage_match:
            stage_values.append((int(stage_match.group(1)), number))
        elif key not in {"unit", "units"}:
            other_values.append(number)

    if stage_values:
        stage_values.sort(key=lambda pair: pair[0])
        if field == "reaction_time_h":
            return sum(number for _, number in stage_values)
        if field == "yield_pct":
            return stage_values[-1][1]
        return stage_values[0][1]
    return other_values[0] if other_values else None


def normalize_batch_numeric_fields(data: dict) -> dict:
    """Return a copy whose BatchRecord numeric fields are scalar or null."""

    normalized = dict(data)
    for field in _BATCH_NUMERIC_FIELDS:
        if field in normalized:
            normalized[field] = _structured_numeric(normalized.get(field), field)
    return normalized


def _structured_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [text for item in value if (text := _structured_text(item))]
        return "; ".join(parts) or None
    if isinstance(value, dict):
        for key in ("value", "name", "identity", "overall", "final"):
            if key in value and (text := _structured_text(value.get(key))):
                return text
        parts = []
        for key, item in value.items():
            if str(key).lower() in {"unit", "units"}:
                continue
            text = _structured_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts) or None
    return str(value)


def normalize_batch_scalar_fields(data: dict) -> dict:
    """Normalize all scalar BatchRecord fields before Pydantic validation."""

    normalized = normalize_batch_numeric_fields(data)
    for field in _BATCH_TEXT_FIELDS:
        if field in normalized:
            normalized[field] = _structured_text(normalized.get(field))
    return normalized


def _context_slice(text: str, start: int, end: int, radius: int = 48) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi].lower()


def infer_scale_mmol(raw_text: str | None, explicit_scale_mmol: Any = None) -> float | None:
    explicit = _coerce_float(explicit_scale_mmol)
    if explicit and explicit > 0:
        return explicit
    text = (raw_text or "").strip()
    if not text:
        return None
    match = _MMOL_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def infer_reaction_time_h(raw_text: str | None, explicit_reaction_time_h: Any = None) -> float | None:
    explicit = _coerce_float(explicit_reaction_time_h)
    if explicit and explicit > 0:
        return explicit
    text = (raw_text or "").strip()
    if not text:
        return None
    match = _TIME_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    unit = match.group(2).lower()
    if unit.startswith("min"):
        return value / 60.0
    return value


def _choose_reaction_volume_candidate(raw_text: str | None) -> tuple[float, str] | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    candidates: list[tuple[int, float, int, str]] = []
    for idx, match in enumerate(_ML_RE.finditer(text)):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value <= 0:
            continue
        context = _context_slice(text, match.start(), match.end())
        score = 0
        if any(word in context for word in _CONTAINER_WORDS):
            score -= 5
        if any(word in context for word in _VOLUME_HINT_WORDS):
            score += 4
        if "added directly to" in context or "in " in context:
            score += 1
        if value <= 5:
            score += 1
        evidence = text[max(0, match.start() - 80) : min(len(text), match.end() + 80)].strip()
        candidates.append((score, value, idx, evidence))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
    return candidates[0][1], candidates[0][3]


def _choose_reaction_volume_mL(raw_text: str | None) -> float | None:
    candidate = _choose_reaction_volume_candidate(raw_text)
    return candidate[0] if candidate else None


def build_batch_quantity_evidence(raw_text: str | None, explicit_data: dict | None = None) -> dict:
    """Return auditable deterministic evidence for engineering-critical fields.

    The values here are intentionally limited to local arithmetic/extraction from
    the protocol text. They are safe to use as authoritative corrections in
    lightweight upstream mode because they are not model-generated design choices.
    """
    text = (raw_text or "").strip()
    explicit_data = explicit_data or {}
    evidence: dict[str, dict] = {}

    scale_match = _MMOL_RE.search(text)
    if scale_match:
        value = float(scale_match.group(1))
        evidence["scale_mmol"] = {
            "value": value,
            "source": "protocol_text",
            "evidence_text": text[max(0, scale_match.start() - 60) : min(len(text), scale_match.end() + 60)].strip(),
        }
    else:
        explicit_scale = _coerce_float(explicit_data.get("scale_mmol"))
        if explicit_scale and explicit_scale > 0:
            evidence["scale_mmol"] = {
                "value": explicit_scale,
                "source": "model_explicit",
                "evidence_text": "",
            }

    time_match = _TIME_RE.search(text)
    if time_match:
        value = float(time_match.group(1))
        unit = time_match.group(2).lower()
        value_h = value / 60.0 if unit.startswith("min") else value
        evidence["reaction_time_h"] = {
            "value": value_h,
            "source": "protocol_text",
            "unit_seen": unit,
            "evidence_text": text[max(0, time_match.start() - 60) : min(len(text), time_match.end() + 60)].strip(),
        }
    else:
        explicit_time = _coerce_float(explicit_data.get("reaction_time_h"))
        if explicit_time and explicit_time > 0:
            evidence["reaction_time_h"] = {
                "value": explicit_time,
                "source": "model_explicit",
                "evidence_text": "",
            }

    explicit_conc_match = _EXPLICIT_CONC_RE.search(text)
    if explicit_conc_match:
        value = float(explicit_conc_match.group(1))
        evidence["concentration_M"] = {
            "value": value,
            "source": "protocol_text_explicit_M",
            "evidence_text": text[
                max(0, explicit_conc_match.start() - 60) : min(len(text), explicit_conc_match.end() + 60)
            ].strip(),
        }
    else:
        volume_candidate = _choose_reaction_volume_candidate(text)
        scale = evidence.get("scale_mmol", {}).get("value")
        if scale and volume_candidate:
            volume_mL, volume_evidence = volume_candidate
            concentration = scale / volume_mL
            evidence["reaction_volume_mL"] = {
                "value": volume_mL,
                "source": "protocol_text",
                "evidence_text": volume_evidence,
            }
            evidence["concentration_M"] = {
                "value": concentration,
                "source": "deterministic_mmol_per_mL",
                "formula": f"{scale:g} mmol / {volume_mL:g} mL = {concentration:g} M",
                "evidence_text": f"{evidence['scale_mmol']['evidence_text']} | {volume_evidence}",
            }
        else:
            explicit_conc = _coerce_float(explicit_data.get("concentration_M"))
            if explicit_conc and explicit_conc > 0:
                evidence["concentration_M"] = {
                    "value": explicit_conc,
                    "source": "model_explicit_no_text_arithmetic",
                    "evidence_text": "",
                }

    return evidence


def apply_authoritative_batch_evidence(data: dict, raw_text: str | None) -> tuple[dict, dict]:
    """Apply protocol-text arithmetic over model guesses and return an audit trail."""
    enriched = dict(data)
    evidence = build_batch_quantity_evidence(raw_text, enriched)
    for field in ("scale_mmol", "reaction_time_h", "concentration_M"):
        item = evidence.get(field)
        if item and item.get("source", "").startswith(("protocol_text", "deterministic")):
            enriched[field] = item["value"]
    return enriched, evidence


def infer_batch_concentration_M(
    raw_text: str | None,
    *,
    explicit_concentration_M: Any = None,
    explicit_scale_mmol: Any = None,
) -> float | None:
    explicit_conc = _coerce_float(explicit_concentration_M)
    if explicit_conc and explicit_conc > 0:
        return explicit_conc

    text = (raw_text or "").strip()
    if not text:
        return None

    conc_match = _EXPLICIT_CONC_RE.search(text)
    if conc_match:
        try:
            value = float(conc_match.group(1))
        except ValueError:
            value = None
        if value and value > 0:
            return value

    scale_mmol = infer_scale_mmol(text, explicit_scale_mmol)
    volume_mL = _choose_reaction_volume_mL(text)
    if not scale_mmol or not volume_mL:
        return None

    concentration_M = scale_mmol / volume_mL
    return concentration_M if concentration_M > 0 else None


def enrich_batch_record_dict(data: dict, raw_text: str | None) -> dict:
    """Fill a few missing engineering-critical fields deterministically."""
    enriched = normalize_batch_scalar_fields(data)
    text = raw_text or enriched.get("raw_text") or enriched.get("reaction_description") or ""

    additives = enriched.get("additives")
    if additives is not None:
        if not isinstance(additives, list):
            additives = [additives]
        normalized_additives: list[str] = []
        for additive in additives:
            if isinstance(additive, str):
                normalized_additives.append(additive)
                continue
            if isinstance(additive, dict):
                identity = next(
                    (
                        additive.get(key)
                        for key in ("name", "reagent", "compound", "additive", "identity")
                        if additive.get(key)
                    ),
                    None,
                )
                normalized_additives.append(
                    str(identity) if identity is not None else str(additive)
                )
                continue
            if additive is not None:
                normalized_additives.append(str(additive))
        enriched["additives"] = normalized_additives

    if not enriched.get("scale_mmol"):
        inferred_scale = infer_scale_mmol(text, enriched.get("scale_mmol"))
        if inferred_scale is not None:
            enriched["scale_mmol"] = inferred_scale

    if not enriched.get("reaction_time_h"):
        inferred_time_h = infer_reaction_time_h(text, enriched.get("reaction_time_h"))
        if inferred_time_h is not None:
            enriched["reaction_time_h"] = inferred_time_h

    if not enriched.get("concentration_M"):
        inferred_conc = infer_batch_concentration_M(
            text,
            explicit_concentration_M=enriched.get("concentration_M"),
            explicit_scale_mmol=enriched.get("scale_mmol"),
        )
        if inferred_conc is not None:
            enriched["concentration_M"] = inferred_conc

    if text and not enriched.get("raw_text"):
        enriched["raw_text"] = text
    return enriched
