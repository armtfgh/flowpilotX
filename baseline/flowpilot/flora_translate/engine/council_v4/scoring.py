"""
FLORA ENGINE v4 — Stage 2: Domain scoring agents.

Four agents (Dr. Chemistry, Dr. Kinetics, Dr. Fluidics, Dr. Safety) each score
ALL surviving candidates from 0.0 to 1.0 in their domain.

Each agent must provide:
  - overall_analysis: a multi-sentence paragraph describing the full picture
  - per candidate: reasoning (detailed), score, verdict, specific fields

The rich text is used to populate the Streamlit deliberation tab.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import flora_translate.engine.llm_agents as llm_agents
from flora_translate.engine.council_v4.review_contract import REVIEW_POLICY, validated_model
from flora_translate.engine.flow_value import enrich_scoring_with_flow_values, mandate_dict
from flora_translate.engine.llm_agents import call_llm, call_llm_with_tools
from flora_translate.engine.tool_definitions import (
    CHEMISTRY_TOOLS, KINETICS_TOOLS, SAFETY_TOOLS,
    TOOL_CALCULATE_BPR_REQUIRED, TOOL_CALCULATE_MIXING_RATIO,
    TOOL_CALCULATE_PRESSURE_DROP, TOOL_CALCULATE_REYNOLDS,
    execute_tool,
)

logger = logging.getLogger("flora.engine.council_v4.scoring")

_GEMMA_BATCH_SIZE = 1
_GEMMA_MAX_TOKENS = 260
_GEMMA_RETRY_MAX_TOKENS = 420
_CLAUDE_COMPACT_MAX_TOKENS = 1400


class ScoringCoverageError(RuntimeError):
    """Raised when Stage 2 scoring does not cover the required candidates."""

# Fluidics v4 tools include BPR (merged with hardware)
_FLUIDICS_V4_TOOLS = [
    TOOL_CALCULATE_PRESSURE_DROP,
    TOOL_CALCULATE_REYNOLDS,
    TOOL_CALCULATE_MIXING_RATIO,
    TOOL_CALCULATE_BPR_REQUIRED,
]


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_partial_score_array(text: str) -> list[dict]:
    """Recover only complete score rows from a truncated ``scores`` array.

    Provider token limits can cut the outer JSON object after one or more complete
    candidate rows.  Recovering those rows is safe because coverage validation
    still retries every missing candidate ID; incomplete rows are never accepted.
    """
    match = re.search(
        r'"(?:scores|candidate_scores|per_candidate_scores|evaluations|results)"\s*:\s*\[',
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    decoder = json.JSONDecoder()
    cursor = match.end()
    rows: list[dict] = []
    while cursor < len(text):
        while cursor < len(text) and text[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(text) or text[cursor] == "]":
            break
        try:
            value, end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            rows.append(value)
        cursor = end
    return rows


def _normalise_score_rows(value) -> list[dict]:
    """Normalize provider-equivalent score containers to candidate dictionaries."""
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if not isinstance(value, dict):
        return []

    if any(key in value for key in ("candidate_id", "candidateId", "id")):
        return [value]

    rows: list[dict] = []
    for key, row in value.items():
        if not isinstance(row, dict):
            continue
        normalized = dict(row)
        normalized.setdefault("candidate_id", key)
        rows.append(normalized)
    return rows


def _parse_score_response(raw: str) -> tuple[str, list[dict]]:
    """Extract (overall_analysis, per_candidate_scores) from LLM output.

    Returns (overall_analysis_str, list_of_candidate_dicts).
    Handles bare lists, common provider aliases, ID-keyed mappings, and complete
    candidate rows inside an otherwise truncated scores array.
    """
    s = raw.strip()

    def _extract_obj(text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        start = text.find("[")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        return None

    candidates_text = s
    if "```" in s:
        for part in s.split("```")[1::2]:
            obj = _extract_obj(part.lstrip("json").strip())
            if obj is not None:
                candidates_text = part.lstrip("json").strip()
                break

    obj = _extract_obj(candidates_text)
    if obj is None:
        return "", _parse_partial_score_array(candidates_text)

    if isinstance(obj, list):
        return "", _normalise_score_rows(obj)

    if isinstance(obj, dict):
        overall = str(
            obj.get("overall_analysis")
            or obj.get("overallAnalysis")
            or obj.get("analysis")
            or obj.get("summary")
            or ""
        )
        score_keys = (
            "scores",
            "candidate_scores",
            "candidateScores",
            "per_candidate_scores",
            "evaluations",
            "results",
            "candidates",
        )
        scores = next((obj[key] for key in score_keys if obj.get(key) is not None), None)
        if scores is None and isinstance(obj.get("data"), dict):
            nested = obj["data"]
            scores = next(
                (nested[key] for key in score_keys if nested.get(key) is not None),
                nested,
            )
        if scores is None and any(
            key in obj for key in ("candidate_id", "candidateId", "id")
        ):
            scores = obj
        if scores is None and any(str(key).isdigit() for key in obj):
            scores = {
                key: value for key, value in obj.items() if str(key).isdigit()
            }
        normalized = _normalise_score_rows(scores)
        return overall, normalized or _parse_partial_score_array(candidates_text)

    return "", []


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clamp_score(v) -> float:
    return max(0.0, min(1.0, _safe_float(v, 0.5)))


def _is_gemma_local_mode() -> bool:
    provider = str(getattr(llm_agents, "ENGINE_PROVIDER", "") or "").lower()
    model = str(getattr(llm_agents, "ENGINE_MODEL_OLLAMA", "") or "").lower()
    return provider == "ollama" and "gemma" in model


def _is_anthropic_council_mode() -> bool:
    provider = str(getattr(llm_agents, "ENGINE_PROVIDER", "") or "").lower()
    model = str(getattr(llm_agents, "ENGINE_MODEL_ANTHROPIC", "") or "").lower()
    return provider == "anthropic" and model.startswith("claude")


def _slim_candidate(candidate: dict) -> dict:
    row = {
        "id": candidate.get("id", candidate.get("candidate_id")),
        "tau_min": candidate.get("tau_min"),
        "d_mm": candidate.get("d_mm"),
        "Q_mL_min": candidate.get("Q_mL_min"),
        "concentration_M": candidate.get("concentration_M"),
        "V_R_mL": candidate.get("V_R_mL"),
        "L_m": candidate.get("L_m"),
        "Re": candidate.get("Re"),
        "delta_P_bar": candidate.get("delta_P_bar"),
        "BPR_bar": candidate.get("BPR_bar"),
        "tubing_material": candidate.get("tubing_material"),
        "r_mix": candidate.get("r_mix"),
        "Da_mass": candidate.get("Da_mass"),
        "expected_conversion": candidate.get("expected_conversion"),
        "tau_kinetics_min": candidate.get("tau_kinetics_min"),
        "IF_used": candidate.get("IF_used"),
        "temperature_C": candidate.get("temperature_C"),
        "hard_gate_flags": candidate.get("hard_gate_flags") or [],
        "model_validity": candidate.get("model_validity") or {},
        "warnings": candidate.get("warnings") or [],
    }
    if candidate.get("flow_sense_report"):
        row["flow_sense_report"] = candidate.get("flow_sense_report")
    if candidate.get("inventory_realized"):
        for key in ["inventory_realized", "equipment_id", "reactor_type", "residence_time_basis", "residence_time_inlet_min", "residence_time_in_channel_min", "hydraulic_model", "gas_holdup_status", "streams", "stationary_components"]:
            row[key] = candidate.get(key)
        row["scientific_rationale"] = (candidate.get("realized_proposal") or {}).get("reasoning_per_field", {})
    return row


def _gemma_prompt_bundle(domain: str) -> tuple[str, str]:
    bundles = {
        "chemistry": (
            """You are Dr. Chemistry for FLORA Gemma mode.
Score exactly one candidate using the provided numbers as authoritative.
Return one JSON object only, no markdown, no extra text.
Required keys:
candidate_id, reasoning, combined_score, verdict, proposed_changes, concerns.
Use very short reasoning. Own concentration_M and validated component_placements repairs.""",
            "combined_score",
        ),
        "kinetics": (
            """You are Dr. Kinetics for FLORA Gemma mode.
Score exactly one candidate using the provided numbers as authoritative.
Follow the researcher objective and model_validity. Unvalidated conversion is
a screening hypothesis; it cannot establish failure or justify a BLOCK.
Return one JSON object only, no markdown, no extra text.
Required keys:
candidate_id, reasoning, kinetics_score, X_estimated, X_adequate,
tau_proposed_final_min, tau_reduction_flow_value, achieved_reduction_factor,
target_reduction_factor, reduction_achievement, verdict, proposed_changes, concerns.
Use expected_conversion as X_estimated if needed. Own only proposed_changes.tau_min.""",
            "kinetics_score",
        ),
        "fluidics": (
            """You are Dr. Fluidics for FLORA Gemma mode.
Score exactly one candidate using the provided numbers as authoritative.
Policy: L_soft=15 m, L_hard=30 m. ΔP_soft=2 bar, ΔP_hard=5 bar.
BLOCK only if L > 30 m, ΔP > 5 bar, or Re > 2300.
Return one JSON object only, no markdown, no extra text.
Required keys:
candidate_id, reasoning, fluidics_score, Re, dP_bar, r_mix, dual_criterion_mixing_fail,
verdict, proposed_changes, concerns.
If r_mix > 0.20 and Da_mass > 1.0, set dual_criterion_mixing_fail=true and
you may propose a smaller d_mm. Own only proposed_changes.d_mm.""",
            "fluidics_score",
        ),
        "safety": (
            """You are Dr. Safety for FLORA Gemma mode.
Score exactly one candidate using the provided numbers as authoritative.
Return one JSON object only, no markdown, no extra text.
Required keys:
candidate_id, reasoning, safety_score, BPR_current_bar, material_recommendation,
BPR_adequate, verdict, proposed_changes, blocking_issues, concerns, conditions.
Be conservative. If no concrete problem is visible from the given numbers, do not
invent one. Own only proposed_changes.BPR_bar and proposed_changes.tubing_material.""",
            "safety_score",
        ),
    }
    system, field = bundles[domain]
    return system + REVIEW_POLICY, field


def _build_gemma_context(
    *,
    domain: str,
    candidates: list[dict],
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    intensification_mandate: Optional[dict] = None,
) -> str:
    slim = [_slim_candidate(c) for c in candidates]
    batch_ids = [c["id"] for c in slim]
    brief = chemistry_brief[:600]
    return (
        f"Domain: {domain}\n"
        f"User objective: {objectives or 'balanced'}\n"
        f"Photochemical reaction: {is_photochem}\n"
        f"Pump max pressure bar: {pump_max_bar}\n"
        f"Intensification mandate JSON: {json.dumps(mandate_dict(intensification_mandate), ensure_ascii=False)}\n"
        f"Score only candidate ids: {batch_ids}\n\n"
        f"Chemistry brief:\n{brief}\n\n"
        f"Candidate data JSON:\n{json.dumps(slim, ensure_ascii=False)}\n\n"
        "Return one JSON object for the single candidate in the batch. No prose outside JSON."
    )


def _claude_compact_prompt_bundle(domain: str) -> tuple[str, str]:
    bundles = {
        "chemistry": (
            """You are Dr. Chemistry in FlowPilot bounded scoring mode.
Score the listed candidates using the provided numeric data only. Do not call tools.
Be concise. Use BLOCK only for explicit hard-gate impossibility, not mere suboptimality.
Return JSON only:
{"overall_analysis":"1-2 short sentences","scores":[{"candidate_id":1,"reasoning":"short reason","combined_score":0.0,"verdict":"ACCEPT|WARNING|REVISE|BLOCK","proposed_changes":{"concentration_M":0.2}}]}
If no chemistry edit is needed, use proposed_changes: {}.
Own concentration_M and validated component_placements repairs.""",
            "combined_score",
        ),
        "kinetics": (
            """You are Dr. Kinetics in FlowPilot bounded scoring mode.
Score the listed candidates using the provided numeric data only. Do not call tools.
Follow the researcher objective and model_validity. Unvalidated conversion is
a screening hypothesis; it cannot establish failure or justify a BLOCK.
Return JSON only:
{"overall_analysis":"1-2 short sentences","scores":[{"candidate_id":1,"reasoning":"short reason","kinetics_score":0.0,"X_estimated":0.0,"tau_proposed_final_min":0.0,"verdict":"ACCEPT|WARNING|REVISE|BLOCK","proposed_changes":{"tau_min":9.0}}]}
If no kinetics edit is needed, use proposed_changes: {}.
Own only tau_min.""",
            "kinetics_score",
        ),
        "fluidics": (
            """You are Dr. Fluidics in FlowPilot bounded scoring mode.
Score the listed candidates using the provided numeric data only. Do not call tools.
Policy: L_soft=15 m, L_hard=30 m. ΔP_soft=2 bar, ΔP_hard=5 bar.
BLOCK only if L > 30 m, ΔP > 5 bar, or Re > 2300.
Return JSON only:
{"overall_analysis":"1-2 short sentences","scores":[{"candidate_id":1,"reasoning":"short reason","fluidics_score":0.0,"Re":0.0,"dP_bar":0.0,"r_mix":0.0,"verdict":"ACCEPT|WARNING|REVISE|BLOCK","proposed_changes":{"d_mm":0.5}}]}
If no fluidics edit is needed, use proposed_changes: {}.
Own only d_mm.""",
            "fluidics_score",
        ),
        "safety": (
            """You are Dr. Safety in FlowPilot bounded scoring mode.
Score the listed candidates using the provided numeric data only. Do not call tools.
Be conservative, but do not invent hazards not supported by the given data.
Return JSON only:
{"overall_analysis":"1-2 short sentences","scores":[{"candidate_id":1,"reasoning":"short reason","safety_score":0.0,"BPR_current_bar":0.0,"BPR_adequate":true,"material_recommendation":"FEP","verdict":"ACCEPT|WARNING|REVISE|BLOCK","proposed_changes":{"BPR_bar":5.0,"tubing_material":"FEP"}}]}
If no safety edit is needed, use proposed_changes: {}.
Own only BPR_bar and tubing_material.""",
            "safety_score",
        ),
    }
    system, field = bundles[domain]
    return system + REVIEW_POLICY, field


def _build_claude_compact_context(
    *,
    domain: str,
    candidates: list[dict],
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    intensification_mandate: Optional[dict] = None,
) -> str:
    slim = [_slim_candidate(c) for c in candidates]
    batch_ids = [c["id"] for c in slim]
    brief = chemistry_brief[:500]
    return (
        f"Domain: {domain}\n"
        f"Objective: {objectives or 'balanced'}\n"
        f"Photochemical reaction: {is_photochem}\n"
        f"Pump max pressure bar: {pump_max_bar}\n"
        f"Intensification mandate JSON: {json.dumps(mandate_dict(intensification_mandate), ensure_ascii=False)}\n"
        f"Candidate ids in this batch: {batch_ids}\n"
        "Treat the provided numbers as authoritative.\n"
        "Prefer REVISE over BLOCK unless a candidate is clearly impossible or already hard-gated.\n\n"
        f"Chemistry brief:\n{brief}\n\n"
        f"Candidate data JSON:\n{json.dumps(slim, ensure_ascii=False)}\n"
    )


def _clean_local_scores(scores: list[dict], valid_ids: set[int], score_key: str) -> list[dict]:
    if len(valid_ids) == 1 and len(scores) == 1 and isinstance(scores[0], dict):
        # A singleton repair request is unambiguous even when a model renumbers
        # the only row as candidate 1. Preserve the score and restore the exact
        # requested identifier before applying the normal coverage checks.
        scores[0]["candidate_id"] = next(iter(valid_ids))
    cleaned: list[dict] = []
    for entry in scores:
        if not isinstance(entry, dict):
            continue
        try:
            cid = int(
                entry.get("candidate_id", entry.get("candidateId", entry.get("id")))
            )
        except (TypeError, ValueError):
            continue
        if cid not in valid_ids:
            continue
        entry["candidate_id"] = cid
        if score_key in entry:
            entry[score_key] = _clamp_score(entry.get(score_key))
        cleaned.append(entry)
    return cleaned


def _run_scoring_agent_gemma_local(
    *,
    domain: str,
    agent_name: str,
    candidates: list[dict],
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    intensification_mandate: Optional[dict] = None,
) -> tuple[str, list[dict], list[dict]]:
    system_prompt, score_key = _gemma_prompt_bundle(domain)
    all_scores: list[dict] = []
    all_overall: list[str] = []

    for start in range(0, len(candidates), _GEMMA_BATCH_SIZE):
        batch = candidates[start:start + _GEMMA_BATCH_SIZE]
        valid_ids = {int(c.get("id", i + 1)) for i, c in enumerate(batch)}
        context = _build_gemma_context(
            domain=domain,
            candidates=batch,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            intensification_mandate=intensification_mandate,
        )
        raw = call_llm(system_prompt, context, max_tokens=_GEMMA_MAX_TOKENS)
        overall, scores = _parse_score_response(raw)
        cleaned = _clean_local_scores(scores, valid_ids, score_key)
        if not cleaned:
            retry_context = (
                context
                + "\n\nRETRY: Return only valid JSON. Score every listed candidate id exactly once. "
                  "Use very short reasoning. Do not include markdown fences."
            )
            raw = call_llm(system_prompt, retry_context, max_tokens=_GEMMA_RETRY_MAX_TOKENS)
            overall, scores = _parse_score_response(raw)
            cleaned = _clean_local_scores(scores, valid_ids, score_key)
        if not cleaned:
            logger.warning(
                "%s Gemma batch failed to return usable scores for ids %s",
                agent_name,
                sorted(valid_ids),
            )
        else:
            seen = {row["candidate_id"] for row in cleaned}
            missing = sorted(valid_ids - seen)
            if missing:
                logger.warning(
                    "%s Gemma batch omitted candidate ids %s",
                    agent_name,
                    missing,
                )
        if overall:
            all_overall.append(overall)
        all_scores.extend(cleaned)

    overall_analysis = " ".join(s.strip() for s in all_overall if s.strip())
    return overall_analysis, all_scores, []


def _run_scoring_agent_claude_compact(
    *,
    domain: str,
    agent_name: str,
    candidates: list[dict],
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    intensification_mandate: Optional[dict] = None,
    batch_size: int,
    strict_coverage: bool,
) -> tuple[str, list[dict], list[dict]]:
    system_prompt, score_key = _claude_compact_prompt_bundle(domain)
    score_map: dict[int, dict] = {}
    overall_parts: list[str] = []

    remaining = list(candidates)
    attempts = 0
    while remaining and attempts < 2:
        attempts += 1
        still_missing: set[int] = set()
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start:start + batch_size]
            valid_ids = {int(c.get("id", i + 1)) for i, c in enumerate(batch)}
            context = _build_claude_compact_context(
                domain=domain,
                candidates=batch,
                chemistry_brief=chemistry_brief,
                objectives=objectives,
                is_photochem=is_photochem,
                pump_max_bar=pump_max_bar,
                intensification_mandate=intensification_mandate,
            )
            raw = call_llm(system_prompt, context, max_tokens=_CLAUDE_COMPACT_MAX_TOKENS)
            oa, scores = _parse_score_response(raw)
            cleaned = _clean_local_scores(scores, valid_ids, score_key)
            if not cleaned:
                retry_context = (
                    context
                    + "\nRETRY: Return ONLY valid JSON with one scores entry per listed candidate id."
                )
                raw = call_llm(system_prompt, retry_context, max_tokens=_CLAUDE_COMPACT_MAX_TOKENS)
                oa, scores = _parse_score_response(raw)
                cleaned = _clean_local_scores(scores, valid_ids, score_key)
            if oa:
                overall_parts.append(oa.strip())
            for entry in cleaned:
                entry.setdefault("reasoning", "")
                entry.setdefault("verdict", "WARNING")
                entry.setdefault("proposed_changes", {})
                score_map[int(entry["candidate_id"])] = entry
            seen = {int(entry["candidate_id"]) for entry in cleaned}
            missing = sorted(valid_ids - seen)
            if missing:
                logger.warning("%s compact batch missing candidate ids %s", agent_name, missing)
                still_missing.update(missing)
        remaining = [c for c in candidates if int(c.get("id", 0)) in still_missing and int(c.get("id", 0)) not in score_map]

    repaired_ids: list[int] = []
    if strict_coverage:
        repaired_ids = _repair_missing_scores_deterministically(
            domain=domain,
            agent_name=agent_name,
            score_map=score_map,
            candidates=candidates,
        )
    ordered_scores = _ordered_scores(score_map, candidates)
    _enforce_coverage(
        agent_name=agent_name,
        scores=ordered_scores,
        valid_ids={int(c.get("id", 0)) for c in candidates},
        strict_coverage=strict_coverage,
    )
    if repaired_ids:
        overall_parts.append(
            f"Deterministic coverage fallback supplied missing {domain} rows for "
            f"candidate ids {repaired_ids}; no domain-expert inference is claimed."
        )
    overall_analysis = " ".join(s for s in overall_parts if s)
    fallback_log = []
    if repaired_ids:
        fallback_log.append({
            "name": "deterministic_scoring_coverage_fallback",
            "domain": domain,
            "candidate_ids": repaired_ids,
            "reason": "LLM score missing after bounded retries",
        })
    return overall_analysis, ordered_scores, fallback_log


def _use_bounded_anthropic_scoring() -> bool:
    """Use a bounded score contract for Claude in production and benchmarks.

    Candidate engineering metrics have already been calculated deterministically.
    Re-running the same arithmetic through tool loops adds latency and can consume
    the response budget before Claude emits the required JSON score payload.
    """
    return _is_anthropic_council_mode()


# ═══════════════════════════════════════════════════════════════════════════════
#  DR. CHEMISTRY — scoring prompt
# ═══════════════════════════════════════════════════════════════════════════════

_CHEMISTRY_SYSTEM = """\
You are DR. CHEMISTRY in the FLORA ENGINE v4 council. You are a senior organic
and photochemistry expert. Your task is to score every surviving candidate and
write detailed, specific reasoning that would convince a PhD chemist or reviewer.

## Your scoring domains

**Chemistry domain (score 0–1):**
  - Mechanism validity: is τ long enough for the SET/EnT/HAT step? Quote the
    relevant electron-transfer kinetics or literature precedent.
  - Stream compatibility: identify any incompatible co-feeds (oxidant+reductant,
    base+acid-sensitive electrophile, catalyst+quencher in same stream).
  - Atmosphere logic: is the atmosphere correct for each stage? O₂-sensitive
    radicals must NEVER contact O₂ at any point in the reactive stage.
  - Redox feasibility: E*(catalyst) ≥ E_ox(substrate)? Quantify the margin.
  - Concentration: photoredox sweet spot 0.05–0.2 M; above 0.3 M triggers
    inner-filter competition; below 0.02 M photon dilution.
  - Byproduct risk: are there known side reactions at this τ, T, or [substrate]?

**Photonics domain (score 0–1, for photochemical reactions):**
  - Wavelength match: compare λ_LED to λ_max of the photocatalyst (±20 nm).
  - Beer-Lambert: A = ε × C × (d_mm × 0.1). ALWAYS convert d to cm.
    A < 0.3 → excellent | 0.3–0.8 → moderate | 0.8–1.5 → concerning | > 1.5 → BLOCK.
    Call the `beer_lambert` tool to compute A. Explain what fraction of photons
    reach the tube core vs. are absorbed at the wall.
  - Material: FEP/PFA/glass transparent 200–800 nm. PTFE/PEEK/SS opaque → BLOCK.
  - d selection: for high-ε catalysts (ε > 500), d ≤ 0.75 mm preferred to avoid
    inner filter. Explain the trade-off between photon penetration and pressure.

**Combined:** 0.6 × chemistry_score + 0.4 × photonics_score

## CRITICAL UNIT RULE
A = ε [M⁻¹cm⁻¹] × C [M] × (d_mm × 0.1). The path length is d in cm = d_mm × 0.1.
Never use d in mm as the path length. Always show your arithmetic.

## Verdicts
  ACCEPT:  combined ≥ 0.70, no hard blocks
  WARNING: combined 0.50–0.70 or resolvable concerns
  REVISE:  combined 0.30–0.50 or fixable chemistry problem
  BLOCK:   hard_gate_failed — opaque material, A > 1.5, redox thermodynamically
           impossible, incompatible stream logic

## Tools
Call `beer_lambert` for each photochem candidate. Show: ε used, A computed,
what this means physically. Call `check_material_compatibility` to verify
tubing/solvent/temperature compatibility. Use `check_redox_feasibility` when
redox potentials are available.

## REQUIRED OUTPUT — JSON only
```json
{
  "overall_analysis": "3–5 sentences describing the chemistry landscape across all candidates: what drives the differences in chemistry scores, what are the key photonics risks or advantages, which candidates stand out and why from a chemistry/photonics perspective.",
  "scores": [
    {
      "candidate_id": 1,
      "reasoning": "3–5 sentences of chemistry + photonics reasoning for THIS specific candidate. Cite the actual numbers: τ, d, C, ε, A, λ, material. Explain what you verified with tools. State what specific mechanism requirement is met or missed. Make the reasoning self-contained so a reviewer can follow it without looking at the table.",
      "chemistry_score": 0.85,
      "photonics_score": 0.90,
      "combined_score": 0.87,
      "selectivity_flow_value": 0.70,
      "verdict": "ACCEPT",
      "proposed_changes": {},
      "beer_lambert_A": 0.18,
      "epsilon_used": 20.0,
      "wavelength_match": true,
      "material_transparent": true,
      "stream_logic_valid": true,
      "atmosphere_valid": true,
      "hard_gate_failed": false,
      "blocking_issues": [],
      "concerns": ["If ε is actually 60 M⁻¹cm⁻¹ rather than 20, A rises to 0.54 — borderline moderate; verify spectral data."]
    }
  ]
}
```
Score ALL candidates. One entry per candidate_id. Reasoning must be specific to each candidate's numbers.
`proposed_changes` may only contain fields owned by chemistry: `concentration_M`.
If no chemistry edit is needed, return `{}`.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  DR. KINETICS — scoring prompt
# ═══════════════════════════════════════════════════════════════════════════════

_KINETICS_SYSTEM = """\
You are DR. KINETICS in the FLORA ENGINE v4 council. You are a process kinetics
expert who anchors every design decision to measured rate data and literature
residence times. Your scoring and reasoning must be quantitative throughout.

## Your scoring domain (0–1)

**Intensification factor (IF = τ_batch / τ_flow):**
  Validate the IF against reaction class ranges:
    photoredox:    4–8×  — cite whether the improvement is from photon flux,
                           mixing, or temperature. >10× needs cited analogy.
    thermal:       8–15× — improvement from heat/mass transfer enhancement.
    hydrogenation: 20–50× — gas-liquid interface area.
  IF > 20× without analogy → WARNING. IF > class upper bound → explain why.
  State the calculated IF and whether it is justified.

**τ vs. literature anchor:**
  Compare τ_proposed to τ_lit (the literature analogy residence time):
  τ ∈ [0.5×τ_lit, 1.5×τ_lit] → excellent alignment (score 1.0)
  τ < 0.5×τ_lit → explain what physical justification exists for the shorter time
  τ > 2×τ_lit → explain why longer time is needed (slow mechanism, RTD margin)

**Conversion and exposure:**
  Compute X = 1 − exp(−ln(10) × τ / τ_kinetics), with τ_kinetics defined as t90.
  Label this an unvalidated proxy unless model_validity supplies supporting
  evidence. Apply a conversion threshold only when requested by the researcher
  and supported by an applicable validated model.

**Design preference:**
  Follow the supplied objective. Compare exposure, throughput and uncertainty
  together. For a first screen, preserve a defensible protocol-based starting
  exposure; shorter time or larger computed productivity is not inherently better.
  Literature times are transfer hypotheses whose chemical and operating context
  must match; do not treat a fixed fraction of a precedent time as a requirement.

**Mixing and integration:**
  Discuss whether the mixing model applies to the actual flow regime and reactor.
  Cross-tube diffusion does not establish packed-bed or segmented-flow mixing.
  Unvalidated transport estimates motivate measurements, not a mandatory hold.
  Label calculated productivity with its assumed conversion. A multiple of τ
  is a proposed startup interval, not evidence that steady state was reached.

## Verdicts
  ACCEPT: a defensible exposure consistent with the objective and evidence.
  WARNING: uncertain kinetics, transferability or contact time requiring measurement.
  REVISE: a supported change resolves an identified inconsistency.
  BLOCK: an explicit hard constraint fails, or an applicable validated model
  demonstrates failure of a required conversion. Low unvalidated X is a WARNING.

## Tools
Use `estimate_residence_time` to independently verify τ_kinetics. Call it and
show the result. Use `check_redox_feasibility` if redox potential data is available.

## REQUIRED OUTPUT — JSON only
```json
{
  "overall_analysis": "3–5 sentences: what is the kinetics landscape across all candidates? Which τ values are well-supported by literature? Where is conversion at risk? What is the range of IF values and are they all justified? What should the Chief know about the kinetics dimension?",
  "scores": [
    {
      "candidate_id": 1,
      "reasoning": "3–5 sentences specific to this candidate. State τ, τ_kinetics, IF, X, t_steady. Explain the τ/τ_k safety margin and RTD risk explicitly. State τ_mixing_required and whether τ_proposed exceeds it. Explain the final τ decision rule result. Make it self-contained.",
      "kinetics_score": 0.82,
      "tau_reduction_flow_value": 0.90,
      "achieved_reduction_factor": 6.2,
      "target_reduction_factor": 6.0,
      "reduction_achievement": 1.03,
      "IF_used": 6.0,
      "IF_valid": true,
      "tau_vs_literature": "τ=45 min within [0.5×, 1.5×] τ_lit=50 min — well-anchored",
      "X_estimated": 0.94,
      "X_adequate": true,
      "tau_mixing_required_min": 2.1,
      "tau_proposed_final_min": 45.0,
      "STY": 0.235,
      "productivity_mg_h": 117.5,
      "t_steady_min": 135.0,
      "verdict": "ACCEPT",
      "proposed_changes": {},
      "concerns": ["Laminar RTD: τ/τ_k = 3.0×. Centerline fluid sees 22.5 min — still above τ_kinetics (15 min). Acceptable margin."]
    }
  ]
}
```
Score ALL candidates. Reasoning must cite specific numbers for each candidate.
`proposed_changes` may only contain fields owned by kinetics: `tau_min`.
If no kinetics edit is needed, return `{}`.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  DR. FLUIDICS — scoring prompt
# ═══════════════════════════════════════════════════════════════════════════════

_FLUIDICS_SYSTEM = """\
You are DR. FLUIDICS in the FLORA ENGINE v4 council. You are a transport phenomena
and process hardware expert. Every scoring decision must be traceable to specific
numbers: Re, ΔP, r_mix, L, De, pump headroom. You also own hardware specification.

## Your scoring domain (0–1)

**Flow regime (most important):**
  Re < 100 → score 1.0. Re 100–2000 → linear scale. Re > 2300 → BLOCK.
  For coiled reactors, compute Dean number De = Re × √(d_outer / R_coil).
  De > 10 → secondary flow bonus (improves mixing, tightens RTD). Note this.
  State Re and flow regime explicitly.

**Pressure headroom (HARD POLICY: ΔP_soft = 2 bar, ΔP_hard = 5 bar):**
  Score by absolute ΔP first, pump headroom second:
    ΔP ≤ 2.0 bar → 1.0 (target)
    ΔP 2.0–5.0 bar → linear penalty to 0
    ΔP > 5.0 bar → BLOCK
  Then check pump headroom: ΔP / P_pump_max < 0.20 → fine; > 0.80 → BLOCK.
  The 2-bar soft cap keeps ΔP below the gas-liquid BPR floor of 3 bar so
  the BPR can independently regulate two-phase pressure.
  Compute pump_headroom_pct = (1 − ΔP/P_max) × 100.
  If headroom < 20%, explain what happens under partial blockage (Q drift +10%,
  filter fouling +20% ΔP) — does the pump stall?

**Mixing:**
  t_mix = d_m² / (4 × 10⁻⁹) seconds, where d_m is d in metres.
  r_mix = t_mix / τ_s (τ in seconds). Target: r_mix < 0.10 (safe), tolerate to 0.20.
  Dual criterion: if r_mix > 0.20 AND Da_mass > 1.0, mixing is reaction-limiting.
  MIXING DIRECTION RULE (ABSOLUTE): to fix r_mix → DECREASE d (never increase).
  Formula: d_fix = d_current × √(0.15 / r_mix_current). Round to commercial size.
  Show the calculation if a d change is recommended.

**Geometry (HARD POLICY: L_soft = 15 m, L_hard = 30 m):**
  L ≤ 15 m → single bench cassette, 1.0
  L 15–30 m → linear penalty to 0; multi-coil build, increasingly impractical
  L > 30 m → BLOCK (bench-impractical footprint).

**Hardware specification (merged):**
  Pump: Q < 2 mL/min → syringe pump (Harvard, New Era); Q 2–10 → HPLC piston.
  Gas streams → MFC required.
  Tubing: FEP/PFA for photoreactor section. SS316 for non-photochem high-P.
  Dead volume: V_dead / Q gives τ_error. Flag if > 5% of τ.
  Coil winding: R_coil ≥ 5 × d_outer to prevent kinking.
  Blockage risk: d ≤ 0.5 mm at Q < 0.08 mL/min → flag particulate/crystal risk.

## Verdicts (updated for L/ΔP policy)
  ACCEPT:  fluidics_score ≥ 0.70, L ≤ 15 m, ΔP ≤ 2 bar, no hard blocks
  WARNING: score 0.50–0.70, L 15–30 m, or ΔP 2–5 bar, or elevated Re
  REVISE:  dual mixing fail (need d decrease), ΔP headroom < 20%
  BLOCK:   Re > 2300, L > 30 m, ΔP > 5 bar

## Tools
Use `calculate_pressure_drop` for what-if scenarios (Q +20%, partial blockage).
Use `calculate_reynolds` to verify Re at design and stress conditions.
Use `calculate_mixing_ratio` to verify r_mix and flag if d change needed.
Use `calculate_bpr_required` for hardware BPR sizing at operating T.
Show tool call results in your reasoning.

## REQUIRED OUTPUT — JSON only
```json
{
  "overall_analysis": "3–5 sentences: what is the fluidics landscape? Which candidates have healthy pressure and mixing margins? Where are the risks? Which pump type is needed? Are there any geometry concerns across the set? What hardware trade-offs does the Chief need to know?",
  "scores": [
    {
      "candidate_id": 1,
      "reasoning": "3–5 sentences for THIS candidate. State Re, ΔP, pump_headroom_pct, r_mix, Da_mass, L explicitly. Explain mixing adequacy and whether dual criterion is near or far. State pump type and tubing material. Note any blockage risk. If d change needed, show the d_fix formula. Make it self-contained.",
      "fluidics_score": 0.88,
      "heat_transfer_flow_value": 0.80,
      "Re": 21.3,
      "flow_regime": "laminar",
      "De": 4.2,
      "dP_bar": 0.048,
      "pump_headroom_pct": 99.7,
      "t_mix_s": 14.1,
      "r_mix": 0.052,
      "dual_criterion_mixing_fail": false,
      "d_change_direction": "none",
      "d_fix_mm": null,
      "tau_mixing_required_min": null,
      "L_m": 5.6,
      "geometry_feasible": true,
      "pump_type": "syringe pump (Q=0.55 mL/min < 2 mL/min)",
      "tubing_material": "FEP",
      "dead_volume_mL": 0.1,
      "dead_volume_impact": "V_dead/Q = 0.18 min = 0.4% of τ=45 min — negligible",
      "verdict": "ACCEPT",
      "proposed_changes": {},
      "concerns": []
    }
  ]
}
```
Score ALL candidates. Reasoning must cite actual numbers for each candidate.
`proposed_changes` may only contain fields owned by fluidics: `d_mm`.
If no fluidics edit is needed, return `{}`.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  DR. SAFETY — scoring prompt
# ═══════════════════════════════════════════════════════════════════════════════

_SAFETY_SYSTEM = """\
You are DR. SAFETY in the FLORA ENGINE v4 council. Safety is Priority 2 in FLORA
(below physical feasibility only). Your job is to score every candidate, identify
every safety-relevant risk, and justify why it can or cannot be managed.
Write as if you are signing off a risk assessment — be specific and thorough.

## Your scoring domain (0–1)

**Thermal safety:**
  Da_thermal = k_ΔH × τ × (−ΔH) / (ρ × Cp × T_ref). Interpret:
  Da_thermal << 0.1 → isothermal, safe. > 0.3 → REVISE (verify heat removal).
  > 1.0 → BLOCK (runaway risk). For most photoredox, thermal hazard is low —
  say so explicitly rather than leaving it unaddressed.

**BPR adequacy:**
  Liquid-only: BPR_set = P_vap(T) + ΔP + 0.5 bar minimum safety margin.
  Gas-liquid:  hard floor = max(3.0, P_vap + ΔP); recommended setting adds
               a 2.0 bar robustness margin. BPR = 3.0 bar is an allowed
               evidence-backed screening point when the hard floor is met.
  Call `calculate_bpr_required` and show the calculated P_min and P_recommended.
  Compare to current BPR setting. State whether BPR is adequate with margin.

**Material compatibility:**
  Call `check_material_compatibility` for the specific solvent + material + T.
  RULES:
  - FEP/PFA mandatory for photochemical reactor section. PTFE = opaque = BLOCK.
  - FEP at ≤ 25°C, pH ≤ 12: fully compatible — do not create false concerns.
  - High pressure (> 20 bar) or Alkaline (pH > 10) at T > 60°C: PFA preferred.
  - SS316: check for halide content before approving.
  - material_rationale MUST state pressure, temperature, pH, and solvent — never generic.

**Atmosphere isolation:**
  Multi-stage processes with different atmospheres: verify physical separation exists.
  State what isolation method is needed (gas-permeable membrane, T-piece, degasser).
  Single-stage processes: note this is not applicable.

**Hazard identification:**
  List every toxic, flammable, reactive, or carcinogenic species explicitly.
  Include: hazard class (CMR, flammable, irritant), handling requirements (extraction,
  inert atmosphere, spill containment), and any downstream inline quench needed.
  If a reaction produces a hazardous intermediate (diazonium, peroxide, organolithium),
  state the downstream quench requirement.

**Pressure drop safety:**
  ΔP headroom < 20% → safety risk (overpressure on restriction). Prefer > 40%.

## Verdicts
  ACCEPT:              safety_score ≥ 0.80, all gates passed
  APPROVED_WITH_CONDITIONS: score 0.60–0.80, manageable conditions explicitly stated
  REVISE:              BPR inadequate, material concern, thermal caution
  BLOCK:               Da_thermal > 1.0, gas-liquid BPR < 3 bar, opaque photoreactor,
                       incompatible material without a safe alternative

## Tools
Call `check_material_compatibility` for each candidate. Show the result.
Call `calculate_bpr_required` to compute BPR need. Show P_vapor, P_min, P_recommended.

## REQUIRED OUTPUT — JSON only
```json
{
  "overall_analysis": "3–5 sentences: what is the safety landscape? Are thermal risks present? What are the key hazard species and how are they managed? Is BPR adequate across all candidates? Any material concerns? What conditions does the lab chemist MUST follow?",
  "scores": [
    {
      "candidate_id": 1,
      "reasoning": "3–5 sentences for THIS candidate. State Da_thermal, BPR_required, BPR_current, material check result, hazard flags, atmosphere requirements. Cite tool results explicitly. Explain why the candidate is safe or what specific conditions must be met. Make it self-contained.",
      "safety_score": 0.92,
      "safety_delta_flow_value": 0.85,
      "Da_thermal": 0.03,
      "thermal_safe": true,
      "BPR_required_bar": 1.8,
      "BPR_current_bar": 3.0,
      "BPR_adequate": true,
      "system_type": "liquid",
      "material_compatible": true,
      "material_recommendation": "FEP",
      "material_rationale": "DMF solvent at 25°C, ambient pressure, pH ~7 — FEP is fully compatible per IDEX/Swagelok compatibility data. No halides, no elevated temperature concern.",
      "atmosphere_isolation_required": false,
      "isolation_method": "N/A — single stage, inert N₂ atmosphere throughout",
      "hazard_flags": [
        "DMF — SVHC reproductive toxin (REACH); use fume hood, closed system, PPE (nitrile gloves + goggles)",
        "Ir(ppy)₃ — potential skin sensitiser; handle as potent pharmaceutical"
      ],
      "verdict": "ACCEPT",
      "proposed_changes": {},
      "blocking_issues": [],
      "conditions": ["Ensure N₂ blanket on all feed reservoirs before pressurising"]
    }
  ]
}
```
Score ALL candidates. Reasoning must cite tool results and specific numbers.
`proposed_changes` may only contain fields owned by safety: `BPR_bar`, `tubing_material`.
If no safety edit is needed, return `{}`.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Common context builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_scoring_context(
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    intensification_mandate: Optional[dict] = None,
) -> str:
    n = len(candidates)
    slim = [_slim_candidate(c) for c in candidates]
    return (
        REVIEW_POLICY + f"\n## Chemistry brief\n{chemistry_brief}\n\n"
        f"## Intensification mandate\n{json.dumps(mandate_dict(intensification_mandate), ensure_ascii=False)}\n\n"
        f"## User objectives\n{objectives or 'balanced'}\n\n"
        f"## Photochemical reaction: {is_photochem}\n\n"
        f"## Pump max pressure: {pump_max_bar} bar\n\n"
        f"## Surviving candidate table ({n} candidates — score ALL)\n"
        f"{table_markdown}\n\n"
        f"## Candidate flow-sense evidence JSON\n{json.dumps(slim, ensure_ascii=False)}\n\n"
        "For inventory_realized candidates, geometry and feed assignments are the actual declared hardware. "
        "Treat tau_min as the labeled nominal liquid-volume index, not a measured multiphase contact time. "
        "Pressure-drop, mixing and conversion proxies cannot establish packed-bed behavior or measured yield. "
        "Distinguish a hard physical/inventory conflict from a heuristic operating margin or an unvalidated intensification target. "
        "Proposed changes must stay within the authoritative inventory; if a different time is needed on a fixed reactor, propose a compatible liquid flow.\n\n"
        "Each scoring agent must include its flow-value contribution field in each score entry: "
        "Dr. Chemistry -> selectivity_flow_value; Dr. Kinetics -> tau_reduction_flow_value plus "
        "achieved_reduction_factor, target_reduction_factor, reduction_achievement; "
        "Dr. Fluidics -> heat_transfer_flow_value; Dr. Safety -> safety_delta_flow_value.\n\n"
        "Score every candidate. Provide detailed reasoning per candidate. "
        "Return JSON with 'overall_analysis' and 'scores' array."
    )


def _build_subset_table(candidates: list[dict]) -> str:
    """Render a markdown table preserving actual candidate ids for subset scoring."""
    if not candidates:
        return "_no candidates_"
    header = (
        "| id | τ min | d mm | Q mL/min | C M | V_R mL | L m | Re | ΔP bar | "
        "BPR bar | material | r_mix | X | prod mg/h | pareto | flags |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for c in candidates:
        flags = []
        if c.get("pareto_front"):
            flags.append("★")
        if c.get("warnings"):
            flags.append(f"⚠{len(c['warnings'])}")
        if not c.get("feasible", True):
            flags.append("✗")
        hg = c.get("hard_gate_flags") or []
        if hg:
            flags.append(f"⛔{len(hg)}")
        rows.append(
            f"| {c.get('id', '')} | {c['tau_min']} | {c['d_mm']} | {c['Q_mL_min']} | "
            f"{c.get('concentration_M', '')} | "
            f"{c['V_R_mL']} | {c['L_m']} | {c['Re']:.0f} | {c['delta_P_bar']:.3f} | "
            f"{c.get('BPR_bar', 0.0)} | {c.get('tubing_material', 'FEP')} | "
            f"{c['r_mix']:.3f} | {c['expected_conversion']:.2f} | "
            f"{c['productivity_mg_h']:.1f} | "
            f"{'✓' if c.get('pareto_front') else ' '} | {' '.join(flags)} |"
        )
    return header + "\n" + "\n".join(rows)


def _ordered_scores(score_map: dict[int, dict], candidates: list[dict]) -> list[dict]:
    ordered: list[dict] = []
    for c in candidates:
        cid = int(c.get("id", 0))
        if cid in score_map:
            ordered.append(score_map[cid])
    return ordered


def _deterministic_coverage_score(domain: str, candidate: dict) -> dict:
    """Return conservative numeric triage when a scoring LLM omits a row.

    This is deliberately not a substitute for domain-expert reasoning. It only
    applies policies already encoded in the bounded prompts, labels its source,
    and leaves non-numeric domains at a neutral warning score. Independent hard
    gates still decide whether a candidate may proceed.
    """
    cid = int(candidate.get("id", 0))
    provenance = {
        "source": "deterministic_coverage_fallback",
        "reason": "LLM score missing after bounded retries",
        "limitations": "Numeric policy triage only; no domain-expert inference was obtained.",
    }
    common = {
        "candidate_id": cid,
        "proposed_changes": {},
        "concerns": ["Domain LLM score unavailable; deterministic coverage fallback used."],
        "score_provenance": provenance,
    }

    if domain == "kinetics":
        conversion = _safe_float(candidate.get("expected_conversion"), 0.0)
        if not validated_model(candidate, "kinetics"):
            score, verdict = 0.65, "WARNING"
        elif conversion < 0.15:
            score, verdict = 0.10, "BLOCK"
        elif conversion < 0.30:
            score, verdict = 0.40, "REVISE"
        else:
            score, verdict = 0.65, "WARNING"
        return {
            **common,
            "reasoning": (
                "Missing kinetics review; unvalidated conversion cannot establish failure."
                if not validated_model(candidate, "kinetics") else
                "Deterministic triage of the declared validated conversion model."
            ),
            "kinetics_score": score,
            "X_estimated": conversion,
            "tau_proposed_final_min": _safe_float(candidate.get("tau_min")),
            "verdict": verdict,
        }

    if domain == "fluidics":
        length_m = _safe_float(candidate.get("L_m"))
        delta_p_bar = _safe_float(candidate.get("delta_P_bar"))
        reynolds = _safe_float(candidate.get("Re"))
        hard_fail = length_m > 30.0 or delta_p_bar > 5.0 or reynolds > 2300.0
        soft_fail = length_m > 15.0 or delta_p_bar > 2.0
        score, verdict = (
            (0.10, "BLOCK") if hard_fail else
            (0.45, "REVISE") if soft_fail else
            (0.65, "WARNING")
        )
        return {
            **common,
            "reasoning": (
                "Deterministic fluidics triage from the declared length, pressure-drop, "
                "and Reynolds-number thresholds."
            ),
            "fluidics_score": score,
            "Re": reynolds,
            "dP_bar": delta_p_bar,
            "r_mix": _safe_float(candidate.get("r_mix")),
            "verdict": verdict,
        }

    if domain == "safety":
        bpr_bar = _safe_float(candidate.get("BPR_bar"))
        return {
            **common,
            "reasoning": (
                "No deterministic rule can replace a chemistry-specific safety review; "
                "the candidate remains a warning and is still subject to independent hard gates."
            ),
            "safety_score": 0.50,
            "BPR_current_bar": bpr_bar,
            "BPR_adequate": None,
            "material_recommendation": str(candidate.get("tubing_material") or ""),
            "verdict": "WARNING",
        }

    return {
        **common,
        "reasoning": (
            "No deterministic rule can replace chemistry review; the candidate remains "
            "a neutral warning and is still subject to independent chemistry hard gates."
        ),
        "combined_score": 0.50,
        "verdict": "WARNING",
    }


def _repair_missing_scores_deterministically(
    *,
    domain: str,
    agent_name: str,
    score_map: dict[int, dict],
    candidates: list[dict],
) -> list[int]:
    """Fill only omitted score rows and return the repaired candidate IDs."""
    repaired: list[int] = []
    for candidate in candidates:
        cid = int(candidate.get("id", 0))
        if cid in score_map:
            continue
        score_map[cid] = _deterministic_coverage_score(domain, candidate)
        repaired.append(cid)
    if repaired:
        logger.error(
            "%s exhausted bounded scoring retries for candidate ids %s; "
            "using explicitly labeled deterministic coverage fallback",
            agent_name,
            repaired,
        )
    return repaired


def _enforce_coverage(
    *,
    agent_name: str,
    scores: list[dict],
    valid_ids: set[int],
    strict_coverage: bool,
) -> None:
    seen = {int(entry.get("candidate_id")) for entry in scores if entry.get("candidate_id") is not None}
    missing = sorted(valid_ids - seen)
    if not missing:
        return
    msg = (
        f"{agent_name} returned scores for {len(seen)}/{len(valid_ids)} candidates; "
        f"missing candidate ids {missing}"
    )
    if strict_coverage:
        raise ScoringCoverageError(msg)
    logger.error("%s — Chief selection will be unreliable.", msg)


def _run_scoring_agent_batched(
    *,
    domain: str,
    agent_name: str,
    system_prompt: str,
    candidates: list[dict],
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    tools: list[dict],
    max_tokens: int,
    batch_size: int,
    strict_coverage: bool,
    intensification_mandate: Optional[dict] = None,
) -> tuple[str, list[dict], list[dict]]:
    """Run one scoring agent over candidate subsets, retrying only missing ids."""
    remaining = list(candidates)
    score_map: dict[int, dict] = {}
    all_tc: list[dict] = []
    overall_parts: list[str] = []
    attempts = 0

    while remaining and attempts < 2:
        attempts += 1
        still_missing: set[int] = set()

        for start in range(0, len(remaining), batch_size):
            batch = remaining[start:start + batch_size]
            batch_table = _build_subset_table(batch)
            batch_context = _build_scoring_context(
                batch,
                batch_table,
                chemistry_brief,
                objectives,
                is_photochem,
                pump_max_bar,
                intensification_mandate,
            )
            valid_ids = {int(c.get("id", i + 1)) for i, c in enumerate(batch)}
            oa, cleaned, tc_log = _run_scoring_agent(
                agent_name,
                system_prompt,
                batch_context,
                tools,
                valid_ids,
                max_tokens,
            )
            if oa:
                overall_parts.append(oa.strip())
            all_tc.extend(tc_log)
            for entry in cleaned:
                score_map[int(entry["candidate_id"])] = entry
            seen = {int(entry["candidate_id"]) for entry in cleaned}
            missing = sorted(valid_ids - seen)
            if missing:
                logger.warning("%s batch missing candidate ids %s", agent_name, missing)
                still_missing.update(missing)

        remaining = [c for c in candidates if int(c.get("id", 0)) in still_missing and int(c.get("id", 0)) not in score_map]

    repaired_ids: list[int] = []
    if strict_coverage:
        repaired_ids = _repair_missing_scores_deterministically(
            domain=domain,
            agent_name=agent_name,
            score_map=score_map,
            candidates=candidates,
        )
    ordered_scores = _ordered_scores(score_map, candidates)
    _enforce_coverage(
        agent_name=agent_name,
        scores=ordered_scores,
        valid_ids={int(c.get("id", 0)) for c in candidates},
        strict_coverage=strict_coverage,
    )
    if repaired_ids:
        overall_parts.append(
            f"Deterministic coverage fallback supplied missing {domain} rows for "
            f"candidate ids {repaired_ids}; no domain-expert inference is claimed."
        )
        all_tc.append({
            "name": "deterministic_scoring_coverage_fallback",
            "domain": domain,
            "candidate_ids": repaired_ids,
            "reason": "LLM score missing after bounded retries",
        })
    overall_analysis = " ".join(part for part in overall_parts if part)
    return overall_analysis, ordered_scores, all_tc


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-agent run functions
# ═══════════════════════════════════════════════════════════════════════════════

def _run_scoring_agent(
    agent_name: str,
    system_prompt: str,
    context: str,
    tools: list[dict],
    valid_ids: set[int],
    max_tokens: int = 4000,
) -> tuple[str, list[dict], list[dict]]:
    """Run one scoring agent with one automatic retry if zero scores are parsed.

    Returns (overall_analysis, score_list, tool_calls_log).
    """
    def _attempt(ctx: str, mt: int, tool_turns: int) -> tuple[str, list[dict], list[dict]]:
        raw, tc_log = call_llm_with_tools(
            system_prompt, ctx,
            tools=tools,
            tool_executor=execute_tool,
            max_tokens=mt,
            max_tool_turns=tool_turns,
        )
        overall_analysis, scores = _parse_score_response(raw)
        cleaned: list[dict] = []
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("candidate_id", entry.get("candidateId", entry.get("id")))
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            if cid not in valid_ids:
                logger.debug("%s: candidate_id=%s not in valid_ids %s — skipped",
                             agent_name, cid, valid_ids)
                continue
            entry["candidate_id"] = cid
            cleaned.append(entry)
        return overall_analysis, cleaned, tc_log

    try:
        oa, cleaned, tc_log = _attempt(context, max_tokens, tool_turns=6)

        # Retry if no candidates were scored — likely a truncated JSON response
        if not cleaned and valid_ids:
            missing_ids = sorted(valid_ids)
            logger.warning(
                "%s returned 0 scores for %d candidates %s — retrying with explicit "
                "reminder and higher token budget",
                agent_name, len(valid_ids), missing_ids,
            )
            retry_context = (
                context
                + f"\n\n⚠️ RETRY: Your previous response contained no per-candidate scores. "
                f"You MUST score ALL {len(valid_ids)} candidates with IDs {missing_ids}. "
                f"Return ONLY valid JSON with 'overall_analysis' and 'scores' array. "
                f"No prose before the JSON. Start your response with '{{'"
            )
            oa, cleaned, tc_log2 = _attempt(retry_context, max_tokens + 2000, tool_turns=8)
            tc_log = tc_log + tc_log2
            if not cleaned:
                logger.error(
                    "%s still returned 0 scores after retry — domain scores will "
                    "default to 0.500. Chief selection will be unreliable.",
                    agent_name,
                )

        return oa, cleaned, tc_log

    except Exception as e:
        logger.warning("%s scoring LLM call failed: %s", agent_name, e)
        return "", [], []


def run_chemistry_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    max_tokens: int = 4000,
    batch_size: int | None = None,
    strict_coverage: bool = False,
    benchmark_claude_compact_mode: bool = False,
    intensification_mandate: Optional[dict] = None,
) -> tuple[str, list[dict], list[dict]]:
    """Run Dr. Chemistry. Returns (overall_analysis, per_candidate_scores, tool_calls)."""
    if _is_gemma_local_mode():
        oa, scores, tc = _run_scoring_agent_gemma_local(
            domain="chemistry",
            agent_name="Dr. Chemistry",
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            intensification_mandate=intensification_mandate,
        )
        logger.info("    Dr. Chemistry scored %d/%d candidates", len(scores), len(candidates))
        return oa, scores, tc
    if _use_bounded_anthropic_scoring():
        oa, scores, tc = _run_scoring_agent_claude_compact(
            domain="chemistry",
            agent_name="Dr. Chemistry",
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            intensification_mandate=intensification_mandate,
            batch_size=batch_size or 3,
            strict_coverage=strict_coverage,
        )
        logger.info("    Dr. Chemistry scored %d/%d candidates", len(scores), len(candidates))
        return oa, scores, tc
    context = _build_scoring_context(
        candidates, table_markdown, chemistry_brief, objectives,
        is_photochem, pump_max_bar, intensification_mandate,
    )
    valid_ids = {c.get("id", i + 1) for i, c in enumerate(candidates)}
    if batch_size and batch_size <= len(candidates):
        oa, scores, tc = _run_scoring_agent_batched(
            domain="chemistry",
            agent_name="Dr. Chemistry",
            system_prompt=_CHEMISTRY_SYSTEM + REVIEW_POLICY,
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            tools=CHEMISTRY_TOOLS,
            max_tokens=max_tokens,
            batch_size=batch_size,
            strict_coverage=strict_coverage,
            intensification_mandate=intensification_mandate,
        )
    else:
        oa, scores, tc = _run_scoring_agent(
            "Dr. Chemistry", _CHEMISTRY_SYSTEM + REVIEW_POLICY, context,
            CHEMISTRY_TOOLS, valid_ids, max_tokens,
        )
        _enforce_coverage(
            agent_name="Dr. Chemistry",
            scores=scores,
            valid_ids={int(v) for v in valid_ids},
            strict_coverage=strict_coverage,
        )
    logger.info("    Dr. Chemistry scored %d/%d candidates", len(scores), len(candidates))
    return oa, scores, tc


def run_kinetics_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    max_tokens: int = 4000,
    batch_size: int | None = None,
    strict_coverage: bool = False,
    benchmark_claude_compact_mode: bool = False,
    intensification_mandate: Optional[dict] = None,
) -> tuple[str, list[dict], list[dict]]:
    """Run Dr. Kinetics. Returns (overall_analysis, per_candidate_scores, tool_calls)."""
    if _is_gemma_local_mode():
        oa, scores, tc = _run_scoring_agent_gemma_local(
            domain="kinetics",
            agent_name="Dr. Kinetics",
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            intensification_mandate=intensification_mandate,
        )
        logger.info("    Dr. Kinetics scored %d/%d candidates", len(scores), len(candidates))
        return oa, scores, tc
    if _use_bounded_anthropic_scoring():
        oa, scores, tc = _run_scoring_agent_claude_compact(
            domain="kinetics",
            agent_name="Dr. Kinetics",
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            intensification_mandate=intensification_mandate,
            batch_size=batch_size or 3,
            strict_coverage=strict_coverage,
        )
        logger.info("    Dr. Kinetics scored %d/%d candidates", len(scores), len(candidates))
        return oa, scores, tc
    context = _build_scoring_context(
        candidates, table_markdown, chemistry_brief, objectives,
        is_photochem, pump_max_bar, intensification_mandate,
    )
    valid_ids = {c.get("id", i + 1) for i, c in enumerate(candidates)}
    if batch_size and batch_size <= len(candidates):
        oa, scores, tc = _run_scoring_agent_batched(
            domain="kinetics",
            agent_name="Dr. Kinetics",
            system_prompt=_KINETICS_SYSTEM + REVIEW_POLICY,
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            tools=KINETICS_TOOLS,
            max_tokens=max_tokens,
            batch_size=batch_size,
            strict_coverage=strict_coverage,
            intensification_mandate=intensification_mandate,
        )
    else:
        oa, scores, tc = _run_scoring_agent(
            "Dr. Kinetics", _KINETICS_SYSTEM + REVIEW_POLICY, context,
            KINETICS_TOOLS, valid_ids, max_tokens,
        )
        _enforce_coverage(
            agent_name="Dr. Kinetics",
            scores=scores,
            valid_ids={int(v) for v in valid_ids},
            strict_coverage=strict_coverage,
        )
    logger.info("    Dr. Kinetics scored %d/%d candidates", len(scores), len(candidates))
    return oa, scores, tc


def run_fluidics_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    max_tokens: int = 4000,
    batch_size: int | None = None,
    strict_coverage: bool = False,
    benchmark_claude_compact_mode: bool = False,
    intensification_mandate: Optional[dict] = None,
) -> tuple[str, list[dict], list[dict]]:
    """Run Dr. Fluidics. Returns (overall_analysis, per_candidate_scores, tool_calls)."""
    if _is_gemma_local_mode():
        oa, scores, tc = _run_scoring_agent_gemma_local(
            domain="fluidics",
            agent_name="Dr. Fluidics",
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            intensification_mandate=intensification_mandate,
        )
        logger.info("    Dr. Fluidics scored %d/%d candidates", len(scores), len(candidates))
        return oa, scores, tc
    if _use_bounded_anthropic_scoring():
        oa, scores, tc = _run_scoring_agent_claude_compact(
            domain="fluidics",
            agent_name="Dr. Fluidics",
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            intensification_mandate=intensification_mandate,
            batch_size=batch_size or 3,
            strict_coverage=strict_coverage,
        )
        logger.info("    Dr. Fluidics scored %d/%d candidates", len(scores), len(candidates))
        return oa, scores, tc
    context = _build_scoring_context(
        candidates, table_markdown, chemistry_brief, objectives,
        is_photochem, pump_max_bar, intensification_mandate,
    )
    valid_ids = {c.get("id", i + 1) for i, c in enumerate(candidates)}
    if batch_size and batch_size <= len(candidates):
        oa, scores, tc = _run_scoring_agent_batched(
            domain="fluidics",
            agent_name="Dr. Fluidics",
            system_prompt=_FLUIDICS_SYSTEM + REVIEW_POLICY,
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            tools=_FLUIDICS_V4_TOOLS,
            max_tokens=max_tokens,
            batch_size=batch_size,
            strict_coverage=strict_coverage,
            intensification_mandate=intensification_mandate,
        )
    else:
        oa, scores, tc = _run_scoring_agent(
            "Dr. Fluidics", _FLUIDICS_SYSTEM + REVIEW_POLICY, context,
            _FLUIDICS_V4_TOOLS, valid_ids, max_tokens,
        )
        _enforce_coverage(
            agent_name="Dr. Fluidics",
            scores=scores,
            valid_ids={int(v) for v in valid_ids},
            strict_coverage=strict_coverage,
        )
    logger.info("    Dr. Fluidics scored %d/%d candidates", len(scores), len(candidates))
    return oa, scores, tc


def run_safety_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    max_tokens: int = 4000,
    batch_size: int | None = None,
    strict_coverage: bool = False,
    benchmark_claude_compact_mode: bool = False,
    intensification_mandate: Optional[dict] = None,
) -> tuple[str, list[dict], list[dict]]:
    """Run Dr. Safety. Returns (overall_analysis, per_candidate_scores, tool_calls)."""
    if _is_gemma_local_mode():
        oa, scores, tc = _run_scoring_agent_gemma_local(
            domain="safety",
            agent_name="Dr. Safety",
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            intensification_mandate=intensification_mandate,
        )
        logger.info("    Dr. Safety scored %d/%d candidates", len(scores), len(candidates))
        return oa, scores, tc
    if _use_bounded_anthropic_scoring():
        oa, scores, tc = _run_scoring_agent_claude_compact(
            domain="safety",
            agent_name="Dr. Safety",
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            intensification_mandate=intensification_mandate,
            batch_size=batch_size or 3,
            strict_coverage=strict_coverage,
        )
        logger.info("    Dr. Safety scored %d/%d candidates", len(scores), len(candidates))
        return oa, scores, tc
    context = _build_scoring_context(
        candidates, table_markdown, chemistry_brief, objectives,
        is_photochem, pump_max_bar, intensification_mandate,
    )
    valid_ids = {c.get("id", i + 1) for i, c in enumerate(candidates)}
    if batch_size and batch_size <= len(candidates):
        oa, scores, tc = _run_scoring_agent_batched(
            domain="safety",
            agent_name="Dr. Safety",
            system_prompt=_SAFETY_SYSTEM + REVIEW_POLICY,
            candidates=candidates,
            chemistry_brief=chemistry_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max_bar,
            tools=SAFETY_TOOLS,
            max_tokens=max_tokens,
            batch_size=batch_size,
            strict_coverage=strict_coverage,
            intensification_mandate=intensification_mandate,
        )
    else:
        oa, scores, tc = _run_scoring_agent(
            "Dr. Safety", _SAFETY_SYSTEM + REVIEW_POLICY, context,
            SAFETY_TOOLS, valid_ids, max_tokens,
        )
        _enforce_coverage(
            agent_name="Dr. Safety",
            scores=scores,
            valid_ids={int(v) for v in valid_ids},
            strict_coverage=strict_coverage,
        )
    logger.info("    Dr. Safety scored %d/%d candidates", len(scores), len(candidates))
    return oa, scores, tc


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator: run all 4 agents
# ═══════════════════════════════════════════════════════════════════════════════

def run_domain_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    batch_size: int | None = None,
    strict_coverage: bool = False,
    benchmark_claude_compact_mode: bool = False,
    intensification_mandate: Optional[dict] = None,
    enabled_agents: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict:
    """Run all four domain scoring agents (Stage 2).

    Returns:
        {
          "chemistry_overall": str,
          "chemistry_scores":   [per-candidate dicts],
          "kinetics_overall":   str,
          "kinetics_scores":    [per-candidate dicts],
          "fluidics_overall":   str,
          "fluidics_scores":    [per-candidate dicts],
          "safety_overall":     str,
          "safety_scores":      [per-candidate dicts],
          "tool_calls": {...},
          "blocked_by_scoring": [candidate_ids with a BLOCK verdict]
        }
    """
    enabled = set(enabled_agents or ("chemistry", "kinetics", "fluidics", "safety"))

    chem_oa, chemistry_scores, chem_tc = run_chemistry_scoring(
        candidates=candidates, table_markdown=table_markdown,
        chemistry_brief=chemistry_brief, objectives=objectives,
        is_photochem=is_photochem, pump_max_bar=pump_max_bar,
        batch_size=batch_size, strict_coverage=strict_coverage,
        benchmark_claude_compact_mode=benchmark_claude_compact_mode,
        intensification_mandate=intensification_mandate,
    ) if "chemistry" in enabled else ("Disabled by execution contract", [], [])
    kin_oa, kinetics_scores, kin_tc = run_kinetics_scoring(
        candidates=candidates, table_markdown=table_markdown,
        chemistry_brief=chemistry_brief, objectives=objectives,
        is_photochem=is_photochem, pump_max_bar=pump_max_bar,
        batch_size=batch_size, strict_coverage=strict_coverage,
        benchmark_claude_compact_mode=benchmark_claude_compact_mode,
        intensification_mandate=intensification_mandate,
    ) if "kinetics" in enabled else ("Disabled by execution contract", [], [])
    flu_oa, fluidics_scores, flu_tc = run_fluidics_scoring(
        candidates=candidates, table_markdown=table_markdown,
        chemistry_brief=chemistry_brief, objectives=objectives,
        is_photochem=is_photochem, pump_max_bar=pump_max_bar,
        batch_size=batch_size, strict_coverage=strict_coverage,
        benchmark_claude_compact_mode=benchmark_claude_compact_mode,
        intensification_mandate=intensification_mandate,
    ) if "fluidics" in enabled else ("Disabled by execution contract", [], [])
    saf_oa, safety_scores, saf_tc = run_safety_scoring(
        candidates=candidates, table_markdown=table_markdown,
        chemistry_brief=chemistry_brief, objectives=objectives,
        is_photochem=is_photochem, pump_max_bar=pump_max_bar,
        batch_size=batch_size, strict_coverage=strict_coverage,
        benchmark_claude_compact_mode=benchmark_claude_compact_mode,
        intensification_mandate=intensification_mandate,
    ) if "safety" in enabled else ("Disabled by execution contract", [], [])

    blocked: set[int] = set()
    for domain_scores in (chemistry_scores, kinetics_scores, fluidics_scores, safety_scores):
        for entry in domain_scores:
            if str(entry.get("verdict", "")).upper() == "BLOCK":
                blocked.add(entry.get("candidate_id", -1))

    result = {
        "chemistry_overall": chem_oa,
        "chemistry_scores": chemistry_scores,
        "kinetics_overall": kin_oa,
        "kinetics_scores": kinetics_scores,
        "fluidics_overall": flu_oa,
        "fluidics_scores": fluidics_scores,
        "safety_overall": saf_oa,
        "safety_scores": safety_scores,
        "tool_calls": {
            "chemistry": chem_tc,
            "kinetics": kin_tc,
            "fluidics": flu_tc,
            "safety": saf_tc,
        },
        "blocked_by_scoring": sorted(blocked),
        "enabled_scoring_agents": sorted(enabled),
        "disabled_scoring_agents": sorted(
            {"chemistry", "kinetics", "fluidics", "safety"} - enabled
        ),
    }
    return enrich_scoring_with_flow_values(
        candidates, result, intensification_mandate=intensification_mandate
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Score lookup helpers (used by Chief)
# ═══════════════════════════════════════════════════════════════════════════════

def get_domain_score(score_list: list[dict], candidate_id: int, score_key: str,
                     default: float = 0.5) -> float:
    for entry in score_list:
        if entry.get("candidate_id") == candidate_id:
            v = entry.get(score_key, default)
            try:
                return max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                return default
    return default


def get_chemistry_combined(scores: list[dict], cid: int) -> float:
    return get_domain_score(scores, cid, "combined_score")


def get_kinetics_score(scores: list[dict], cid: int) -> float:
    return get_domain_score(scores, cid, "kinetics_score")


def get_fluidics_score(scores: list[dict], cid: int) -> float:
    return get_domain_score(scores, cid, "fluidics_score")


def get_safety_score(scores: list[dict], cid: int) -> float:
    return get_domain_score(scores, cid, "safety_score")


def geometry_practicality_score(candidate: dict) -> float:
    L = float(candidate.get("L_m") or 0.0)
    V_R = float(candidate.get("V_R_mL") or 0.0)
    L_score = max(0.0, min(1.0, 1.0 - (L - 10.0) / 15.0)) if L > 10.0 else 1.0
    V_score = max(0.0, min(1.0, 1.0 - (V_R - 10.0) / 40.0)) if V_R > 10.0 else 1.0
    return 0.5 * L_score + 0.5 * V_score


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3.5 — Revision Agent
# ═══════════════════════════════════════════════════════════════════════════════

_REVISION_SYSTEM = """\
You are the REVISION ENGINEER in the FLORA ENGINE v4 council (Stage 3.5).
Your role: synthesize domain expert REVISE/BLOCK verdicts for the selected winner
candidate into concrete, minimal, justified parameter edits.

## Rules — only change what was explicitly flagged

**τ (residence time) — revise only if Dr. Kinetics gave REVISE/BLOCK for low X:**
  Target X = 0.90. τ_revised = −τ_kinetics_min × ln(1 − 0.90) / ln(10) = τ_kinetics_min.
  Cap at min(4 × τ_original, 300 min). Show arithmetic.

**d (tube diameter) — revise only if Dr. Fluidics flagged r_mix > 0.20:**
  d_fix = d_current × √(0.15 / r_mix_current).
  Round DOWN to nearest commercial size: [0.5, 0.75, 1.0, 1.6, 2.0, 3.0 mm].
  NEVER increase d to fix mixing. Show arithmetic.

**BPR — revise only if Dr. Safety flagged BPR_adequate=false:**
  Liquid-only: BPR_bar = P_vap(T) + ΔP + 0.5 bar safety margin.
  Gas-liquid:  BPR_bar = max(3.0, P_vap + ΔP) + 2.0 bar when adding the
               recommended robustness margin; 3.0 bar remains the hard floor.
  If agent reported BPR_required_bar, set BPR = BPR_required_bar + 0.5.
  Show arithmetic.

**tubing_material — revise only if Dr. Safety or Dr. Chemistry flagged incompatibility:**
  Photochem → FEP or PFA (never PTFE/SS).
  High-T/high-P (> 80°C AND > 10 bar) → PFA.
  Standard thermal → FEP (preferred) or PFA.
  State the reason.

**concentration_M — revise only if Dr. Chemistry explicitly flagged concentration issue:**
  Photoredox: sweet spot 0.05–0.20 M. Adjust to midpoint of valid range.

## Important
- If ALL flagged issues are already BPR/material (rule-based) and no kinetics/fluidics REVISE
  exists, propose BPR and material only.
- If no parameter needs changing (all verdicts ACCEPT/WARNING), output empty proposed_changes.
- Be conservative: a WARNING alone does not justify revision.

## REQUIRED OUTPUT — JSON only
```json
{
  "proposed_changes": {
    "tau_min": 127.3,
    "d_mm": 0.75,
    "BPR_bar": 0.6,
    "tubing_material": "FEP"
  },
  "unchanged_fields": ["concentration_M", "temperature_C"],
  "change_rationale": {
    "tau_min": "Dr. Kinetics REVISE: X=0.63 < 0.85 target. τ_kinetics=55 min. τ_revised = 55 min for 90% under the unvalidated first-order t90 model.",
    "d_mm": "Dr. Fluidics REVISE: r_mix=0.31. d_fix = 1.6×√(0.15/0.31) = 1.6×0.695 = 1.11 → rounded down to 1.0 mm.",
    "BPR_bar": "Dr. Safety: BPR_adequate=false. BPR_required=0.11 bar + 0.5 margin = 0.6 bar.",
    "tubing_material": "Dr. Safety: recommended FEP for DMF at 120°C, pH neutral."
  },
  "domains_that_triggered_revision": ["kinetics", "fluidics", "safety"]
}
```
"""


def run_revision_stage(
    winner: dict,
    scoring: dict,
    chemistry_brief: str,
    is_photochem: bool,
    is_gas_liquid: bool,
    pump_max_bar: float,
    solvent: str,
    temperature_C: float,
    concentration_M: float,
    extinction_coeff_M_cm: Optional[float] = None,
    batch_yield_fraction: Optional[float] = None,
    batch_time_min: Optional[float] = None,
    translation_policy: str = "intensify",
    measured_evidence_available: bool = False,
    measured_tau_floor_min: Optional[float] = None,
    pump_max_flow_mL_min: Optional[float] = None,
    gas_liquid_max_bpr_bar: float = 50.0,
) -> Optional[dict]:
    """Stage 3.5: Revision Agent proposes parameter edits for the council winner.

    Examines REVISE/BLOCK verdicts from all domain agents on the winning candidate
    and proposes concrete parameter fixes (τ, d, BPR, material). Recomputes all
    derived metrics (Re, ΔP, r_mix, X, V_R, L) via compute_metrics().

    Returns a revised candidate dict (same id, updated fields + revision metadata),
    or None if no revisions are warranted.
    """
    cid = winner.get("id")

    # Collect domain verdicts and specifics for the winner
    revision_domains: list[str] = []
    domain_specifics: dict[str, dict] = {}

    for domain, score_list_key in [
        ("chemistry", "chemistry_scores"),
        ("kinetics",  "kinetics_scores"),
        ("fluidics",  "fluidics_scores"),
        ("safety",    "safety_scores"),
    ]:
        entries = scoring.get(score_list_key, [])
        entry = next((e for e in entries if e.get("candidate_id") == cid), None)
        if entry:
            verdict = str(entry.get("verdict", "ACCEPT")).upper()
            domain_specifics[domain] = entry
            if verdict in ("REVISE", "BLOCK"):
                revision_domains.append(domain)

    if not revision_domains:
        logger.info("  Stage 3.5: winner id=%d has no REVISE/BLOCK verdicts — skipping", cid)
        return None

    # Build context for revision agent
    lines = [
        f"## Winner candidate (id={cid})",
        f"τ={winner.get('tau_min')} min | d={winner.get('d_mm')} mm | "
        f"Q={winner.get('Q_mL_min', 0):.4f} mL/min | "
        f"Re={winner.get('Re', 0):.1f} | r_mix={winner.get('r_mix', 0):.3f} | "
        f"X={winner.get('expected_conversion', 0):.2f} | "
        f"BPR={winner.get('BPR_bar', 0)} bar | material={winner.get('tubing_material', 'FEP')} | "
        f"solvent={solvent} | T={temperature_C}°C | C={concentration_M} M",
        "",
        f"## Chemistry brief\n{chemistry_brief}",
        "",
        f"## Domain verdicts requiring revision: {', '.join(revision_domains)}",
    ]

    for domain in revision_domains:
        entry = domain_specifics[domain]
        verdict = str(entry.get("verdict", "?")).upper()
        reasoning = entry.get("reasoning", "")
        lines.append(f"\n### {domain.upper()} — {verdict}")
        lines.append(f"Reasoning: {reasoning[:400]}")

        if domain == "kinetics":
            lines.append(
                f"X_estimated={entry.get('X_estimated')} | "
                f"tau_proposed_final_min={entry.get('tau_proposed_final_min')} | "
                f"tau_kinetics_min={winner.get('tau_kinetics_min')}"
            )
        elif domain == "fluidics":
            lines.append(
                f"r_mix={entry.get('r_mix')} | d_fix_mm={entry.get('d_fix_mm')} | "
                f"d_change_direction={entry.get('d_change_direction', '')}"
            )
        elif domain == "safety":
            lines.append(
                f"BPR_required_bar={entry.get('BPR_required_bar')} | "
                f"BPR_current_bar={entry.get('BPR_current_bar')} | "
                f"BPR_adequate={entry.get('BPR_adequate')} | "
                f"material_recommendation={entry.get('material_recommendation')}"
            )
        elif domain == "chemistry":
            lines.append(
                f"beer_lambert_A={entry.get('beer_lambert_A')} | "
                f"material_transparent={entry.get('material_transparent')} | "
                f"blocking_issues={entry.get('blocking_issues', [])}"
            )

    lines.append("\n\nPropose concrete parameter edits only for flagged issues. Output JSON only.")
    context = "\n".join(lines)

    revision_data: dict = {}
    try:
        raw = call_llm(_REVISION_SYSTEM + REVIEW_POLICY, context, max_tokens=700)
        s = raw.strip()
        if "```" in s:
            for part in s.split("```")[1::2]:
                try:
                    revision_data = json.loads(part.lstrip("json").strip())
                    break
                except json.JSONDecodeError:
                    continue
        if not revision_data:
            try:
                revision_data = json.loads(s)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.warning("Revision agent LLM call failed: %s", e)
        return None

    proposed = revision_data.get("proposed_changes", {})
    if not proposed:
        logger.info("  Stage 3.5: revision agent proposed no changes for winner id=%d", cid)
        return None

    # Build revised candidate by recomputing metrics with new τ/d if changed
    new_tau = float(proposed.get("tau_min", winner.get("tau_min")))
    new_d   = float(proposed.get("d_mm",   winner.get("d_mm")))

    current_tau = float(winner.get("tau_min", 0.0) or 0.0)
    if (
        (translation_policy or "").lower() == "evidence_first"
        and not measured_evidence_available
        and current_tau > 0
        and new_tau < current_tau - 1e-6
    ):
        logger.warning(
            "  Stage 3.5 evidence-first guard rejected unsupported tau reduction "
            "for winner id=%d: %.3f -> %.3f min",
            cid,
            current_tau,
            new_tau,
        )
        new_tau = current_tau
    if (
        measured_tau_floor_min is not None
        and measured_tau_floor_min > 0
        and new_tau < measured_tau_floor_min
    ):
        logger.warning(
            "  Stage 3.5 measured-evidence guard raised tau for winner id=%d: "
            "%.3f -> %.3f min",
            cid,
            new_tau,
            measured_tau_floor_min,
        )
        new_tau = float(measured_tau_floor_min)
    tau_k = float(winner.get("tau_kinetics_min") or new_tau or current_tau or 1.0)
    if (
        (translation_policy or "").lower() == "intensify"
        and new_tau > current_tau + 1e-6
        and tau_k > 0
    ):
        if batch_yield_fraction is not None and batch_yield_fraction > 0:
            x_target = min(0.90, max(float(batch_yield_fraction) * 1.10, 0.65))
        else:
            x_target = 0.85
        # Required τ to hit x_target under first-order kinetics k = -ln(0.1)/tau_k:
        # x_target = 1 - exp(-tau / (tau_k / -ln(0.1)))
        # → tau = tau_k * ln(1/(1-x_target)) / ln(10)
        import math as _math
        tau_for_target = tau_k * _math.log(1.0 / max(1.0 - x_target, 1e-6)) / _math.log(10.0)
        # Cap the proposed bump at the target-required τ. If LLM proposed
        # more (i.e. chasing X > x_target), clamp down to x_target's tau.
        if new_tau > tau_for_target * 1.05:
            logger.info(
                "  Stage 3.5 X-target cap: proposed τ=%.1f min > τ_for_X=%.1f min "
                "(x_target=%.2f from batch_yield=%s); clamping to %.1f min.",
                new_tau, tau_for_target, x_target,
                f"{batch_yield_fraction:.2f}" if batch_yield_fraction else "default",
                tau_for_target,
            )
            new_tau = round(tau_for_target, 2)
    orig_Q  = float(winner.get("Q_mL_min") or 0.01)
    tau_k   = float(winner.get("tau_kinetics_min") or new_tau)
    IF_used = float(winner.get("IF_used") or 6.0)
    MW      = float(winner.get("assumed_MW") or 250.0)
    proposed_bpr = float(proposed.get("BPR_bar", winner.get("BPR_bar", 0.0)))
    proposed_material = proposed.get(
        "tubing_material",
        winner.get("tubing_material", "FEP"),
    )
    if (
        abs(new_tau - current_tau) <= 1e-9
        and abs(new_d - float(winner.get("d_mm") or new_d)) <= 1e-9
        and abs(proposed_bpr - float(winner.get("BPR_bar", 0.0) or 0.0)) <= 1e-9
        and proposed_material == winner.get("tubing_material", "FEP")
    ):
        logger.info(
            "  Stage 3.5: all proposed changes were rejected or no-op for winner id=%d",
            cid,
        )
        return None

    try:
        from flora_translate.engine.sampling import compute_metrics
        revised = compute_metrics(
            tau_min          = new_tau,
            d_mm             = new_d,
            Q_mL_min         = orig_Q,
            solvent          = solvent,
            temperature_C    = temperature_C,
            concentration_M  = concentration_M,
            assumed_MW       = MW,
            IF_used          = IF_used,
            tau_kinetics_min = tau_k,
            pump_max_bar     = pump_max_bar,
            is_photochem     = is_photochem,
            extinction_coeff_M_cm = extinction_coeff_M_cm,
            tau_source       = "revision_agent_stage_3.5",
            is_gas_liquid    = is_gas_liquid,
            BPR_bar           = proposed_bpr,
            target_gas_equiv_inlet = float(
                winner.get("target_gas_equiv_inlet") or 1.0
            ),
            gas_reagent_fraction = float(
                winner.get("gas_reagent_fraction") or 1.0
            ),
        )
    except Exception as e:
        logger.warning("compute_metrics failed during revision: %s — using shallow copy", e)
        revised = dict(winner)

    # Carry forward winner identity and non-physics fields
    revised["id"]             = cid
    revised["pareto_front"]   = True
    revised["feasible"]       = True
    revised["hard_gate_flags"]  = []
    revised["hard_gate_status"] = "PASS"
    # BPR and material are not recomputed by compute_metrics — apply from proposed
    revised["BPR_bar"]        = proposed_bpr
    revised["tubing_material"]= proposed_material
    # Revision provenance
    revised["revision_applied"]   = True
    revised["revision_rationale"] = revision_data.get("change_rationale", {})
    revised["revision_domains"]   = revision_data.get("domains_that_triggered_revision",
                                                       revision_domains)

    try:
        from flora_translate.engine.sampling import hard_filter
        feasible, violations, warnings = hard_filter(
            revised,
            is_photochem=is_photochem,
            is_gas_liquid=is_gas_liquid,
            pump_max_bar=pump_max_bar,
            BPR_bar=float(revised.get("BPR_bar", 0.0)),
            max_flow_rate_mL_min=pump_max_flow_mL_min,
            gas_liquid_max_bpr_bar=gas_liquid_max_bpr_bar,
        )
        if not feasible:
            logger.warning(
                "  Stage 3.5 rejected unsafe revision for winner id=%d: %s",
                cid,
                "; ".join(violations),
            )
            return None
        revised["warnings"] = warnings
    except Exception as e:
        logger.warning("hard_filter failed during revision validation: %s", e)

    logger.info(
        "  Stage 3.5 applied: id=%d | τ %.1f→%.1f min | d %.2f→%.2f mm | "
        "BPR %.2f→%.2f bar | mat %s→%s | X %.2f→%.2f | domains=%s",
        cid,
        winner.get("tau_min", 0), revised.get("tau_min", 0),
        winner.get("d_mm", 0),    revised.get("d_mm", 0),
        winner.get("BPR_bar", 0), revised.get("BPR_bar", 0),
        winner.get("tubing_material", "?"), revised.get("tubing_material", "?"),
        winner.get("expected_conversion", 0), revised.get("expected_conversion", 0),
        revision_domains,
    )
    return revised
