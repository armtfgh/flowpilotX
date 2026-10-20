"""
FLORA ENGINE v4 — Chief Engineer and CouncilV4 orchestrator.

Selection logic:
  1. Hard-gate violations are FLAGS, not removals — all candidates reach scoring.
     Flagged candidates carry hard_gate_flags so every agent sees the reasons.
     Only scoring-blocked and Skeptic CRITICAL disqualifications remove candidates.
  2. Compute weighted combined score:
       combined = 0.25*chemistry + 0.20*kinetics + 0.20*fluidics
                + 0.20*safety + 0.15*geometry_practicality
  3. Apply objective modifier (multiply domain score by 1.5 for user objective).
  4. Tiebreaker: safety → uncertainty → tau_literature proximity.
  5. If scores are close, select one defensible first screen and retain alternatives.

After selection: run DFMEA on winning candidate (Dr. Safety).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict
from typing import Optional

import flora_translate.config as cfg
from flora_translate.config import (
    ENGINE_PROVIDER,
    FLOW_MAX_TAU_TO_BATCH_RATIO,
    FLOW_TRANSLATION_POLICY,
)
from flora_translate.design_calculator import DesignCalculator, DesignCalculations
from flora_translate.engine.flow_value import attach_flow_sense_reports, pvs_for_candidate
from flora_translate.engine.llm_agents import call_llm, call_llm_with_tools
from flora_translate.engine.tool_definitions import CHIEF_TOOLS, execute_tool
from flora_translate.engine.council_v4.designer import (
    V4_GAS_LIQUID_MAX_BPR_BAR,
    _apply_v4_hard_gates,
    run_designer_v4,
    run_problem_framing,
)
from flora_translate.engine.council_v4.scoring import (
    run_domain_scoring,
    run_revision_stage,
    get_chemistry_combined, get_kinetics_score,
    get_fluidics_score, get_safety_score,
    geometry_practicality_score,
)
from flora_translate.engine.council_v4.skeptic import run_skeptic_audit
from flora_translate.engine.sampling import compute_metrics, format_candidate_table, hard_filter
from flora_translate.intake_agent import historical_text_from_package, intake_context_block
from flora_translate.inventory_constraints import available_pressure_settings, enforce_reactor_inventory
from flora_translate.schemas import (
    BatchRecord, ChemistryPlan, DesignInputPackage, FlowProposal, LabInventory,
    DesignCandidate, CouncilMessage, DeliberationLog,
    AgentDeliberation, FieldProposal,
)
from flora_translate.engine.council_v4.execution_config import CouncilExecutionConfig
from flora_translate.engine.council_v4.review_contract import REVIEW_POLICY, diverse_candidates, validated_model, candidate_fingerprint

logger = logging.getLogger("flora.engine.council_v4.chief")

_CHIEF_MAX_TOKENS = 4000


def _bypassed_skeptic_audit(candidates: list[dict]) -> dict:
    """Return an explicit no-audit record for controlled ablation runs."""
    return {
        "calculation_errors": [],
        "threshold_errors": [],
        "scope_violations": [],
        "bpr_errors": [],
        "flow_sense_errors": [],
        "process_value_scores": [],
        "weak_pool_report": None,
        "verdict": "BYPASSED",
        "all_errors": [],
        "disqualify_recommendations": [],
        "disqualify_ids": [],
        "audit_summary": (
            f"Skeptic audit disabled by execution contract for {len(candidates)} candidates."
        ),
        "council_may_proceed": True,
        "module_disabled": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Weighted scoring
# ═══════════════════════════════════════════════════════════════════════════════

_WEIGHTS = {
    "chemistry": 0.25,
    "kinetics":  0.20,
    "fluidics":  0.20,
    "safety":    0.20,
    "geometry":  0.15,
}

_OBJECTIVE_MODIFIER: dict[str, tuple[str, float]] = {
    "conversion":   ("kinetics",   1.5),
    "yield":        ("kinetics",   1.5),
    "throughput":   ("geometry",   1.5),  # productivity is embedded in geometry/fluidics
    "footprint":    ("geometry",   1.5),
    "de-risk":      ("safety",     1.5),
    "safe":         ("safety",     1.5),
    "balanced":     ("chemistry",  1.0),  # no boost
}


def _objective_key(objectives: str) -> tuple[str, float]:
    """Map user objectives string to (domain_to_boost, boost_factor)."""
    obj_lower = objectives.lower()
    for kw, (domain, factor) in _OBJECTIVE_MODIFIER.items():
        if kw in obj_lower:
            return domain, factor
    return "chemistry", 1.0  # balanced default — no boost


def compute_weighted_scores(
    candidates: list[dict],
    scoring: dict,
    objectives: str,
    disqualify_ids: set[int],
    batch_time_min: Optional[float] = None,
    translation_policy: str = FLOW_TRANSLATION_POLICY,
    max_tau_to_batch_ratio: float = FLOW_MAX_TAU_TO_BATCH_RATIO,
) -> list[dict]:
    """Compute combined weighted scores for all surviving candidates.

    Returns list of {candidate_id, combined, chemistry, kinetics, fluidics,
                     safety, geometry, objective_boosted, disqualified}.
    """
    boost_domain, boost_factor = _objective_key(objectives)
    active_domains = tuple(
        scoring.get("enabled_scoring_agents")
        or ("chemistry", "kinetics", "fluidics", "safety")
    )

    results: list[dict] = []
    for c in candidates:
        cid = c.get("id", 0)
        disq = cid in disqualify_ids

        chem = get_chemistry_combined(scoring["chemistry_scores"], cid)
        kin  = get_kinetics_score(scoring["kinetics_scores"], cid)
        flu  = get_fluidics_score(scoring["fluidics_scores"], cid)
        saf  = get_safety_score(scoring["safety_scores"], cid)
        geo  = geometry_practicality_score(c)

        # Apply objective modifier
        scores = {"chemistry": chem, "kinetics": kin, "fluidics": flu,
                  "safety": saf, "geometry": geo}
        if boost_domain == "geometry" or boost_domain in active_domains:
            scores[boost_domain] = min(1.0, scores[boost_domain] * boost_factor)

        active_weight = _WEIGHTS["geometry"] + sum(
            _WEIGHTS[domain] for domain in active_domains
        )
        legacy_combined = (
            _WEIGHTS["geometry"] * scores["geometry"]
            + sum(_WEIGHTS[domain] * scores[domain] for domain in active_domains)
        ) / active_weight
        domain_mean = sum(scores[domain] for domain in active_domains) / len(active_domains)
        pvs = pvs_for_candidate(scoring, cid)
        if pvs <= 0:
            pvs = float((c.get("flow_sense_report") or {}).get("process_value_score") or 0.0)
        combined = (
            cfg.CHIEF_SCORE_WEIGHT_DOMAIN * legacy_combined
            + cfg.CHIEF_SCORE_WEIGHT_PVS * pvs
            + cfg.CHIEF_SCORE_WEIGHT_GEOMETRY * scores["geometry"]
        )
        intensification_penalty_applied = False
        if (
            (translation_policy or "").lower() == "intensify"
            and batch_time_min is not None
            and batch_time_min > 0
            and float(c.get("tau_min", 0.0) or 0.0) > batch_time_min * max_tau_to_batch_ratio
        ):
            combined *= 0.5
            intensification_penalty_applied = True

        results.append({
            "candidate_id": cid,
            "combined": round(combined, 4),
            "legacy_domain_combined": round(legacy_combined, 4),
            "domain_mean": round(domain_mean, 4),
            "PVS": round(pvs, 4),
            "chemistry": round(scores["chemistry"], 4),
            "kinetics":  round(scores["kinetics"], 4),
            "fluidics":  round(scores["fluidics"], 4),
            "safety":    round(scores["safety"], 4),
            "geometry":  round(scores["geometry"], 4),
            "objective_boosted": boost_domain,
            "disqualified": disq,
            "intensification_penalty_applied": intensification_penalty_applied,
            "enabled_scoring_agents": list(active_domains),
        })

    # Sort: disqualified last, then by combined score descending
    results.sort(key=lambda r: (r["disqualified"], -r["combined"]))

    # Scoring health check: warn if ALL domain scores for a candidate are at the
    # 0.5 default — this indicates a silent scoring failure, not a real assessment.
    all_default_domains = set(active_domains)
    for r in results:
        if not r["disqualified"] and all(
            abs(r[d] - 0.5) < 0.01 for d in all_default_domains
        ):
            logger.error(
                "SCORING FAILURE: candidate %d has all domain scores at 0.500 default — "
                "domain agents returned no scores. Chief selection will be unreliable. "
                "Check scoring agent logs for truncated JSON or API errors.",
                r["candidate_id"],
            )

    return results


def _resolve_disqualify_ids(
    candidates: list[dict],
    scoring: dict,
    audit: dict,
) -> tuple[set[int], set[int]]:
    """A numerical audit does not overturn a specialist's substantive block."""
    audit_disqualified = {
        int(candidate_id)
        for candidate_id in (audit.get("disqualify_ids") or [])
    }
    scoring_blocked = {
        int(candidate_id)
        for candidate_id in (scoring.get("blocked_by_scoring") or [])
    }
    hard_disqualified = {int(c["id"]) for c in candidates if c.get("hard_gate_flags") or c.get("feasible") is False}
    return audit_disqualified | scoring_blocked | hard_disqualified, set()


def _select_winner_id(chief_data: dict, weighted_scores: list[dict], valid_ids: set):
    """Preserve an explicit no-winner decision; recover only absent/invalid IDs."""
    if "selected_candidate_id" in chief_data and chief_data["selected_candidate_id"] is None:
        return None
    try:
        winner = int(chief_data.get("selected_candidate_id"))
    except (TypeError, ValueError):
        winner = None
    if winner in valid_ids:
        return winner
    winner, reason = _deterministic_resolve([r for r in weighted_scores if r.get("candidate_id") in valid_ids])
    chief_data["selected_candidate_id"] = winner
    chief_data["selection_recovery_reason"] = reason
    return winner


# ═══════════════════════════════════════════════════════════════════════════════
#  DFMEA prompt (Dr. Safety, post-selection)
# ═══════════════════════════════════════════════════════════════════════════════

_DFMEA_SYSTEM = """\
You are DR. SAFETY performing a post-selection DFMEA for the winning flow chemistry
design candidate. Identify the top 3 failure modes, their severity and likelihood,
and recommend 1-3 targeted validation experiments.

## REQUIRED OUTPUT — JSON only
```json
{
  "selected_candidate_id": 3,
  "failure_modes": [
    {
      "rank": 1,
      "mode": "Pump cavitation at low Q",
      "cause": "Vapour pressure of solvent exceeds pump inlet pressure at T > 40°C",
      "severity": 8,
      "likelihood": 4,
      "RPN": 32,
      "mitigation": "Pressurise feed reservoir to 0.5 bar above vapour pressure"
    }
  ],
  "single_points_of_failure": ["BPR valve — blockage stops all flow"],
  "validation_experiments": [
    "Run blank solvent at Q=0.5 mL/min for 60 min; monitor pressure for drift > 5%"
  ]
}
```
"""


def run_dfmea(
    winner_id: int,
    winner: dict,
    chemistry_brief: str,
    safety_score_entry: Optional[dict] = None,
) -> dict:
    """Run DFMEA on the selected candidate. Returns DFMEA dict."""
    context = (
        f"## Selected candidate\n"
        f"id={winner_id} | tau={winner.get('tau_min')} min | "
        f"d={winner.get('d_mm')} mm | Q={winner.get('Q_mL_min')} mL/min | "
        f"V_R={winner.get('V_R_mL')} mL | L={winner.get('L_m')} m | "
        f"dP={winner.get('delta_P_bar', 0):.3f} bar | "
        f"Re={winner.get('Re', 0):.0f} | X={winner.get('expected_conversion', 0):.2f}\n\n"
        f"## Chemistry brief\n{chemistry_brief}\n\n"
        f"## Safety assessment from scoring phase\n"
        f"{json.dumps(safety_score_entry or {}, indent=2)[:600]}\n\n"
        "Identify top 3 failure modes. Output JSON only."
    )
    if winner.get("inventory_realized"):
        from .scoring import _slim_candidate
        context += "\n\n## Authoritative realized hardware and feeds\n" + json.dumps(_slim_candidate(winner))
        context += "\nResidence-time indices and hydraulic/conversion proxies are not measurements. Preserve the supplied hardware and identify unresolved commissioning data."
    dfmea: dict = {
        "selected_candidate_id": winner_id,
        "failure_modes": [],
        "single_points_of_failure": [],
        "validation_experiments": [],
    }
    try:
        raw = call_llm(_DFMEA_SYSTEM, context, max_tokens=1600)
        s = raw.strip()
        if "```" in s:
            for part in s.split("```")[1::2]:
                try:
                    dfmea.update(json.loads(part.lstrip("json").strip()))
                    break
                except json.JSONDecodeError:
                    continue
        else:
            try:
                dfmea.update(json.loads(s))
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.warning("DFMEA LLM call failed: %s", e)
    return dfmea


# ═══════════════════════════════════════════════════════════════════════════════
#  Chief LLM call
# ═══════════════════════════════════════════════════════════════════════════════

_CHIEF_SYSTEM_V4 = """\
You are the CHIEF ENGINEER of the FLORA ENGINE v4 council. You are the senior
process engineer who closes the stage-gated evaluation and selects the final design.
You do not argue domain science and do not invent new parameter values. You select
from audited, scored, surviving candidates — and you explain your reasoning fully.

## Your selection logic
1. Disqualified candidates (hard gates + Skeptic CRITICAL errors) are off the table.
2. You receive pre-computed weighted scores. The final combined score now includes
   domain adequacy, Process Value Score (PVS), and geometry. PVS asks whether the
   candidate has a plausible flow benefit; its proxies do not demonstrate improvement over batch.
3. Walk through the top 3 candidates explicitly: what each scored, where they
   differ, and which domain drove the difference.
4. If top two are within 5% combined: tiebreak by (a) safety score, (b) design
   fit to the researcher objective, (c) a defensible protocol-based starting exposure. Literature similarity alone does not validate kinetic transfer.
5. If surviving candidates are within 10%, select ONE defensible starting
   experiment using the researcher objective and retain alternatives in the design
   envelope. Set selected_candidate_id to its ID and selection_flag to SCREEN.
   Similar scores or unvalidated kinetics do not mean that no screen can be chosen.
   Use selected_candidate_id=null only when no available candidate can be selected
   because of an unresolved hard constraint; then state the actual constraint.
   A selected SCREEN is not an experimentally validated optimum.
6. If flow chemistry is not justified for this reaction, state explicitly:
   "Batch is currently the better engineering choice for this reaction."
7. If you need clarification on a specific domain issue before deciding, set
   `ask_expert` to the domain name ("kinetics" or "chemistry") and state your
   question in `ask_expert_question`. The council will answer and you will be
   called again with the response. Use this AT MOST ONCE.

## Conflict resolution (when domain scores conflict)
  Safety > Chemistry > Kinetics > Fluidics > Hardware (advisory only)

## Pump flowrate derivation (REQUIRED)
After selecting the winner, derive each pump's flowrate using:
  Q_i = (ṅ_limiting × eq_i) / C_feed_i
where ṅ_limiting = P_batch / (Y × 60) in mmol/min (provided in chemistry brief).
If ṅ_limiting is not provided, derive it from the limiting-component feed: Q_liquid × C_limiting. Do not multiply substrate delivery by product yield.
Show the derivation explicitly for each stream in `pump_flowrates`.

## What to write in selection_rationale
Write 5–8 sentences that:
  - Name the winner and runner-ups with their combined scores
  - Explain what drove the winner's highest domain score
  - Explain what held back the runner-up(s)
  - State how the user objective influenced the tie-breaking
  - Name the 1–2 most important remaining uncertainties a lab chemist should test
  - State what the FIRST experiment on the bench should verify

## REQUIRED OUTPUT — JSON only
```json
{
  "selected_candidate_id": 3,
  "runner_up_ids": [5, 1],
  "selection_justification": "Selected because heat transfer and productivity are improved over batch, achieving 6.2x residence-time reduction at acceptable pressure-drop and safety risk.",
  "selection_flag": "OK",
  "selection_rationale": "5-8 sentences.",
  "weighted_scores": {},
  "resolved_tradeoffs": [
    "id=3 vs id=5: id=3 has higher safety score (0.93 vs 0.78) because BPR margin is 2.1 bar vs 0.4 bar."
  ],
  "overridden_concerns": [
    {"agent": "DR. FLUIDICS", "concern": "r_mix=0.12 slightly elevated", "override_reason": "Da_mass=0.4 < 1.0 so dual criterion not triggered"}
  ],
  "remaining_uncertainties": [
    "k estimated from batch; flow enhancement factor of 10× is class-typical but not experimentally verified"
  ],
  "requires_new_experiment": false,
  "experiment_recommendation": "Run blank solvent at design Q for 30 min, then switch to reaction mixture. Sample at t=3τ and t=5τ.",
  "design_envelope": {
    "tau_range_min": [2, 5], "Q_range_mL_min": [0.8, 1.2],
    "d_range_mm": [0.75, 1.0], "T_range_C": [20, 30], "BPR_range_bar": [0, 2]
  },
  "final_consensus": {
    "tau_min": 3.0, "Q_mL_min": 1.0, "d_mm": 1.0,
    "V_R_mL": 3.0, "L_m": 3.82, "BPR_bar": 0.0,
    "material": "FEP", "X_estimated": 0.94, "productivity_mg_h": 250.0
  },
  "pump_flowrates": [
    {
      "stream_label": "A",
      "role": "substrate",
      "Q_mL_min": 0.500,
      "derivation": "ṅ_lim=0.333 mmol/min × eq=1.0 / C_feed=0.667 M = 0.500 mL/min",
      "C_feed_M": 0.667,
      "molar_equiv": 1.0
    },
    {
      "stream_label": "B",
      "role": "reagent",
      "Q_mL_min": 0.500,
      "derivation": "ṅ_lim=0.333 mmol/min × eq=1.0 / C_feed=0.667 M = 0.500 mL/min",
      "C_feed_M": 0.667,
      "molar_equiv": 1.0
    }
  ],
  "n_limiting_mmol_min": 0.333,
  "Q_total_mL_min": 1.000,
  "ask_expert": null,
  "ask_expert_question": null
}
```

## Multi-stage guidance (only if n_stages > 1)
For multi-stage processes, each stage has its OWN independently derived τ from its
own chemistry class and batch time. τ per stage is NEVER a fraction of a shared total.
Provide `stage_d_mm` only — one diameter per stage.
Do NOT provide `stage_tau_fractions` — residence times come from the design calculator.

Tool available: `compute_design_envelope` — call on 2–3 top candidates before deciding.
Show your envelope comparison in resolved_tradeoffs.
"""


def _parse_chief(raw: str) -> dict:
    s = raw.strip()
    if "```" in s:
        for part in s.split("```")[1::2]:
            cleaned = part.lstrip("json").strip()
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return _normalise_chief_selection(parsed)
            except json.JSONDecodeError:
                continue
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return _normalise_chief_selection(parsed)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(s[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(s[start:i + 1])
                        if isinstance(parsed, dict):
                            return _normalise_chief_selection(parsed)
                    except json.JSONDecodeError:
                        break
    # The selected ID is emitted first by the Chief contract. If only the tail
    # was truncated, retain that explicit choice and let downstream candidate
    # validation plus deterministic rationale completion enforce the contract.
    match = re.search(
        r'"(?:selected_candidate_id|selectedCandidateId|winner_id|selected_id)"\s*:\s*(null|"?\d+"?)',
        s,
    )
    if match:
        return {
            "selected_candidate_id": None if match.group(1) == "null" else int(match.group(1).strip('"')),
            "response_incomplete": True,
        }
    return {}


def _normalise_chief_selection(data: dict) -> dict:
    """Normalize equivalent provider field names without changing the decision."""
    normalized = dict(data)
    nested = normalized.get("selection")
    if isinstance(nested, dict):
        for key, value in nested.items():
            normalized.setdefault(key, value)
    if "selected_candidate_id" not in normalized:
        for alias in ("selectedCandidateId", "winner_id", "selected_id"):
            if normalized.get(alias) is not None:
                normalized["selected_candidate_id"] = normalized[alias]
                break
    return normalized


def _run_chief_llm(
    candidates: list[dict],
    weighted_scores: list[dict],
    scoring: dict,
    audit: dict,
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    disqualify_ids: set[int],
    n_limiting_mmol_min: Optional[float] = None,
    P_batch_mmol_h: Optional[float] = None,
    expert_answer: Optional[str] = None,
) -> tuple[dict, list[dict]]:
    surviving = [r for r in weighted_scores if not r["disqualified"]]

    score_table = (
        "| id | combined | PVS | domain | chemistry | kinetics | fluidics | safety | geometry |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in surviving:
        score_table += (
            f"| {r['candidate_id']} | {r['combined']:.3f} | "
            f"{r.get('PVS', 0.0):.3f} | {r.get('legacy_domain_combined', 0.0):.3f} | "
            f"{r['chemistry']:.3f} | {r['kinetics']:.3f} | "
            f"{r['fluidics']:.3f} | {r['safety']:.3f} | {r['geometry']:.3f} |\n"
        )

    molar_flow_block = ""
    if n_limiting_mmol_min is not None:
        molar_flow_block = (
            f"\n## Molar flow (for pump flowrate derivation)\n"
            f"ṅ_limiting = {n_limiting_mmol_min:.4f} mmol/min\n"
            f"P_batch = {P_batch_mmol_h:.2f} mmol/h\n"
            f"Use Q_i = (ṅ_limiting × eq_i) / C_feed_i for each pump.\n"
        )
    if any(c.get("inventory_realized") for c in candidates):
        from flora_translate.engine.council_v4.scoring import _slim_candidate
        molar_flow_block = (
            "\n## Inventory-realized candidates\n"
            + json.dumps([_slim_candidate(c) for c in candidates], ensure_ascii=False)
            + "\nThe candidate feed rates, concentrations and gas equivalents are jointly solved on the declared hardware. "
            "Use these rates to derive molar delivery; do not replace them with a batch-productivity heuristic. "
            "Generic design-envelope tools do not override fixed reactor geometry. "
            "If no candidate is acceptable, return selected_candidate_id=null with the unresolved constraints.\n"
        )

    expert_block = ""
    if expert_answer:
        expert_block = f"\n## Expert answer to your question\n{expert_answer}\n"

    context = (
        f"## User objectives\n{objectives or 'balanced'}\n\n"
        f"## Chemistry brief\n{chemistry_brief}\n\n"
        f"## Candidate table\n{table_markdown}\n\n"
        f"## Weighted scores (objective-adjusted)\n{score_table}\n"
        f"## Skeptic audit summary\n{audit.get('audit_summary', '')}\n\n"
        f"## Skeptic disqualify recommendations\n"
        + "\n".join(
            f"  - id {r['candidate_id']}: {r['reason']}"
            for r in audit.get("disqualify_recommendations", [])
        )
        + molar_flow_block
        + expert_block
        + "\n\nSelect the winner and derive pump flowrates. Include a concrete selection_justification tied to the researcher objective. Distinguish a proposed flow advantage from demonstrated performance. A chosen first screen must have its selected_candidate_id, even when alternatives are equally plausible. Output JSON only."
    )

    decision, tc = {}, []
    try:
        raw, tc = call_llm_with_tools(
            _CHIEF_SYSTEM_V4 + REVIEW_POLICY, context,
            tools=CHIEF_TOOLS,
            tool_executor=execute_tool,
            max_tokens=_CHIEF_MAX_TOKENS,
            max_tool_turns=4,
        )
        decision = _parse_chief(raw)
        screen_id = decision.get("first_screen_candidate_id")
        eligible = {c["id"] for c in candidates if c["id"] not in disqualify_ids}
        if ("selected_candidate_id" in decision and decision["selected_candidate_id"] is None
                and screen_id in eligible and str(decision.get("selection_flag", "")).upper() == "SCREEN"):
            # Resolve contradictory output once. This never turns a rejection
            # into acceptance without a new explicit model decision.
            clarification = (
                context + "\n\nYour previous decision was internally ambiguous:\n" + raw
                + "\nClarify the decision once. selected_candidate_id=null means no design may be "
                "delivered, but first_screen_candidate_id names a screen. If you choose that "
                "candidate for a first screen, put its ID in selected_candidate_id and keep "
                "selection_flag=SCREEN. You may reject all candidates by keeping null and "
                "stating the unresolved hard constraint. Do not claim experimental validation "
                "or change any candidate. Return one complete decision JSON."
            )
            revised_raw, revised_calls = call_llm_with_tools(
                _CHIEF_SYSTEM_V4 + REVIEW_POLICY, clarification, tools=CHIEF_TOOLS,
                tool_executor=execute_tool, max_tokens=_CHIEF_MAX_TOKENS, max_tool_turns=1)
            decision = _parse_chief(revised_raw)
            if decision.get("selected_candidate_id") not in eligible:
                decision["selected_candidate_id"] = None
            tc.extend(revised_calls)
        return decision, tc
    except Exception as e:
        logger.warning("Chief v4 LLM call failed: %s", e)
        return decision, tc


def _run_expert_askback(
    domain: str,
    question: str,
    chemistry_brief: str,
    candidates: list[dict],
    scoring: dict,
) -> str:
    """Run a targeted single-question query to a domain expert for the Chief."""
    _EXPERT_SYSTEM = (
        f"You are DR. {domain.upper()} in the FLORA council. "
        f"The Chief Engineer has a focused question for you. "
        f"Answer concisely (3–5 sentences) with specific numbers where relevant. "
        f"Do not repeat the question. Answer directly."
    )
    # Build a brief candidate summary
    top_ids = [c.get('id') for c in candidates[:3]]
    scores_subset = [
        e for e in scoring.get(f"{domain.lower()}_scores", [])
        if e.get("candidate_id") in top_ids
    ]
    context = (
        f"## Chemistry brief\n{chemistry_brief}\n\n"
        f"## Chief Engineer's question\n{question}\n\n"
        f"## Relevant domain scores (top candidates)\n"
        + "\n".join(
            f"- id={e.get('candidate_id')}: score={e.get(f'{domain.lower()}_score', '?')}, "
            f"reasoning={str(e.get('reasoning', ''))[:200]}"
            for e in scores_subset
        )
        + "\n\nAnswer the question directly."
    )
    try:
        answer = call_llm(_EXPERT_SYSTEM, context, max_tokens=800)
        logger.info("  Chief ask-back (%s) answered: %s", domain, answer[:100])
        return answer.strip()
    except Exception as e:
        logger.warning("Expert ask-back failed: %s", e)
        return f"Expert answer unavailable ({e})"


def _deterministic_resolve(weighted_scores: list[dict]) -> tuple[Optional[int], str]:
    """Deterministic fallback: pick highest combined non-disqualified score."""
    survivors = [r for r in weighted_scores if not r["disqualified"]]
    if not survivors:
        return None, "No surviving candidates after disqualification."
    best = survivors[0]  # already sorted descending by combined
    return best["candidate_id"], (
        f"Deterministic selection: id={best['candidate_id']} "
        f"(combined={best['combined']:.3f}, safety={best['safety']:.3f})"
    )


def _ensure_selection_justification(
    chief_data: dict,
    *,
    winner_id: Optional[int],
    winner: Optional[dict],
    weighted_scores: list[dict],
    intensification_mandate: dict,
) -> None:
    if winner_id is None or not winner:
        return
    text = str(chief_data.get("selection_justification") or chief_data.get("selection_rationale") or "").strip()
    weak = not text or text.lower() in {
        "selected because it scored highest overall.",
        "selected because it is the best candidate.",
    }
    if weak:
        text = (
            f"Candidate {winner_id} was selected for an initial experimental screen. "
            "The reviewed operating point is recorded below; its yield and "
            "performance relative to batch require experimental measurement."
        )
    chief_data["selection_justification"] = text
    # A starting design need not demonstrate an experimental advantage before
    # it can be tested. Do not infer approval or an objective failure from a
    # keyword in the rationale, or promote a SCREEN decision to OK.
    chief_data.setdefault("selection_flag", "SCREEN")


def _refresh_selection_justification_from_winner(
    chief_data: dict,
    *,
    winner_id: int,
    winner: dict,
    weighted_scores: list[dict],
    intensification_mandate: dict,
) -> None:
    _ensure_selection_justification(
        chief_data, winner_id=winner_id, winner=winner,
        weighted_scores=weighted_scores, intensification_mandate=intensification_mandate,
    )
    # The caller has obtained renewed approval of a changed operating point.
    # Preserve that review and its flag; append arithmetic, not a fabricated
    # chemical or safety justification based on a heuristic score.
    batch_time = _positive_float(winner.get("batch_time_min"))
    tau = _positive_float(winner.get("tau_min"))
    time_note = f"nominal exposure={tau:.2f} min"
    if batch_time > 0:
        time_note += f", recorded batch duration={batch_time:.2f} min"
    chief_data["selection_justification"] += (
        f" Reviewed operating point: {time_note}, "
        f"d={_positive_float(winner.get('d_mm')):.2f} mm, "
        f"Q={_positive_float(winner.get('Q_mL_min')):.3f} mL/min, "
        f"L={_positive_float(winner.get('L_m')):.2f} m. "
        "These design quantities do not establish conversion, yield or an experimental advantage."
    )


def _intensification_feasibility_precheck(
    *,
    batch_time_min: float,
    tau_kinetics_min: float,
    intensification_mandate: dict,
    translation_policy: str,
    calc: Optional[DesignCalculations] = None,
    candidate_tau_min: float | None = None,
) -> Optional[dict]:
    if (translation_policy or "").lower() != "intensify":
        return None
    if batch_time_min <= 0 or tau_kinetics_min <= 0:
        return None
    target = max(float(intensification_mandate.get("tau_reduction_target") or 2.0), 1.0)
    tau_ceiling = batch_time_min / target
    if tau_kinetics_min <= tau_ceiling:
        return None
    required_to_ceiling_ratio = (tau_kinetics_min / tau_ceiling) if tau_ceiling > 0 else float("inf")
    marginal_boundary_conflict = required_to_ceiling_ratio <= 1.10
    candidate_tau = _positive_float(candidate_tau_min)
    candidate_conversion = None
    candidate_under_ceiling = candidate_tau > 0 and candidate_tau <= tau_ceiling
    if candidate_tau > 0:
        try:
            candidate_conversion = 1.0 - math.exp(-2.303 * candidate_tau / max(tau_kinetics_min, 1e-9))
        except OverflowError:
            candidate_conversion = 0.0
    candidate_worth_council = (
        candidate_under_ceiling
        and candidate_conversion is not None
        and candidate_conversion >= 0.30
    )
    kinetic_uncertain = False
    uncertainty_reasons: list[str] = []
    step2_values: dict = {}
    if calc is not None:
        for step in getattr(calc, "steps", []) or []:
            if getattr(step, "step", None) == 2:
                step2_values = dict(getattr(step, "values", {}) or {})
                break
        raw_ifs = step2_values.get("raw_analogy_IFs") or []
        corrected_ifs = step2_values.get("analogy_IFs") or []
        if step2_values.get("IF_floor_applied") or any(_positive_float(v) < 1.0 for v in raw_ifs):
            kinetic_uncertain = True
            uncertainty_reasons.append("sub-unity analogy IF values were floored to 1.0")
        if (
            _positive_float(step2_values.get("IF_analogy")) <= 1.0
            and _positive_float(step2_values.get("IF_class")) >= 3.0
        ):
            kinetic_uncertain = True
            uncertainty_reasons.append("analogy IF is batch-equivalent while class IF predicts intensification")
        if raw_ifs or corrected_ifs:
            uncertainty_reasons.append(
                f"raw_analogy_IFs={raw_ifs}; corrected_analogy_IFs={corrected_ifs}"
            )
    if marginal_boundary_conflict:
        kinetic_uncertain = True
        uncertainty_reasons.append(
            "tau_kinetics is within 10% of the intensification ceiling; treating as screen-required boundary case"
        )
    if candidate_worth_council:
        kinetic_uncertain = True
        uncertainty_reasons.append(
            f"design-space candidate tau={candidate_tau:.1f} min is under the intensification ceiling "
            f"with projected X≈{candidate_conversion:.2f}; delegating kinetics risk to council"
        )
    status = (
        "KINETIC_ANCHOR_UNCERTAIN_SCREEN_REQUIRED"
        if kinetic_uncertain
        else "INFEASIBLE_WITH_CURRENT_KINETIC_ANCHOR"
    )
    diagnosis_prefix = (
        "Current kinetic anchor is uncertain, not a hard infeasibility: "
        if kinetic_uncertain
        else ""
    )
    # Build the diagnosis. When a design-space candidate exists under the
    # mandate ceiling, the analogy-derived tau_kinetics is no longer the
    # authoritative kinetic statement — surface the candidate-supported
    # tau as the operational reference so the council message doesn't
    # mislead downstream agents (and the user reading logs).
    if candidate_worth_council:
        diagnosis = (
            f"{diagnosis_prefix}analogy kinetic anchor τ={tau_kinetics_min:.1f} min exceeds "
            f"the intensification ceiling τ≤{tau_ceiling:.1f} min "
            f"({target:.1f}× faster than {batch_time_min:.1f} min batch), but the upstream "
            f"design-space search already found a candidate at τ={candidate_tau:.1f} min "
            f"with projected X≈{candidate_conversion:.2f} — proceed and let the council "
            f"evaluate it as a screening hypothesis."
        )
    else:
        diagnosis = (
            f"{diagnosis_prefix}current kinetics require tau={tau_kinetics_min:.1f} min, but the "
            f"intensification mandate limits tau to <= {tau_ceiling:.1f} min "
            f"({target:.1f}x faster than {batch_time_min:.1f} min batch)."
        )
    return {
        "status": status,
        "hard_block": not kinetic_uncertain,
        "kinetic_anchor_quality": "UNCERTAIN" if kinetic_uncertain else "CONSTRAINING",
        "uncertainty_reasons": uncertainty_reasons,
        "batch_time_min": round(batch_time_min, 3),
        "tau_kinetics_min": round(tau_kinetics_min, 3),
        "tau_reduction_target": round(target, 3),
        "tau_intensification_ceiling_min": round(tau_ceiling, 3),
        "required_to_ceiling_ratio": round(required_to_ceiling_ratio, 3) if tau_ceiling > 0 else None,
        "candidate_tau_min": round(candidate_tau, 3) if candidate_tau > 0 else None,
        "candidate_projected_conversion": round(candidate_conversion, 3) if candidate_conversion is not None else None,
        "candidate_under_ceiling": bool(candidate_under_ceiling),
        "candidate_worth_council": bool(candidate_worth_council),
        "minimum_flow_advantage": intensification_mandate.get("minimum_flow_advantage", "productivity"),
        "diagnosis": diagnosis,
        "recommended_next_steps": [
            (
                "Council proceeds with the design-space candidate; outcome is a SCREEN_REQUIRED hypothesis."
                if candidate_worth_council
                else "Generate intensified screen candidates, but do not mark the selected point as engine-validated without experimental support."
                if kinetic_uncertain
                else "Do not select a normal flow candidate from this kinetic anchor."
            ),
            "Run a small flow screen to measure conversion at sub-batch tau values.",
            "Test intensification levers: higher temperature, lower viscosity/dilution strategy, smaller ID or enhanced mixing, and staged/recycle operation if chemically justified.",
            "Revisit analogy/kinetic anchor if it came from weak local-model reasoning rather than measured flow precedent.",
        ],
    }


def _positive_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _float_or_default(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tube_volume_mL(length_m: float, d_mm: float) -> float:
    area_mm2 = math.pi * (d_mm / 2.0) ** 2
    # 1 mL = 1000 mm3 and 1 m = 1000 mm, so mL = area_mm2 * length_m.
    return max(area_mm2 * length_m, 0.0)


def _build_screen_required_payload(
    *,
    current: FlowProposal,
    calc: DesignCalculations,
    batch_time_min: float,
    intensification_mandate: dict,
    solvent: str,
    reason: str,
    feasibility_diagnostic: Optional[dict] = None,
) -> dict:
    """Build a small experimental screen when no validated flow design is defensible.

    These are not final designs. They are bounded experiments to measure whether
    the reaction can actually be intensified at sub-batch residence times.
    """
    target = max(_positive_float(intensification_mandate.get("tau_reduction_target"), 2.0), 1.0)
    if feasibility_diagnostic:
        tau_ceiling = _positive_float(feasibility_diagnostic.get("tau_intensification_ceiling_min"))
    else:
        tau_ceiling = batch_time_min / target if batch_time_min > 0 else 0.0
    if tau_ceiling <= 0:
        tau_ceiling = max(_positive_float(current.residence_time_min, 5.0) * 0.5, 1.0)
    if batch_time_min > 0:
        tau_ceiling = min(tau_ceiling, max(batch_time_min * cfg.BATCH_PROXIMITY_THRESHOLD * 0.95, 0.5))
    tau_floor = max(0.5, tau_ceiling / 3.0)
    tau_mid = (tau_floor + tau_ceiling) / 2.0

    base_temp = _positive_float(current.temperature_C, 25.0)
    concentration = _positive_float(current.concentration_M) or _positive_float(calc.concentration_M, 0.1)
    bpr = _positive_float(current.BPR_bar)
    tubing_material = current.tubing_material or "FEP"
    primary_advantage = str(intensification_mandate.get("minimum_flow_advantage") or "productivity")

    templates = [
        ("S1", "target-ceiling small-ID baseline", tau_ceiling, 0.75, 10.0, base_temp),
        ("S2", "mid-tau small-ID intensified", tau_mid, 0.75, 10.0, base_temp + 10.0),
        ("S3", "short-tau high surface-area challenge", tau_floor, 0.50, 8.0, base_temp + 20.0),
        ("S4", "target-ceiling mid-ID practical comparator", tau_ceiling, 1.00, 10.0, base_temp + 10.0),
        ("S5", "large-ID dispersion/control comparator", tau_ceiling, 1.60, 8.0, base_temp),
    ]

    candidates: list[dict] = []
    for idx, (label, purpose, tau, d_mm, length, temp) in enumerate(templates, start=1):
        tau = max(float(tau), 0.1)
        volume = _tube_volume_mL(length, d_mm)
        q = volume / tau
        candidates.append({
            "id": idx,
            "screen_id": label,
            "purpose": purpose,
            "tau_min": round(tau, 3),
            "batch_time_min": round(batch_time_min, 3) if batch_time_min > 0 else None,
            "target_reduction_factor": round(target, 3),
            "Q_mL_min": round(q, 4),
            "V_R_mL": round(volume, 3),
            "d_mm": round(d_mm, 3),
            "L_m": round(length, 3),
            "temperature_C": round(temp, 1),
            "concentration_M": round(concentration, 4),
            "BPR_bar": round(bpr, 2),
            "tubing_material": tubing_material,
            "primary_flow_advantage": primary_advantage,
            "status": "SCREEN_ONLY_NOT_ENGINE_VALIDATED",
            "rationale": (
                f"Screen point for testing {primary_advantage} at tau below batch; "
                "advance only if measured conversion/yield supports this residence time."
            ),
        })

    attach_flow_sense_reports(
        candidates,
        batch_time_min=batch_time_min,
        batch_concentration_M=getattr(calc, "concentration_M", None),
        solvent_name=solvent,
        intensification_mandate=intensification_mandate,
    )

    return {
        "screen_required": True,
        "screen_reason": reason,
        "screen_candidates": candidates,
        "screen_acceptance_criteria": [
            "Measure conversion, isolated/assay yield, impurity profile, pressure drop, and visual clogging for every screen point.",
            "Do not promote a point to final design unless tau_flow is below the batch-proximity threshold and measured conversion/yield meets the project target.",
            "If all screen points underperform, report the reaction as not currently flow-intensifiable under the available kinetic anchor and require manual chemistry review.",
        ],
        "feasibility_diagnostic": feasibility_diagnostic or {},
    }


def _downgrade_uncertain_kinetic_hard_gates(candidates: list[dict]) -> None:
    """Keep repaired-kinetic flags visible without disqualifying screen candidates."""
    uncertain_patterns = (
        "X=",
        "X_minimum",
        "insufficient conversion",
        "dual-criterion mixing fail",
        "Da_mass",
    )
    for candidate in candidates or []:
        flags = [str(flag) for flag in (candidate.get("hard_gate_flags") or [])]
        uncertain_flags = [
            flag for flag in flags
            if any(pattern in flag for pattern in uncertain_patterns)
        ]
        remaining = [flag for flag in flags if flag not in uncertain_flags]
        if uncertain_flags:
            existing = [str(flag) for flag in (candidate.get("screen_uncertain_flags") or [])]
            candidate["screen_uncertain_flags"] = existing + uncertain_flags
            candidate["hard_gate_flags"] = remaining
            candidate["hard_gate_status"] = (
                "PASS_WITH_SCREEN_WARNINGS" if not remaining else "FLAGGED: " + "; ".join(remaining)
            )
            candidate["kinetic_anchor_uncertain"] = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-selection expert refinement loop
# ═══════════════════════════════════════════════════════════════════════════════

_COMMERCIAL_D_MM = [0.50, 0.75, 1.00, 1.60, 2.00, 3.00]
_FIELD_OWNER = {
    "component_placements": "chemistry",
    "tau_min": "kinetics",
    "d_mm": "fluidics",
    "BPR_bar": "safety",
    "tubing_material": "safety",
    "concentration_M": "chemistry",
}
_AGENT_TO_DOMAIN = {
    "DR. CHEMISTRY": "chemistry",
    "DR. KINETICS": "kinetics",
    "DR. FLUIDICS": "fluidics",
    "DR. SAFETY": "safety",
}
_DOMAIN_PRIORITY = {
    "kinetics": 0,
    "fluidics": 1,
    "safety": 2,
    "chemistry": 3,
}


def _round_down_commercial_d(value_mm: float) -> float:
    eligible = [d for d in _COMMERCIAL_D_MM if d <= value_mm + 1e-9]
    return eligible[-1] if eligible else _COMMERCIAL_D_MM[0]


def _build_domain_blocklist(audit: dict) -> dict[int, set[str]]:
    blocked: dict[int, set[str]] = defaultdict(set)
    for err in audit.get("all_errors", []):
        if err.get("severity") not in ("CRITICAL", "HIGH"):
            continue
        cid = err.get("candidate_id")
        domain = _AGENT_TO_DOMAIN.get(str(err.get("agent", "")).upper())
        if cid is not None and domain:
            blocked[int(cid)].add(domain)
    return blocked


def _entry_by_candidate(entries: list[dict], candidate_id: int) -> Optional[dict]:
    return next((e for e in entries if e.get("candidate_id") == candidate_id), None)


def _extract_explicit_patch(
    entry: dict,
    domain: str,
    *,
    candidate: Optional[dict] = None,
    batch_concentration_M: Optional[float] = None,
) -> tuple[dict, dict]:
    """Extract field patches the domain agent explicitly emitted.

    Applies Fix #1 (lock C in Refinement Board): concentration_M patches are
    REJECTED unless they meet at least one of:
      - the patch increases C (we never block dilution-reverse moves);
      - the new C is >= 50% of the batch concentration;
      - the agent flagged an explicit thermal/precipitation/inner-filter reason
        in `concentration_change_reason` or in `concerns`.
    This prevents the historical failure mode where Dr. Chemistry would
    silently drop C from 0.5 M → 0.044 M for every candidate, killing
    productivity and creating a manufactured low-C regime nobody asked for.
    """
    patch: dict = {}
    rationale: dict = {}
    raw = entry.get("proposed_changes")
    if not isinstance(raw, dict):
        return patch, rationale

    batch_c = float(batch_concentration_M) if batch_concentration_M else None
    current_c = None
    if candidate is not None:
        try:
            current_c = float(candidate.get("concentration_M", 0.0) or 0.0)
        except (TypeError, ValueError):
            current_c = None
    if current_c is None and batch_c is not None:
        current_c = batch_c

    for field, value in raw.items():
        if _FIELD_OWNER.get(field) != domain:
            continue
        if field == "component_placements":
            if candidate and candidate.get("inventory_realized") and isinstance(value, list) and value:
                patch[field] = value
                rationale[field] = "Explicit structural repair; validate against chemistry and inventory, then re-review."
            continue
        if field == "d_mm":
            try:
                proposed_d = float(value)
            except (TypeError, ValueError):
                continue
            if proposed_d < _COMMERCIAL_D_MM[0] or proposed_d > _COMMERCIAL_D_MM[-1]:
                rationale["d_mm_rejected"] = (
                    f"Proposed d={proposed_d:.3f} mm is outside the supported "
                    f"commercial range {_COMMERCIAL_D_MM[0]:.2f}-"
                    f"{_COMMERCIAL_D_MM[-1]:.2f} mm."
                )
                continue
            commercial_d = _round_down_commercial_d(proposed_d)
            patch[field] = commercial_d
            rationale[field] = (
                f"Explicit diameter patch normalized to commercial size "
                f"{commercial_d:.2f} mm."
            )
            continue
        if field == "concentration_M":
            try:
                new_c = float(value)
            except (TypeError, ValueError):
                continue
            if new_c <= 0:
                continue
            # Always allow if it's a no-op or an increase
            if current_c is not None and new_c >= current_c - 1e-9:
                patch[field] = new_c
                rationale[field] = "Explicit C patch (non-reducing)."
                continue
            # Reduction: require justification
            ratio = (new_c / batch_c) if batch_c else 1.0
            justified = False
            justify_reasons = []
            reasons_blob = " ".join(str(x).lower() for x in (entry.get("concerns") or []))
            change_reason = str(entry.get("concentration_change_reason") or "").lower()
            if change_reason:
                justify_reasons.append(change_reason)
            try:
                A = float(entry.get("beer_lambert_A") or 0.0)
            except (TypeError, ValueError):
                A = 0.0
            if A > 1.5:
                justified = True
                justify_reasons.append(f"inner_filter A={A:.2f}>1.5")
            if any(k in change_reason or k in reasons_blob for k in (
                "thermal", "exotherm", "runaway", "heat removal",
                "precipita", "solubility", "saturation_limit",
                "inner filter", "beer-lambert", "photon",
            )):
                justified = True
            # Always allow if reduction is <= 50% (modest dilution)
            if batch_c and ratio >= 0.5:
                justified = True
                justify_reasons.append(f"C={new_c:.3f}M is ≥50% of batch C={batch_c:.3f}M")
            if justified:
                patch[field] = new_c
                rationale[field] = (
                    "Explicit C reduction allowed: " + "; ".join(justify_reasons or ["modest"])
                )
            # else: silently dropped — Refinement Board cannot collapse C
            continue
        patch[field] = value
        rationale[field] = "Explicit patch proposed by domain scorer."
    return patch, rationale


def _derive_domain_patch(
    *,
    domain: str,
    entry: dict,
    candidate: dict,
    concentration_M: float,
    allow_warning_refinement: bool = False,
    strong_revision_mode: bool = False,
    batch_concentration_M: Optional[float] = None,
    translation_policy: str = "evidence_first",
) -> tuple[dict, dict]:
    verdict = str(entry.get("verdict", "ACCEPT")).upper()
    # concentration_M source-of-truth for the gating check below: prefer the
    # explicit batch concentration when provided (this is the protocol's
    # ground truth); fall back to the in-flight reactor concentration only
    # when no batch C is available.
    batch_c_for_gate = float(batch_concentration_M) if batch_concentration_M else float(concentration_M)
    patch, rationale = _extract_explicit_patch(
        entry, domain,
        candidate=candidate, batch_concentration_M=batch_c_for_gate,
    )
    active_verdicts = {"REVISE", "BLOCK"}
    if allow_warning_refinement:
        active_verdicts.add("WARNING")
    if strong_revision_mode:
        active_verdicts.update({"WARNING", "ACCEPT"})
    if verdict not in active_verdicts:
        return patch, rationale

    if domain == "kinetics" and "tau_min" not in patch:
        tau_target = entry.get("tau_proposed_final_min") or entry.get("tau_mixing_required_min")
        try:
            tau_target_f = float(tau_target)
        except (TypeError, ValueError):
            tau_target_f = 0.0
        if tau_target_f > float(candidate.get("tau_min", 0.0)) * 1.02:
            patch["tau_min"] = round(tau_target_f, 2)
            rationale["tau_min"] = (
                f"Dr. Kinetics {verdict}: raise tau to the explicit final target "
                f"{tau_target_f:.2f} min."
            )

    if domain == "fluidics" and "d_mm" not in patch:
        direction = str(entry.get("d_change_direction", "none")).lower()
        d_fix = entry.get("d_fix_mm")
        try:
            d_fix_f = float(d_fix)
        except (TypeError, ValueError):
            d_fix_f = 0.0
        current_d = float(candidate.get("d_mm", 0.0))
        if direction == "decrease" and d_fix_f > 0:
            d_rounded = _round_down_commercial_d(d_fix_f)
            if d_rounded < current_d - 1e-6:
                patch["d_mm"] = d_rounded
                rationale["d_mm"] = (
                    f"Dr. Fluidics {verdict}: decrease d from {current_d:.2f} mm "
                    f"to commercial size {d_rounded:.2f} mm from d_fix={d_fix_f:.2f} mm."
                )

    if domain == "safety":
        if "BPR_bar" not in patch and entry.get("BPR_adequate") is False:
            required = entry.get("BPR_required_bar")
            try:
                required_f = float(required)
            except (TypeError, ValueError):
                required_f = 0.0
            system_type = str(entry.get("system_type", "")).lower()
            if required_f > 0:
                bpr = required_f + 0.5
                if "gas" in system_type:
                    bpr = max(5.0, bpr)
                current_bpr = float(candidate.get("BPR_bar", 0.0))
                if abs(bpr - current_bpr) > 0.05:
                    patch["BPR_bar"] = round(bpr, 2)
                    rationale["BPR_bar"] = (
                        f"Dr. Safety {verdict}: current BPR inadequate; set to "
                        f"{bpr:.2f} bar from required={required_f:.2f} bar."
                    )
        if "tubing_material" not in patch:
            material = entry.get("material_recommendation")
            if material:
                current_mat = str(candidate.get("tubing_material", "FEP"))
                if str(material).upper() != current_mat.upper():
                    patch["tubing_material"] = str(material)
                    rationale["tubing_material"] = (
                        f"Dr. Safety {verdict}: switch tubing material from "
                        f"{current_mat} to {material}."
                    )

    if domain == "chemistry" and "concentration_M" not in patch:
        # Auto-derived C-drop: only fires for SEVERE inner-filter (A > 1.5),
        # not the previous trigger of A > 1.0. The earlier threshold caused
        # routine C drops to 0.20 M even when the absorbance was modest,
        # contributing to the over-dilution cascade.
        try:
            A = float(entry.get("beer_lambert_A"))
        except (TypeError, ValueError):
            A = 0.0
        current_c = float(candidate.get("concentration_M", concentration_M) or concentration_M)
        if A > 1.5 and current_c > 0.20:
            # Don't drop below 50% of batch C unless A is extreme (>3.0)
            target_c = max(0.20, 0.5 * batch_c_for_gate) if A <= 3.0 else 0.20
            target_c = min(target_c, current_c)
            if target_c < current_c - 1e-6:
                patch["concentration_M"] = round(target_c, 3)
                rationale["concentration_M"] = (
                    f"Dr. Chemistry {verdict}: severe inner-filter A={A:.2f}>1.5; "
                    f"C reduced from {current_c:.2f} → {target_c:.2f} M "
                    f"(floor = 50% of batch unless A > 3.0)."
                )

    if domain == "kinetics" and "tau_min" in patch:
        try:
            proposed_tau = float(patch["tau_min"])
        except (TypeError, ValueError):
            proposed_tau = 0.0
        report = candidate.get("flow_sense_report") or {}
        batch_time = float(candidate.get("batch_time_min") or report.get("batch_time_min") or 0.0)
        target = float(report.get("target_reduction_factor") or 1.0)
        ceiling = batch_time / target if batch_time > 0 and target > 1.0 else batch_time
        if (
            (translation_policy or "").lower() == "intensify"
            and ceiling > 0
            and proposed_tau > ceiling
        ):
            patch.pop("tau_min", None)
            rationale["tau_min_rejected"] = (
                f"Dr. Kinetics proposed tau={proposed_tau:.2f} min, but the "
                f"intensification ceiling is {ceiling:.2f} min. Candidate should "
                "be penalized/disqualified rather than de-intensified."
            )

    return patch, rationale


def _tag_pareto_front(candidates: list[dict]) -> None:
    objectives = [
        ("productivity_mg_h", "max"),
        ("L_m", "min"),
        ("r_mix", "min"),
    ]
    for a in candidates:
        dominated = False
        for b in candidates:
            if a is b:
                continue
            strictly_better = False
            for key, direction in objectives:
                av = a.get(key, 0.0)
                bv = b.get(key, 0.0)
                if direction == "max":
                    if bv < av:
                        strictly_better = False
                        break
                    if bv > av:
                        strictly_better = True
                else:
                    if bv > av:
                        strictly_better = False
                        break
                    if bv < av:
                        strictly_better = True
            if strictly_better:
                dominated = True
                break
        a["pareto_front"] = not dominated


def _candidate_revision_variants(
    *,
    candidate: dict,
    cid: int,
    domain_patch_map: dict[str, tuple[dict, dict]],
    strong_revision_mode: bool,
    max_descendants_per_candidate: int,
) -> list[dict]:
    variants: list[dict] = []

    combined_patch: dict = {}
    combined_rationale: dict = {}
    combined_domains: list[str] = []
    for domain in ("chemistry", "kinetics", "fluidics", "safety"):
        patch, rationale = domain_patch_map.get(domain, ({}, {}))
        if not patch:
            continue
        combined_patch.update(patch)
        combined_rationale.update(rationale)
        combined_domains.append(domain)

    if combined_patch:
        variants.append({
            "variant_key": f"{cid}_merged",
            "patch": combined_patch,
            "rationale": combined_rationale,
            "domains": combined_domains,
            "mode": "merged",
        })

    if strong_revision_mode:
        ranked_domains = sorted(
            (
                domain for domain, (patch, _) in domain_patch_map.items()
                if patch
            ),
            key=lambda domain: (_DOMAIN_PRIORITY.get(domain, 99), domain),
        )
        for domain in ranked_domains:
            patch, rationale = domain_patch_map[domain]
            variants.append({
                "variant_key": f"{cid}_{domain}",
                "patch": patch,
                "rationale": rationale,
                "domains": [domain],
                "mode": "domain_focus",
            })

    seen: set[str] = set()
    deduped: list[dict] = []
    for variant in variants:
        key = json.dumps(variant["patch"], sort_keys=True, ensure_ascii=False)
        if not key or key == "{}" or key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
        if len(deduped) >= max_descendants_per_candidate:
            break
    return deduped


def _materialize_revised_candidate(
    *,
    candidate: dict,
    patch: dict,
    rationale: dict,
    revision_domains: list[str],
    solvent: str,
    temperature_C: float,
    concentration_M: float,
    assumed_MW: float,
    IF_used: float,
    pump_max_bar: float,
    is_photochem: bool,
    is_gas_liquid: bool,
    extinction_coeff_M_cm: Optional[float],
    new_id: int,
    parent_id: int,
    variant_mode: str,
    gas_liquid_max_bpr_bar: float,
    pump_max_flow_mL_min: Optional[float],
) -> dict:
    if candidate.get("inventory_realized"):
        from flora_translate.engine.council_v4.inventory_candidates import materialize
        try:
            revised = materialize(candidate["_realization_context"], patch=patch, candidate_id=new_id)
        except ValueError as exc:
            return {**candidate, "feasible": False, "violations": [str(exc)]}
        revised.update(parent_id=parent_id, variant_mode=variant_mode,
            revision_applied_preselection=True, revision_domains_preselection=revision_domains,
            revision_rationale_preselection=rationale, revision_changes_preselection=patch)
        return revised
    new_tau = _positive_float(
        patch.get("tau_min"),
        _positive_float(candidate.get("tau_min"), 1.0),
    )
    new_d = _positive_float(
        patch.get("d_mm"),
        _positive_float(candidate.get("d_mm"), 1.0),
    )
    new_Q = _positive_float(
        patch.get("Q_mL_min"),
        _positive_float(candidate.get("Q_mL_min"), 0.01),
    )
    new_c = _positive_float(
        patch.get("concentration_M"),
        _positive_float(candidate.get("concentration_M"), concentration_M),
    )
    new_bpr = max(
        0.0,
        _float_or_default(
            patch.get("BPR_bar"),
            _float_or_default(candidate.get("BPR_bar"), 0.0),
        ),
    )
    new_mat = str(patch.get("tubing_material", candidate.get("tubing_material", "FEP")))

    revised = compute_metrics(
        tau_min=new_tau,
        d_mm=new_d,
        Q_mL_min=new_Q,
        solvent=solvent,
        temperature_C=temperature_C,
        concentration_M=new_c,
        assumed_MW=assumed_MW,
        IF_used=IF_used,
        tau_kinetics_min=float(candidate.get("tau_kinetics_min") or new_tau),
        pump_max_bar=pump_max_bar,
        is_photochem=is_photochem,
        extinction_coeff_M_cm=extinction_coeff_M_cm,
        tau_source="preselection_revision_loop",
        is_gas_liquid=is_gas_liquid,
        BPR_bar=new_bpr,
        target_gas_equiv_inlet=_positive_float(
            candidate.get("target_gas_equiv_inlet"),
            1.0,
        ),
        gas_reagent_fraction=_positive_float(
            candidate.get("gas_reagent_fraction"),
            1.0,
        ),
    )
    feasible, violations, warnings = hard_filter(
        revised,
        is_photochem=is_photochem,
        is_gas_liquid=is_gas_liquid,
        pump_max_bar=pump_max_bar,
        BPR_bar=new_bpr,
        max_flow_rate_mL_min=pump_max_flow_mL_min,
        gas_liquid_max_bpr_bar=gas_liquid_max_bpr_bar,
    )
    revised["id"] = new_id
    revised["parent_id"] = parent_id
    revised["variant_mode"] = variant_mode
    revised["feasible"] = feasible
    revised["violations"] = violations
    revised["warnings"] = warnings
    revised["BPR_bar"] = round(new_bpr, 2)
    revised["tubing_material"] = new_mat
    revised["concentration_M"] = round(new_c, 4)
    revised["temperature_C"] = candidate.get("temperature_C", temperature_C)
    revised["revision_applied_preselection"] = True
    revised["revision_domains_preselection"] = sorted(set(revision_domains))
    revised["revision_rationale_preselection"] = rationale
    revised["revision_changes_preselection"] = patch
    return revised


def _run_candidate_refinement_loop(
    *,
    candidates: list[dict],
    scoring: dict,
    audit: dict,
    solvent: str,
    temperature_C: float,
    concentration_M: float,
    assumed_MW: float,
    IF_used: float,
    pump_max_bar: float,
    is_photochem: bool,
    is_gas_liquid: bool,
    extinction_coeff_M_cm: Optional[float],
    allow_warning_refinement: bool = False,
    strong_revision_mode: bool = False,
    branching_revision_mode: bool = False,
    max_descendants_per_candidate: int = 2,
    max_total_revised_candidates: Optional[int] = None,
    batch_concentration_M: Optional[float] = None,
    kinetic_x_minimum: float = 0.50,
    gas_liquid_max_bpr_bar: float = V4_GAS_LIQUID_MAX_BPR_BAR,
    pump_max_flow_mL_min: Optional[float] = None,
    translation_policy: str = "evidence_first",
) -> tuple[list[dict], dict]:
    blocked_domains = _build_domain_blocklist(audit)
    revised_candidates: list[dict] = []
    change_rows: list[dict] = []
    changed_ids: list[int] = []
    next_candidate_id = max((int(c.get("id", 0)) for c in candidates), default=0) + 1

    for candidate in candidates:
        cid = int(candidate.get("id"))
        skipped_domains: list[str] = []
        domain_patch_map: dict[str, tuple[dict, dict]] = {}

        domain_entries = {
            "chemistry": _entry_by_candidate(scoring.get("chemistry_scores", []), cid),
            "kinetics": _entry_by_candidate(scoring.get("kinetics_scores", []), cid),
            "fluidics": _entry_by_candidate(scoring.get("fluidics_scores", []), cid),
            "safety": _entry_by_candidate(scoring.get("safety_scores", []), cid),
        }

        for domain, entry in domain_entries.items():
            if not entry:
                continue
            if domain in blocked_domains.get(cid, set()):
                skipped_domains.append(domain)
                continue
            patch, rationale = _derive_domain_patch(
                domain=domain,
                entry=entry,
                candidate=candidate,
                concentration_M=concentration_M,
                allow_warning_refinement=allow_warning_refinement,
                strong_revision_mode=strong_revision_mode,
                batch_concentration_M=batch_concentration_M,
                translation_policy=translation_policy,
            )
            if not patch:
                continue
            domain_patch_map[domain] = (patch, rationale)

        if not domain_patch_map:
            revised = dict(candidate)
            revised["revision_applied_preselection"] = False
            revised["revision_domains_preselection"] = []
            revised["revision_rationale_preselection"] = {}
            revised["parent_id"] = cid
            revised["variant_mode"] = "original"
            revised_candidates.append(revised)
            if skipped_domains:
                change_rows.append({
                    "candidate_id": cid,
                    "changes": {},
                    "rationale": {},
                    "domains": [],
                    "skipped_domains": skipped_domains,
                })
            continue

        if branching_revision_mode:
            original = dict(candidate)
            original["revision_applied_preselection"] = False
            original["revision_domains_preselection"] = []
            original["revision_rationale_preselection"] = {}
            original["parent_id"] = cid
            original["variant_mode"] = "original"
            revised_candidates.append(original)

        variants = _candidate_revision_variants(
            candidate=candidate,
            cid=cid,
            domain_patch_map=domain_patch_map,
            strong_revision_mode=strong_revision_mode,
            max_descendants_per_candidate=max_descendants_per_candidate,
        )

        primary_changes: dict = {}
        primary_rationale: dict = {}
        primary_domains: list[str] = []
        descendant_ids: list[int] = []
        rejected_variants: list[dict] = []
        for idx, variant in enumerate(variants):
            new_id = cid if (idx == 0 and not branching_revision_mode) else next_candidate_id
            if new_id != cid or branching_revision_mode:
                next_candidate_id += 1
            revised = _materialize_revised_candidate(
                candidate=candidate,
                patch=variant["patch"],
                rationale=variant["rationale"],
                revision_domains=variant["domains"],
                solvent=solvent,
                temperature_C=temperature_C,
                concentration_M=concentration_M,
                assumed_MW=assumed_MW,
                IF_used=IF_used,
                pump_max_bar=pump_max_bar,
                is_photochem=is_photochem,
                is_gas_liquid=is_gas_liquid,
                extinction_coeff_M_cm=extinction_coeff_M_cm,
                new_id=new_id,
                parent_id=cid,
                variant_mode=variant["mode"],
                gas_liquid_max_bpr_bar=gas_liquid_max_bpr_bar,
                pump_max_flow_mL_min=pump_max_flow_mL_min,
            )
            if not revised.get("feasible", False):
                rejected_variants.append(
                    {
                        "variant_key": variant["variant_key"],
                        "patch": variant["patch"],
                        "violations": revised.get("violations") or [],
                    }
                )
                continue
            revised_candidates.append(revised)
            descendant_ids.append(new_id)
            if not primary_changes:
                primary_changes = variant["patch"]
                primary_rationale = variant["rationale"]
                primary_domains = variant["domains"]
        if not descendant_ids and not branching_revision_mode:
            original = dict(candidate)
            original["revision_applied_preselection"] = False
            original["revision_domains_preselection"] = []
            original["revision_rationale_preselection"] = {
                "rejected_revision_variants": rejected_variants,
            }
            original["parent_id"] = cid
            original["variant_mode"] = "original_after_rejected_revision"
            revised_candidates.append(original)
        if descendant_ids:
            changed_ids.append(cid)
        change_rows.append({
            "candidate_id": cid,
            "changes": primary_changes,
            "rationale": primary_rationale,
            "domains": sorted(set(primary_domains)),
            "skipped_domains": skipped_domains,
            "descendant_ids": descendant_ids,
            "descendant_count": len(descendant_ids),
            "rejected_variants": rejected_variants,
            "rejected_variant_count": len(rejected_variants),
            "branching_revision_mode": branching_revision_mode,
        })

    truncated = False
    dropped_candidate_ids: list[int] = []
    retained_candidate_ids: list[int] = []
    if max_total_revised_candidates is not None and max_total_revised_candidates > 0:
        def _variant_priority(row: dict) -> tuple[int, int, int, int]:
            variant_mode = str(row.get("variant_mode", "original"))
            mode_rank = {
                "merged": 0,
                "domain_focus": 1,
                "original": 2,
            }.get(variant_mode, 3)
            return (
                0 if bool(row.get("feasible", True)) else 1,
                0 if bool(row.get("revision_applied_preselection")) else 1,
                mode_rank,
                int(row.get("id", 0)),
            )

        ranked_candidates = sorted(revised_candidates, key=_variant_priority)
        retained = ranked_candidates[:max_total_revised_candidates]
        dropped = ranked_candidates[max_total_revised_candidates:]
        if dropped:
            truncated = True
            retained_candidate_ids = [int(r.get("id", 0)) for r in retained]
            dropped_candidate_ids = [int(r.get("id", 0)) for r in dropped]
            revised_candidates = retained

    revised_candidates, diversity = diverse_candidates(revised_candidates, candidates, max_total_revised_candidates)
    for revised in revised_candidates:
        _apply_v4_hard_gates(
            [revised],
            pump_max_bar=pump_max_bar,
            is_photochem=is_photochem,
            is_gas_liquid=is_gas_liquid,
            BPR_bar=float(revised.get("BPR_bar", 0.0)),
            X_minimum=kinetic_x_minimum,
            tubing_material=str(revised.get("tubing_material", "FEP")),
        )
    _tag_pareto_front(revised_candidates)

    changed = len(changed_ids)
    summary = {
        "diversity": diversity,
        "changed_candidate_ids": sorted(changed_ids),
        "changed_count": changed,
        "candidate_changes": change_rows,
        "had_changes": bool(changed),
        "summary": (
            f"Pre-selection expert refinement applied bounded edits to {changed}/"
            f"{len(candidates)} candidates before rescoring."
            if changed
            else "Pre-selection expert refinement found no bounded edits to apply."
        ),
        "branching_revision_mode": branching_revision_mode,
        "strong_revision_mode": strong_revision_mode,
        "final_candidate_count": len(revised_candidates),
        "max_total_revised_candidates": max_total_revised_candidates,
        "truncated_to_max_total_revised_candidates": truncated,
        "retained_candidate_ids": retained_candidate_ids,
        "dropped_candidate_ids": dropped_candidate_ids,
        "dropped_candidate_count": len(dropped_candidate_ids),
    }
    return revised_candidates, summary


# ═══════════════════════════════════════════════════════════════════════════════
#  Apply winner to FlowProposal
# ═══════════════════════════════════════════════════════════════════════════════

_ALLOWED_PATCH_FIELDS = {
    "residence_time_min", "flow_rate_mL_min", "tubing_ID_mm",
    "BPR_bar", "temperature_C", "concentration_M",
    "tubing_material", "wavelength_nm", "deoxygenation_method",
    "mixer_type", "reactor_volume_mL",
}


def _apply_winner(
    proposal: FlowProposal,
    winner: dict,
    extra_changes: dict,
    chief_data: Optional[dict] = None,
) -> tuple[FlowProposal, dict]:
    if winner.get("inventory_realized"):
        chosen = FlowProposal.model_validate(winner["realized_proposal"])
        chosen.inventory_constraints["council_reviewed_parameters"] = {
            key: getattr(chosen, key) for key in ["flow_rate_mL_min", "reactor_volume_mL", "tubing_ID_mm", "temperature_C", "BPR_bar", "concentration_M"]
        }
        chosen.inventory_constraints["council_reviewed_validation"] = winner["realization_validation"]
        from flora_translate.design_evidence import execution_state, REASONING_FIELDS, reasoning_snapshot
        chosen.inventory_constraints["council_reviewed_execution_state"] = execution_state(chosen)
        if chief_data:
            for key, destination in (("selection_rationale", "council_selection"),
                                     ("selection_justification", "council_selection_justification")):
                if chief_data.get(key):
                    chosen.decision_rationale[destination] = {
                        "text": chief_data[key], "source": "council_selection",
                        "parameter_snapshot": reasoning_snapshot(chosen.model_dump()),
                        "dependencies": list(REASONING_FIELDS), "status": "conditions_preserved",
                    }
        return chosen, {"inventory_candidate_id": winner["id"], "source": "exact inventory-realized candidate reviewed by council"}
    changes: dict = {
        "residence_time_min": str(round(winner["tau_min"], 2)),
        "flow_rate_mL_min":   str(round(winner["Q_mL_min"], 4)),
        "tubing_ID_mm":       str(round(winner["d_mm"], 3)),
        "reactor_volume_mL":  str(round(winner["V_R_mL"], 3)),
    }
    for k, v in (extra_changes or {}).items():
        if k in _ALLOWED_PATCH_FIELDS and k not in changes:
            try:
                current = proposal.model_dump().get(k)
                if isinstance(current, (int, float)):
                    float(v)
                changes[k] = str(v)
            except (TypeError, ValueError):
                pass

    data = proposal.model_dump()
    for field, value_str in changes.items():
        if field not in data:
            continue
        try:
            current = data[field]
            if isinstance(current, float):
                data[field] = float(value_str)
            elif isinstance(current, int):
                data[field] = int(float(value_str))
            else:
                data[field] = value_str
        except (ValueError, TypeError):
            data[field] = value_str

    # ── Per-stage diameter decisions (multi-step, from Chief LLM) ──────────
    if chief_data:
        d_per_stage = chief_data.get("stage_d_mm") or []
        if d_per_stage:
            existing_params = data.get("stage_parameters") or []
            existing_by_sn = {
                p.get("stage_number"): p
                for p in existing_params
                if isinstance(p, dict) and "stage_number" in p
            }
            stage_params = []
            for i, d_val in enumerate(d_per_stage, start=1):
                sp: dict = dict(existing_by_sn.get(i, {"stage_number": i}))
                sp["stage_number"] = i
                sp.pop("tau_fraction", None)
                if d_val:
                    try:
                        sp["d_mm"] = float(d_val)
                    except (TypeError, ValueError):
                        pass
                stage_params.append(sp)
            data["stage_parameters"] = stage_params
            logger.info(
                "  Chief: per-stage d_mm applied — %s",
                [sp.get("d_mm") for sp in stage_params],
            )

        # ── Apply pump flowrates from Chief derivation ──────────────────────
        pump_flowrates = chief_data.get("pump_flowrates") or []
        if pump_flowrates and data.get("streams"):
            pf_by_label = {
                pf["stream_label"].upper(): pf
                for pf in pump_flowrates
                if isinstance(pf, dict) and pf.get("stream_label")
            }
            for stream in data.get("streams", []):
                label = (stream.get("stream_label") or "").upper()
                if label in pf_by_label:
                    pf = pf_by_label[label]
                    try:
                        stream["flow_rate_mL_min"] = float(pf["Q_mL_min"])
                        if pf.get("derivation"):
                            stream["reasoning"] = pf["derivation"]
                    except (TypeError, ValueError):
                        pass
            logger.info(
                "  Chief: pump flowrates applied — %s",
                {pf.get("stream_label"): pf.get("Q_mL_min") for pf in pump_flowrates},
            )

            # Normalise so REACTOR-FEED pump rates (excluding quench streams)
            # sum exactly to Q_total_winner.
            #
            # Why exclude quench streams from the sum?  The council winner's Q
            # is the design flow through the MAIN reactor. Quench streams are
            # injected post-reactor and must NOT be counted in the reactor
            # inlet — otherwise we scale the main feeds down incorrectly.
            q_winner = winner.get("Q_mL_min") or 0.0
            _GAS_ROLES_SET = {
                "n2", "n₂", "nitrogen", "o2", "o₂", "oxygen", "co2", "co₂",
                "h2", "h₂", "hydrogen", "ar", "argon", "helium", "air",
                "compressed air", "mfc", "n2 gas", "o2 gas", "gas injection",
                "gas stream", "gas feed", "co", "carbon monoxide", "syngas",
                "ozone", "o3", "o₃", "chlorine", "cl2", "cl₂", "ammonia",
                "nh3", "nh₃", "hcl gas", "hydrogen chloride", "so2", "so₂",
                "sulfur dioxide", "ethylene", "acetylene",
            }
            _QUENCH_ROLE_KEYWORDS = ("quench", "neutraliz", "workup", "post-reactor")

            def _is_quench_role(role: str) -> bool:
                rl = (role or "").lower()
                return any(kw in rl for kw in _QUENCH_ROLE_KEYWORDS)

            def _is_gas_role(role: str) -> bool:
                rl = (role or "").lower()
                if any(kw in rl for kw in ("degassed", "deoxygenated")):
                    return False
                words = set(re.sub(r"[^a-z0-9₂]+", " ", rl).split())
                if words & {"gas", "mfc", "gaseous", "vapor", "vapour"}:
                    return True
                return any(term in rl for term in _GAS_ROLES_SET if len(term) > 2)

            reactor_feed_streams = [
                s for s in data.get("streams", [])
                if s.get("flow_rate_mL_min")
                and (s.get("stream_label") or "").upper() in pf_by_label
                and not _is_gas_role(s.get("pump_role") or "")
                and not _is_quench_role(s.get("pump_role") or "")
            ]
            q_sum = sum(s["flow_rate_mL_min"] for s in reactor_feed_streams)
            if q_winner > 0 and q_sum > 0 and abs(q_sum - q_winner) / q_winner > 0.02:
                scale = q_winner / q_sum
                for s in reactor_feed_streams:
                    old_rate = s["flow_rate_mL_min"]
                    new_rate = round(old_rate * scale, 4)
                    s["flow_rate_mL_min"] = new_rate
                    # Keep the derivation text consistent with the displayed rate.
                    orig = s.get("reasoning") or ""
                    scale_note = (
                        f" → scaled ×{scale:.3f} to {new_rate:.4f} mL/min "
                        f"so Σ reactor-feed pumps = Q_reactor_inlet = "
                        f"{q_winner:.4f} mL/min"
                    )
                    if orig and "scaled ×" not in orig:
                        s["reasoning"] = orig.rstrip() + scale_note
                    elif not orig:
                        s["reasoning"] = (
                            f"Chief-derived Q_i = {old_rate:.4f} mL/min"
                            + scale_note
                        )
                logger.info(
                    "  Chief: reactor-feed flowrates renormalised ×%.4f "
                    "(sum %.4f → %.4f mL/min; quench streams untouched)",
                    scale, q_sum, q_winner,
                )

    # ── Propagate council-selected τ into stage_parameters ─────────────────
    # Done unconditionally (not gated on chief_data) so the multi-step topology
    # builder always sees the council τ for Stage 1, even when the Chief LLM
    # returned no pump flowrates.
    winner_tau = winner.get("tau_min")
    if winner_tau:
        stage_params = data.get("stage_parameters") or []
        existing_by_sn = {
            p["stage_number"]: p for p in stage_params
            if isinstance(p, dict) and "stage_number" in p
        }
        if 1 not in existing_by_sn:
            new_entry: dict = {"stage_number": 1}
            existing_by_sn[1] = new_entry
            stage_params = [new_entry] + list(stage_params)
        existing_by_sn[1].setdefault("residence_time_min", round(winner_tau, 3))
        data["stage_parameters"] = [
            p for p in stage_params if isinstance(p, dict) and "stage_number" in p
        ]

    return FlowProposal(**data), changes


def _bench_stage_start(recorder, name: str, details: Optional[dict] = None):
    if recorder and hasattr(recorder, "start_stage"):
        try:
            recorder.start_stage(name, details or {})
        except Exception:
            pass


def _bench_stage_end(recorder, name: str, details: Optional[dict] = None, status: str = "completed"):
    if recorder and hasattr(recorder, "end_stage"):
        try:
            recorder.end_stage(name, details or {}, status=status)
        except Exception:
            pass


def _bench_snapshot(recorder, name: str, payload):
    if recorder and hasattr(recorder, "save_snapshot"):
        try:
            recorder.save_snapshot(name, payload)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Build AgentDeliberation records for UI compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def _verdict_icon(verdict: str) -> str:
    v = str(verdict).upper()
    return {"ACCEPT": "✅", "WARNING": "⚠️", "REVISE": "🔄", "BLOCK": "🚫",
            "APPROVED_WITH_CONDITIONS": "✅⚠️"}.get(v, "❓")


def _score_bar(score: float, width: int = 10) -> str:
    score = max(0.0, min(1.0, _positive_float(score, 0.0)))
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled) + f" {score:.2f}"


def _fmt_num(value, fmt: str, default: Optional[float] = None) -> Optional[str]:
    """Format numeric LLM/tool fields without crashing on text placeholders."""
    try:
        if value in (None, ""):
            return None if default is None else format(default, fmt)
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return None if default is None else format(default, fmt)


def _stringify_issue(value) -> str:
    """Render LLM/tool issue payloads as safe one-line strings."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        preferred = [
            "description", "concern", "issue", "reason", "message",
            "error_type", "field", "value",
        ]
        parts = [str(value[k]) for k in preferred if value.get(k) is not None]
        if parts:
            return " — ".join(parts)
        return "; ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return "; ".join(_stringify_issue(v) for v in value if v is not None)
    return str(value)


def _issue_list(values, fallback: str) -> list[str]:
    if not values:
        return [fallback]
    if isinstance(values, (str, dict)):
        values = [values]
    return [text for text in (_stringify_issue(v) for v in values) if text] or [fallback]


def _build_chemistry_cot(scoring: dict) -> str:
    overall = scoring.get("chemistry_overall", "")
    scores = scoring.get("chemistry_scores", [])
    lines = []
    if overall:
        lines.append(f"### Overall Chemistry Assessment\n\n{overall}\n")
    lines.append("---\n### Per-Candidate Analysis\n")
    for e in scores:
        cid = e.get("candidate_id", "?")
        verdict = e.get("verdict", "?")
        icon = _verdict_icon(verdict)
        chem_s = e.get("chemistry_score", 0.0) or 0.0
        phot_s = e.get("photonics_score", 0.0) or 0.0
        comb_s = e.get("combined_score", 0.0) or 0.0
        reasoning = e.get("reasoning", "")
        A = e.get("beer_lambert_A")
        eps = e.get("epsilon_used")
        wl = e.get("wavelength_match")
        mat = e.get("material_transparent")
        blocking = _issue_list(e.get("blocking_issues"), "")
        concerns = _issue_list(e.get("concerns"), "")

        lines.append(f"#### {icon} Candidate {cid} — {verdict}")
        lines.append(
            f"| Domain | Score |\n|---|---|\n"
            f"| Chemistry | {_score_bar(chem_s)} |\n"
            f"| Photonics | {_score_bar(phot_s)} |\n"
            f"| **Combined** | **{_score_bar(comb_s)}** |\n"
        )
        if reasoning:
            lines.append(f"{reasoning}\n")
        details = []
        if A is not None:
            A_fmt = _fmt_num(A, ".4f")
            if A_fmt is not None:
                details.append(f"- Beer-Lambert A = **{A_fmt}** (ε = {eps} M⁻¹cm⁻¹)")
        if wl is not None:
            details.append(f"- Wavelength match: {'✓' if wl else '✗'}")
        if mat is not None:
            details.append(f"- Material transparent: {'✓' if mat else '✗ (BLOCK)'}")
        if details:
            lines.append("\n".join(details) + "\n")
        if any(blocking):
            lines.append("**⛔ Blocking issues:**\n" + "\n".join(f"- {b}" for b in blocking) + "\n")
        if any(concerns):
            lines.append("**⚠️ Concerns:**\n" + "\n".join(f"- {c}" for c in concerns) + "\n")
        lines.append("---\n")
    return "\n".join(lines)


def _build_kinetics_cot(scoring: dict) -> str:
    overall = scoring.get("kinetics_overall", "")
    scores = scoring.get("kinetics_scores", [])
    lines = []
    if overall:
        lines.append(f"### Overall Kinetics Assessment\n\n{overall}\n")
    lines.append("---\n### Per-Candidate Analysis\n")
    for e in scores:
        cid = e.get("candidate_id", "?")
        verdict = e.get("verdict", "?")
        icon = _verdict_icon(verdict)
        kin_s = e.get("kinetics_score", 0.0) or 0.0
        reasoning = e.get("reasoning", "")
        X = e.get("X_estimated")
        IF = e.get("IF_used")
        tau_vs_lit = e.get("tau_vs_literature", "")
        tau_mix = e.get("tau_mixing_required_min")
        tau_final = e.get("tau_proposed_final_min")
        t_steady = e.get("t_steady_min")
        prod = e.get("productivity_mg_h")
        concerns = _issue_list(e.get("concerns"), "")

        lines.append(f"#### {icon} Candidate {cid} — {verdict}")
        lines.append(f"**Kinetics score:** {_score_bar(kin_s)}\n")
        if reasoning:
            lines.append(f"{reasoning}\n")
        details = []
        if X is not None:
            X_f = _positive_float(X, -1.0)
            if X_f >= 0:
                x_ok = "✓" if X_f >= 0.85 else ("⚠️" if X_f >= 0.70 else "✗")
                details.append(f"- Conversion X = **{X_f:.3f}** {x_ok}")
        if IF is not None:
            details.append(f"- Intensification factor IF = **{IF}×**  {'✓' if e.get('IF_valid') else '⚠️'}")
        if tau_vs_lit:
            details.append(f"- τ vs literature: {tau_vs_lit}")
        if tau_mix is not None:
            tau_mix_fmt = _fmt_num(tau_mix, ".1f")
            if tau_mix_fmt is not None:
                details.append(f"- τ_mixing_required = {tau_mix_fmt} min")
        if tau_final is not None:
            tau_final_fmt = _fmt_num(tau_final, ".1f")
            if tau_final_fmt is not None:
                details.append(f"- τ_final (decision rule) = **{tau_final_fmt} min**")
        if t_steady is not None:
            t_steady_fmt = _fmt_num(t_steady, ".0f")
            if t_steady_fmt is not None:
                details.append(f"- Steady-state wait = {t_steady_fmt} min (3×τ)")
        if prod is not None:
            prod_fmt = _fmt_num(prod, ".1f")
            if prod_fmt is not None:
                details.append(f"- Productivity = {prod_fmt} mg/h")
        if details:
            lines.append("\n".join(details) + "\n")
        if any(concerns):
            lines.append("**⚠️ Concerns:**\n" + "\n".join(f"- {c}" for c in concerns) + "\n")
        lines.append("---\n")
    return "\n".join(lines)


def _build_fluidics_cot(scoring: dict) -> str:
    overall = scoring.get("fluidics_overall", "")
    scores = scoring.get("fluidics_scores", [])
    lines = []
    if overall:
        lines.append(f"### Overall Fluidics & Hardware Assessment\n\n{overall}\n")
    lines.append("---\n### Per-Candidate Analysis\n")
    for e in scores:
        cid = e.get("candidate_id", "?")
        verdict = e.get("verdict", "?")
        icon = _verdict_icon(verdict)
        flu_s = e.get("fluidics_score", 0.0) or 0.0
        reasoning = e.get("reasoning", "")
        Re = e.get("Re")
        regime = e.get("flow_regime", "")
        dP = e.get("dP_bar")
        headroom = e.get("pump_headroom_pct")
        r_mix = e.get("r_mix")
        dual_fail = e.get("dual_criterion_mixing_fail", False)
        d_dir = e.get("d_change_direction", "none")
        d_fix = e.get("d_fix_mm")
        L = e.get("L_m")
        pump = e.get("pump_type", "")
        mat = e.get("tubing_material", "")
        dv_impact = e.get("dead_volume_impact", "")
        concerns = _issue_list(e.get("concerns"), "")

        lines.append(f"#### {icon} Candidate {cid} — {verdict}")
        lines.append(f"**Fluidics score:** {_score_bar(flu_s)}\n")
        if reasoning:
            lines.append(f"{reasoning}\n")
        details = []
        if Re is not None:
            Re_fmt = _fmt_num(Re, ".0f")
            if Re_fmt is not None:
                details.append(f"- Re = **{Re_fmt}** ({regime})")
        if dP is not None and headroom is not None:
            dP_fmt = _fmt_num(dP, ".3f")
            headroom_f = _positive_float(headroom, -1.0)
            if dP_fmt is not None and headroom_f >= 0:
                hp_icon = "✓" if headroom_f > 40 else ("⚠️" if headroom_f > 20 else "✗")
                details.append(f"- ΔP = {dP_fmt} bar | Pump headroom = **{headroom_f:.1f}%** {hp_icon}")
        if r_mix is not None:
            r_mix_f = _positive_float(r_mix, -1.0)
            if r_mix_f >= 0:
                mix_icon = "✓" if r_mix_f < 0.10 else ("⚠️" if r_mix_f < 0.20 else "✗")
                details.append(f"- r_mix = **{r_mix_f:.3f}** {mix_icon}{' — dual-criterion FAIL ⛔' if dual_fail else ''}")
        if d_dir and d_dir != "none":
            details.append(f"- d change needed: **{d_dir}**" + (f" → d_fix = {d_fix} mm" if d_fix else ""))
        if L is not None:
            L_fmt = _fmt_num(L, ".1f")
            if L_fmt is not None:
                details.append(f"- L = {L_fmt} m")
        if pump:
            details.append(f"- Pump: {pump}")
        if mat:
            details.append(f"- Tubing material: {mat}")
        if dv_impact:
            details.append(f"- Dead volume: {dv_impact}")
        if details:
            lines.append("\n".join(details) + "\n")
        if any(concerns):
            lines.append("**⚠️ Concerns:**\n" + "\n".join(f"- {c}" for c in concerns) + "\n")
        lines.append("---\n")
    return "\n".join(lines)


def _build_safety_cot(scoring: dict) -> str:
    overall = scoring.get("safety_overall", "")
    scores = scoring.get("safety_scores", [])
    lines = []
    if overall:
        lines.append(f"### Overall Safety Assessment\n\n{overall}\n")
    lines.append("---\n### Per-Candidate Analysis\n")
    for e in scores:
        cid = e.get("candidate_id", "?")
        verdict = e.get("verdict", "?")
        icon = _verdict_icon(verdict)
        saf_s = e.get("safety_score", 0.0) or 0.0
        reasoning = e.get("reasoning", "")
        Da_th = e.get("Da_thermal")
        BPR_req = e.get("BPR_required_bar")
        BPR_cur = e.get("BPR_current_bar")
        BPR_ok = e.get("BPR_adequate")
        mat_rec = e.get("material_recommendation", "")
        mat_rat = e.get("material_rationale", "")
        atm_req = e.get("atmosphere_isolation_required", False)
        iso_method = e.get("isolation_method", "")
        hazards = e.get("hazard_flags") or []
        blocking = e.get("blocking_issues") or []
        conditions = e.get("conditions") or []

        lines.append(f"#### {icon} Candidate {cid} — {verdict}")
        lines.append(f"**Safety score:** {_score_bar(saf_s)}\n")
        if reasoning:
            lines.append(f"{reasoning}\n")
        details = []
        if Da_th is not None:
            Da_th_f = _positive_float(Da_th, -1.0)
            if Da_th_f >= 0:
                th_icon = "✓" if Da_th_f < 0.1 else ("⚠️" if Da_th_f < 1.0 else "✗")
                details.append(f"- Da_thermal = **{Da_th_f:.4f}** {th_icon}")
        if BPR_req is not None and BPR_cur is not None:
            BPR_req_fmt = _fmt_num(BPR_req, ".2f")
            BPR_cur_fmt = _fmt_num(BPR_cur, ".2f")
            if BPR_req_fmt is not None and BPR_cur_fmt is not None:
                bpr_icon = "✓" if BPR_ok else "✗ REVISE"
                details.append(
                    f"- BPR required = {BPR_req_fmt} bar | current = {BPR_cur_fmt} bar {bpr_icon}"
                )
        if mat_rec:
            details.append(f"- Material: **{mat_rec}**" + (f" — {mat_rat}" if mat_rat else ""))
        if atm_req:
            details.append(f"- Atmosphere isolation required: {iso_method or 'see conditions'}")
        if details:
            lines.append("\n".join(details) + "\n")
        if hazards:
            lines.append("**🔶 Hazard flags:**\n" + "\n".join(f"- {h}" for h in hazards) + "\n")
        if blocking:
            lines.append("**⛔ Blocking issues:**\n" + "\n".join(f"- {b}" for b in blocking) + "\n")
        if conditions:
            lines.append("**📋 Required conditions:**\n" + "\n".join(f"- {c}" for c in conditions) + "\n")
        lines.append("---\n")
    return "\n".join(lines)


def _build_skeptic_cot(audit: dict) -> str:
    lines = []
    summary = audit.get("audit_summary", "")
    if summary:
        lines.append(f"### Arithmetic & Consistency Audit\n\n{summary}\n")

    all_errors = audit.get("all_errors", [])
    if not all_errors:
        lines.append("✅ **All arithmetic checks passed.** No calculation errors, threshold "
                     "violations, or scope issues detected.")
        return "\n".join(lines)

    for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        errs = [e for e in all_errors if e.get("severity") == severity]
        if not errs:
            continue
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(severity, "⚪")
        lines.append(f"\n#### {icon} {severity} severity ({len(errs)} issues)\n")
        for e in errs:
            lines.append(
                f"**[{e.get('error_type', '?')}]** candidate {e.get('candidate_id', '?')} "
                f"({e.get('agent', '?')})\n\n> {e.get('description', '')}\n"
            )

    disq = audit.get("disqualify_recommendations", [])
    if disq:
        lines.append(f"\n#### 🚫 Disqualification Recommendations\n")
        for r in disq:
            lines.append(f"- Candidate {r['candidate_id']}: {r['reason']}\n")

    return "\n".join(lines)


def _build_refinement_cot(refinement: dict) -> str:
    lines = []
    summary = refinement.get("summary", "")
    if summary:
        lines.append(f"### Pre-selection Expert Refinement\n\n{summary}\n")
    rows = refinement.get("candidate_changes") or []
    if not rows:
        lines.append("No bounded candidate edits were applied before final selection.")
        return "\n".join(lines)

    changed_rows = [r for r in rows if r.get("changes")]
    skipped_rows = [r for r in rows if r.get("skipped_domains")]

    if changed_rows:
        lines.append("### Applied Candidate Edits\n")
        for row in changed_rows:
            cid = row.get("candidate_id")
            domains = ", ".join(row.get("domains") or [])
            lines.append(f"#### Candidate {cid} ({domains or 'unknown domains'})")
            for field, value in (row.get("changes") or {}).items():
                reason = (row.get("rationale") or {}).get(field, "")
                lines.append(f"- **{field}** → `{value}`: {reason}")
            lines.append("")

    if skipped_rows:
        lines.append("### Skipped Domains Due To Audit Findings\n")
        for row in skipped_rows:
            lines.append(
                f"- Candidate {row.get('candidate_id')}: "
                f"{', '.join(row.get('skipped_domains') or [])}"
            )

    return "\n".join(lines)


def _build_chief_cot(
    chief_data: dict,
    weighted_scores: list[dict],
    winner_id: Optional[int],
    dfmea: Optional[dict],
) -> str:
    lines = []

    rationale = chief_data.get("selection_rationale", "")
    if rationale:
        lines.append(f"### Selection Rationale\n\n{rationale}\n")

    # Scoring health warning — shown prominently if domain agents failed
    survivors = [r for r in weighted_scores if not r["disqualified"]]
    all_default_domains = {"chemistry", "kinetics", "fluidics", "safety"}
    failed_scoring = [
        r["candidate_id"] for r in survivors
        if all(abs(r[d] - 0.5) < 0.01 for d in all_default_domains)
    ]
    if failed_scoring:
        lines.append(
            f"\n> ⚠️ **SCORING FAILURE** — domain agents returned no per-candidate scores "
            f"for candidates {failed_scoring}. All domain scores are at the 0.500 default. "
            f"The selection below is based on geometry only and has no engineering basis. "
            f"Re-run or check the scoring agent logs.\n"
        )

    # Score comparison table
        if survivors:
            lines.append("### Weighted Score Comparison\n")
            lines.append("| id | combined | chem | kin | fluid | safety | geo | boosted |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in survivors:
                winner_marker = " ★" if r["candidate_id"] == winner_id else ""
                lines.append(
                f"| **{r['candidate_id']}{winner_marker}** | **{_positive_float(r.get('combined')):.3f}** | "
                f"{_positive_float(r.get('chemistry')):.3f} | {_positive_float(r.get('kinetics')):.3f} | "
                f"{_positive_float(r.get('fluidics')):.3f} | {_positive_float(r.get('safety')):.3f} | "
                f"{_positive_float(r.get('geometry')):.3f} | {r.get('objective_boosted', '')} |"
                )
        lines.append("")

    # Resolved trade-offs
    tradeoffs = chief_data.get("resolved_tradeoffs") or []
    if tradeoffs:
        lines.append("### Resolved Trade-offs\n")
        for t in tradeoffs:
            lines.append(f"- {t}")
        lines.append("")

    # Overridden concerns
    overrides = chief_data.get("overridden_concerns") or []
    if overrides:
        lines.append("### Overridden Concerns\n")
        for o in (overrides if isinstance(overrides[0], dict) else
                  [{"agent": "?", "concern": str(x), "override_reason": ""} for x in overrides]):
            lines.append(
                f"- **{o.get('agent', '?')}**: {o.get('concern', '')} "
                f"→ *{o.get('override_reason', '')}*"
            )
        lines.append("")

    # Remaining uncertainties
    uncerts = chief_data.get("remaining_uncertainties") or []
    if uncerts:
        lines.append("### Remaining Uncertainties\n")
        for u in uncerts:
            lines.append(f"- {u}")
        lines.append("")

    # Pump flowrates derived by chief
    pump_frs = chief_data.get("pump_flowrates") or []
    if pump_frs:
        lines.append("### Pump Flowrate Derivation\n")
        lines.append("| Stream | Role | Q (mL/min) | Derivation |")
        lines.append("|---|---|---|---|")
        for pf in pump_frs:
            if isinstance(pf, dict):
                lines.append(
                    f"| {pf.get('stream_label', '?')} | {pf.get('role', '?')} | "
                    f"**{pf.get('Q_mL_min', '?')}** | {pf.get('derivation', '')} |"
                )
        n_lim_shown = chief_data.get("n_limiting_mmol_min")
        if n_lim_shown:
            lines.append(f"\nṅ_limiting = **{n_lim_shown} mmol/min** | Q_total = **{chief_data.get('Q_total_mL_min', '?')} mL/min**\n")

    # Ask-back
    ask_d = chief_data.get("ask_expert")
    ask_q = chief_data.get("ask_expert_question")
    if ask_d and ask_q:
        lines.append(f"\n### Chief Ask-Back\n\nDomain consulted: **{ask_d}**\n\nQuestion: {ask_q}\n")

    # Validation experiment
    exp = chief_data.get("experiment_recommendation")
    if exp:
        lines.append(f"### Recommended First Experiment\n\n{exp}\n")

    # DFMEA
    if dfmea and dfmea.get("failure_modes"):
        lines.append("### DFMEA — Failure Mode Analysis\n")
        lines.append("| Rank | Mode | Cause | Severity | Likelihood | RPN | Mitigation |")
        lines.append("|---|---|---|---|---|---|---|")
        for fm in dfmea["failure_modes"]:
            lines.append(
                f"| {fm.get('rank', '?')} | {fm.get('mode', '')} | {fm.get('cause', '')} | "
                f"{fm.get('severity', '?')} | {fm.get('likelihood', '?')} | "
                f"**{fm.get('RPN', '?')}** | {fm.get('mitigation', '')} |"
            )
        lines.append("")
        spof = dfmea.get("single_points_of_failure") or []
        if spof:
            lines.append("**Single points of failure:**\n" + "\n".join(f"- {s}" for s in spof) + "\n")
        val_exp = dfmea.get("validation_experiments") or []
        if val_exp:
            lines.append("**Validation experiments:**\n" + "\n".join(f"1. {v}" for v in val_exp) + "\n")

    return "\n".join(lines)


def _to_deliberations_v4(
    designer_result: dict,
    scoring: dict,
    audit: dict,
    weighted_scores: list[dict],
    winner_id: Optional[int],
    candidates: list[dict],
    dfmea: Optional[dict],
    chief_data: dict,
    tool_calls: dict,
    revision_result: Optional[dict] = None,
    preselection_refinement: Optional[dict] = None,
) -> list[list[AgentDeliberation]]:
    """Convert v4 council output to UI-compatible AgentDeliberation lists with rich markdown."""
    round1: list[AgentDeliberation] = []

    # ── Designer ─────────────────────────────────────────────────────────────
    n_surv = len(designer_result.get("survivors", []))
    n_disq = len(designer_result.get("disqualified", []))
    ps = designer_result.get("problem_statement", {})
    disq_list = designer_result.get("disqualified", [])

    designer_cot_lines = [
        f"### Stage 0 — Problem Framing\n",
        f"**Reaction class:** {ps.get('reaction_class', 'unknown')}  \n"
        f"**Flags:** {', '.join(ps.get('special_flags', [])) or 'none'}  \n"
        f"**Flow justified:** {'Yes' if ps.get('flow_justified', True) else '⚠️ Questionable — ' + ps.get('flow_justification_note', '')}",
        f"\n### Stage 1 — Candidate Matrix\n",
        f"**Strategy:** {designer_result.get('strategy_reasoning', '')}  \n"
        f"**Candidates to council:** {n_surv}  \n"
        f"**Hard-gate flagged (proceed with warnings):** {n_disq}",
    ]
    if disq_list:
        designer_cot_lines.append("\n**Hard-gate flag details** *(candidates still evaluated — flags inform scoring agents)*:")
        for d in disq_list[:8]:
            cand = d.get("candidate", {})
            designer_cot_lines.append(
                f"- ⛔ Candidate {cand.get('id', '?')} (τ={cand.get('tau_min', '?')} min, "
                f"d={cand.get('d_mm', '?')} mm): {d.get('reason', '')}"
            )

    envelope = designer_result.get("design_envelope_preliminary", {})
    if envelope:
        designer_cot_lines.append(
            f"\n**Preliminary design envelope:** τ ∈ {envelope.get('tau_range', '?')} min | "
            f"d ∈ {envelope.get('d_range', '?')} mm | Q ∈ {envelope.get('Q_range', '?')} mL/min"
        )

    round1.append(AgentDeliberation(
        agent="DesignerV4", agent_display_name="Designer",
        round=1,
        chain_of_thought="\n".join(designer_cot_lines),
        findings=[
            f"Candidates to council: {n_surv} (all proceed to scoring)",
            f"Hard-gate flagged: {n_disq} carry warning flags visible to all agents",
        ],
        status="ACCEPT",
    ))

    if preselection_refinement and (
        preselection_refinement.get("had_changes")
        or preselection_refinement.get("candidate_changes")
    ):
        changed = preselection_refinement.get("changed_count", 0)
        round1.append(AgentDeliberation(
            agent="CandidateRefinementV4",
            agent_display_name="Candidate Refinement Board",
            round=1,
            chain_of_thought=_build_refinement_cot(preselection_refinement),
            findings=[
                f"Bounded expert edits applied to {changed} candidate(s) before the final scoring pass",
                "All revised candidates were recomputed deterministically before rescoring",
            ],
            concerns=[
                f"Candidate {row.get('candidate_id')}: skipped {', '.join(row.get('skipped_domains') or [])}"
                for row in (preselection_refinement.get("candidate_changes") or [])
                if row.get("skipped_domains")
            ],
            status="REVISE" if changed else "ACCEPT",
        ))

    # ── Dr. Chemistry ─────────────────────────────────────────────────────────
    chem_blocked = [e for e in scoring.get("chemistry_scores", [])
                    if str(e.get("verdict", "")).upper() == "BLOCK"]
    n_chem = len(scoring.get("chemistry_scores", []))

    round1.append(AgentDeliberation(
        agent="DrChemistryV4", agent_display_name="Dr. Chemistry",
        round=1,
        chain_of_thought=_build_chemistry_cot(scoring),
        findings=[
            f"Scored {n_chem} candidates across chemistry and photonics domains",
            f"Blocked: {len(chem_blocked)} | Accepted: {n_chem - len(chem_blocked)}",
        ],
        concerns=[
            f"**Candidate {e['candidate_id']} BLOCKED:** "
            f"{', '.join(_issue_list(e.get('blocking_issues'), 'unspecified'))}"
            for e in chem_blocked
        ],
        status="ACCEPT" if not chem_blocked else "WARNING",
        tool_calls=scoring.get("tool_calls", {}).get("chemistry", []),
    ))

    # ── Dr. Kinetics ──────────────────────────────────────────────────────────
    kin_blocked = [e for e in scoring.get("kinetics_scores", [])
                   if str(e.get("verdict", "")).upper() == "BLOCK"]
    n_kin = len(scoring.get("kinetics_scores", []))

    round1.append(AgentDeliberation(
        agent="DrKineticsV4", agent_display_name="Dr. Kinetics",
        round=1,
        chain_of_thought=_build_kinetics_cot(scoring),
        findings=[
            f"Scored {n_kin} candidates across kinetics, conversion, and integration domains",
            f"Blocked: {len(kin_blocked)} | Accepted: {n_kin - len(kin_blocked)}",
        ],
        concerns=[
            f"**Candidate {e['candidate_id']} BLOCKED:** "
            f"{', '.join(_issue_list(e.get('concerns'), 'X below minimum'))}"
            for e in kin_blocked
        ],
        status="ACCEPT" if not kin_blocked else "WARNING",
        tool_calls=scoring.get("tool_calls", {}).get("kinetics", []),
    ))

    # ── Dr. Fluidics ──────────────────────────────────────────────────────────
    flu_blocked = [e for e in scoring.get("fluidics_scores", [])
                   if str(e.get("verdict", "")).upper() == "BLOCK"]
    n_flu = len(scoring.get("fluidics_scores", []))

    round1.append(AgentDeliberation(
        agent="DrFluidicsV4", agent_display_name="Dr. Fluidics",
        round=1,
        chain_of_thought=_build_fluidics_cot(scoring),
        findings=[
            f"Scored {n_flu} candidates across fluidics, transport phenomena, and hardware",
            f"Blocked: {len(flu_blocked)} | Accepted: {n_flu - len(flu_blocked)}",
        ],
        concerns=[
            f"**Candidate {e['candidate_id']} BLOCKED:** "
            f"{', '.join(_issue_list(e.get('concerns'), 'Re or geometry limit'))}"
            for e in flu_blocked
        ],
        status="ACCEPT" if not flu_blocked else "WARNING",
        tool_calls=scoring.get("tool_calls", {}).get("fluidics", []),
    ))

    # ── Dr. Safety ────────────────────────────────────────────────────────────
    saf_blocked = [e for e in scoring.get("safety_scores", [])
                   if str(e.get("verdict", "")).upper() == "BLOCK"]
    n_saf = len(scoring.get("safety_scores", []))

    round1.append(AgentDeliberation(
        agent="DrSafetyV4", agent_display_name="Dr. Safety",
        round=1,
        chain_of_thought=_build_safety_cot(scoring),
        findings=[
            f"Scored {n_saf} candidates across thermal safety, BPR, materials, and hazards",
            f"Blocked: {len(saf_blocked)} | Accepted/Conditional: {n_saf - len(saf_blocked)}",
        ],
        concerns=[
            f"**Candidate {e['candidate_id']} BLOCKED:** "
            f"{', '.join(_issue_list(e.get('blocking_issues'), 'safety gate failed'))}"
            for e in saf_blocked
        ],
        status="ACCEPT" if not saf_blocked else "WARNING",
        tool_calls=scoring.get("tool_calls", {}).get("safety", []),
    ))

    # ── Skeptic ───────────────────────────────────────────────────────────────
    errors = audit.get("all_errors", [])
    critical_count = sum(1 for e in errors if e.get("severity") == "CRITICAL")
    high_count = sum(1 for e in errors if e.get("severity") == "HIGH")

    round1.append(AgentDeliberation(
        agent="SkepticV4", agent_display_name="Skeptic",
        round=1,
        chain_of_thought=_build_skeptic_cot(audit),
        findings=[
            f"Audit complete: {len(errors)} issues found "
            f"({critical_count} CRITICAL, {high_count} HIGH)",
            f"Disqualification recommendations: {len(audit.get('disqualify_recommendations', []))} candidates",
            f"Council may proceed: {'✅ Yes' if audit.get('council_may_proceed') else '🔴 No — CRITICAL errors must be resolved'}",
        ],
        concerns=[
            f"[{e['severity']}] Candidate {e.get('candidate_id')} — "
            f"{e.get('error_type', '?')}: {e['description'][:200]}"
            for e in errors if e.get("severity") in ("CRITICAL", "HIGH")
        ],
        status=("REVISE" if not audit.get("council_may_proceed") else
                ("WARNING" if errors else "ACCEPT")),
    ))

    # ── Revision Agent (Stage 3.5) ────────────────────────────────────────────
    if revision_result is not None:
        rev_rationale = revision_result.get("revision_rationale", {})
        rev_domains   = revision_result.get("revision_domains", [])
        rev_lines = [
            "### Stage 3.5 — Parameter Revision\n",
            f"**Domains that triggered revision:** {', '.join(rev_domains)}\n",
            "**Changes applied to winner:**\n",
        ]
        if rev_rationale:
            for param, reason in rev_rationale.items():
                rev_lines.append(f"- **{param}**: {reason}")
        else:
            rev_lines.append("(No detailed rationale provided)")

        round1.append(AgentDeliberation(
            agent="RevisionAgent", agent_display_name="Revision Engineer",
            round=1,
            chain_of_thought="\n".join(rev_lines),
            findings=[
                f"Revised {len(rev_rationale)} parameter(s) on winner id={winner_id}: "
                + ", ".join(f"{k}={v}" for k, v in {
                    "tau_min": revision_result.get("tau_min"),
                    "d_mm": revision_result.get("d_mm"),
                    "BPR_bar": revision_result.get("BPR_bar"),
                    "tubing_material": revision_result.get("tubing_material"),
                }.items() if k in rev_rationale)
            ],
            status="REVISE",
        ))

    # ── Chief Engineer + DFMEA ────────────────────────────────────────────────
    round2: list[AgentDeliberation] = []
    if winner_id is not None:
        w_entry = next((r for r in weighted_scores if r["candidate_id"] == winner_id), {})

        round2.append(AgentDeliberation(
            agent="ChiefV4", agent_display_name="Chief Engineer",
            round=2,
            chain_of_thought=_build_chief_cot(chief_data, weighted_scores, winner_id, dfmea),
            findings=[
                f"**Winner: Candidate {winner_id}** | Combined score: {w_entry.get('combined', '?'):.3f}",
                f"PVS: {w_entry.get('PVS', 0.0):.3f} | Selection flag: {chief_data.get('selection_flag', 'OK')}",
                f"Runner-ups: {chief_data.get('runner_up_ids', [])}",
                f"Objective boosted: {w_entry.get('objective_boosted', 'none')}",
            ],
            concerns=[str(u) for u in (chief_data.get("remaining_uncertainties") or [])[:5]],
            status="ACCEPT",
            tool_calls=tool_calls.get("chief", []),
        ))

    return [round1, round2] if round2 else [round1]


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_summary_v4(
    designer_result: dict,
    scoring: dict,
    audit: dict,
    weighted_scores: list[dict],
    winner_id: Optional[int],
    candidates: list[dict],
    chief_data: dict,
    dfmea: Optional[dict],
    objectives: str,
    preselection_refinement: Optional[dict] = None,
) -> str:
    lines = ["## Council v4 — Stage-gated Engineering Design Summary"]
    lines.append(f"\n**Objectives**: {objectives}")
    lines.append(f"**Designer**: {len(designer_result.get('survivors', []))} survivors, "
                 f"{len(designer_result.get('disqualified', []))} hard-gate disqualified")
    lines.append(f"\n### Candidate Shortlist\n{designer_result.get('table_markdown', '')}")

    if preselection_refinement:
        lines.append("\n### Pre-selection Expert Refinement")
        lines.append(preselection_refinement.get("summary", ""))
        changed_rows = [
            row for row in (preselection_refinement.get("candidate_changes") or [])
            if row.get("changes")
        ]
        for row in changed_rows:
            rendered = ", ".join(
                f"{field}={value}" for field, value in (row.get("changes") or {}).items()
            )
            lines.append(f"- id {row.get('candidate_id')}: {rendered}")

    lines.append("\n### Weighted Scores (all surviving candidates)")
    lines.append("| id | combined | PVS | chemistry | kinetics | fluidics | safety | geometry | disq |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in weighted_scores:
        lines.append(
            f"| {r['candidate_id']} | {_positive_float(r.get('combined')):.3f} | "
            f"{_positive_float(r.get('PVS')):.3f} | "
            f"{_positive_float(r.get('chemistry')):.3f} | {_positive_float(r.get('kinetics')):.3f} | "
            f"{_positive_float(r.get('fluidics')):.3f} | {_positive_float(r.get('safety')):.3f} | "
            f"{_positive_float(r.get('geometry')):.3f} | "
            f"{'✗' if r['disqualified'] else ''} |"
        )

    lines.append(f"\n### Skeptic Audit\n{audit.get('audit_summary', '')}")
    if audit.get("all_errors"):
        for e in audit["all_errors"]:
            if e.get("severity") in ("CRITICAL", "HIGH"):
                lines.append(f"- [{e['severity']}] id {e.get('candidate_id')}: {e['description'][:200]}")

    if winner_id is not None:
        w = next((c for c in candidates if c.get("id") == winner_id), {})
        lines.append(f"\n### Winner — candidate id={winner_id}")
        lines.append(
            f"- τ={w.get('tau_min')} min, d={w.get('d_mm')} mm, "
            f"Q={w.get('Q_mL_min')} mL/min, V_R={w.get('V_R_mL')} mL, "
            f"L={w.get('L_m')} m"
        )
        lines.append(
            f"- Re={_positive_float(w.get('Re')):.0f}, ΔP={_positive_float(w.get('delta_P_bar')):.3f} bar, "
            f"r_mix={_positive_float(w.get('r_mix')):.3f}, X={_positive_float(w.get('expected_conversion')):.2f}, "
            f"prod={_positive_float(w.get('productivity_mg_h')):.1f} mg/h"
        )
        lines.append(f"- **Rationale**: {chief_data.get('selection_rationale', '')}")
        lines.append(f"- **Flow-value justification**: {chief_data.get('selection_justification', '')}")

        if chief_data.get("remaining_uncertainties"):
            lines.append("- **Remaining uncertainties**:")
            for u in chief_data["remaining_uncertainties"]:
                lines.append(f"  - {u}")

        if dfmea and dfmea.get("failure_modes"):
            lines.append("\n### DFMEA — Top Failure Modes")
            for fm in dfmea["failure_modes"]:
                lines.append(
                    f"- [{fm.get('rank')}] {fm.get('mode')}: "
                    f"severity={fm.get('severity')}, likelihood={fm.get('likelihood')}, "
                    f"RPN={fm.get('RPN')} | mitigation: {fm.get('mitigation', '')}"
                )
            if dfmea.get("validation_experiments"):
                lines.append("\n### Validation Experiments")
                for exp in dfmea["validation_experiments"]:
                    lines.append(f"- {exp}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_chemistry_brief(
    batch: BatchRecord,
    plan: Optional[ChemistryPlan],
    proposal: FlowProposal,
) -> str:
    rc = (plan.reaction_class if plan else "unknown") or "unknown"
    mech = (plan.mechanism_type if plan else "") or "—"
    n_stages = plan.n_stages if plan else 1
    o2 = bool(plan and plan.oxygen_sensitive)
    quench = bool(plan and plan.quench_required)
    atm = ""
    if plan and plan.stages:
        atm = " | atm: " + "/".join(s.atmosphere for s in plan.stages)
    return (
        f"reaction_class: {rc} | mechanism: {mech} | n_stages: {n_stages} | "
        f"O₂-sensitive: {o2} | quench_required: {quench}{atm}\n"
        f"solvent: {proposal.streams[0].solvent if proposal.streams else 'EtOH'} | "
        f"T: {proposal.temperature_C}°C | C: {proposal.concentration_M} M"
    )


def _extract_extinction_coeff(
    batch: BatchRecord,
    plan: Optional[ChemistryPlan],
) -> Optional[float]:
    for attr in ("extinction_coeff_M_cm", "epsilon_M_cm", "photocatalyst_epsilon"):
        for obj in (batch, plan):
            if obj is None:
                continue
            v = getattr(obj, attr, None)
            try:
                if v is not None and float(v) > 0:
                    return float(v)
            except (TypeError, ValueError):
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  CouncilV4 — main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class CouncilV4:
    """Stage-gated multi-agent scoring council.

    Usage (same API as CouncilV3):
        result, calc = CouncilV4().run(proposal, batch, analogies, inventory,
                                        chemistry_plan=..., calculations=...,
                                        objectives="de-risk first-run")
    """

    def run(
        self,
        proposal: FlowProposal,
        batch_record: BatchRecord,
        analogies: list[dict],
        inventory: LabInventory,
        chemistry_plan: Optional[ChemistryPlan] = None,
        calculations: Optional[DesignCalculations] = None,
        objectives: str = "balanced",
        max_loop_rounds: int = 3,
        fixed_candidates: Optional[list[dict]] = None,
        candidate_budget: int = 12,
        allow_warning_refinement: bool = False,
        design_space_seed_candidates: Optional[list[dict]] = None,
        benchmark_recorder=None,
        benchmark_strict_scoring: bool = False,
        benchmark_scoring_batch_size: Optional[int] = None,
        benchmark_claude_compact_mode: bool = False,
        benchmark_strong_revision_mode: bool = False,
        benchmark_branching_revision_mode: bool = False,
        benchmark_max_descendants_per_candidate: int = 2,
        benchmark_max_total_revised_candidates: Optional[int] = None,
        intake_package: Optional[DesignInputPackage | dict] = None,
        execution_config: Optional[CouncilExecutionConfig | dict] = None,
        council_backflow_review: bool = True,
        council_physics_profile: Optional[dict] = None,
        hard_constraints: object = None,
    ) -> tuple[DesignCandidate, DesignCalculations]:
        """Run the full council pipeline.

        If *fixed_candidates* is provided, Stage 1 (Designer) is skipped and
        those candidates are passed directly to Stage 2 (domain scoring).
        Use this for ablation studies comparing 1-candidate vs N-candidate modes.
        """

        from flora_translate.scientific_evidence import enabled
        if enabled(chemistry_plan):
            from flora_translate.engine.council_v4.scientific import run_scientific_council
            return run_scientific_council(proposal, batch_record, chemistry_plan,
                inventory, analogies, objectives, candidate_budget, intake_package,
                backflow_review=council_backflow_review, physics_profile=council_physics_profile)
        execution = CouncilExecutionConfig.coerce(execution_config)
        current = proposal.model_copy(deep=True)
        log = DeliberationLog()
        _bench_snapshot(
            benchmark_recorder,
            "council_execution_config",
            execution.provenance(),
        )

        evidence_calibration = None
        if intake_package is not None:
            try:
                from flora_translate.experiment_loop import (
                    extract_experiments_from_text,
                    refine_from_experimental_campaign,
                )

                evidence_experiments = extract_experiments_from_text(
                    historical_text_from_package(intake_package)
                )
                if len(evidence_experiments) >= 2:
                    evidence_result = refine_from_experimental_campaign(
                        {"proposal": current.model_dump()},
                        evidence_experiments,
                        target_yield_pct=75.0,
                        target_conversion_pct=75.0,
                        target_selectivity_pct=85.0,
                    )
                    current = FlowProposal(**evidence_result.refined_result["proposal"])
                    current, evidence_inventory_report = enforce_reactor_inventory(
                        current,
                        inventory,
                    )
                    evidence_calibration = current.evidence_calibration or {}
                    calculations = None
                    logger.info(
                        "  Council v4 — measured-evidence initialization: "
                        "best tau=%.3f min, requested next tau=%.3f min, "
                        "inventory-feasible tau=%.3f min%s",
                        float(evidence_calibration.get("best_tau_min") or 0.0),
                        float(evidence_calibration.get("recommended_tau_min") or 0.0),
                        float(current.residence_time_min or 0.0),
                        " (pump constrained)"
                        if (
                            (current.inventory_constraints or {})
                            .get("flow_recalculation", {})
                            .get("pump_flow_clamped")
                        )
                        else "",
                    )
            except Exception as exc:
                logger.warning("Council measured-evidence initialization skipped: %s", exc)

        # ── Ensure calculator center-point ──────────────────────────────────
        if calculations is None:
            calculations = DesignCalculator().run(
                batch_record, chemistry_plan, current, inventory,
                analogies=analogies,
                target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                target_tubing_ID_mm=current.tubing_ID_mm or None,
            )
        calc = calculations

        # ── Extract context ─────────────────────────────────────────────────
        is_photochem = bool(
            (chemistry_plan and "photo" in (chemistry_plan.reaction_class or "").lower()) or
            (current.wavelength_nm and current.wavelength_nm > 0)
        )
        is_gas_liquid = bool(getattr(calc, "is_gas_liquid", False))
        # Distinguish "O2 inhibits the reaction" (needs degassing) from
        # "O2 is a reagent" (needs MFC + BPR + gas-liquid hardware). The
        # framing LLM has historically conflated these — `O2_sensitive`
        # was being emitted for aerobic oxidations where O2 is the
        # oxidant. Read both flags explicitly and pass them downstream.
        is_O2_sensitive = bool(chemistry_plan and chemistry_plan.oxygen_sensitive)
        is_O2_reagent = bool(chemistry_plan and getattr(chemistry_plan, "o2_is_reagent", False))
        # Derive o2_is_reagent from stream contents if the chemistry agent
        # didn't set it explicitly — a gas stream containing O2/air is
        # definitive evidence.
        if not is_O2_reagent and chemistry_plan is not None:
            for stream in getattr(chemistry_plan, "stream_logic", []) or []:
                phase = (getattr(stream, "phase", "") or "").lower()
                reagents = [str(r).lower() for r in (getattr(stream, "reagents", []) or [])]
                if phase == "gas" and any(("o2" in r or "o₂" in r or "air" == r or "oxygen" in r) for r in reagents):
                    is_O2_reagent = True
                    break
        # If O2 is the reagent it cannot simultaneously be an inhibitor.
        # The reaction may still need controlled O2 dosing, but the
        # "degassing required" semantic does not apply.
        if is_O2_reagent and is_O2_sensitive:
            logger.info(
                "    O2 flag disambiguation: O2 is REAGENT — clearing inhibits-O2 flag "
                "(chemistry agent set both, likely conflating semantics)"
            )
            is_O2_sensitive = False

        # Fix #5: does this chemistry actually use a photocatalyst? Drives
        # the Skeptic's ε plausibility threshold (1000 vs 5000).
        has_photocatalyst = False
        if chemistry_plan is not None:
            for reagent in (getattr(chemistry_plan, "reagents", []) or []):
                role = (getattr(reagent, "role", "") or "").lower()
                if "photocatalyst" in role or "photo-catalyst" in role or "sensitizer" in role:
                    has_photocatalyst = True
                    break
            # Heuristic backup: look in batch description for explicit "no photocatalyst" mention
            if not has_photocatalyst and is_photochem:
                desc = (getattr(batch_record, "reaction_description", "") or "").lower()
                if any(t in desc for t in ("no photocatalyst", "photocatalyst-free", "without photocatalyst", "catalyst-free")):
                    has_photocatalyst = False
                elif any(t in desc for t in ("ir(ppy)", "ru(bpy)", "eosin", "acridinium", "4cz")):
                    has_photocatalyst = True

        solvent = (
            (chemistry_plan.stages[0].solvent if chemistry_plan and chemistry_plan.stages else "")
            or batch_record.solvent
            or next((s.solvent for s in current.streams if s.solvent), "")
        )

        tau_center = calc.residence_time_min or current.residence_time_min or 30.0
        tau_lit = getattr(calc, "tau_analogy_min", None) or getattr(calc, "tau_class_min", None)
        tau_kinetics = getattr(calc, "tau_kinetics_min", None) or tau_center
        batch_time_min = (
            (getattr(batch_record, "reaction_time_h", None) or 0.0) * 60.0
            or (getattr(calc, "batch_time_s", 0.0) or 0.0) / 60.0
        )
        translation_policy = FLOW_TRANSLATION_POLICY
        kinetic_x_minimum = (
            0.50 if (translation_policy or "").lower() == "intensify" else 0.0
        )
        IF_used = calc.intensification_factor or 6.0
        assumed_MW = getattr(batch_record, "product_MW", None) or 250.0
        pump_max = calc.pump_max_bar or 20.0
        target_gas_equiv = _positive_float(
            getattr(calc, "target_gas_equiv_inlet", None),
            1.0,
        )
        gas_reagent_fraction = _positive_float(
            getattr(calc, "gas_reagent_fraction", None),
            1.0,
        )
        inventory_bpr_values = [
            _positive_float(value)
            for value in available_pressure_settings(inventory)
            if _positive_float(value) > 0
        ]
        gas_liquid_max_bpr = (
            max(inventory_bpr_values)
            if inventory_bpr_values
            else min(pump_max, V4_GAS_LIQUID_MAX_BPR_BAR)
        )
        selected_pump = (current.inventory_constraints or {}).get("selected_pump") or {}
        pump_min_flow = _positive_float(
            selected_pump.get("min_flow_rate_mL_min"),
            0.05,
        )
        inventory_pump_max_flows = [
            _positive_float(getattr(pump, "max_flow_rate_mL_min", None))
            for pump in (getattr(inventory, "pumps", None) or [])
            if _positive_float(getattr(pump, "max_flow_rate_mL_min", None)) > 0
        ]
        pump_max_flow = _positive_float(
            selected_pump.get("max_flow_rate_mL_min"),
            max(inventory_pump_max_flows) if inventory_pump_max_flows else 0.0,
        ) or None
        ext_coeff = _extract_extinction_coeff(batch_record, chemistry_plan)
        chem_brief = _build_chemistry_brief(batch_record, chemistry_plan, current)
        intake_block = intake_context_block(intake_package) if intake_package is not None else ""
        if intake_block:
            chem_brief += "\n\n" + intake_block
        reaction_class = (chemistry_plan.reaction_class if chemistry_plan else "unknown") or "unknown"
        intensification_mandate = (
            chemistry_plan.intensification_mandate.model_dump()
            if chemistry_plan and getattr(chemistry_plan, "intensification_mandate", None)
            else {}
        )
        if intensification_mandate:
            chem_brief += (
                "\nintensification_mandate: "
                + json.dumps(intensification_mandate, ensure_ascii=False)
            )
        if evidence_calibration:
            feasibility_diagnostic = None
        else:
            feasibility_diagnostic = _intensification_feasibility_precheck(
                batch_time_min=batch_time_min,
                tau_kinetics_min=tau_kinetics,
                intensification_mandate=intensification_mandate,
                translation_policy=translation_policy,
                calc=calc,
                candidate_tau_min=current.residence_time_min,
            )

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 0 — Problem Framing
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Stage 0: Problem framing")
        _bench_stage_start(benchmark_recorder, "council_stage_0_problem_framing")
        problem_statement = run_problem_framing(
            reaction_class=reaction_class,
            is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
            is_O2_sensitive=is_O2_sensitive,
            is_O2_reagent=is_O2_reagent,
            tau_center_min=tau_center, tau_lit_min=tau_lit,
            solvent=solvent, temperature_C=current.temperature_C,
            concentration_M=current.concentration_M,
            objectives=objectives,
            batch_time_min=batch_time_min,
            protocol=batch_record.raw_text or batch_record.reaction_description or "",
            intake_context=intake_block,
            inventory=inventory.model_dump() if inventory else {},
        )
        logger.info(
            "    Problem framing: reaction_class=%s | flags=%s | flow_justified=%s",
            problem_statement.get("reaction_class"),
            problem_statement.get("special_flags"),
            problem_statement.get("flow_justified"),
        )
        _bench_stage_end(
            benchmark_recorder,
            "council_stage_0_problem_framing",
            {
                "reaction_class": problem_statement.get("reaction_class"),
                "flow_justified": problem_statement.get("flow_justified"),
                "special_flags": problem_statement.get("special_flags", []),
            },
        )
        # The intensification precheck is diagnostic only — it never blocks
        # the council. Adjudicating whether a flow design delivers a credible
        # advantage is the job of Skeptic + Chief + DFMEA, with the full
        # candidate matrix and scoring in hand. The single-point tau-ratio
        # test the precheck performs is strictly less informed than that, so
        # we surface its output as problem-statement context and let the
        # council make the call.
        if feasibility_diagnostic is not None:
            logger.warning(
                "Council v4: intensification feasibility diagnostic — %s",
                feasibility_diagnostic["diagnosis"],
            )
            problem_statement.setdefault("ambiguities", [])
            problem_statement["ambiguities"].append(feasibility_diagnostic["diagnosis"])
            problem_statement["kinetic_anchor_quality"] = feasibility_diagnostic.get("kinetic_anchor_quality")
            problem_statement["screen_required"] = True
            if benchmark_recorder is not None:
                _bench_snapshot(
                    benchmark_recorder,
                    "stage0_intensification_feasibility",
                    feasibility_diagnostic,
                )
        if intake_block:
            problem_statement["intake_context"] = intake_block

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 1 — Designer: Candidate Matrix  (or bypass with fixed_candidates)
        # ═══════════════════════════════════════════════════════════════════
        # Any non-None diagnostic now feeds Designer the intensification ceiling
        # so it samples under it. The status string distinction the previous
        # version cared about was only relevant when hard_block was a thing.
        uncertain_kinetics_screen = feasibility_diagnostic is not None
        uncertain_redesign_instructions = None
        if uncertain_kinetics_screen:
            tau_ceiling = _positive_float(feasibility_diagnostic.get("tau_intensification_ceiling_min"))
            target = _positive_float(feasibility_diagnostic.get("tau_reduction_target"), 2.0)
            uncertain_redesign_instructions = {
                "tau_ceiling": tau_ceiling,
                "tau_floor": max(0.5, tau_ceiling / 3.0) if tau_ceiling > 0 else 0.5,
                "required_advantage": feasibility_diagnostic.get("minimum_flow_advantage", "productivity"),
                "note": (
                    "Kinetic anchor is uncertain because analogy IF values were repaired/floored. "
                    "Generate intensified screen candidates under the mandate ceiling; do not treat "
                    "calculated conversion from the repaired tau anchor as a hard gate."
                ),
                "target_reduction_factor": target,
            }
        elif evidence_calibration:
            evidence_floor = _positive_float(
                evidence_calibration.get("best_tau_min")
            )
            uncertain_redesign_instructions = {
                "tau_floor": evidence_floor,
                "measured_evidence_override": True,
                "note": (
                    "Measured campaign evidence is authoritative. Do not generate "
                    "or select a residence time below the best measured anchor while "
                    "the measured response remains below target."
                ),
            }
        _bench_stage_start(
            benchmark_recorder,
            "council_stage_1_designer",
            {"candidate_budget": candidate_budget, "fixed_candidates": len(fixed_candidates or [])},
        )
        if fixed_candidates is not None:
            logger.info(
                "  Council v4 — Stage 1: BYPASSED (fixed_candidates=%d provided)",
                len(fixed_candidates),
            )
            survivors = fixed_candidates
            attach_flow_sense_reports(
                survivors,
                batch_time_min=batch_time_min,
                batch_concentration_M=getattr(calc, "concentration_M", None),
                solvent_name=solvent,
                intensification_mandate=intensification_mandate,
            )
            table_markdown = format_candidate_table(survivors, max_rows=len(survivors))
            designer_result = {
                "survivors":                  survivors,
                "disqualified":               [],
                "table_markdown":             table_markdown,
                "strategy_reasoning":         f"Stage 1 bypassed — {len(survivors)} fixed candidate(s) provided",
                "design_envelope_preliminary":{},
                "problem_statement":          problem_statement,
                "pool_metadata":              {"pool_quality": "FIXED", "candidates_generated": len(survivors)},
            }
        elif inventory and inventory.reactors and (not chemistry_plan or len(chemistry_plan.stages or []) <= 1):
            from flora_translate.engine.council_v4.inventory_candidates import generate_inventory_candidates
            designer_result = generate_inventory_candidates(
                current, batch_record, chemistry_plan, inventory, hard_constraints,
                count=candidate_budget, tau_kinetics_min=tau_kinetics, assumed_MW=assumed_MW,
                IF_used=IF_used, pump_max_bar=pump_max, is_photochem=is_photochem,
                batch_time_min=batch_time_min,
            )
            designer_result["problem_statement"] = problem_statement
        else:
            logger.info("  Council v4 — Stage 1: Designer candidate matrix")
            designer_result = run_designer_v4(
                reaction_class=reaction_class,
                is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
                is_O2_sensitive=is_O2_sensitive,
                tau_center_min=tau_center, tau_lit_min=tau_lit,
                tau_kinetics_min=tau_kinetics,
                d_center_mm=current.tubing_ID_mm or 1.0,
                Q_center_mL_min=current.flow_rate_mL_min or 1.0,
                solvent=solvent, temperature_C=current.temperature_C,
                concentration_M=current.concentration_M,
                assumed_MW=assumed_MW, IF_used=IF_used,
                pump_max_bar=pump_max,
                pump_min_flow_mL_min=pump_min_flow,
                pump_max_flow_mL_min=pump_max_flow,
                BPR_bar=current.BPR_bar or 0.0,
                batch_time_min=batch_time_min,
                translation_policy=translation_policy,
                extinction_coeff_M_cm=ext_coeff,
                tubing_material=current.tubing_material or "FEP",
                X_minimum=0.0 if uncertain_kinetics_screen else kinetic_x_minimum,
                N_target=candidate_budget,
                problem_statement=problem_statement,
                intensification_mandate=intensification_mandate,
                redesign_instructions=uncertain_redesign_instructions,
                target_gas_equiv_inlet=target_gas_equiv,
                gas_reagent_fraction=gas_reagent_fraction,
                gas_liquid_max_bpr_bar=gas_liquid_max_bpr,
            )
            if (
                feasibility_diagnostic is not None
                and not feasibility_diagnostic.get("hard_block", True)
            ):
                designer_result["feasibility_diagnostic"] = feasibility_diagnostic

        survivors = designer_result["survivors"]
        if evidence_calibration:
            evidence_floor = _positive_float(evidence_calibration.get("best_tau_min"))
            if evidence_floor > 0:
                evidence_rejected = []
                evidence_eligible = []
                for candidate in survivors:
                    candidate_tau = _positive_float(candidate.get("tau_min"))
                    if candidate_tau + 1e-9 < evidence_floor:
                        reason = (
                            "measured-evidence residence-time floor: "
                            f"tau={candidate_tau:.3f} < {evidence_floor:.3f} min"
                        )
                        candidate.setdefault("hard_gate_flags", []).append(reason)
                        candidate["hard_gate_status"] = "BLOCKED: " + reason
                        evidence_rejected.append({"candidate": candidate, "reason": reason})
                    else:
                        evidence_eligible.append(candidate)
                if evidence_eligible:
                    survivors = evidence_eligible
                    designer_result["survivors"] = survivors
                    designer_result.setdefault("disqualified", []).extend(evidence_rejected)
                    logger.info(
                        "    Evidence floor %.3f min: %d candidates retained, %d blocked",
                        evidence_floor,
                        len(evidence_eligible),
                        len(evidence_rejected),
                    )
                else:
                    logger.warning(
                        "Council Designer produced no candidate at/above the %.3f min "
                        "measured-evidence floor",
                        evidence_floor,
                    )
        if uncertain_kinetics_screen:
            _downgrade_uncertain_kinetic_hard_gates(survivors)
            _downgrade_uncertain_kinetic_hard_gates(designer_result.get("disqualified", []))
        _bench_snapshot(
            benchmark_recorder,
            "stage1_designer_result",
            {
                "strategy_reasoning": designer_result.get("strategy_reasoning"),
                "problem_statement": designer_result.get("problem_statement"),
                "pool_metadata": designer_result.get("pool_metadata", {}),
                "survivors": survivors,
                "disqualified": designer_result.get("disqualified", []),
            },
        )
        _bench_stage_end(
            benchmark_recorder,
            "council_stage_1_designer",
            {
                "survivor_count": len(survivors),
                "disqualified_count": len(designer_result.get("disqualified", [])),
            },
            status="completed" if survivors else "empty",
        )
        if not survivors:
            if designer_result.get("pool_metadata", {}).get("pool_quality") == "INVENTORY_BOUND":
                current.inventory_constraints["council_selection_block"] = "No candidate passed inventory realization; inspect the saved rejected candidates."
                return self._fallback(current, chemistry_plan, calc, log, designer_result,
                    reason="no inventory-feasible candidate", batch_record=batch_record, inventory=inventory, analogies=analogies)
            # FALLBACK: Council Designer's own sampling found 0 feasible
            # candidates. Before skipping the council entirely (which loses all
            # domain-agent input), try seeding from the upstream Design Space
            # grid search. Those candidates already passed the same unified
            # feasibility checks (post-refactor: design_space.py and
            # sampling.py share compute_metrics + hard_filter), so they ARE
            # genuine alternates — they just landed outside the LLM's chosen
            # sampling envelope.
            seed_pool: list[dict] = list(design_space_seed_candidates or [])
            if evidence_calibration:
                evidence_floor = _positive_float(evidence_calibration.get("best_tau_min"))
                seed_pool = [
                    candidate
                    for candidate in seed_pool
                    if _positive_float(candidate.get("tau_min")) + 1e-9 >= evidence_floor
                ]
            if seed_pool:
                logger.warning(
                    "Council v4: Designer sampling returned 0 survivors — seeding from "
                    "Design Space grid (%d feasible candidates available)",
                    len(seed_pool),
                )
                # Run the same v4 hard-gate logic to attach flags / status.
                # Seeds already passed sampling.hard_filter, so most should
                # pass the v4 gates too; any that fail still surface their
                # flags to agents.
                attach_flow_sense_reports(
                    seed_pool,
                    batch_time_min=batch_time_min,
                    batch_concentration_M=getattr(calc, "concentration_M", None),
                    solvent_name=solvent,
                    intensification_mandate=intensification_mandate,
                )
                seed_survivors, seed_flagged = _apply_v4_hard_gates(
                    seed_pool, pump_max_bar=pump_max,
                    is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
                    BPR_bar=current.BPR_bar or 0.0,
                    X_minimum=0.0 if uncertain_kinetics_screen else kinetic_x_minimum,
                    tubing_material=current.tubing_material or "FEP",
                )
                if seed_survivors:
                    survivors = seed_survivors
                    designer_result["survivors"] = survivors
                    designer_result["disqualified"] = seed_flagged
                    designer_result["table_markdown"] = format_candidate_table(
                        survivors, max_rows=len(survivors)
                    )
                    designer_result.setdefault("pool_metadata", {})
                    designer_result["pool_metadata"]["pool_quality"] = "DESIGN_SPACE_FALLBACK"
                    designer_result["pool_metadata"]["candidates_generated"] = len(survivors)
                    designer_result["pool_metadata"]["fallback_source"] = "design_space"
                    designer_result["strategy_reasoning"] = (
                        (designer_result.get("strategy_reasoning") or "")
                        + " | Designer sampling empty → seeded from Design Space grid."
                    )
                    logger.info(
                        "    Design Space fallback: %d survivors seeded into council",
                        len(survivors),
                    )
                    _bench_snapshot(
                        benchmark_recorder,
                        "stage1_design_space_fallback",
                        {
                            "seed_count": len(seed_pool),
                            "survivor_count": len(survivors),
                            "flagged_count": len(seed_flagged),
                        },
                    )
                else:
                    logger.warning(
                        "Council v4: Design Space fallback also empty after v4 hard gates — "
                        "%d flagged: %s",
                        len(seed_flagged),
                        "; ".join((f.get("reason") or "")[:80] for f in seed_flagged[:3]),
                    )
            if not survivors:
                logger.warning(
                    "Council v4: no survivors after hard-gate filter (incl. Design Space fallback) "
                    "— requiring experimental screen"
                )
                screen_payload = _build_screen_required_payload(
                    current=current,
                    calc=calc,
                    batch_time_min=batch_time_min,
                    intensification_mandate=intensification_mandate,
                    solvent=solvent,
                    reason="no usable candidates after filtering",
                    feasibility_diagnostic=designer_result.get("feasibility_diagnostic"),
                )
                designer_result["screen_required_report"] = screen_payload
                _bench_snapshot(benchmark_recorder, "stage1_screen_required", screen_payload)
                return self._fallback(
                    current, chemistry_plan, calc, log, designer_result,
                    reason="screen required: no usable candidates after filtering",
                    batch_record=batch_record, inventory=inventory, analogies=analogies,
                )
        if survivors and all(
            any("X=" in str(flag) and "X_minimum" in str(flag) for flag in (c.get("hard_gate_flags") or []))
            for c in survivors
        ):
            logger.warning(
                "Council v4: all %d Designer candidates fail the insufficient-conversion hard gate — "
                "skipping domain scoring and falling back for manual review.",
                len(survivors),
            )
            if benchmark_recorder is not None:
                _bench_snapshot(
                    benchmark_recorder,
                    "stage1_pool_rejected_pre_scoring",
                    {
                        "reason": "all candidates fail insufficient-conversion hard gate",
                        "survivor_count": len(survivors),
                        "candidate_ids": [c.get("id") for c in survivors],
                    },
                )
            screen_payload = _build_screen_required_payload(
                current=current,
                calc=calc,
                batch_time_min=batch_time_min,
                intensification_mandate=intensification_mandate,
                solvent=solvent,
                reason="all candidates fail insufficient-conversion hard gate",
                feasibility_diagnostic=designer_result.get("feasibility_diagnostic"),
            )
            designer_result["screen_required_report"] = screen_payload
            _bench_snapshot(benchmark_recorder, "stage1_screen_required", screen_payload)
            return self._fallback(
                current, chemistry_plan, calc, log, designer_result,
                reason="screen required: all candidates fail insufficient-conversion hard gate",
                batch_record=batch_record, inventory=inventory, analogies=analogies,
            )

        table_markdown = designer_result["table_markdown"]

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 2 — Domain Scoring (max max_loop_rounds iterations if needed)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Stage 2: Domain scoring (%d survivors)", len(survivors))
        _bench_stage_start(benchmark_recorder, "council_stage_2_domain_scoring", {"survivor_count": len(survivors)})
        scoring_batch_size = benchmark_scoring_batch_size
        if scoring_batch_size is None and candidate_budget > 1:
            scoring_batch_size = 3 if candidate_budget <= 6 else 4
        initial_scoring = run_domain_scoring(
            candidates=survivors,
            table_markdown=table_markdown,
            chemistry_brief=chem_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max,
            batch_size=scoring_batch_size,
            strict_coverage=benchmark_strict_scoring,
            benchmark_claude_compact_mode=benchmark_claude_compact_mode,
            intensification_mandate=intensification_mandate,
            enabled_agents=execution.enabled_scoring_agents,
        )
        _bench_snapshot(benchmark_recorder, "stage2_initial_scoring", initial_scoring)
        _bench_stage_end(
            benchmark_recorder,
            "council_stage_2_domain_scoring",
            {"blocked_by_scoring": initial_scoring.get("blocked_by_scoring", [])},
        )

        # Remove scoring-blocked candidates from survivors for audit
        scoring_blocked_ids = set(initial_scoring.get("blocked_by_scoring", []))

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 3 — Skeptic Audit
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Stage 3: Skeptic audit")
        _bench_stage_start(benchmark_recorder, "council_stage_3_skeptic_audit")
        initial_audit = run_skeptic_audit(
            candidates=survivors,
            chemistry_scores=initial_scoring["chemistry_scores"],
            kinetics_scores=initial_scoring["kinetics_scores"],
            fluidics_scores=initial_scoring["fluidics_scores"],
            safety_scores=initial_scoring["safety_scores"],
            is_gas_liquid=is_gas_liquid,
            concentration_M=current.concentration_M,
            pump_max_bar=pump_max,
            batch_time_min=batch_time_min,
            batch_concentration_M=getattr(calc, "concentration_M", None),
            solvent_name=solvent,
            translation_policy=translation_policy,
            max_tau_to_batch_ratio=FLOW_MAX_TAU_TO_BATCH_RATIO,
            intensification_mandate=intensification_mandate,
            pool_metadata=designer_result.get("pool_metadata", {}),
            process_value_scores=initial_scoring.get("process_value_scores", []),
            has_photocatalyst=has_photocatalyst,
        ) if execution.enable_skeptic else _bypassed_skeptic_audit(survivors)

        weak_pool_cycles = 0
        weak_pool_history: list[dict] = []
        while (
            initial_audit.get("verdict") == "WEAK_POOL"
            and fixed_candidates is None
            and weak_pool_cycles < cfg.MAX_WEAK_POOL_CYCLES
        ):
            weak_pool_cycles += 1
            report = initial_audit.get("weak_pool_report") or {}
            weak_pool_history.append(report)
            logger.warning(
                "  Council v4 — Skeptic WEAK_POOL cycle %d/%d: %s",
                weak_pool_cycles,
                cfg.MAX_WEAK_POOL_CYCLES,
                "; ".join(report.get("specific_failures", []))[:180],
            )
            redesigned_result = run_designer_v4(
                reaction_class=reaction_class,
                is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
                is_O2_sensitive=is_O2_sensitive,
                tau_center_min=tau_center, tau_lit_min=tau_lit,
                tau_kinetics_min=tau_kinetics,
                d_center_mm=current.tubing_ID_mm or 1.0,
                Q_center_mL_min=current.flow_rate_mL_min or 1.0,
                solvent=solvent, temperature_C=current.temperature_C,
                concentration_M=current.concentration_M,
                assumed_MW=assumed_MW, IF_used=IF_used,
                pump_max_bar=pump_max,
                BPR_bar=current.BPR_bar or 0.0,
                batch_time_min=batch_time_min,
                translation_policy=translation_policy,
                extinction_coeff_M_cm=ext_coeff,
                tubing_material=current.tubing_material or "FEP",
                N_target=candidate_budget,
                problem_statement=problem_statement,
                intensification_mandate=intensification_mandate,
                redesign_instructions=report.get("regeneration_instructions") or {},
            )
            redesigned_survivors = redesigned_result["survivors"]
            if not redesigned_survivors:
                # A failed regeneration must not erase the last viable pool.
                # Preserve the prior candidates/scoring/audit and let the Chief
                # make a bounded SCREEN_REQUIRED decision from real options.
                initial_audit["weak_pool_redesign_exhausted"] = True
                initial_audit.setdefault("all_errors", []).append({
                    "agent": "SKEPTIC",
                    "candidate_id": None,
                    "error_type": "WEAK_POOL_REDESIGN_EMPTY",
                    "description": (
                        "Weak-pool regeneration produced no feasible candidates; "
                        "the previous viable pool was retained for selection."
                    ),
                    "severity": "HIGH",
                })
                logger.warning(
                    "  Council v4 — weak-pool redesign returned no candidates; "
                    "retaining the previous %d-candidate pool",
                    len(survivors),
                )
                break
            designer_result = redesigned_result
            survivors = redesigned_survivors
            table_markdown = designer_result["table_markdown"]
            initial_scoring = run_domain_scoring(
                candidates=survivors,
                table_markdown=table_markdown,
                chemistry_brief=chem_brief,
                objectives=objectives,
                is_photochem=is_photochem,
                pump_max_bar=pump_max,
                batch_size=scoring_batch_size,
                strict_coverage=benchmark_strict_scoring,
                benchmark_claude_compact_mode=benchmark_claude_compact_mode,
                intensification_mandate=intensification_mandate,
                enabled_agents=execution.enabled_scoring_agents,
            )
            initial_audit = run_skeptic_audit(
                candidates=survivors,
                chemistry_scores=initial_scoring["chemistry_scores"],
                kinetics_scores=initial_scoring["kinetics_scores"],
                fluidics_scores=initial_scoring["fluidics_scores"],
                safety_scores=initial_scoring["safety_scores"],
                is_gas_liquid=is_gas_liquid,
                concentration_M=current.concentration_M,
                pump_max_bar=pump_max,
                batch_time_min=batch_time_min,
                batch_concentration_M=getattr(calc, "concentration_M", None),
                solvent_name=solvent,
                translation_policy=translation_policy,
                max_tau_to_batch_ratio=FLOW_MAX_TAU_TO_BATCH_RATIO,
                intensification_mandate=intensification_mandate,
                pool_metadata=designer_result.get("pool_metadata", {}),
                process_value_scores=initial_scoring.get("process_value_scores", []),
                has_photocatalyst=has_photocatalyst,
            ) if execution.enable_skeptic else _bypassed_skeptic_audit(survivors)
        weak_pool_exhausted = (
            initial_audit.get("verdict") == "WEAK_POOL"
            and weak_pool_cycles >= cfg.MAX_WEAK_POOL_CYCLES
        )
        if weak_pool_exhausted:
            initial_audit["council_may_proceed"] = False
            initial_audit["audit_summary"] += (
                " Maximum WEAK_POOL cycles reached; manual review required."
            )
            block_reason = (
                "Pipeline unable to generate adequate flow design after maximum "
                "WEAK_POOL redesign cycles; manual review required."
            )
            initial_audit.setdefault("all_errors", []).append({
                "agent": "SKEPTIC",
                "candidate_id": None,
                "error_type": "WEAK_POOL_MAX_CYCLES",
                "description": block_reason,
                "severity": "CRITICAL",
            })
            initial_audit["disqualify_ids"] = sorted({int(c.get("id", 0)) for c in survivors})
            initial_audit["disqualify_recommendations"] = [
                {"candidate_id": int(c.get("id", 0)), "reason": block_reason}
                for c in survivors
            ]
        if weak_pool_history:
            initial_audit["weak_pool_history"] = weak_pool_history

        # If CRITICAL errors found, log and continue (v4 selects best available)
        if not initial_audit["council_may_proceed"]:
            logger.warning(
                "  Council v4 — Skeptic found CRITICAL errors: %s. "
                "Proceeding with disqualification of affected candidates.",
                [e["description"][:80] for e in initial_audit["all_errors"]
                 if e.get("severity") == "CRITICAL"]
            )
        _bench_snapshot(benchmark_recorder, "stage3_initial_audit", initial_audit)
        _bench_stage_end(
            benchmark_recorder,
            "council_stage_3_skeptic_audit",
            {
                "council_may_proceed": initial_audit.get("council_may_proceed"),
                "disqualify_ids": initial_audit.get("disqualify_ids", []),
                "error_count": len(initial_audit.get("all_errors", [])),
            },
        )
        if weak_pool_exhausted:
            screen_payload = _build_screen_required_payload(
                current=current,
                calc=calc,
                batch_time_min=batch_time_min,
                intensification_mandate=intensification_mandate,
                solvent=solvent,
                reason="maximum weak-pool redesign cycles exhausted",
                feasibility_diagnostic=designer_result.get(
                    "feasibility_diagnostic"
                ),
            )
            designer_result["screen_required_report"] = screen_payload
            _bench_snapshot(
                benchmark_recorder,
                "stage3_screen_required_weak_pool_exhausted",
                {
                    **screen_payload,
                    "weak_pool_cycles": weak_pool_cycles,
                    "weak_pool_history": weak_pool_history,
                },
            )
            return self._fallback(
                current,
                chemistry_plan,
                calc,
                log,
                designer_result,
                reason=(
                    "screen required: maximum weak-pool redesign cycles "
                    "exhausted"
                ),
                batch_record=batch_record,
                inventory=inventory,
                analogies=analogies,
            )

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 3.5 — Pre-selection expert refinement + rescoring
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Stage 3.5: Pre-selection expert refinement")
        _bench_stage_start(
            benchmark_recorder,
            "council_stage_3_5_preselection_refinement",
            {"allow_warning_refinement": allow_warning_refinement},
        )
        if execution.enable_preselection_refinement:
            revised_survivors, refinement_summary = _run_candidate_refinement_loop(
                candidates=survivors,
                scoring=initial_scoring,
                audit=initial_audit,
                solvent=solvent,
                temperature_C=current.temperature_C,
                concentration_M=current.concentration_M,
                assumed_MW=assumed_MW,
                IF_used=IF_used,
                pump_max_bar=pump_max,
                is_photochem=is_photochem,
                is_gas_liquid=is_gas_liquid,
                extinction_coeff_M_cm=ext_coeff,
                allow_warning_refinement=allow_warning_refinement,
                strong_revision_mode=benchmark_strong_revision_mode,
                branching_revision_mode=benchmark_branching_revision_mode,
                max_descendants_per_candidate=benchmark_max_descendants_per_candidate,
                max_total_revised_candidates=benchmark_max_total_revised_candidates,
                batch_concentration_M=getattr(batch_record, "concentration_M", None) or getattr(calc, "concentration_M", None),
                kinetic_x_minimum=kinetic_x_minimum,
                gas_liquid_max_bpr_bar=gas_liquid_max_bpr,
                pump_max_flow_mL_min=pump_max_flow,
                translation_policy=translation_policy,
            )
        else:
            revised_survivors = survivors
            refinement_summary = {
                "had_changes": False,
                "changed_candidate_ids": [],
                "final_candidate_count": len(survivors),
                "module_disabled": True,
            }
        _bench_snapshot(benchmark_recorder, "stage3_5_refinement_summary", refinement_summary)
        attach_flow_sense_reports(
            revised_survivors,
            batch_time_min=batch_time_min,
            batch_concentration_M=getattr(calc, "concentration_M", None),
            solvent_name=solvent,
            intensification_mandate=intensification_mandate,
        )
        if uncertain_kinetics_screen:
            _downgrade_uncertain_kinetic_hard_gates(revised_survivors)

        refinement_applied = bool(refinement_summary.get("had_changes"))
        if refinement_applied:
            logger.info(
                "  Council v4 — rescoring %d revised candidates after bounded expert edits",
                len(revised_survivors),
            )
            revised_table_markdown = format_candidate_table(
                revised_survivors, max_rows=len(revised_survivors)
            )
            final_scoring = run_domain_scoring(
                candidates=revised_survivors,
                table_markdown=revised_table_markdown,
                chemistry_brief=chem_brief,
                objectives=objectives,
                is_photochem=is_photochem,
                pump_max_bar=pump_max,
                batch_size=scoring_batch_size,
                strict_coverage=benchmark_strict_scoring,
                benchmark_claude_compact_mode=benchmark_claude_compact_mode,
                intensification_mandate=intensification_mandate,
                enabled_agents=execution.enabled_scoring_agents,
            )
            final_audit = run_skeptic_audit(
                candidates=revised_survivors,
                chemistry_scores=final_scoring["chemistry_scores"],
                kinetics_scores=final_scoring["kinetics_scores"],
                fluidics_scores=final_scoring["fluidics_scores"],
                safety_scores=final_scoring["safety_scores"],
                is_gas_liquid=is_gas_liquid,
                concentration_M=current.concentration_M,
                pump_max_bar=pump_max,
                batch_time_min=batch_time_min,
                batch_concentration_M=getattr(calc, "concentration_M", None),
                solvent_name=solvent,
                translation_policy=translation_policy,
                max_tau_to_batch_ratio=FLOW_MAX_TAU_TO_BATCH_RATIO,
                intensification_mandate=intensification_mandate,
                pool_metadata=designer_result.get("pool_metadata", {}),
                process_value_scores=final_scoring.get("process_value_scores", []),
                has_photocatalyst=has_photocatalyst,
            ) if execution.enable_skeptic else _bypassed_skeptic_audit(revised_survivors)
            survivors_for_selection = revised_survivors
            table_for_selection = revised_table_markdown
            _bench_snapshot(benchmark_recorder, "stage3_5_final_scoring", final_scoring)
            _bench_snapshot(benchmark_recorder, "stage3_5_final_audit", final_audit)
        else:
            final_scoring = initial_scoring
            final_audit = initial_audit
            survivors_for_selection = survivors
            table_for_selection = table_markdown
        _bench_stage_end(
            benchmark_recorder,
            "council_stage_3_5_preselection_refinement",
            {
                "refinement_applied": refinement_applied,
                "changed_candidate_ids": refinement_summary.get("changed_candidate_ids", []),
                "final_candidate_count": refinement_summary.get("final_candidate_count"),
                "truncated_to_max_total_revised_candidates": refinement_summary.get(
                    "truncated_to_max_total_revised_candidates", False
                ),
                "dropped_candidate_count": refinement_summary.get("dropped_candidate_count", 0),
            },
        )

        # Retain both hard-gate failures and explicit specialist/audit rejections.
        disqualify_ids, scoring_block_overrides = _resolve_disqualify_ids(
            survivors_for_selection,
            final_scoring,
            final_audit,
        )
        if scoring_block_overrides:
            final_audit["scoring_block_overrides"] = sorted(
                scoring_block_overrides
            )
            final_audit.setdefault("all_errors", []).append({
                "agent": "CHIEF",
                "candidate_id": None,
                "error_type": "ALL_SCORING_BLOCKS_OVERRIDDEN",
                "description": (
                    "All candidates were blocked only by stochastic domain "
                    "verdicts despite passing deterministic audit. Blocks were "
                    "retained as concerns but not used to erase the full pool."
                ),
                "severity": "WARNING",
            })
            logger.warning(
                "Council v4: preserving %d deterministic-audit candidates "
                "because scoring blocks covered the entire eligible pool",
                len(scoring_block_overrides),
            )
        valid_candidate_ids = {
            cid for cid in (int(c.get("id") or 0) for c in survivors_for_selection)
            if cid and cid not in disqualify_ids
        }
        if survivors_for_selection and not valid_candidate_ids:
            current.inventory_constraints["council_selection_block"] = "All candidates were disqualified by specialist review or the deterministic audit."
            logger.warning(
                "Council v4: all %d candidates were disqualified before Chief selection — "
                "requiring experimental screen",
                len(survivors_for_selection),
            )
            screen_payload = _build_screen_required_payload(
                current=current,
                calc=calc,
                batch_time_min=batch_time_min,
                intensification_mandate=intensification_mandate,
                solvent=solvent,
                reason="all candidates disqualified before Chief selection",
                feasibility_diagnostic=designer_result.get("feasibility_diagnostic"),
            )
            designer_result["screen_required_report"] = screen_payload
            _bench_snapshot(
                benchmark_recorder,
                "stage4_screen_required_all_candidates_disqualified",
                {
                    **screen_payload,
                    "disqualify_ids": sorted(disqualify_ids),
                    "candidate_ids": [c.get("id") for c in survivors_for_selection],
                },
            )
            return self._fallback(
                current, chemistry_plan, calc, log, designer_result,
                reason="screen required: all candidates disqualified before Chief selection",
                batch_record=batch_record, inventory=inventory, analogies=analogies,
            )

        # ═══════════════════════════════════════════════════════════════════
        #  CHIEF ENGINEER — Weighted Selection (with optional ask-back)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Chief Engineer: weighted selection")
        _bench_stage_start(benchmark_recorder, "council_stage_4_chief_selection")
        weighted_scores = compute_weighted_scores(
            candidates=survivors_for_selection,
            scoring=final_scoring,
            objectives=objectives,
            disqualify_ids=disqualify_ids,
            batch_time_min=batch_time_min,
            translation_policy=translation_policy,
            max_tau_to_batch_ratio=FLOW_MAX_TAU_TO_BATCH_RATIO,
        )

        # Pass ṅ_limiting from calculator so chief can derive per-pump Q
        n_lim = getattr(calc, "n_molar_flow_mmol_min", None)
        P_batch_val = getattr(calc, "P_batch_mmol_h", None)

        if execution.enable_chief_llm:
            chief_data, chief_tc = _run_chief_llm(
                candidates=survivors_for_selection,
                weighted_scores=weighted_scores,
                scoring=final_scoring,
                audit=final_audit,
                table_markdown=table_for_selection,
                chemistry_brief=chem_brief,
                objectives=objectives,
                disqualify_ids=disqualify_ids,
                n_limiting_mmol_min=n_lim,
                P_batch_mmol_h=P_batch_val,
            )
        else:
            deterministic_id, deterministic_reason = _deterministic_resolve(weighted_scores)
            chief_data = {
                "selected_candidate_id": deterministic_id,
                "selection_rationale": deterministic_reason,
                "selection_flag": "DETERMINISTIC_ABLATION",
                "final_consensus": {},
                "module_disabled": True,
            }
            chief_tc = []

        # Ask-back round: chief may request one expert clarification (max once)
        ask_domain = chief_data.get("ask_expert")
        ask_question = chief_data.get("ask_expert_question")
        if ask_domain and ask_question and ask_domain.lower() in ("kinetics", "chemistry"):
            logger.info(
                "  Chief ask-back: requesting %s expert for: %s",
                ask_domain, str(ask_question)[:100],
            )
            expert_answer = _run_expert_askback(
                domain=ask_domain,
                question=ask_question,
                chemistry_brief=chem_brief,
                candidates=survivors_for_selection,
                scoring=final_scoring,
            )
            chief_data2, chief_tc2 = _run_chief_llm(
                candidates=survivors_for_selection,
                weighted_scores=weighted_scores,
                scoring=final_scoring,
                audit=final_audit,
                table_markdown=table_for_selection,
                chemistry_brief=chem_brief,
                objectives=objectives,
                disqualify_ids=disqualify_ids,
                n_limiting_mmol_min=n_lim,
                P_batch_mmol_h=P_batch_val,
                expert_answer=f"[{ask_domain.upper()} EXPERT] {expert_answer}",
            )
            if chief_data2.get("selected_candidate_id") is not None:
                chief_data = chief_data2
                chief_tc = chief_tc + chief_tc2
                logger.info("  Chief re-ran after ask-back — updated decision applied")

        valid_survivor_ids = {
            c.get("id") for c in survivors_for_selection if c.get("id") not in disqualify_ids
        }
        winner_id = _select_winner_id(chief_data, weighted_scores, valid_survivor_ids)

        if winner_id is None:
            current.inventory_constraints["council_selection_block"] = chief_data.get("selection_justification") or chief_data.get("selection_rationale") or "The council did not approve a candidate."
            _bench_snapshot(benchmark_recorder, "stage4_chief_rejection", chief_data)
            screen_payload = _build_screen_required_payload(
                current=current,
                calc=calc,
                batch_time_min=batch_time_min,
                intensification_mandate=intensification_mandate,
                solvent=solvent,
                reason="Chief could not select a valid candidate",
                feasibility_diagnostic=designer_result.get("feasibility_diagnostic"),
            )
            designer_result["screen_required_report"] = screen_payload
            _bench_snapshot(benchmark_recorder, "stage4_screen_required_no_winner", screen_payload)
            return self._fallback(
                current, chemistry_plan, calc, log, designer_result,
                reason="screen required: Chief could not select a valid candidate",
                batch_record=batch_record, inventory=inventory, analogies=analogies,
            )

        winner = next(c for c in survivors_for_selection if c.get("id") == winner_id)
        _ensure_selection_justification(
            chief_data,
            winner_id=winner_id,
            winner=winner,
            weighted_scores=weighted_scores,
            intensification_mandate=intensification_mandate,
        )
        _bench_snapshot(
            benchmark_recorder,
            "stage4_chief_selection",
            {
                "weighted_scores": weighted_scores,
                "chief_data": chief_data,
                "disqualify_ids": sorted(disqualify_ids),
            },
        )
        _bench_stage_end(
            benchmark_recorder,
            "council_stage_4_chief_selection",
            {"winner_id": winner_id, "disqualify_ids": sorted(disqualify_ids)},
        )
        logger.info(
            "  Council v4 — Winner: id=%d τ=%s min, d=%s mm, Q=%s mL/min, X=%s",
            winner_id, winner.get("tau_min"), winner.get("d_mm"),
            winner.get("Q_mL_min"), winner.get("expected_conversion"),
        )

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 3.5 — Revision Agent: fix REVISE/BLOCK issues on winner
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Stage 3.5: Revision Agent on winner id=%d", winner_id)
        _bench_stage_start(benchmark_recorder, "council_stage_5_revision_agent", {"winner_id": winner_id})
        # Batch yield as a fraction (e.g., 0.75 for 75% reported yield) —
        # used to cap the Revision Engineer's τ bump to a realistic X target.
        _batch_yield_pct = getattr(batch_record, "yield_pct", None) or 0.0
        _batch_yield_fraction = (float(_batch_yield_pct) / 100.0) if _batch_yield_pct else None
        revision_result = run_revision_stage(
            winner           = winner,
            scoring          = final_scoring,
            chemistry_brief  = chem_brief,
            is_photochem     = is_photochem,
            is_gas_liquid    = is_gas_liquid,
            pump_max_bar     = pump_max,
            solvent          = solvent,
            temperature_C    = current.temperature_C,
            concentration_M  = current.concentration_M,
            extinction_coeff_M_cm = ext_coeff,
            batch_yield_fraction = _batch_yield_fraction,
            batch_time_min   = batch_time_min,
            translation_policy = translation_policy,
            measured_evidence_available = bool(evidence_calibration),
            measured_tau_floor_min = (
                _positive_float(evidence_calibration.get("best_tau_min"))
                if evidence_calibration
                else None
            ),
            pump_max_flow_mL_min = pump_max_flow,
            gas_liquid_max_bpr_bar = gas_liquid_max_bpr,
        ) if execution.enable_winner_revision else None
        if revision_result is not None:
            if winner.get("inventory_realized"):
                from flora_translate.engine.council_v4.inventory_candidates import materialize
                patch = {k: revision_result[k] for k in ["tau_min", "Q_mL_min", "d_mm", "V_R_mL", "BPR_bar", "temperature_C", "concentration_M", "tubing_material"] if k in revision_result and revision_result[k] != winner.get(k)}
                try:
                    revision_result = materialize(winner["_realization_context"], patch=patch, candidate_id=winner_id)
                except ValueError as exc:
                    _bench_snapshot(benchmark_recorder, "stage5_inventory_revision_rejected", {"patch": patch, "reason": str(exc)})
                    revision_result = None
        if revision_result is not None and candidate_fingerprint(revision_result) != candidate_fingerprint(winner):
            # A post-selection edit is a new operating point. It cannot inherit
            # the old review or be called approved merely because arithmetic closes.
            _apply_v4_hard_gates([revision_result], pump_max_bar=pump_max,
                is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
                BPR_bar=revision_result.get("BPR_bar", 0), X_minimum=kinetic_x_minimum,
                tubing_material=revision_result.get("tubing_material", "FEP"))
            review_scoring = run_domain_scoring(candidates=[revision_result],
                table_markdown=format_candidate_table([revision_result]), chemistry_brief=chem_brief,
                objectives=objectives, is_photochem=is_photochem, pump_max_bar=pump_max,
                strict_coverage=True, intensification_mandate=intensification_mandate,
                enabled_agents=execution.enabled_scoring_agents)
            review_audit = run_skeptic_audit(candidates=[revision_result],
                chemistry_scores=review_scoring["chemistry_scores"], kinetics_scores=review_scoring["kinetics_scores"],
                fluidics_scores=review_scoring["fluidics_scores"], safety_scores=review_scoring["safety_scores"],
                is_gas_liquid=is_gas_liquid, concentration_M=revision_result.get("concentration_M", current.concentration_M),
                pump_max_bar=pump_max, batch_time_min=batch_time_min, solvent_name=solvent,
                translation_policy=translation_policy, intensification_mandate=intensification_mandate,
                has_photocatalyst=has_photocatalyst) if execution.enable_skeptic else _bypassed_skeptic_audit([revision_result])
            rejected, _ = _resolve_disqualify_ids([revision_result], review_scoring, review_audit)
            review_weights = compute_weighted_scores(candidates=[revision_result], scoring=review_scoring,
                objectives=objectives, disqualify_ids=rejected, batch_time_min=batch_time_min,
                translation_policy=translation_policy)
            review_chief, review_calls = ({"selected_candidate_id": None}, [])
            if not rejected:
                if execution.enable_chief_llm:
                    review_chief, review_calls = _run_chief_llm(candidates=[revision_result],
                        weighted_scores=review_weights, scoring=review_scoring, chemistry_brief=chem_brief,
                        objectives=objectives, audit=review_audit, disqualify_ids=rejected,
                        table_markdown=format_candidate_table([revision_result]))
                else:
                    review_chief = {"selected_candidate_id": winner_id, "selection_rationale": "Revised candidate passed the configured deterministic selection and domain review."}
            _bench_snapshot(benchmark_recorder, "stage5_revision_review", {
                "scoring": review_scoring, "audit": review_audit, "chief": review_chief,
                "disqualified_ids": sorted(rejected)})
            # Renewed approval must be explicit; an absent/malformed Chief reply
            # cannot be replaced by the original selection's weighted ranking.
            if rejected or review_chief.get("selected_candidate_id") != winner_id:
                current.inventory_constraints["council_selection_block"] = "Post-selection revision did not receive renewed approval."
                return self._fallback(current, chemistry_plan, calc, log, designer_result,
                    reason="screen required: revised candidate rejected on renewed review",
                    batch_record=batch_record, inventory=inventory, analogies=analogies)
            chief_tc.extend(review_calls)
            chief_data = review_chief
            for domain in ("chemistry", "kinetics", "fluidics", "safety"):
                key = domain + "_scores"
                final_scoring[key] = [e for e in final_scoring[key] if e.get("candidate_id") != winner_id] + review_scoring[key]
            final_audit = review_audit
            weighted_scores = [r for r in weighted_scores if r.get("candidate_id") != winner_id] + review_weights
            survivors_for_selection = [revision_result if c.get("id") == winner_id else c
                                       for c in survivors_for_selection]
        if revision_result is not None:
            winner_before_revision = winner
            winner = revision_result
            for key in ("batch_time_min", "temperature_C", "concentration_M"):
                if winner.get(key) is None and winner_before_revision.get(key) is not None:
                    winner[key] = winner_before_revision[key]
            if uncertain_kinetics_screen and not winner.get("inventory_realized"):
                tau_ceiling = _positive_float(
                    (feasibility_diagnostic or {}).get("tau_intensification_ceiling_min")
                )
                if tau_ceiling > 0 and _positive_float(winner.get("tau_min")) > tau_ceiling:
                    winner["screen_repair_note"] = (
                        f"Revision tau {_positive_float(winner.get('tau_min')):.3g} min exceeded "
                        f"uncertain-kinetics screen ceiling {tau_ceiling:.3g} min; the reviewed setting is retained without a post-review mutation."
                    )
            logger.info(
                "  Stage 3.5 applied revisions to winner id=%d: τ→%.1f min, d→%.2f mm, BPR→%.2f bar",
                winner_id, winner.get("tau_min", 0), winner.get("d_mm", 0), winner.get("BPR_bar", 0),
            )
            _refresh_selection_justification_from_winner(
                chief_data,
                winner_id=winner_id,
                winner=winner,
                weighted_scores=weighted_scores,
                intensification_mandate=intensification_mandate,
            )
        revision_snapshot = revision_result if revision_result is not None else {
            "revision_applied": False,
            "module_disabled": not execution.enable_winner_revision,
        }
        _bench_snapshot(benchmark_recorder, "stage5_revision_result", revision_snapshot)
        _bench_stage_end(
            benchmark_recorder,
            "council_stage_5_revision_agent",
            {
                "winner_id": winner_id,
                "revision_applied": revision_result is not None,
                "module_disabled": not execution.enable_winner_revision,
            },
        )

        # ═══════════════════════════════════════════════════════════════════
        #  DFMEA on winner
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — DFMEA on winning candidate id=%d", winner_id)
        _bench_stage_start(benchmark_recorder, "council_stage_6_dfmea", {"winner_id": winner_id})
        safety_entry = next(
            (e for e in final_scoring["safety_scores"] if e.get("candidate_id") == winner_id),
            None,
        )
        dfmea = run_dfmea(
            winner_id=winner_id,
            winner=winner,
            chemistry_brief=chem_brief,
            safety_score_entry=safety_entry,
        ) if execution.enable_dfmea else {
            "failure_modes": [],
            "single_points_of_failure": [],
            "validation_experiments": [],
            "module_disabled": True,
        }
        _bench_snapshot(benchmark_recorder, "stage6_dfmea", dfmea)
        _bench_stage_end(
            benchmark_recorder,
            "council_stage_6_dfmea",
            {"winner_id": winner_id, "failure_mode_count": len(dfmea.get("failure_modes", []))},
        )

        # ═══════════════════════════════════════════════════════════════════
        #  Apply winner + re-run calculator
        # ═══════════════════════════════════════════════════════════════════
        _bench_stage_start(benchmark_recorder, "council_stage_7_apply_winner", {"winner_id": winner_id})
        final_consensus = chief_data.get("final_consensus") or {}
        extra_changes: dict = {}
        for field_map in [
            ("BPR_bar", "BPR_bar"),
            ("material", "tubing_material"),
        ]:
            fc_key, patch_key = field_map
            if fc_key in final_consensus:
                extra_changes[patch_key] = str(final_consensus[fc_key])

        # Revision Stage 3.5: also apply BPR and material from revised winner
        # (these are not in chief final_consensus, but were set by revision agent)
        # Revision values are deterministic engineering patches and must win over
        # Chief LLM final_consensus when they conflict.
        if revision_result is not None:
            if revision_result.get("BPR_bar") is not None:
                extra_changes["BPR_bar"] = str(revision_result["BPR_bar"])
            if revision_result.get("tubing_material"):
                extra_changes["tubing_material"] = revision_result["tubing_material"]

        current, changes = _apply_winner(
            current, winner, extra_changes, chief_data=chief_data,
        )
        log.all_changes_applied.update(changes)

        calc = DesignCalculator().run(
            batch_record, chemistry_plan, current, inventory,
            analogies=analogies,
            target_flow_rate_mL_min=current.flow_rate_mL_min or None,
            target_tubing_ID_mm=current.tubing_ID_mm or None,
            target_residence_time_min=current.residence_time_min or None,
        )
        if not winner.get("inventory_realized"):
            current = DesignCalculator.annotate_proposal_with_calculations(current, calc)
        _bench_stage_end(
            benchmark_recorder,
            "council_stage_7_apply_winner",
            {
                "winner_id": winner_id,
                "applied_changes": changes,
                "final_residence_time_min": current.residence_time_min,
                "final_flow_rate_mL_min": current.flow_rate_mL_min,
                "final_tubing_ID_mm": current.tubing_ID_mm,
            },
        )

        # ── Build output ────────────────────────────────────────────────────
        tool_calls = {
            "chief": chief_tc,
            **final_scoring.get("tool_calls", {}),
        }
        all_round_delibs = _to_deliberations_v4(
            designer_result=designer_result,
            scoring=final_scoring,
            audit=final_audit,
            weighted_scores=weighted_scores,
            winner_id=winner_id,
            candidates=survivors_for_selection,
            dfmea=dfmea,
            chief_data=chief_data,
            tool_calls=tool_calls,
            revision_result=revision_result,
            preselection_refinement=refinement_summary,
        )
        log.rounds = all_round_delibs

        # Skeptic trade-off summary
        log.trade_off_summary = final_audit.get("audit_summary", "")
        log.trade_off_matrix = (
            "| id | combined | chemistry | kinetics | fluidics | safety | geometry |\n"
            "|---|---|---|---|---|---|---|\n" +
            "\n".join(
                f"| {r['candidate_id']} | {_positive_float(r.get('combined')):.3f} | "
                f"{_positive_float(r.get('chemistry')):.3f} | {_positive_float(r.get('kinetics')):.3f} | "
                f"{_positive_float(r.get('fluidics')):.3f} | {_positive_float(r.get('safety')):.3f} | "
                f"{_positive_float(r.get('geometry')):.3f} |"
                for r in weighted_scores
            )
        )
        log.total_rounds = len(all_round_delibs)
        log.consensus_reached = final_audit["council_may_proceed"]

        summary_candidates = [
            winner if c.get("id") == winner_id else c
            for c in survivors_for_selection
        ]
        log.summary = _build_summary_v4(
            designer_result=designer_result,
            scoring=final_scoring,
            audit=final_audit,
            weighted_scores=weighted_scores,
            winner_id=winner_id,
            candidates=summary_candidates,
            chief_data=chief_data,
            dfmea=dfmea,
            objectives=objectives,
            preselection_refinement=refinement_summary,
        )

        legacy = [
            CouncilMessage(
                agent=d.agent, status=d.status, field="design",
                value=d.chain_of_thought[:200],
                concern=(d.concerns[0] if d.concerns else ""),
                revision_required=(d.status == "REVISE"),
            )
            for rnd in all_round_delibs
            for d in rnd
        ]

        uncertain_kinetics = not validated_model(winner, "kinetics") or bool(uncertain_kinetics_screen) or (
            (designer_result.get("feasibility_diagnostic") or {}).get("status")
            == "KINETIC_ANCHOR_UNCERTAIN_SCREEN_REQUIRED"
        )
        current.engine_validated = not uncertain_kinetics
        current.confidence = "LOW" if uncertain_kinetics else ("HIGH" if winner.get("pareto_front") else "MEDIUM")
        if uncertain_kinetics:
            current.safety_flags.append("SCREEN_REQUIRED: kinetic anchor uncertain; selected point is a screening hypothesis")
            current.chemistry_notes = (
                (current.chemistry_notes or "").rstrip()
                + "\nSelected point is a screening hypothesis because the kinetic model is unvalidated at these conditions; measure conversion and transport before claiming process performance."
            ).strip()

        return DesignCandidate(
            proposal=current,
            chemistry_plan=chemistry_plan,
            council_messages=legacy,
            council_rounds=log.total_rounds,
            safety_report={
                "total_checks": len(final_audit.get("all_errors", [])),
                "critical": len([e for e in final_audit.get("all_errors", [])
                                 if e.get("severity") == "CRITICAL"]),
                "dfmea_modes": len(dfmea.get("failure_modes", [])) if dfmea else 0,
                "validation_experiments": dfmea.get("validation_experiments", []) if dfmea else [],
                "screen_required": uncertain_kinetics,
                "feasibility_diagnostic": designer_result.get("feasibility_diagnostic") or {},
            },
            unit_operations=[],
            pid_description="",
            deliberation_log=log,
        ), calc

    def _fallback(
        self,
        current: FlowProposal,
        chemistry_plan: Optional[ChemistryPlan],
        calc: DesignCalculations,
        log: DeliberationLog,
        designer_result: dict,
        reason: str = "no usable candidates after filtering",
        batch_record: Optional[BatchRecord] = None,
        inventory: Optional[LabInventory] = None,
        analogies: Optional[list[dict]] = None,
    ) -> tuple[DesignCandidate, DesignCalculations]:
        n_surv = len(designer_result.get("survivors", []))
        n_disq = len(designer_result.get("disqualified", []))
        diagnostic = designer_result.get("feasibility_diagnostic") or {}
        screen_report = designer_result.get("screen_required_report") or {}
        screen_required = bool(screen_report.get("screen_required"))
        log.summary = (
            f"Council v4 fallback: {reason}. "
            f"Designer: {n_surv} survivors, {n_disq} hard-gate disqualified. "
            "Keeping calculator center-point."
        )
        if diagnostic:
            log.summary += " " + str(diagnostic.get("diagnosis", ""))
        if screen_required:
            log.summary += " Experimental screen required before final design selection."
        current.engine_validated = False
        current.safety_flags.append(f"ENGINE_FALLBACK: {reason}")
        if screen_required:
            current.safety_flags.append(f"SCREEN_REQUIRED: {screen_report.get('screen_reason') or reason}")
            current.chemistry_notes = (
                (current.chemistry_notes or "").rstrip()
                + "\nNo engine-validated final design was selected. Run the attached screen matrix first."
            ).strip()
        if batch_record is not None and inventory is not None:
            calc = DesignCalculator().run(
                batch_record, chemistry_plan, current, inventory,
                analogies=analogies or [],
                target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                target_tubing_ID_mm=current.tubing_ID_mm or None,
                target_residence_time_min=current.residence_time_min or None,
            )
        current = DesignCalculator.annotate_proposal_with_calculations(current, calc)
        return DesignCandidate(
            proposal=current,
            chemistry_plan=chemistry_plan,
            council_messages=[],
            council_rounds=0,
            safety_report={
                "fallback_reason": reason,
                "feasibility_diagnostic": diagnostic,
                "screen_required": screen_required,
                "screen_reason": screen_report.get("screen_reason") if screen_required else None,
                "screen_candidates": screen_report.get("screen_candidates", []) if screen_required else [],
                "screen_acceptance_criteria": (
                    screen_report.get("screen_acceptance_criteria", []) if screen_required else []
                ),
            },
            unit_operations=[],
            pid_description="",
            deliberation_log=log,
        ), calc
