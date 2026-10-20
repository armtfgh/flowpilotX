"""
FLORA ENGINE v3 — Chief Engineer orchestrator.

Flow:
  1. Extract center-point + chemistry context from the upstream calculator.
  2. Designer → sampling strategy + candidate shortlist.
  3. Expert panel round 1 → 4 specialist picks.
  4. Skeptic round 1 → assumption attacks + code-block rejections.
  5. Expert panel round 2 → specialists see Skeptic's attacks; may switch or defend.
  6. Skeptic round 2 → final attacks.
  7. Chief resolves against user-stated objectives → winning candidate + FlowProposal patch.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Optional

from flora_translate.config import ENGINE_PROVIDER
from flora_translate.design_calculator import DesignCalculator, DesignCalculations
from flora_translate.engine.llm_agents import call_llm, call_llm_with_tools
from flora_translate.engine.tool_definitions import CHIEF_TOOLS, execute_tool
from flora_translate.engine.council_v3.designer import run_designer
from flora_translate.engine.council_v3.expert import run_expert_panel
from flora_translate.engine.council_v3.skeptic import run_skeptic, sanity_code_check
from flora_translate.schemas import (
    BatchRecord, ChemistryPlan, FlowProposal, LabInventory,
    DesignCandidate, CouncilMessage, DeliberationLog,
    AgentDeliberation, FieldProposal,
)

logger = logging.getLogger("flora.engine.council_v3.chief")


# ═══════════════════════════════════════════════════════════════════════════════
#  Chief's conflict-resolution prompt
# ═══════════════════════════════════════════════════════════════════════════════

_CHIEF_SYSTEM = """\
You are the **Chief Engineer** of the FLORA ENGINE council. You are the senior
process engineer who closes the debate and picks the winning design.

You DO NOT re-derive numbers and you DO NOT second-guess the specialists'
domain judgement within their domains. Your job is exactly one thing:
**weigh specialist disagreement against the user's stated objectives
and pick ONE candidate.**

## How to resolve
1. Start from the surviving picks (those not blocked by Skeptic code rules).
2. For each, read: the Expert's reasoning for it + the Skeptic's outstanding
   attacks on it.
3. Map to the user objectives:
   • **de-risk first-run** → favor the pick with highest safety margins
     (ΔP headroom, r_mix, X well above threshold), accept lower productivity.
   • **yield-oriented** → favor τ with comfortable margin over τ_kinetics
     and clear mechanism fit (Kinetics/Chemistry picks).
   • **throughput-oriented** → favor higher productivity_mg_h with acceptable
     (not minimal) margins.
   • **balanced / unstated** → favor the Pareto-front pick with the fewest
     Skeptic HIGH-severity attacks.
4. When two picks remain roughly equal, break the tie by:
   (a) Skeptic severity count first (fewer HIGH attacks wins)
   (b) Pareto-front membership second
   (c) Kinetics advocate's pick third (τ-anchor is the hardest physical
       constraint in photoredox/thermal)

## REQUIRED OUTPUT — JSON ONLY
```json
{
  "winning_pick_id": 3,
  "winner_reasoning": "2-4 sentences citing objectives, specialist picks, skeptic attacks.",
  "open_risks": ["risk that chemist should watch for in the lab"],
  "changes_to_apply": {
     "residence_time_min": "45.0",
     "flow_rate_mL_min": "0.55",
     "tubing_ID_mm": "0.75"
  }
}
```

Only fields present in FlowProposal can be in `changes_to_apply`. Allowed:
residence_time_min, flow_rate_mL_min, tubing_ID_mm, BPR_bar, temperature_C,
concentration_M, tubing_material, wavelength_nm, deoxygenation_method,
mixer_type, reactor_volume_mL.

All values in changes_to_apply must be strings.

## Tool available
Use `compute_design_envelope` on each surviving pick to assess its operating window width
before choosing. A wider envelope means more robustness to lab variability (pump drift,
temperature fluctuation). Call it on 2-3 candidates, then decide.
"""


def _parse_chief(raw: str) -> dict:
    s = raw.strip()
    if "```" in s:
        for part in s.split("```")[1::2]:
            cleaned = part.lstrip("json").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(s)
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
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  Deterministic fallback resolver (used if LLM fails)
# ═══════════════════════════════════════════════════════════════════════════════

def _deterministic_resolve(
    expert_picks: list[dict],
    skeptic: dict,
    candidates: list[dict],
) -> tuple[Optional[int], str]:
    """Majority vote over non-blocked picks, tiebroken by attack count."""
    blocked = set(skeptic.get("blocked_picks", []))
    votes = Counter(
        p["pick_id"] for p in expert_picks
        if p.get("pick_id") is not None and p["pick_id"] not in blocked
    )
    if not votes:
        return None, "No surviving picks — all blocked by Skeptic code rules."

    # Attack severity per candidate
    high_attacks: Counter = Counter()
    for a in skeptic.get("llm_attacks", []):
        if a.get("severity") == "HIGH":
            high_attacks[a.get("pick_id")] += 1

    # Sort: most votes desc, fewest HIGH attacks asc, Pareto first, id asc
    def rank(pid: int) -> tuple:
        return (
            -votes[pid],
            high_attacks[pid],
            0 if candidates[pid - 1].get("pareto_front") else 1,
            pid,
        )

    winner = sorted(votes, key=rank)[0]
    reason = (
        f"Majority vote: {dict(votes)}. "
        f"Winner id={winner} with {high_attacks[winner]} HIGH attacks "
        f"(Pareto={'yes' if candidates[winner-1].get('pareto_front') else 'no'})."
    )
    return winner, reason


def _build_tradeoff_matrix(
    expert_picks: list[dict],
    skeptic: dict,
    candidates: list[dict],
    pump_max_bar: float,
) -> str:
    """Build a concise comparison table of surviving picks for the Chief."""
    blocked = set(skeptic.get("blocked_picks", []))
    high_per_id: Counter = Counter(
        a["pick_id"] for a in skeptic.get("llm_attacks", [])
        if a.get("severity") == "HIGH"
    )
    # Collect unique surviving pick ids, preserving order
    seen: set = set()
    surviving_ids: list[int] = []
    for p in expert_picks:
        pid = p.get("pick_id")
        if pid is not None and pid not in blocked and pid not in seen:
            surviving_ids.append(pid)
            seen.add(pid)

    if not surviving_ids:
        return "_All picks blocked._"

    # Advocates per id
    advocates: dict[int, list[str]] = {}
    for p in expert_picks:
        pid = p.get("pick_id")
        if pid in seen:
            advocates.setdefault(pid, []).append(p.get("domain", "?"))

    header = "| id | Advocates | τ (min) | Q (mL/min) | d (mm) | X | ΔP (bar) | ΔP head% | r_mix | L (m) | Pareto | HIGH attacks |"
    sep    = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows   = [header, sep]
    for pid in surviving_ids:
        c = candidates[pid - 1]
        dP = c.get("delta_P_bar", 0.0)
        head_pct = round(100 * (1 - dP / pump_max_bar), 1) if pump_max_bar > 0 else "?"
        rows.append(
            f"| {pid} | {'+'.join(advocates.get(pid, ['?']))} "
            f"| {c.get('tau_min','?')} | {c.get('Q_mL_min','?')} "
            f"| {c.get('d_mm','?')} | {c.get('expected_conversion',0):.2f} "
            f"| {dP:.3f} | {head_pct}% "
            f"| {c.get('r_mix',0):.3f} | {c.get('L_m','?')} "
            f"| {'★' if c.get('pareto_front') else ''} "
            f"| {high_per_id.get(pid, 0)} |"
        )
    return "\n".join(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Chief LLM call
# ═══════════════════════════════════════════════════════════════════════════════

def _run_chief_llm(
    expert_picks: list[dict],
    skeptic: dict,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    trade_off_matrix: str = "",
    max_tokens: int = 1000,
) -> tuple[dict, list[dict]]:
    """Run the Chief LLM call.

    Returns (chief_data_dict, tool_calls_log).
    """
    blocked = set(skeptic.get("blocked_picks", []))
    surviving = [p for p in expert_picks
                 if p.get("pick_id") is not None and p["pick_id"] not in blocked]

    picks_str = "\n".join(
        f"- {p['domain']} → id {p['pick_id']}: {p.get('pick_reason','')[:200]}"
        for p in surviving
    ) or "_none surviving_"

    attacks_str = "\n".join(
        f"- id {a['pick_id']} ({a['severity']}): {a['assumption_at_risk']} "
        f"[mitigation: {a.get('mitigation','')}]"
        for a in skeptic.get("llm_attacks", [])
    ) or "_none_"

    blocked_str = "\n".join(f"- id {pid}" for pid in sorted(blocked)) or "_none_"

    context = (
        "## User objectives\n" + (objectives or "balanced (no explicit preference stated)") +
        "\n\n## Chemistry brief\n" + chemistry_brief +
        "\n\n## Full candidate table\n" + table_markdown +
        "\n\n## Surviving picks — comparison matrix\n" + (trade_off_matrix or "_not available_") +
        "\n\n## Surviving Expert picks (with reasoning)\n" + picks_str +
        "\n\n## Skeptic's outstanding attacks\n" + attacks_str +
        "\n\n## Blocked by hard code rules (not available)\n" + blocked_str +
        "\n\nChoose ONE winner. Output JSON only."
    )

    try:
        raw, tool_calls_log = call_llm_with_tools(
            _CHIEF_SYSTEM, context,
            tools=CHIEF_TOOLS,
            tool_executor=execute_tool,
            max_tokens=max_tokens,
            max_tool_turns=4,
        )
        return _parse_chief(raw), tool_calls_log
    except Exception as e:
        logger.warning("Chief LLM call failed: %s", e)
        return {}, []


# ═══════════════════════════════════════════════════════════════════════════════
#  Extract chemistry brief + extinction coeff from upstream state
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
    """Try to pull ε from the batch record or chemistry plan. Return None if unknown."""
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
#  Build AgentDeliberation records for UI compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def _to_deliberations(
    designer: dict,
    expert_picks_rounds: list[list[dict]],
    skeptic_rounds: list[dict],
    winner_id: Optional[int],
    candidates: list[dict],
    tool_calls_by_agent: Optional[dict] = None,
) -> list[list[AgentDeliberation]]:
    """Convert council_v3 output into per-round AgentDeliberation lists for UI.

    Returns a list of rounds, each round is a list of AgentDeliberation.
    Round 1: Designer + Expert round 1 + Skeptic round 1
    Round 2 (if exists): Expert round 2 + Skeptic round 2
    Final: Chief Engineer (appended to last round)
    """
    if tool_calls_by_agent is None:
        tool_calls_by_agent = {}

    all_rounds: list[list[AgentDeliberation]] = []
    n_expert_rounds = len(expert_picks_rounds)

    for rnd_idx in range(n_expert_rounds):
        rnd_num = rnd_idx + 1
        round_delibs: list[AgentDeliberation] = []

        # Designer only in round 1
        if rnd_idx == 0:
            n_feasible = len(designer.get("feasible", []))
            n_infeasible = len(designer.get("infeasible", []))
            round_delibs.append(AgentDeliberation(
                agent="DesignerV3", agent_display_name="Designer",
                round=1,
                chain_of_thought=(
                    f"Sampled {n_feasible + n_infeasible} candidates; "
                    f"{n_feasible} feasible after hard filter. "
                    f"Strategy: {designer.get('strategy_reasoning', '')}"
                ),
                findings=[
                    f"Strategy: τ ∈ [{designer['strategy']['tau_low_factor']:.2f}×, "
                    f"{designer['strategy']['tau_high_factor']:.2f}×], "
                    f"n_tau={designer['strategy']['n_tau']}, "
                    f"d≤{designer['strategy']['d_exclude_above_mm']} mm"
                ],
                status="ACCEPT",
            ))

        # Expert picks for this round
        for p in expert_picks_rounds[rnd_idx]:
            domain = p.get("domain", "UNKNOWN")
            status = "ACCEPT" if p.get("status") == "OK" else "WARNING"
            key = f"expert_{domain}_r{rnd_num}"
            tc = tool_calls_by_agent.get(key, p.get("tool_calls", []))
            round_delibs.append(AgentDeliberation(
                agent=f"Expert_{domain}_V3",
                agent_display_name=p["specialist_name"],
                round=rnd_num,
                chain_of_thought=p.get("pick_reason", ""),
                findings=[f"Advocates candidate id={p.get('pick_id')}"],
                concerns=p.get("concerns_on_other_picks", []),
                status=status,
                tool_calls=tc,
            ))

        # Skeptic for this round
        if rnd_idx < len(skeptic_rounds):
            sk = skeptic_rounds[rnd_idx]
            attacks = sk.get("llm_attacks", [])
            blocks = sk.get("blocked_picks", [])
            cot_lines = [f"Skeptic summary: {sk.get('summary', '')}"]
            for a in attacks[:8]:
                cot_lines.append(
                    f"  • id {a['pick_id']} ({a['severity']}): {a['assumption_at_risk']}"
                )
            if blocks:
                cot_lines.append(f"  • Blocked by code: {blocks}")
            sk_status = "REVISE" if blocks else ("WARNING" if attacks else "ACCEPT")
            findings_list = [f"{len(attacks)} attacks, {len(blocks)} hard blocks"]
            ts = sk.get("trade_off_summary", "")
            if ts:
                findings_list.append(f"Trade-off: {ts}")
            key = f"skeptic_r{rnd_num}"
            tc = tool_calls_by_agent.get(key, sk.get("tool_calls", []))
            round_delibs.append(AgentDeliberation(
                agent="SkepticV3", agent_display_name="Skeptic",
                round=rnd_num,
                chain_of_thought="\n".join(cot_lines),
                findings=findings_list,
                concerns=[f"id {a['pick_id']}: {a['assumption_at_risk']}" for a in attacks],
                status=sk_status,
                tool_calls=tc,
            ))

        all_rounds.append(round_delibs)

    # Chief goes in the last round
    if winner_id is not None and 1 <= winner_id <= len(candidates):
        w = candidates[winner_id - 1]
        chief_tc = tool_calls_by_agent.get("chief", [])
        chief_delib = AgentDeliberation(
            agent="ChiefV3", agent_display_name="Chief Engineer",
            round=n_expert_rounds,
            chain_of_thought=(
                f"Winner: candidate id={winner_id} — "
                f"τ={w['tau_min']} min, d={w['d_mm']} mm, Q={w['Q_mL_min']} mL/min, "
                f"X={w['expected_conversion']:.2f}, prod={w['productivity_mg_h']:.1f} mg/h."
            ),
            findings=[f"Final design id={winner_id}"],
            status="ACCEPT",
            tool_calls=chief_tc,
        )
        if all_rounds:
            all_rounds[-1].append(chief_delib)
        else:
            all_rounds.append([chief_delib])

    return all_rounds


# ═══════════════════════════════════════════════════════════════════════════════
#  Apply winning candidate to FlowProposal
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
    extra_changes: dict[str, str],
) -> tuple[FlowProposal, dict[str, str]]:
    """Apply winning candidate's (τ, d, Q, V_R) to FlowProposal and merge LLM patches."""
    changes: dict[str, str] = {
        "residence_time_min": str(round(winner["tau_min"], 2)),
        "flow_rate_mL_min":    str(round(winner["Q_mL_min"], 4)),
        "tubing_ID_mm":        str(round(winner["d_mm"], 3)),
        "reactor_volume_mL":   str(round(winner["V_R_mL"], 3)),
    }
    # Chief's LLM-proposed overrides (for fields we don't set from candidate)
    for k, v in (extra_changes or {}).items():
        if k in _ALLOWED_PATCH_FIELDS and k not in changes:
            try:
                # Coerce to float if the target is float — keep string otherwise
                current = proposal.model_dump().get(k)
                if isinstance(current, (int, float)):
                    float(v)  # validate
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

    return FlowProposal(**data), changes


# ═══════════════════════════════════════════════════════════════════════════════
#  Main orchestrator class
# ═══════════════════════════════════════════════════════════════════════════════

class CouncilV3:
    """Advocacy-based council orchestrator.

    Usage:
        result, calc = CouncilV3().run(proposal, batch, analogies, inventory,
                                        chemistry_plan, calculations,
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
        max_loop_rounds: int = 2,
    ) -> tuple[DesignCandidate, DesignCalculations]:

        current = proposal.model_copy(deep=True)
        log = DeliberationLog()

        # ── Ensure calculator center-point ──────────────────────────────────
        if calculations is None:
            calculations = DesignCalculator().run(
                batch_record, chemistry_plan, current, inventory,
                analogies=analogies,
                target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                target_tubing_ID_mm=current.tubing_ID_mm or None,
            )
        calc = calculations

        # ── Extract context for Designer ────────────────────────────────────
        is_photochem = bool(
            (chemistry_plan and "photo" in (chemistry_plan.reaction_class or "").lower()) or
            (current.wavelength_nm and current.wavelength_nm > 0)
        )
        is_gas_liquid = bool(getattr(calc, "is_gas_liquid", False))
        is_O2_sensitive = bool(chemistry_plan and chemistry_plan.oxygen_sensitive)

        solvent = (
            chemistry_plan.stages[0].solvent
            if chemistry_plan and chemistry_plan.stages
            else (current.streams[0].solvent if current.streams else "EtOH")
        )

        tau_center = calc.residence_time_min or current.residence_time_min or 30.0
        tau_lit = getattr(calc, "tau_analogy_min", None) or getattr(calc, "tau_class_min", None)
        tau_kinetics = getattr(calc, "tau_kinetics_min", None) or tau_center
        IF_used = calc.intensification_factor or 6.0
        assumed_MW = getattr(batch_record, "product_MW", None) or 250.0
        pump_max = calc.pump_max_bar or 20.0
        ext_coeff = _extract_extinction_coeff(batch_record, chemistry_plan)

        chem_brief = _build_chemistry_brief(batch_record, chemistry_plan, current)

        # ═══════════════════════════════════════════════════════════════════
        #  STEP 1 — Designer
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v3 — Step 1: Designer (sampling strategy + candidates)")
        designer = run_designer(
            reaction_class=(chemistry_plan.reaction_class if chemistry_plan else "unknown"),
            is_photochem=is_photochem,
            is_gas_liquid=is_gas_liquid,
            is_O2_sensitive=is_O2_sensitive,
            tau_center_min=tau_center,
            tau_lit_min=tau_lit,
            tau_kinetics_min=tau_kinetics,
            d_center_mm=current.tubing_ID_mm or 1.0,
            Q_center_mL_min=current.flow_rate_mL_min or 1.0,
            solvent=solvent,
            temperature_C=current.temperature_C,
            concentration_M=current.concentration_M,
            assumed_MW=assumed_MW,
            IF_used=IF_used,
            pump_max_bar=pump_max,
            BPR_bar=current.BPR_bar or 0.0,
            extinction_coeff_M_cm=ext_coeff,
            N_target=12,
        )
        candidates = designer["feasible"]
        if not candidates:
            logger.warning("Council v3: no feasible candidates from Designer — falling back to center")
            return self._fallback(current, chemistry_plan, calc, log, designer)

        # ═══════════════════════════════════════════════════════════════════
        #  STEPS 2-5 — Expert ⇄ Skeptic loop (max 2 rounds)
        # ═══════════════════════════════════════════════════════════════════
        expert_rounds: list[list[dict]] = []
        skeptic_rounds: list[dict] = []
        skeptic_notes: list[str] = []
        peer_concerns_for_next: list[dict] = []
        tool_calls_by_agent: dict = {}

        for rnd in range(1, max_loop_rounds + 1):
            logger.info("  Council v3 — Round %d/%d: Expert panel", rnd, max_loop_rounds)
            picks = run_expert_panel(
                candidates=candidates,
                table_markdown=designer["table_markdown"],
                chemistry_brief=chem_brief,
                objectives=objectives,
                is_photochem=is_photochem,
                prior_picks=expert_rounds[-1] if expert_rounds else None,
                skeptic_notes=skeptic_notes or None,
                peer_concerns=peer_concerns_for_next if expert_rounds else None,
            )
            expert_rounds.append(picks)
            # Collect expert tool calls
            for p in picks:
                domain = p.get("domain", "UNKNOWN")
                key = f"expert_{domain}_r{rnd}"
                tool_calls_by_agent[key] = p.get("tool_calls", [])

            logger.info("  Council v3 — Round %d/%d: Skeptic", rnd, max_loop_rounds)
            sk = run_skeptic(
                candidates=candidates,
                expert_picks=picks,
                table_markdown=designer["table_markdown"],
                chemistry_brief=chem_brief,
                is_photochem=is_photochem,
                is_gas_liquid=is_gas_liquid,
                pump_max_bar=pump_max,
                BPR_bar=current.BPR_bar or 0.0,
                tubing_material=current.tubing_material or "FEP",
            )
            skeptic_rounds.append(sk)
            # Collect skeptic tool calls
            tool_calls_by_agent[f"skeptic_r{rnd}"] = sk.get("tool_calls", [])

            # Convergence: stop if no HIGH attacks, no blocks
            high_attacks = [a for a in sk.get("llm_attacks", [])
                            if a.get("severity") == "HIGH"]
            unique_picks = {p["pick_id"] for p in picks if p.get("pick_id") is not None}
            # Converge only when consensus (≤1 unique pick) AND no HIGH attacks AND no blocks
            if len(unique_picks) <= 1 and not high_attacks and not sk.get("blocked_picks"):
                logger.info("  Council v3 — Consensus + clean Skeptic, converged at round %d", rnd)
                break
            elif not high_attacks and not sk.get("blocked_picks"):
                logger.info(
                    "  Council v3 — Skeptic clean but %d specialists disagree (ids: %s) — continuing",
                    len(unique_picks), sorted(unique_picks),
                )

            # Collect peer concerns for round 2 Expert context
            peer_concerns_for_next: list[dict] = []
            for p in picks:
                for concern in (p.get("concerns_on_other_picks") or []):
                    # Parse "id X: ..." format to extract pick_id if possible
                    peer_concerns_for_next.append({
                        "domain": p.get("domain", "?"),
                        "pick_id": p.get("pick_id"),
                        "concern": str(concern)[:200],
                    })

            # Build notes for next round's Expert
            skeptic_notes = [
                f"{a['specialist']} pick id={a['pick_id']}: {a['assumption_at_risk']}"
                for a in high_attacks
            ]

        # ═══════════════════════════════════════════════════════════════════
        #  STEP 6 — Chief resolution
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v3 — Step 6: Chief resolution")
        trade_off_matrix = _build_tradeoff_matrix(
            expert_picks=expert_rounds[-1],
            skeptic=skeptic_rounds[-1],
            candidates=candidates,
            pump_max_bar=pump_max,
        )
        logger.info("  Council v3 — Trade-off matrix:\n%s", trade_off_matrix)
        chief_data, chief_tool_calls = _run_chief_llm(
            expert_picks=expert_rounds[-1],
            skeptic=skeptic_rounds[-1],
            candidates=candidates,
            table_markdown=designer["table_markdown"],
            chemistry_brief=chem_brief,
            objectives=objectives,
            trade_off_matrix=trade_off_matrix,
        )
        tool_calls_by_agent["chief"] = chief_tool_calls
        winner_id = chief_data.get("winning_pick_id")
        try:
            winner_id = int(winner_id) if winner_id is not None else None
        except (TypeError, ValueError):
            winner_id = None
        if winner_id is None or not (1 <= winner_id <= len(candidates)):
            winner_id, reason = _deterministic_resolve(
                expert_rounds[-1], skeptic_rounds[-1], candidates,
            )
            logger.info("  Chief LLM failed — deterministic resolution: %s", reason)
            chief_data["winner_reasoning"] = reason
        else:
            logger.info(
                "  Chief chose id=%d: %s",
                winner_id, chief_data.get("winner_reasoning", "")[:160],
            )

        if winner_id is None:
            return self._fallback(current, chemistry_plan, calc, log, designer)

        winner = candidates[winner_id - 1]

        # ═══════════════════════════════════════════════════════════════════
        #  STEP 7 — Apply winner to FlowProposal, re-run calculator
        # ═══════════════════════════════════════════════════════════════════
        current, changes = _apply_winner(
            current, winner, chief_data.get("changes_to_apply", {}) or {},
        )
        log.all_changes_applied.update(changes)
        calc = DesignCalculator().run(
            batch_record, chemistry_plan, current, inventory,
            analogies=analogies,
            target_flow_rate_mL_min=current.flow_rate_mL_min or None,
            target_tubing_ID_mm=current.tubing_ID_mm or None,
            target_residence_time_min=current.residence_time_min or None,
        )

        # ── Build output ────────────────────────────────────────────────────
        all_round_delibs = _to_deliberations(
            designer, expert_rounds, skeptic_rounds, winner_id, candidates,
            tool_calls_by_agent=tool_calls_by_agent,
        )
        log.rounds = all_round_delibs
        # Store trade-off data for UI
        log.trade_off_matrix = trade_off_matrix
        log.trade_off_summary = skeptic_rounds[-1].get("trade_off_summary", "")
        log.total_rounds = len(expert_rounds)
        log.consensus_reached = True
        log.summary = self._build_summary(
            designer, expert_rounds, skeptic_rounds, winner_id, candidates,
            chief_data, objectives,
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

        current.engine_validated = True
        current.confidence = "HIGH" if winner.get("pareto_front") else "MEDIUM"

        return DesignCandidate(
            proposal=current,
            chemistry_plan=chemistry_plan,
            council_messages=legacy,
            council_rounds=log.total_rounds,
            safety_report={
                "total_checks": len(skeptic_rounds[-1].get("code_checks", {}))
                                if skeptic_rounds else 0,
                "blocks": skeptic_rounds[-1].get("blocked_picks", []) if skeptic_rounds else [],
            },
            unit_operations=[],
            pid_description="",
            deliberation_log=log,
        ), calc

    # ── fallback ───────────────────────────────────────────────────────────
    def _fallback(
        self,
        current: FlowProposal,
        chemistry_plan: Optional[ChemistryPlan],
        calc: DesignCalculations,
        log: DeliberationLog,
        designer: dict,
    ) -> tuple[DesignCandidate, DesignCalculations]:
        log.summary = (
            "Council v3 fallback: no feasible candidate. "
            "Keeping calculator center-point. "
            f"Designer inspected {len(designer.get('feasible', []))} feasible "
            f"+ {len(designer.get('infeasible', []))} infeasible samples."
        )
        current.engine_validated = False
        return DesignCandidate(
            proposal=current,
            chemistry_plan=chemistry_plan,
            council_messages=[],
            council_rounds=0,
            safety_report={},
            unit_operations=[],
            pid_description="",
            deliberation_log=log,
        ), calc

    # ── summary ───────────────────────────────────────────────────────────
    def _build_summary(
        self,
        designer: dict,
        expert_rounds: list[list[dict]],
        skeptic_rounds: list[dict],
        winner_id: int,
        candidates: list[dict],
        chief_data: dict,
        objectives: str,
    ) -> str:
        lines = ["## Council v3 — Advocacy Deliberation Summary"]
        lines.append(f"\n**Objectives**: {objectives}")
        lines.append(f"\n**Designer strategy**: {designer.get('strategy_reasoning','')}")
        lines.append(
            f"**Candidates generated**: {len(designer.get('feasible',[]))} feasible"
        )

        lines.append("\n### Shortlist")
        lines.append(designer["table_markdown"])

        for i, (picks, sk) in enumerate(zip(expert_rounds, skeptic_rounds), 1):
            lines.append(f"\n### Round {i} — Expert picks")
            for p in picks:
                lines.append(
                    f"- **{p['specialist_name']}** → id {p.get('pick_id')}: "
                    f"{p.get('pick_reason','')[:200]}"
                )
            lines.append(f"\n### Round {i} — Skeptic")
            for a in sk.get("llm_attacks", []):
                lines.append(
                    f"- id {a['pick_id']} ({a['severity']}): {a['assumption_at_risk']} "
                    f"→ {a.get('mitigation','')}"
                )
            if sk.get("blocked_picks"):
                lines.append(f"- Blocked by code: {sk['blocked_picks']}")

        w = candidates[winner_id - 1]
        lines.append(f"\n### Winner — candidate id={winner_id}")
        lines.append(
            f"- τ={w['tau_min']} min, d={w['d_mm']} mm, Q={w['Q_mL_min']} mL/min, "
            f"V_R={w['V_R_mL']} mL, L={w['L_m']} m"
        )
        lines.append(
            f"- Re={w['Re']:.0f}, ΔP={w['delta_P_bar']:.3f} bar, "
            f"r_mix={w['r_mix']:.3f}, X={w['expected_conversion']:.2f}, "
            f"prod={w['productivity_mg_h']:.1f} mg/h"
        )
        lines.append(f"- **Chief**: {chief_data.get('winner_reasoning','')}")
        if chief_data.get("open_risks"):
            lines.append("- **Open risks for lab**:")
            for r in chief_data["open_risks"]:
                lines.append(f"  - {r}")

        return "\n".join(lines)
