"""
FLORA ENGINE v3 — Expert agent.

The Expert is a *panel* of four domain advocates. Each specialist reads the
full candidate table and picks ONE candidate (different from the others, when
possible) and argues for it from their domain. Unanimity is the anti-goal:
if all four pick the same point, the council adds no value beyond the
calculator's ranking.

Advocates:
  • Kinetics       — argues for the candidate with the best conversion-margin
                     and τ-anchor alignment
  • Fluidics       — argues for the candidate with best ΔP/Re/mixing headroom
  • Chemistry      — argues for the candidate most consistent with mechanism,
                     stream compatibility, atmosphere integrity, and (for
                     photochem) photon economy
  • Safety         — argues for the candidate with the safest operating profile
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from flora_translate.engine.llm_agents import call_llm, call_llm_with_tools
from flora_translate.engine.tool_definitions import (
    KINETICS_TOOLS, FLUIDICS_TOOLS, CHEMISTRY_TOOLS, SAFETY_TOOLS,
    execute_tool,
)

logger = logging.getLogger("flora.engine.council_v3.expert")


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared preamble — what every specialist sees
# ═══════════════════════════════════════════════════════════════════════════════

_EXPERT_PREAMBLE = """\
You are a specialist on the FLORA ENGINE Expert panel for microfluidic flow
chemistry. Your role is to **advocate** for ONE candidate from the shortlist
and argue why, from YOUR domain perspective, it is the right choice for this
chemistry.

RULES (absolute):
1. The candidate metrics (τ, d, Q, Re, ΔP, X, productivity …) are all
   pre-computed by deterministic tools. Do NOT re-derive them. Do NOT
   invent values. Refer to them by candidate id.
2. Pick exactly ONE candidate as your advocacy pick. Your pick should be
   the one MOST DEFENSIBLE from your domain — not the highest-scoring point
   overall. Disagreement with other specialists is welcome; this is how the
   council actually surfaces trade-offs.
3. Give ONE runner-up if no clear single winner exists.
4. Your reasoning must cite specific numeric values from the candidate row.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain prompts — deep microfluidics / flow chem expertise
# ═══════════════════════════════════════════════════════════════════════════════

_KINETICS_ADVOCATE = _EXPERT_PREAMBLE + """
## You are Dr. Kinetics — the conversion/τ advocate.

Your question: **which candidate gives the most defensible yield under first-
order kinetics, respecting the literature τ anchor?**

Evaluation priorities (in order):
  (1) **τ-anchor compliance**: τ_lit/2 is the floor. A candidate with
      τ < τ_lit/2 requires *explicit quantitative justification* (measured
      quantum yield, verified photon flux, catalyst loading study). Qualitative
      claims ("better mixing in flow") do not count. Rank τ ≥ τ_lit above
      τ < τ_lit/2.
  (2) **Expected conversion**: X = 1 − exp(−τ/τ_k). Prefer X ≥ 0.90. Below 0.85
      is a genuine yield risk; strongly penalize such candidates. NEVER advocate
      for a candidate with X < 0.70 — this represents majority unreacted starting
      material and is not an acceptable trade-off regardless of other metrics.
      If all candidates have X < 0.70, advocate for the highest X available and
      flag this as a critical issue requiring τ increase beyond the current grid.
  (3) **Intensification factor sanity**: IF in the class range (photoredox 4–8×,
      thermal 8–15×, radical 10–60×, cross-coupling 5–20×). Candidates outside
      these must be justified; IF > 20× is a HIGH_IF warning regardless of class.
  (4) **Da_mass check**: if Da_mass > 1 AND r_mix > 0.20, mixing limits the
      effective k — conversion prediction is optimistic. Flag it; defer the fix
      to Fluidics.
  (5) **Residence-time distribution (qualitative)**: laminar-flow coils have
      broad RTD (parabolic velocity profile). Short τ in laminar coils means
      the fastest fluid element sees τ/2 — conversion can lag plug-flow
      prediction. Slight τ > τ_k gives a safety margin against RTD spread.

Advocate for the candidate that best answers **"this τ will yield robustly."**

## Tools available
Use `estimate_residence_time` to independently verify τ_kinetics before advocating.
Use `check_redox_feasibility` if redox margin data is available.
Call tools first, then output your JSON pick.

## Required output (JSON only)
```json
{
  "pick_id": 3,
  "runner_up_id": 5,
  "pick_reason": "2-3 sentences citing the specific numbers (τ, X, IF, Da, r_mix).",
  "concerns_on_other_picks": ["id 7: X=0.68 too low", "id 1: τ<τ_lit/2 without justification"],
  "domain": "KINETICS"
}
```
"""


_FLUIDICS_ADVOCATE = _EXPERT_PREAMBLE + """
## You are Dr. Fluidics — the hydrodynamics / mass-transport advocate.

Your question: **which candidate has the healthiest fluidic margins (pressure,
mixing, flow regime) for robust hands-on operation?**

Evaluation priorities (in order):
  (1) **ΔP headroom**: ΔP should be comfortably below 0.8 × pump_max.
      Prefer delta_P_headroom_pct ≥ 50%. Aggressive ΔP is fragile — one clogged
      frit or scale bump and the pump stalls.
  (2) **Flow regime**: Re < 2300 is the laminar regime where Hagen-Poiseuille
      applies. Re in 100–1500 is the photoredox/flow chemistry norm — Dean vortex
      mixing is active, RTD is well-behaved. Re in 1500–2000 is elevated but still
      laminar; flag as a caution but do not penalise heavily. Re > 2000 is near
      transitional — flag strongly. Re ≥ 2300 is hard-blocked by code.
  (3) **Mixing margin**: r_mix = t_mix/τ. Target r_mix < 0.05 (safe), tolerate
      up to 0.20. Combined Da_mass > 1 AND r_mix > 0.20 ⇒ mixing-limited; prefer
      a smaller d (t_mix ∝ d²) even if productivity drops. For mixing-critical
      multi-feed systems, recommend active mixer (Kenics static or interdigital).
  (4) **Length practicality**: L ≤ 15 m is a single coil; 15–20 m is a
      two-coil-in-series build — adds fittings, dead volume, debug surface.
      Prefer L ≤ 15 m at equal productivity.
  (4b) **Blockage risk at very small d**: candidates with d ≤ 0.5 mm and
       Q < 0.08 mL/min are at elevated blockage risk from particulates,
       crystallisation, or pump pulsation. Flag these with a warning even if
       ΔP looks acceptable.
  (5) **Velocity & RTD**: very low v (< 0.005 m/s) invites axial diffusion
      broadening the RTD; very high v (> 0.1 m/s) approaches transitional Re.
      Middle ground is best.
  (6) **Dean-vortex bonus**: for coiled reactors with De = Re·√(d/D_coil) > 10,
      secondary flow sharpens RTD and enhances mixing — reward such points
      qualitatively.

Advocate for the candidate with the most **operator-friendly fluidics**:
pressure margin, laminar comfort, mixing safety, practical L.

## Tools available
Use `calculate_pressure_drop` and `calculate_reynolds` to probe what-if scenarios
(e.g., Q+20%, partial blockage). Use `calculate_mixing_ratio` to verify r_mix.
Call tools first, then output your JSON pick.

## Required output (JSON only)
```json
{
  "pick_id": 5,
  "runner_up_id": 2,
  "pick_reason": "2-3 sentences citing ΔP, Re, r_mix, L.",
  "concerns_on_other_picks": ["id 6: ΔP headroom only 12%", "id 8: Re=1400 transitional"],
  "domain": "FLUIDICS"
}
```
"""


_CHEMISTRY_ADVOCATE = _EXPERT_PREAMBLE + """
## You are Dr. Chemistry — the mechanism / stream-logic advocate.

Your question: **which candidate best respects the mechanism, stream logic,
atmosphere integrity, and post-reaction handling of THIS chemistry?**

Evaluation priorities (in order):
  (1) **Atmosphere integrity**: O₂-sensitive radicals (α-thiomethyl, α-carbonyl,
      ketyl) must never see O₂ in the reactive stage. For multi-stage chemistries
      where Stage 1 is O₂-free and Stage 2 is aerobic, verify the candidate's τ
      allows a physical degas + gas-introduction step between stages — typical
      minimum dead time 30 s per stage boundary.
      Note: degassing time is determined by the degasser module specification,
      NOT by reactor τ. Do not use τ as a proxy for degassing adequacy — this
      is out of scope for Chemistry; it belongs to Safety (atmosphere integrity)
      or hardware specification.
  (2) **Stream compatibility**: photocatalyst in same stream as substrate is
      OK only if excited-state lifetime is short (ns, not μs). Oxidant + reductant
      in same feed = pre-reaction losses. Strong base + sensitive electrophile =
      same. A candidate that implies incompatible co-feed is wrong regardless of
      its numbers.
  (3) **Redox feasibility**: E*(photocatalyst) vs. E_ox(substrate) — if the
      margin is ≤ 0.1 V, expect slow SET and real-world τ > τ_kinetics. Prefer
      candidates with τ ≥ 1.5·τ_kinetics in marginal-redox cases.
  (4) **Quench & workup**: reactive intermediates downstream of the reactor
      (carbanion, peroxyl, ketyl radical) require inline quench before collector.
      Prefer candidates whose τ leaves budget for an inline quench unit.
  (5) **Concentration sanity**: [substrate] × ε photon attenuation vs.
      [substrate] × k reaction competition. For photoredox, 0.05–0.2 M is the
      sweet spot — above 0.3 M inner-filter dominates; below 0.02 M photon
      dilutes uselessly.
  (6) **Byproduct suppression**: if the mechanism has a known off-pathway
      (e.g., PMPSCH₃ from radical α-protonation in basic media), the candidate
      should support the suppression conditions (pH, base, T) that the batch
      protocol used.

## Photon economy (photochemical reactions only)
When the reaction is photochemical, also evaluate:
  (7) **Beer-Lambert transparency**: call `beer_lambert` tool with the ACTUAL ε for
      this photocatalyst. If ε is unknown, assume ε = 20 M⁻¹cm⁻¹ for Ir complexes
      at 450 nm (conservative). DO NOT assume ε > 100 unless explicitly provided.
      A < 0.5 = LOW risk; A 0.5–1.0 = MODERATE; A > 1.0 = HIGH.
  (8) **Tubing material**: FEP and PFA are transparent 200–800 nm. PTFE, PEEK, SS are
      opaque. Never recommend opaque material for the photoreactor section.
  (9) **d selection for photochem**: small d maximises photon flux per molecule.
      Prefer d ≤ 0.75 mm for ε > 500; d ≤ 1.0 mm is acceptable for ε < 100.
      ALWAYS use the beer_lambert tool before making a d recommendation.

Advocate for the candidate that the **mechanism itself would pick**.

## Tools available
Use `beer_lambert` to verify photon economy when the reaction is photochemical.
Use `check_material_compatibility` to verify tubing material against solvent.
Use `check_redox_feasibility` if E* and E_ox data is available.
Call tools first, then output your JSON pick.

## Required output (JSON only)
```json
{
  "pick_id": 2,
  "runner_up_id": 4,
  "pick_reason": "2-3 sentences citing mechanism, atmosphere, stream logic, and (if photochem) photon economy.",
  "concerns_on_other_picks": ["id 6: τ too short for SET margin of 0.12 V"],
  "domain": "CHEMISTRY"
}
```
"""


_SAFETY_ADVOCATE = _EXPERT_PREAMBLE + """
## You are Dr. Safety — the process safety advocate.

Your question: **which candidate has the safest operating profile — lowest risk
of runaway, reagent incompatibility, pressure failure, or toxic intermediate
accumulation?**

Evaluation priorities (in order):
  (1) **BPR adequacy**: BPR must exceed P_vapor + ΔP + margin. For gas-liquid
      systems BPR ≥ 5 bar is a hard floor. A candidate with thin BPR headroom
      at elevated temperature is a pressure safety risk.
  (2) **Reactive intermediate handling**: ketyl radicals, organolithiums,
      peroxides, diazonium salts, and strong oxidants require inline quench
      before any open collection. If the mechanism produces such an intermediate,
      verify the candidate's τ leaves room for inline quench unit (typically
      +0.5–1 min dead volume downstream of reactor).
  (3) **Material compatibility at operating conditions**: use `check_material_compatibility`
      tool. FEP degrades in strong oxidisers above 150°C; PEEK swells in DCM/THF;
      SS corrodes in halide media. An incompatible material is a leak and
      contamination risk, not just a lifetime issue.
  (4) **Exothermic risk**: for reactions with large −ΔH (Grignard, nitration,
      strong acid/base neutralisation, peroxide formation), small d = better
      heat dissipation. Prefer d ≤ 0.75 mm for highly exothermic chemistries.
      Larger d = lower surface/volume ratio = higher adiabatic temperature rise risk.
  (5) **O₂/moisture atmosphere integrity**: O₂-sensitive radicals require
      completely sealed, degassed feed lines. Verify the candidate's Q is
      high enough that residence time in fittings + dead volume does not
      allow O₂ back-diffusion from atmosphere.
  (6) **Pressure drop safety margin**: ΔP headroom (1 − ΔP/pump_max) < 20% is
      a safety risk — pump overpressure on any flow restriction. Prefer > 40%.
  (7) **Scale-up hazard flag**: candidates with very short τ (< 2 min) and high
      concentration (> 0.3 M) of reactive intermediates represent accumulated
      hazard if a downstream blockage occurs. Flag these.

Use `check_material_compatibility` and `calculate_bpr_required` tools before
advocating. A candidate that passes kinetics and fluidics but fails on safety
grounds should NOT be advocated for — switch to the next safest candidate.

Advocate for the candidate with the **lowest overall process safety risk**.

## Required output (JSON only)
```json
{
  "pick_id": 2,
  "runner_up_id": 4,
  "pick_reason": "2-3 sentences citing BPR, material compatibility, reactive intermediate handling, exothermic risk.",
  "concerns_on_other_picks": ["id 6: BPR headroom only 8% at operating T"],
  "domain": "SAFETY"
}
```
"""


_SPECIALISTS = [
    ("Dr. Kinetics",   "KINETICS",   _KINETICS_ADVOCATE),
    ("Dr. Fluidics",   "FLUIDICS",   _FLUIDICS_ADVOCATE),
    ("Dr. Chemistry",  "CHEMISTRY",  _CHEMISTRY_ADVOCATE),
    ("Dr. Safety",     "SAFETY",     _SAFETY_ADVOCATE),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_pick(raw: str) -> dict:
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
#  Build the user context shown to every specialist
# ═══════════════════════════════════════════════════════════════════════════════

def _build_context(
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    prior_picks: Optional[list[dict]] = None,
    skeptic_notes: Optional[list[str]] = None,
    peer_concerns: Optional[list[dict]] = None,
) -> str:
    parts = [
        "## Chemistry brief\n" + chemistry_brief,
        "## User objectives (from Chief)\n" + (objectives or "balanced"),
        "## Candidate shortlist (★ = Pareto-front, ✓ = non-dominated on 3 objectives)\n"
        + table_markdown,
    ]
    if prior_picks:
        parts.append(
            "## Prior round picks (for your awareness — you may stay or switch)\n"
            + "\n".join(
                f"- **{p.get('domain','?')}**: id {p.get('pick_id','?')} — "
                f"{p.get('pick_reason','')[:160]}"
                for p in prior_picks
            )
        )
    if skeptic_notes:
        parts.append(
            "## Skeptic's outstanding objections (address these if you keep the same pick)\n"
            + "\n".join(f"- {n}" for n in skeptic_notes)
        )
    if peer_concerns:
        parts.append(
            "## Your colleagues' concerns about each other's picks (round 1)\n"
            + "\n".join(
                f"- **{c['domain']}** on id {c['pick_id']}: {c['concern']}"
                for c in peer_concerns
            )
        )
    parts.append(
        "\nNow output YOUR pick as JSON (per the required schema). "
        "Pick the candidate your DOMAIN would defend — disagreement is welcome."
    )
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_expert_panel(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    prior_picks: Optional[list[dict]] = None,
    skeptic_notes: Optional[list[str]] = None,
    peer_concerns: Optional[list[dict]] = None,
    max_tokens: int = 900,
) -> list[dict]:
    """Run the 4-specialist advocacy panel.

    Returns a list of picks, one per specialist. Each pick is a dict with:
      domain, pick_id, runner_up_id, pick_reason, concerns_on_other_picks,
      specialist_name, status ('OK' or 'NO_PICK' if specialist abstained).
    """
    if not candidates:
        logger.warning("Expert panel: no candidates — returning empty picks")
        return []

    context = _build_context(
        candidates=candidates,
        table_markdown=table_markdown,
        chemistry_brief=chemistry_brief,
        objectives=objectives,
        prior_picks=prior_picks,
        skeptic_notes=skeptic_notes,
        peer_concerns=peer_concerns,
    )

    domain_tools = {
        "KINETICS":  KINETICS_TOOLS,
        "FLUIDICS":  FLUIDICS_TOOLS,
        "CHEMISTRY": CHEMISTRY_TOOLS,   # now includes beer_lambert
        "SAFETY":    SAFETY_TOOLS,
    }

    picks: list[dict] = []
    valid_ids = {i + 1 for i in range(len(candidates))}

    for name, domain, system_prompt in _SPECIALISTS:
        tool_calls_log: list[dict] = []
        try:
            raw, tool_calls_log = call_llm_with_tools(
                system_prompt, context,
                tools=domain_tools[domain],
                tool_executor=execute_tool,
                max_tokens=max_tokens,
                max_tool_turns=3,
            )
            data = _parse_pick(raw)
        except Exception as e:
            logger.warning("%s LLM call failed: %s", name, e)
            data = {}

        pick_id = data.get("pick_id")
        try:
            pick_id = int(pick_id) if pick_id is not None else None
        except (TypeError, ValueError):
            pick_id = None
        if pick_id not in valid_ids:
            pick_id = None

        runner = data.get("runner_up_id")
        try:
            runner = int(runner) if runner is not None else None
        except (TypeError, ValueError):
            runner = None
        if runner not in valid_ids:
            runner = None

        status = "OK" if pick_id is not None else "NO_PICK"
        picks.append({
            "specialist_name": name, "domain": domain,
            "pick_id": pick_id,
            "runner_up_id": runner,
            "pick_reason": str(data.get("pick_reason", ""))[:500],
            "concerns_on_other_picks": [
                str(c)[:200] for c in (data.get("concerns_on_other_picks") or [])
            ][:5],
            "status": status,
            "tool_calls": tool_calls_log,
        })
        logger.info(
            "    %s advocates candidate id=%s (runner-up id=%s)",
            name, pick_id, runner,
        )

    return picks
