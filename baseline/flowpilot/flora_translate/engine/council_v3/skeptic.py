"""
FLORA ENGINE v3 — Skeptic agent.

The Skeptic attacks **assumptions**, not arithmetic. Re and ΔP are already
computed by tools; checking those is deterministic code (done here as
`sanity_code_check`). The Skeptic's LLM role is to ask the questions a
senior flow chemist would ask in a design review:

  • Is k really constant across τ? (photocatalyst bleaching, substrate
    inhibition, intermediate accumulation).
  • Is ε known or guessed? (wrong ε by 3× is common).
  • Is the mixing model valid here? (Da = k·d²/D assumes single-liquid,
    diffusion-limited — fails for gas-liquid, fails for two-phase, fails
    for suspensions).
  • Are RTD effects ignored? (broad RTD in coils at low Re vs. plug-flow
    assumption of X = 1 − exp(−τ/τ_k)).
  • Is the safety floor actually respected? (BPR ≥ 5 bar for gas-liquid,
    FEP/PFA for photoreactor, atmosphere integrity across stages).

Also includes the deterministic safety/rule check — these are not debatable
and do not need an LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from flora_translate.engine.llm_agents import call_llm, call_llm_with_tools
from flora_translate.engine.tool_definitions import SKEPTIC_TOOLS, execute_tool

logger = logging.getLogger("flora.engine.council_v3.skeptic")


# ═══════════════════════════════════════════════════════════════════════════════
#  Deterministic safety / rule check — no LLM needed
# ═══════════════════════════════════════════════════════════════════════════════

_OPAQUE_MATERIALS = {"PTFE", "PEEK", "SS", "SS316", "HASTELLOY"}
_PHOTO_OK_MATERIALS = {"FEP", "PFA", "GLASS", "QUARTZ"}


def sanity_code_check(
    candidate: dict,
    *,
    is_photochem: bool,
    is_gas_liquid: bool,
    pump_max_bar: float,
    BPR_bar: float = 0.0,
    tubing_material: str = "FEP",
) -> dict:
    """Hard, non-debatable rule checks on one candidate.

    Returns:
      {
        "blocks":   ["reason 1", "reason 2"],   # candidate MUST be rejected
        "warnings": ["..."],                     # candidate may proceed with note
      }
    """
    blocks: list[str] = []
    warnings: list[str] = []

    # Photoreactor material rule (hard)
    mat = (tubing_material or "FEP").upper()
    if is_photochem and mat in _OPAQUE_MATERIALS:
        blocks.append(
            f"tubing_material={mat} is opaque — photoreactor requires FEP/PFA"
        )

    # Gas-liquid BPR rule (hard)
    if is_gas_liquid and BPR_bar < 5.0:
        blocks.append(
            f"BPR={BPR_bar} bar < 5 bar minimum for gas-liquid flow "
            "(liquid film collapses, slug flow fails)"
        )

    # Photochem Beer-Lambert (hard)
    if is_photochem and candidate.get("d_mm", 0) > 1.0:
        blocks.append(
            f"d={candidate['d_mm']} mm > 1.0 mm — inner-filter effect blocks "
            "photon penetration"
        )

    # Pressure headroom (hard at 95% of pump, warn at 80%)
    dP = candidate.get("delta_P_bar", 0.0)
    if dP >= 0.95 * pump_max_bar:
        blocks.append(f"ΔP={dP:.3f} bar > 95% of pump_max — no operating margin")
    elif dP >= 0.8 * pump_max_bar:
        warnings.append(f"ΔP={dP:.3f} bar ≥ 80% pump_max — thin margin")

    # Reynolds turbulent (hard)
    re = candidate.get("Re", 0.0)
    if re >= 2300:
        blocks.append(f"Re={re:.0f} ≥ 2300 — laminar-flow equations invalid")
    elif re > 1500:
        warnings.append(f"Re={re:.0f} > 1500 — approaching transitional; RTD model weakened above 2000")

    # Very narrow tubing blockage risk
    d_mm = candidate.get("d_mm", 1.0)
    Q = candidate.get("Q_mL_min", 1.0)
    if d_mm <= 0.5 and Q < 0.08:
        warnings.append(
            f"d={d_mm} mm + Q={Q:.3f} mL/min — elevated blockage risk from particulates "
            "or crystallisation; verify reagent solubility and use inline filter"
        )

    # Bench geometry (hard)
    if candidate.get("L_m", 0) > 20.0:
        blocks.append(f"L={candidate['L_m']:.1f} m > 20 m — exceeds single-coil bench limit")
    if candidate.get("V_R_mL", 0) > 25.0:
        blocks.append(f"V_R={candidate['V_R_mL']:.1f} mL > 25 mL — exceeds single-reactor bench limit")

    # Mixing failure (warn only — Fluidics Expert may prefer larger d)
    if candidate.get("Da_mass", 0) > 1.0 and candidate.get("r_mix", 0) > 0.20:
        warnings.append(
            f"Mixing-limited: Da_mass={candidate['Da_mass']:.2f}, r_mix="
            f"{candidate['r_mix']:.2f} — consider smaller d or static mixer"
        )

    # Conversion gate
    X = candidate.get("expected_conversion", 1.0)
    if X < 0.70:
        blocks.append(f"X={X:.2f} < 0.70 — unacceptable conversion; majority unreacted starting material")
    elif X < 0.80:
        warnings.append(f"X={X:.2f} < 0.80 — τ likely too short; yield risk")

    return {"blocks": blocks, "warnings": warnings}


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM skeptic — assumption attacks
# ═══════════════════════════════════════════════════════════════════════════════

_SKEPTIC_SYSTEM = """\
You are **The Skeptic** on the FLORA ENGINE council — a senior flow chemistry
design reviewer whose only job is to find the **assumptions that would break
this design in the lab**. You do NOT redo arithmetic; Re, ΔP, V_R, L are
already computed by tools and are correct. You attack premises.

## Your expertise — the assumption audit
You know every place a microfluidic design quietly assumes something that
only sometimes holds:

• **Rate constant k is not actually constant**: photocatalysts bleach under
  extended irradiation (especially 4CzIPN, Eosin Y, some Ir complexes beyond
  ~2 h continuous); metal catalysts deactivate by intermediate binding;
  radicals can reach chain-termination regime at high [R·]. If a candidate's
  τ is long (> 1 h equivalent batch), ask "does k hold for the last 30% of τ?"
• **ε / quantum yield assumed**: most batch papers don't report ε at the LED
  wavelength. A 3× error in ε propagates to a 3× error in predicted photon
  absorption → wrong d choice. If the design uses a headline ε from the
  spectrum peak, the LED wavelength may see ε/3 or less.
• **First-order kinetics is a fit, not a truth**: many flow designs use
  X = 1 − exp(−τ/τ_k). This assumes single-limiting-step first-order.
  Radical chains are often zero-order in substrate above saturation;
  bimolecular recombinations are second-order; photoredox is photon-flux-
  limited (zero-order in substrate at high C). Challenge candidates whose
  X calculation assumes first-order without justification.
• **Plug-flow assumption hides RTD**: X = 1 − exp(−τ/τ_k) is the plug-flow
  answer. Laminar flow in a straight tube has parabolic RTD: the centerline
  sees τ/2, the wall sees ∞ (effectively). Coiling + Dean vortex narrows
  RTD but doesn't remove it. Short τ designs (τ < 3 × τ_k) are vulnerable
  to the fast-moving core NOT converting. Flag when the "safety margin" of
  τ/τ_k is thin (< 2×).
• **Mixing model breaks in multi-phase**: Da_mass = k·d²/D assumes single
  liquid phase and diffusion-limited mixing. It is wrong for:
   – gas-liquid slug flow (use kLa framework, not D)
   – packed-bed (use inter-particle Peclet)
   – suspensions / heterogeneous catalysts
   – strong viscosity contrasts between streams
  If the candidate is multi-phase, flag that the single-liquid mixing model
  is not the right tool.
• **BPR recycle: gas dissolution unknown**: in gas-liquid systems, O₂/H₂/CO₂
  solubility at 5 bar is much higher than at 1 atm — the flow "C" of gas in
  the liquid is NOT the 1-atm Henry's law value. Reaction rates computed
  from atmospheric C may undershoot by 3–10×. Flag this when gas-liquid.
• **Atmosphere integrity across stages**: multi-stage "one-pot" designs in
  flow REQUIRE physical isolation between atmospheres. A single tube that
  switches from Ar to air cannot be trusted — back-diffusion of O₂ through
  the liquid slug is real. Demand an explicit gas-segmentation point.
• **Hardware availability ignored**: a candidate with d = 0.50 mm at Q =
  0.08 mL/min needs a precise low-flow syringe pump (Harvard Apparatus, New
  Era) — not every lab has this. Flag candidates whose Q is below the
  practical floor of common HPLC pumps (~0.05–0.1 mL/min) when no syringe
  pump is stated.
• **Scale-out fragility**: if the candidate picks d = 0.50 mm for photon
  economy, two trains in parallel for 2× throughput require 2× the LED
  footprint, not 2× the tube length. Flag when scale-out is clearly
  compromised.
• **Material fatigue**: FEP swells in aromatic/chlorinated solvents above
  ~60 °C; PFA is better. Long runs at elevated T can compromise FEP.

## What to output — for each specialist's pick in priority order
For each Expert pick, identify AT MOST TWO most-likely-to-bite assumptions
and one concrete mitigation each. If a pick passes cleanly, say so — do NOT
manufacture objections. Over-skepticism is as bad as rubber-stamping.

## Comparative narrative — trade_off_summary
After attacking individual picks, write ONE paragraph that reads ACROSS all picks:
- Where do the specialists disagree (which candidates, which metrics)?
- What is the core trade-off the disagreement reveals (e.g. τ vs ΔP, yield safety vs operational comfort)?
- What does the lab chemist need to decide in order to choose?
Be specific: cite the candidate ids and the actual numbers.

## Tools available to you
You have access to calculation tools. Use them to QUANTIFY your attacks — do not just
state "the margin looks thin", compute it. Examples:
- Use `estimate_residence_time` to compute τ_kinetics independently and verify the τ/τ_k margin
- Use `beer_lambert` to compute absorbance at a WORST-CASE ε (e.g., ε/3 for off-peak LED)
- Use `calculate_pressure_drop` to probe what 20% Q increase does to ΔP headroom
- Use `calculate_reynolds` to verify Re stays laminar at operating conditions

Call tools BEFORE writing your JSON. Ground every HIGH or MEDIUM attack in a tool result.

## REQUIRED OUTPUT — JSON ONLY
```json
{
  "attacks": [
    {
      "pick_id": 3,
      "specialist": "KINETICS",
      "assumption_at_risk": "First-order rate with τ/τ_k margin of 1.4× — vulnerable to laminar RTD leakage",
      "mitigation": "Add a Dean-coil reactor (De > 10) OR increase τ to 2×τ_k",
      "severity": "MEDIUM"
    }
  ],
  "clean_picks": [{"pick_id": 5, "specialist": "FLUIDICS"}],
  "summary": "One-sentence overall read.",
  "trade_off_summary": "One paragraph: where do the picks disagree? What metric drives that gap? What does the lab chemist need to decide between them?"
}
```

Severity = HIGH (pick should be demoted), MEDIUM (pick OK with mitigation),
LOW (note for lab chemist only).
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Parse helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_skeptic(raw: str) -> dict:
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
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_skeptic(
    *,
    candidates: list[dict],
    expert_picks: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    is_photochem: bool,
    is_gas_liquid: bool,
    pump_max_bar: float,
    BPR_bar: float = 0.0,
    tubing_material: str = "FEP",
    max_tokens: int = 1200,
) -> dict:
    """Run the Skeptic on the Expert's picks.

    Returns:
      {
        "code_checks": { pick_id: {"blocks": [...], "warnings": [...]}, ... },
        "llm_attacks": [...],          # LLM-generated assumption attacks
        "clean_picks": [...],
        "summary": "...",
        "blocked_picks": [pick_id, ...],   # picks demoted by code blocks (hard)
        "tool_calls": [...],               # tool calls made by Skeptic LLM
      }
    """
    # ── Deterministic rule check on every pick (hard blocks) ─────────────────
    code_checks: dict[int, dict] = {}
    blocked: set[int] = set()
    for p in expert_picks:
        pid = p.get("pick_id")
        if pid is None or not (1 <= pid <= len(candidates)):
            continue
        cand = candidates[pid - 1]
        chk = sanity_code_check(
            cand, is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
            pump_max_bar=pump_max_bar, BPR_bar=BPR_bar,
            tubing_material=tubing_material,
        )
        code_checks[pid] = chk
        if chk["blocks"]:
            blocked.add(pid)
            logger.info("    Skeptic (code): id=%d BLOCKED: %s", pid, chk["blocks"])

    # ── LLM attack on non-blocked picks ─────────────────────────────────────
    picks_for_llm = [
        p for p in expert_picks
        if p.get("pick_id") is not None and p["pick_id"] not in blocked
    ]
    if not picks_for_llm:
        return {
            "code_checks": code_checks,
            "llm_attacks": [],
            "clean_picks": [],
            "summary": "All picks blocked by hard code rules — see blocks.",
            "trade_off_summary": "",
            "blocked_picks": sorted(blocked),
            "tool_calls": [],
        }

    picks_summary = "\n".join(
        f"- {p['domain']} picks id={p['pick_id']}: {p.get('pick_reason','')[:200]}"
        for p in picks_for_llm
    )
    code_summary = "\n".join(
        f"- id {pid}: warnings={chk['warnings']}"
        for pid, chk in code_checks.items() if chk.get("warnings")
    ) or "_none_"

    context = (
        "## Chemistry brief\n" + chemistry_brief +
        "\n\n## Candidate table (reference)\n" + table_markdown +
        "\n\n## Expert picks to attack\n" + picks_summary +
        "\n\n## Code-level warnings already flagged (for context only)\n" + code_summary +
        "\n\nAttack each pick's weakest assumption. Be specific. JSON only."
    )

    llm_attacks: list[dict] = []
    clean_picks: list[dict] = []
    summary = ""
    trade_off_summary = ""
    tool_calls_log: list[dict] = []
    try:
        raw, tool_calls_log = call_llm_with_tools(
            _SKEPTIC_SYSTEM, context,
            tools=SKEPTIC_TOOLS,
            tool_executor=execute_tool,
            max_tokens=max_tokens,
            max_tool_turns=4,
        )
        data = _parse_skeptic(raw)
        for a in data.get("attacks", []) or []:
            if isinstance(a, dict) and a.get("pick_id") is not None:
                llm_attacks.append({
                    "pick_id": a.get("pick_id"),
                    "specialist": str(a.get("specialist", ""))[:40],
                    "assumption_at_risk": str(a.get("assumption_at_risk", ""))[:400],
                    "mitigation": str(a.get("mitigation", ""))[:300],
                    "severity": str(a.get("severity", "MEDIUM")).upper(),
                })
        clean_picks = [
            {"pick_id": c.get("pick_id"), "specialist": c.get("specialist", "")}
            for c in (data.get("clean_picks") or [])
            if isinstance(c, dict) and c.get("pick_id") is not None
        ]
        summary = str(data.get("summary", ""))[:300]
        trade_off_summary = str(data.get("trade_off_summary", ""))[:600]
    except Exception as e:
        logger.warning("Skeptic LLM call failed: %s", e)

    for a in llm_attacks:
        logger.info(
            "    Skeptic attack on id=%s (%s): %s",
            a["pick_id"], a["severity"], a["assumption_at_risk"][:100],
        )

    return {
        "code_checks": code_checks,
        "llm_attacks": llm_attacks,
        "clean_picks": clean_picks,
        "summary": summary,
        "trade_off_summary": trade_off_summary,
        "blocked_picks": sorted(blocked),
        "tool_calls": tool_calls_log,
    }
