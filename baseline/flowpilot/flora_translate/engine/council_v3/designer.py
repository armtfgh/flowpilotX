"""
FLORA ENGINE v3 — Designer agent.

Role: strategy selection, not value selection.
The Designer LLM reads the chemistry + calculator center-point and chooses a
SAMPLING STRATEGY (τ range factors, which tubing IDs to include, how many
τ samples, log vs linear spacing). The actual numbers come from
`sampling.generate_candidates` — a deterministic tool. The LLM never invents
a τ, d, Q, Re, ΔP, or productivity.

Output: a shortlist of 10–15 fully-metriced feasible candidates.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from flora_translate.engine.llm_agents import call_llm
from flora_translate.engine.sampling import (
    generate_candidates,
    format_candidate_table,
)

logger = logging.getLogger("flora.engine.council_v3.designer")


# ═══════════════════════════════════════════════════════════════════════════════
#  Microfluidics / flow chemistry expert prompt
# ═══════════════════════════════════════════════════════════════════════════════

_DESIGNER_SYSTEM = """\
You are **The Designer** — a senior process engineer specializing in
microfluidic flow chemistry design for laboratory-scale synthesis.

## Your expertise
You have deep, working knowledge of:

• **Hydrodynamics in microchannels**  Hagen-Poiseuille (ΔP ∝ L·Q/d⁴), the
  laminar regime (Re < ~2300 in coils; typically Re ≪ 100 in bench microreactors),
  Dean-vortex enhancement in coiled tubes (De = Re·√(d/D_coil)), Taylor dispersion
  axial broadening at low Re.
• **Mixing & mass transfer**  Diffusion-limited mixing time t_mix ≈ d²/(4·D_AB)
  in laminar flow (very slow for d > 0.5 mm; halving d gives 4× faster mixing).
  The Damköhler number Da = k·d²/D_AB compares reaction to diffusion.
  Action threshold: Da > 1 AND r_mix = t_mix/τ > 0.20 — require active mixing
  or smaller d. Below either threshold, diffusion completes within τ.
• **Photochemistry geometry**  Beer-Lambert A = ε·C·ℓ where ℓ = tubing ID.
  Inner-filter risk: A > 0.5 moderate, A > 1.0 severe. For photoredox
  (Ir/Ru/4CzIPN with typical ε 500–5000 M⁻¹cm⁻¹), d ≤ 1.0 mm is mandatory
  and d ≤ 0.75 mm is preferred. FEP/PFA only in the photoreactor (transparent);
  PTFE is opaque and kills photon transport.
• **Gas-liquid flow**  Slug/Taylor flow in microchannels at moderate Re.
  Requires BPR ≥ 5 bar to keep the reactor liquid-filled and prevent phase
  separation. d ≥ 0.75 mm preferred for reliable slug formation. Mass transfer
  (kLa) is excellent compared to batch (>10× better).
• **Residence time & intensification**  τ_flow ≪ τ_batch because reactor-scale
  mass-/heat-transfer limits evaporate. Class-typical intensification factors
  (IF = τ_batch / τ_flow):
    Photoredox 4–8× (up to 10× with quantitative justification);
    Thermal 8–15×; Hydrogenation 20–50×; Radical 10–60×; Cross-coupling 5–20×.
  Pushing IF > class upper bound requires cited precedent, not optimism.
• **Bench hardware envelope**  Single FEP/PFA coil realistic limits: L ≤ 20 m,
  V_R ≤ 25 mL, pump_max ~20 bar (syringe) or ~100 bar (HPLC piston);
  syringe pumps practical floor ≈ 0.05 mL/min; MFCs for gas streams.
• **Reactor geometry choice**  Coil-around-LED-strip is the de facto standard
  for photoredox. Chip reactors for very short τ or high photon flux.
  Packed-bed for heterogeneous catalysis. Falling-film for gas-liquid + photon.

## Your job — output a SAMPLING STRATEGY (not numbers)
Given the calculator's center point (τ_center, d_center, Q_center) and the
chemistry context, decide **how wide** to sample and **which d values** to
include. Your judgment is about coverage: where in (τ, d) space is the
engineering trade-off actually interesting for THIS chemistry?

Guidance for your choices:

  • **Short-τ aggressive chemistries** (high-photon-flux photoredox, fast radical,
    fast thermal): sample down to τ_center/3 to expose "can we go even shorter?"
  • **Long-τ chemistries** (slow thermal, cross-coupling, aged catalyst risk):
    sample up to 2× or further to explore conversion-first designs.
  • **O₂-sensitive photoredox**: bias toward small d (0.5, 0.75 mm) to prioritize
    photon flux; exclude d > 1.0 mm outright.
  • **Gas-liquid (O₂, H₂, CO₂)**: exclude d < 0.75 mm (slug flow needs headroom);
    keep BPR floor in mind.
  • **Heat-sensitive / exothermic**: bias toward small d (better heat dissipation).
  • **Long-τ + literature anchor**: include τ_lit and τ_lit/2 explicitly —
    never sample only below the literature floor without justification.

Use **log-spaced** τ sampling when the range spans > 3× (e.g., 10–100 min);
**linear** when narrower (e.g., 20–40 min). More n_tau (6–8) when the chemistry
is genuinely uncertain; fewer (4–5) when the calculator's center is well-anchored.

## REQUIRED OUTPUT — JSON ONLY, no prose outside the JSON block
```json
{
  "reasoning": "2-3 sentences on WHY this sampling strategy fits the chemistry.",
  "tau_low_factor": 0.3,
  "tau_high_factor": 2.0,
  "n_tau": 5,
  "tau_log_spaced": true,
  "d_exclude_above_mm": 1.0,
  "include_long_L_fraction": true
}
```

Rules:
- `tau_low_factor` in [0.15, 0.8], `tau_high_factor` in [1.2, 3.0]
- `n_tau` ∈ [3, 8]
- `d_exclude_above_mm`: set to 1.0 for photochem, 2.0 for liquid, null otherwise
- `include_long_L_fraction`: true to include L_fraction up to 0.95 (long coils allowed)
"""


_DEFAULT_STRATEGY = {
    "reasoning": "Default balanced sampling — calculator center-point is trusted.",
    "tau_low_factor": 0.4,
    "tau_high_factor": 2.0,
    "n_tau": 5,
    "tau_log_spaced": True,
    "d_exclude_above_mm": None,
    "include_long_L_fraction": True,
}


def _parse_strategy(raw: str) -> dict:
    """Extract the strategy JSON from the LLM response."""
    s = raw.strip()
    # Strip fences
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
    # Brace match
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


def _sanitize_strategy(data: dict) -> dict:
    """Clamp all strategy values to safe ranges."""
    s = dict(_DEFAULT_STRATEGY)
    s.update({k: v for k, v in data.items() if k in _DEFAULT_STRATEGY})
    # Clamps
    try:
        s["tau_low_factor"] = max(0.15, min(0.8, float(s["tau_low_factor"])))
    except (TypeError, ValueError):
        s["tau_low_factor"] = 0.4
    try:
        s["tau_high_factor"] = max(1.2, min(3.0, float(s["tau_high_factor"])))
    except (TypeError, ValueError):
        s["tau_high_factor"] = 2.0
    try:
        s["n_tau"] = int(max(3, min(8, int(s["n_tau"]))))
    except (TypeError, ValueError):
        s["n_tau"] = 5
    s["tau_log_spaced"] = bool(s.get("tau_log_spaced", True))
    try:
        v = s.get("d_exclude_above_mm")
        s["d_exclude_above_mm"] = float(v) if v is not None else None
    except (TypeError, ValueError):
        s["d_exclude_above_mm"] = None
    s["include_long_L_fraction"] = bool(s.get("include_long_L_fraction", True))
    return s


# ═══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_designer(
    *,
    reaction_class: str,
    is_photochem: bool,
    is_gas_liquid: bool,
    is_O2_sensitive: bool,
    tau_center_min: float,
    tau_lit_min: Optional[float],
    tau_kinetics_min: float,
    d_center_mm: float,
    Q_center_mL_min: float,
    solvent: str,
    temperature_C: float,
    concentration_M: float,
    assumed_MW: float,
    IF_used: float,
    pump_max_bar: float,
    BPR_bar: float = 0.0,
    extinction_coeff_M_cm: Optional[float] = None,
    N_target: int = 12,
) -> dict:
    """Run the Designer agent.

    Returns:
        {
          "strategy": {...},
          "strategy_reasoning": "...",
          "feasible": [ ... candidate dicts ... ],
          "infeasible": [ ... ],
          "table_markdown": "...",
        }
    """
    # ── Build a concise chemistry brief for the LLM ──────────────────────────
    chem_brief = (
        f"reaction_class={reaction_class} | photochem={is_photochem} | "
        f"gas_liquid={is_gas_liquid} | O2_sensitive={is_O2_sensitive}\n"
        f"τ_center={tau_center_min:.1f} min | τ_lit="
        f"{'%.1f' % tau_lit_min if tau_lit_min else 'none'} min | "
        f"τ_kinetics(90% X)={tau_kinetics_min:.1f} min | IF={IF_used:.1f}\n"
        f"d_center={d_center_mm} mm | Q_center={Q_center_mL_min:.3f} mL/min\n"
        f"solvent={solvent} | T={temperature_C}°C | C={concentration_M} M | "
        f"MW={assumed_MW:.0f} g/mol | pump_max={pump_max_bar} bar"
    )
    if is_photochem:
        chem_brief += (
            f"\nε(photocatalyst)={extinction_coeff_M_cm or 'not provided'} M⁻¹cm⁻¹"
        )

    user_msg = (
        "## Chemistry & calculator center-point\n" + chem_brief +
        "\n\nChoose the sampling strategy. Output JSON only."
    )

    # ── LLM call ─────────────────────────────────────────────────────────────
    strategy = dict(_DEFAULT_STRATEGY)
    strategy_reasoning = strategy["reasoning"]
    try:
        raw = call_llm(_DESIGNER_SYSTEM, user_msg, max_tokens=500)
        parsed = _parse_strategy(raw)
        if parsed:
            strategy = _sanitize_strategy(parsed)
            strategy_reasoning = strategy.get("reasoning", strategy_reasoning)
    except Exception as e:
        logger.warning("Designer LLM call failed — using default strategy: %s", e)

    logger.info(
        "    Designer strategy: τ∈[%.2f×, %.2f×], n_tau=%d, %s, d≤%s mm",
        strategy["tau_low_factor"], strategy["tau_high_factor"],
        strategy["n_tau"], "log" if strategy["tau_log_spaced"] else "lin",
        strategy["d_exclude_above_mm"],
    )

    # ── Generate candidates via deterministic tool ───────────────────────────
    L_fractions = (
        [0.40, 0.60, 0.80, 0.95]
        if strategy["include_long_L_fraction"]
        else [0.40, 0.60, 0.80]
    )

    feasible, infeasible = generate_candidates(
        tau_center_min=tau_center_min,
        tau_lit_min=tau_lit_min,
        solvent=solvent, temperature_C=temperature_C,
        concentration_M=concentration_M, assumed_MW=assumed_MW,
        IF_used=IF_used, tau_kinetics_min=tau_kinetics_min,
        pump_max_bar=pump_max_bar, is_photochem=is_photochem,
        is_gas_liquid=is_gas_liquid, BPR_bar=BPR_bar,
        extinction_coeff_M_cm=extinction_coeff_M_cm,
        tau_low_factor=strategy["tau_low_factor"],
        tau_high_factor=strategy["tau_high_factor"],
        n_tau=strategy["n_tau"],
        tau_log_spaced=strategy["tau_log_spaced"],
        d_exclude_above_mm=strategy["d_exclude_above_mm"],
        L_fractions=L_fractions,
        N_target=N_target,
    )

    table = format_candidate_table(feasible, max_rows=N_target)
    logger.info(
        "    Designer: %d feasible / %d infeasible candidates",
        len(feasible), len(infeasible),
    )

    return {
        "strategy": strategy,
        "strategy_reasoning": strategy_reasoning,
        "feasible": feasible,
        "infeasible": infeasible,
        "table_markdown": table,
    }
