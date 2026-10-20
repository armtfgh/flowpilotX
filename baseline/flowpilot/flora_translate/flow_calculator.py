"""
FLORA-Translate — Flow Chemistry Calculator.

Computes engineering numbers from batch record + proposed conditions.
Results are injected into the translation prompt so the LLM reasons
from concrete numbers rather than guessing.

Key outputs:
  - Estimated residence time (from kinetics + intensification factor)
  - Reynolds number (laminar/turbulent check)
  - Damköhler number estimate (reaction vs mixing timescale)
  - BPR requirement (boiling point vs operating temperature)
  - Heat transfer flag (exotherm estimate)
  - Photon flux estimate (for photochem)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

from flora_translate.config import (
    SOLVENT_VISCOSITY_cP,
    SOLVENT_DENSITY_g_mL,
    SOLVENT_BOILING_POINT_C,
    INCOMPATIBLE_COMBOS,
)


# ── Intensification factors from flow chemistry literature ────────────────────
# How much faster is flow vs batch for each chemistry type?
# Conservative estimates — used when no analogy is available.
CHEMISTRY_INTENSIFICATION: dict[str, float] = {
    "photoredox":          48.0,   # photon penetration depth improvement
    "photocatalysis":      48.0,
    "photochem":           30.0,
    "thermal":             10.0,   # heat transfer improvement
    "cross-coupling":      15.0,
    "hydrogenation":       50.0,   # gas-liquid mass transfer
    "electrochemistry":    20.0,   # electrode surface:volume ratio
    "biocatalysis":         8.0,   # gentle, enzyme stability
    "radical":             30.0,
    "default":             20.0,
}

# Estimated mixing times (seconds) for different reactor types
MIXING_TIME_S: dict[str, float] = {
    "coil":         5.0,
    "chip":         0.5,
    "packed_bed":   2.0,
    "CSTR":        30.0,
    "default":      5.0,
}


@dataclass
class FlowCalculations:
    """All calculated engineering parameters for one reaction."""

    # ── Kinetics ──────────────────────────────────────────────────────────────
    batch_time_min: float | None = None
    intensification_factor: float | None = None    # batch_time / flow_rt
    estimated_rt_min: float | None = None          # estimated residence time
    rt_range_min: tuple[float, float] | None = None  # conservative range

    # ── Fluid mechanics ───────────────────────────────────────────────────────
    reynolds_number: float | None = None
    flow_regime: str = ""                          # "laminar" / "transitional"
    solvent_viscosity_cP: float | None = None
    solvent_density: float | None = None

    # ── Damköhler number ─────────────────────────────────────────────────────
    damkohler_number: float | None = None          # Da = τ_reaction / τ_mixing
    damkohler_interpretation: str = ""             # what Da means for this reaction

    # ── Thermal ───────────────────────────────────────────────────────────────
    bpr_required: bool = False
    bpr_minimum_bar: float | None = None           # minimum BPR to prevent boiling
    solvent_boiling_point_C: float | None = None
    heat_transfer_flag: bool = False               # likely exothermic?

    # ── Photochemistry ────────────────────────────────────────────────────────
    photon_limited: bool = False
    recommended_tubing_ID_mm: float | None = None  # penetration depth constraint

    # ── Material compatibility ────────────────────────────────────────────────
    material_warnings: list[str] = field(default_factory=list)

    # ── Summary for prompt injection ─────────────────────────────────────────
    calculation_notes: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Format calculations as a concise block for LLM injection."""
        lines = ["## Pre-computed Engineering Calculations"]
        lines.append("Use these numbers directly — do NOT contradict them without justification.\n")

        if self.estimated_rt_min is not None:
            lines.append(
                f"**Estimated residence time:** {self.estimated_rt_min:.1f} min "
                f"(range: {self.rt_range_min[0]:.0f}–{self.rt_range_min[1]:.0f} min)"
            )
            if self.intensification_factor:
                lines.append(
                    f"  → Based on intensification factor = {self.intensification_factor:.0f}× "
                    f"(batch {self.batch_time_min:.0f} min → flow)"
                )

        if self.reynolds_number is not None:
            lines.append(
                f"**Reynolds number:** Re ≈ {self.reynolds_number:.0f} ({self.flow_regime})"
            )
            lines.append(
                f"  → Calculated for a 1 mm ID tube at the proposed flow rate. "
                f"Re < 2300 = laminar; Re > 2300 = turbulent. "
                f"Laminar flow is standard in microreactors."
            )

        if self.damkohler_number is not None:
            lines.append(
                f"**Damköhler number:** Da ≈ {self.damkohler_number:.1f}"
            )
            lines.append(f"  → {self.damkohler_interpretation}")

        if self.bpr_required:
            lines.append(
                f"**BPR REQUIRED:** Solvent bp = {self.solvent_boiling_point_C:.0f}°C, "
                f"operating temp > bp − 20°C → minimum BPR = {self.bpr_minimum_bar:.0f} bar"
            )
        else:
            lines.append("**BPR:** Not required at operating temperature (solvent safely below boiling point)")

        if self.photon_limited:
            lines.append(
                f"**Photochemistry tubing:** Recommended ID ≤ {self.recommended_tubing_ID_mm:.1f} mm "
                f"for adequate light penetration depth"
            )

        if self.material_warnings:
            lines.append("**Material incompatibilities:**")
            for w in self.material_warnings:
                lines.append(f"  ⚠ {w}")

        for note in self.calculation_notes:
            lines.append(f"**Note:** {note}")

        return "\n".join(lines)


def _lookup_solvent(solvent: str | None, table: dict) -> float | None:
    if not solvent:
        return None
    sl = solvent.lower().strip()
    for key, val in table.items():
        if key.lower() in sl or sl in key.lower():
            return val
    return None


def _detect_chemistry_type(batch_record) -> str:
    """Infer chemistry type from batch record fields."""
    desc = " ".join([
        str(batch_record.reaction_description or ""),
        str(batch_record.photocatalyst or ""),
    ]).lower()

    if any(k in desc for k in ["photoredox", "ir(ppy", "ru(bpy", "photocatal", "4czipn", "eosin"]):
        return "photoredox"
    if any(k in desc for k in ["hν", "light", "uv", "led", "irradiat", "photochem"]):
        return "photochem"
    if any(k in desc for k in ["pd", "suzuki", "heck", "cross-coupl", "buchwald"]):
        return "cross-coupling"
    if any(k in desc for k in ["h2", "hydrogenat", "transfer hydrogenat"]):
        return "hydrogenation"
    if any(k in desc for k in ["electrochem", "anodic", "cathodic", "electrode"]):
        return "electrochemistry"
    if any(k in desc for k in ["enzyme", "lipase", "biocatal", "whole cell"]):
        return "biocatalysis"
    if any(k in desc for k in ["radical", "aibn", "peroxide"]):
        return "radical"
    return "default"


def calculate(
    batch_record,
    proposed_flow_rate_mL_min: float = 0.1,
    proposed_tubing_ID_mm: float = 1.0,
    reactor_type: str = "coil",
    is_photochem: bool = False,
) -> FlowCalculations:
    """
    Compute engineering numbers from batch record + proposed geometry.

    Parameters
    ----------
    batch_record : BatchRecord
    proposed_flow_rate_mL_min : float
        Starting flow rate assumption (updated by council later).
    proposed_tubing_ID_mm : float
        Starting tubing ID assumption.
    reactor_type : str
        One of 'coil', 'chip', 'packed_bed', 'CSTR'.
    is_photochem : bool
        Whether this is a photochemical reaction.
    """
    calc = FlowCalculations()
    notes = []

    # ── Solvent properties ────────────────────────────────────────────────────
    solvent = batch_record.solvent
    mu = _lookup_solvent(solvent, SOLVENT_VISCOSITY_cP)   # cP = mPa·s
    rho = _lookup_solvent(solvent, SOLVENT_DENSITY_g_mL)  # g/mL
    bp  = _lookup_solvent(solvent, SOLVENT_BOILING_POINT_C)

    calc.solvent_viscosity_cP = mu
    calc.solvent_density       = rho
    calc.solvent_boiling_point_C = bp

    # ── Kinetics: residence time estimate ─────────────────────────────────────
    batch_time_h = batch_record.reaction_time_h
    if batch_time_h and batch_time_h > 0:
        batch_time_min = batch_time_h * 60
        calc.batch_time_min = batch_time_min

        chemistry_type = _detect_chemistry_type(batch_record)
        factor = CHEMISTRY_INTENSIFICATION.get(chemistry_type, CHEMISTRY_INTENSIFICATION["default"])
        calc.intensification_factor = factor

        estimated_rt = batch_time_min / factor
        calc.estimated_rt_min = round(estimated_rt, 1)
        # Conservative range: factor/2 to factor*2
        calc.rt_range_min = (
            round(batch_time_min / (factor * 2), 1),
            round(batch_time_min / (factor / 2), 1),
        )
        notes.append(
            f"Residence time estimated from batch time ({batch_time_min:.0f} min) "
            f"÷ intensification factor ({factor:.0f}×) = {estimated_rt:.1f} min. "
            f"Intensification factor from {chemistry_type} literature median."
        )
    else:
        notes.append("No batch reaction time provided — residence time set from analogy data only.")

    # ── Reynolds number ───────────────────────────────────────────────────────
    if mu and rho and proposed_flow_rate_mL_min > 0:
        # Convert: Q [mL/min] → [m³/s], D [mm] → [m], μ [cP] → [Pa·s]
        Q_m3s   = proposed_flow_rate_mL_min * 1e-6 / 60   # m³/s
        D_m     = proposed_tubing_ID_mm * 1e-3              # m
        A_m2    = math.pi * (D_m / 2) ** 2                 # m²
        v_ms    = Q_m3s / A_m2                              # m/s
        mu_Pas  = mu * 1e-3                                 # Pa·s
        rho_kgm3 = rho * 1000                              # kg/m³

        Re = rho_kgm3 * v_ms * D_m / mu_Pas
        calc.reynolds_number = round(Re, 0)
        calc.flow_regime = (
            "laminar" if Re < 2300
            else "transitional" if Re < 4000
            else "turbulent"
        )
    else:
        notes.append("Cannot compute Re — solvent viscosity or density not in database.")

    # ── Damköhler number ─────────────────────────────────────────────────────
    tau_mixing_s = MIXING_TIME_S.get(reactor_type, MIXING_TIME_S["default"])
    if calc.estimated_rt_min:
        tau_reaction_s = calc.estimated_rt_min * 60
        Da = tau_reaction_s / tau_mixing_s
        calc.damkohler_number = round(Da, 1)

        if Da < 1:
            calc.damkohler_interpretation = (
                f"Da < 1: mixing is SLOWER than reaction. Risk of incomplete mixing "
                f"before reaction completes. Consider a shorter {reactor_type} or a chip reactor."
            )
        elif Da < 10:
            calc.damkohler_interpretation = (
                f"Da = {Da:.0f}: mixing and reaction on comparable timescale. "
                f"Adequate for a {reactor_type} reactor."
            )
        else:
            calc.damkohler_interpretation = (
                f"Da >> 1: reaction is slow relative to mixing. Mixing is not limiting. "
                f"Standard {reactor_type} is appropriate."
            )

    # ── BPR requirement ───────────────────────────────────────────────────────
    op_temp_C = batch_record.temperature_C
    if bp and op_temp_C and op_temp_C > (bp - 20):
        calc.bpr_required = True
        # Clausius-Clapeyron approximation: ~1 bar per 20°C above bp is rough
        # Use simple rule: BPR = 1 + (T - (bp-20)) * 0.05 bar (conservative)
        excess = op_temp_C - (bp - 20)
        calc.bpr_minimum_bar = round(1.0 + excess * 0.05, 1)
        notes.append(
            f"BPR required: operating at {op_temp_C}°C, solvent bp = {bp}°C. "
            f"Minimum BPR ≈ {calc.bpr_minimum_bar} bar."
        )
    elif bp:
        calc.bpr_required = False

    # ── Photochemistry tubing ID ──────────────────────────────────────────────
    if is_photochem:
        calc.photon_limited = True
        # Light penetration depth in typical photoredox: ~2-5 mm for FEP/PFA
        # For homogeneous solutions, ~1-2 mm is a common practical guideline
        calc.recommended_tubing_ID_mm = 1.0
        notes.append(
            "Photochemical reaction: tubing ID should be ≤ 1.0 mm for adequate "
            "light penetration (Beer-Lambert law). FEP or PFA preferred."
        )

    # ── Material compatibility check ──────────────────────────────────────────
    for (material, sol_key), concern in INCOMPATIBLE_COMBOS.items():
        if solvent and sol_key.lower() in (solvent or "").lower():
            calc.material_warnings.append(f"{material} + {sol_key}: {concern}")

    calc.calculation_notes = notes
    return calc
