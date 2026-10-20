"""
FLORA ENGINE — Domain calculation tools.

These functions are AUTHORITATIVE for their specific computations.
They run before any LLM call; results are pre-injected into the triage report.
No agent should ever re-derive what a tool already computes.

All functions return plain dicts — JSON-serialisable for prompt injection.
"""

from __future__ import annotations
import math
import re

from flora_translate.config import (
    SOLVENT_VISCOSITY_cP,
    SOLVENT_DENSITY_g_mL,
    SOLVENT_BOILING_POINT_C,
    INCOMPATIBLE_COMBOS,
)
from flora_translate.design_calculator import ANTOINE, GAS_LIQUID_MIN_BPR_BAR, INTENSIFICATION

PI = math.pi
D_MOLECULAR = 1.0e-9   # m²/s — diffusion coefficient for small organics in liquid
PUMP_MAX_DEFAULT = 20.0  # bar — conservative default when no inventory


# ── Solvent property lookup (handles mixed solvents like "EtOH/water 5:1") ───

def _solvent_prop(solvent: str, lut: dict, default: float) -> float:
    """Look up a solvent property, averaging components for mixed solvents."""
    # Direct match first
    for key, val in lut.items():
        if key.lower() == solvent.lower().strip():
            return val
    # Substring match
    for key, val in lut.items():
        if key.lower() in solvent.lower():
            return val
    # Mixed solvent: split by / or , and average found components
    parts = re.split(r"[/,]", solvent)
    vals = []
    for p in parts:
        p = p.strip().split("(")[0].strip()
        for key, val in lut.items():
            if key.lower() in p.lower():
                vals.append(val)
                break
    return sum(vals) / len(vals) if vals else default


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 1 — Beer-Lambert light absorption
# ═══════════════════════════════════════════════════════════════════════════════

def beer_lambert(
    concentration_M: float,
    extinction_coeff_M_cm: float,
    path_length_mm: float,
) -> dict:
    """Compute Beer-Lambert absorption for photocatalyst in a tube.

    The optical path length equals the tubing ID.
    Inner filter risk threshold: A > 0.5 (>68% absorbed in outer shell).

    Returns:
        absorbance, transmission, inner_filter_risk, recommended_action
    """
    path_cm = path_length_mm / 10.0
    A = extinction_coeff_M_cm * concentration_M * path_cm
    T = 10.0 ** (-A)
    pct = (1.0 - T) * 100.0

    if A > 1.0:
        risk, action = "HIGH", f"Reduce ID to ≤0.5 mm or dilute to ≤{0.5 / (extinction_coeff_M_cm * path_cm):.3f} M"
    elif A > 0.5:
        risk, action = "MODERATE", "Consider reducing ID to 0.75 mm; monitor inner filter effect"
    else:
        risk, action = "LOW", "No action required"

    return {
        "absorbance": round(A, 4),
        "transmission": round(T, 4),
        "percent_absorbed": round(pct, 2),
        "inner_filter_risk": risk,
        "threshold_exceeded": A > 0.5,
        "recommended_action": action,
        "formula": f"A = ε·C·l = {extinction_coeff_M_cm}×{concentration_M}×{path_cm:.3f} cm = {A:.4f}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 2 — Material compatibility
# ═══════════════════════════════════════════════════════════════════════════════

_TEMP_LIMITS = {
    "FEP": 200.0, "PFA": 260.0, "PTFE": 260.0,
    "SS316": 450.0, "SS": 450.0, "HASTELLOY": 400.0,
    "PEEK": 150.0, "GLASS": 200.0, "PDMS": 150.0,
}

_EXTRA_INCOMPATIBLE = {
    ("PEEK", "DCM"):          "PEEK swells in DCM — use SS or PFA",
    ("PEEK", "THF"):          "PEEK swells in THF at elevated temperature",
    ("SS", "HCL"):            "SS corrodes in HCl — use PTFE/PFA",
    ("SS316", "CHLORIDE"):    "SS316 susceptible to pitting in chloride media",
    ("FEP", "BUFFER"):        "FEP may become brittle in strongly alkaline buffers above 80°C",
    ("PDMS", "ORGANICS"):     "PDMS swells in most organic solvents — use glass or FEP chip",
}


def check_material_compatibility(
    material: str,
    solvent: str,
    temperature_C: float,
) -> dict:
    """Check tubing/fitting compatibility with solvent at operating temperature."""
    mat = material.upper()
    max_T = next((v for k, v in _TEMP_LIMITS.items() if k == mat), 200.0)
    temp_ok = temperature_C < max_T

    concern = None
    # Config table
    for (m, s), msg in INCOMPATIBLE_COMBOS.items():
        if m.upper() == mat and s.lower() in solvent.lower():
            concern = msg
            break
    # Extended table
    if not concern:
        for (m, s), msg in _EXTRA_INCOMPATIBLE.items():
            if m == mat and s in solvent.upper():
                concern = msg
                break

    return {
        "material": material,
        "solvent": solvent,
        "compatible": concern is None and temp_ok,
        "concern": concern,
        "temperature_limit_C": max_T,
        "temperature_ok": temp_ok,
        "temperature_margin_C": round(max_T - temperature_C, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 3 — BPR sizing
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_bpr_required(
    temperature_C: float,
    solvent: str,
    delta_P_system_bar: float,
    is_gas_liquid: bool = False,
) -> dict:
    """Calculate required BPR setting using Antoine equation.

    Adds 0.5 bar margin for liquid-only and 2.0 bar for gas-liquid.
    Gas-liquid systems use a 3 bar hard floor before the recommended margin.
    """
    sol_key = next((k for k in ANTOINE if k.lower() in solvent.lower()), None)
    if sol_key:
        A, B, C = ANTOINE[sol_key]
        P_vap_bar = (10.0 ** (A - B / (C + temperature_C))) * 0.00133322
    else:
        P_vap_bar = 0.10

    bp = _solvent_prop(solvent, SOLVENT_BOILING_POINT_C, 100.0)
    requires_for_T = temperature_C > (bp - 20.0)

    # Protocol mandates BPR_recommended = BPR_minimum + 2.0 bar for gas-liquid
    # (O₂, H₂, CO₂) to maintain liquid-filled reactor and prevent phase separation.
    # For liquid-only: BPR_minimum + 0.5 bar is sufficient.
    margin = 2.0 if is_gas_liquid else 0.5
    P_min = P_vap_bar + delta_P_system_bar + margin
    if is_gas_liquid:
        P_min = max(P_min, GAS_LIQUID_MIN_BPR_BAR + margin)

    return {
        "required": requires_for_T or is_gas_liquid,
        "reason": (
            f"gas-liquid system — BPR mandatory ({GAS_LIQUID_MIN_BPR_BAR + margin:.1f} bar recommended)" if is_gas_liquid else
            f"T = {temperature_C}°C > bp − 20°C = {bp - 20:.0f}°C" if requires_for_T else
            "not required (liquid-only, below boiling threshold)"
        ),
        "P_vapor_bar": round(P_vap_bar, 4),
        "P_min_bar": round(P_min, 2),
        "P_recommended_bar": round(P_min, 1),   # P_min already includes margin
        "solvent_bp_C": bp,
        "gas_liquid": is_gas_liquid,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 4 — Mixing ratio
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_mixing_ratio(
    tubing_ID_mm: float,
    residence_time_min: float,
    diffusivity_m2_s: float = D_MOLECULAR,
) -> dict:
    """Compute mixing ratio t_mix / τ.

    Rule: actionable ONLY when ratio > 0.20 AND Da_mass > 1.
    Below 0.20, diffusion completes well within τ.
    """
    d_m = tubing_ID_mm * 1e-3
    t_mix_s = d_m ** 2 / (4.0 * diffusivity_m2_s)
    tau_s = residence_time_min * 60.0
    ratio = t_mix_s / tau_s if tau_s > 0 else float("inf")
    THRESHOLD = 0.20

    return {
        "mixing_time_s": round(t_mix_s, 2),
        "residence_time_s": round(tau_s, 1),
        "mixing_ratio": round(ratio, 4),
        "threshold": THRESHOLD,
        "actionable": ratio > THRESHOLD,
        "interpretation": (
            f"ratio = {ratio:.3f} > {THRESHOLD} — mixing genuinely limiting (reduce ID or use active mixer)"
            if ratio > THRESHOLD else
            f"ratio = {ratio:.3f} < {THRESHOLD} — mixing completes within τ, no action needed"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 5 — Reynolds number
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_reynolds(
    flow_rate_mL_min: float,
    tubing_ID_mm: float,
    solvent: str,
    temperature_C: float = 25.0,
) -> dict:
    """Reynolds number for flow in a circular tube."""
    mu_cP = _solvent_prop(solvent, SOLVENT_VISCOSITY_cP, 1.0)
    rho_g_mL = _solvent_prop(solvent, SOLVENT_DENSITY_g_mL, 1.0)
    mu = mu_cP * 1e-3
    rho = rho_g_mL * 1000.0
    Q = flow_rate_mL_min * 1e-6 / 60.0
    d = tubing_ID_mm * 1e-3
    v = 4.0 * Q / (PI * d ** 2) if d > 0 else 0.0
    Re = rho * v * d / mu if mu > 0 else 0.0

    return {
        "Re": round(Re, 2),
        "velocity_m_s": round(v, 7),
        "viscosity_cP": round(mu_cP, 3),
        "density_g_mL": round(rho_g_mL, 3),
        "flow_regime": "turbulent" if Re > 2300 else ("transitional" if Re > 1000 else "laminar"),
        "laminar": Re < 2300,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 6 — Hagen-Poiseuille pressure drop
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_pressure_drop(
    flow_rate_mL_min: float,
    tubing_ID_mm: float,
    length_m: float,
    solvent: str,
) -> dict:
    """Hagen-Poiseuille: ΔP = 128·μ·L·Q / (π·d⁴)"""
    mu_cP = _solvent_prop(solvent, SOLVENT_VISCOSITY_cP, 1.0)
    mu = mu_cP * 1e-3
    Q = flow_rate_mL_min * 1e-6 / 60.0
    d = tubing_ID_mm * 1e-3
    dP_Pa = 128.0 * mu * length_m * Q / (PI * d ** 4) if d > 0 else 0.0
    dP_bar = dP_Pa * 1e-5

    return {
        "delta_P_Pa": round(dP_Pa, 3),
        "delta_P_bar": round(dP_bar, 5),
        "formula": "ΔP = 128·μ·L·Q / (π·d⁴)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 7 — Redox feasibility
# ═══════════════════════════════════════════════════════════════════════════════

def check_redox_feasibility(
    E_excited_V_SCE: float,
    E_substrate_ox_V_SCE: float,
    mode: str = "oxidative",
) -> dict:
    """Check SET thermodynamic feasibility.

    Oxidative quenching: E*[PC] ≥ E_ox[substrate]  →  feasible
    """
    if mode == "oxidative":
        feasible = E_excited_V_SCE >= E_substrate_ox_V_SCE
        margin = E_excited_V_SCE - E_substrate_ox_V_SCE
    else:
        feasible = E_excited_V_SCE <= E_substrate_ox_V_SCE
        margin = E_substrate_ox_V_SCE - E_excited_V_SCE

    return {
        "feasible": feasible,
        "margin_V": round(margin, 3),
        "mode": mode,
        "verdict": (
            f"SET feasible — {margin:.3f} V driving force" if feasible else
            f"SET not feasible — {abs(margin):.3f} V shortfall"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 8 — Residence time estimation
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_residence_time(
    batch_time_h: float,
    reaction_class: str,
    if_override: float | None = None,
) -> dict:
    """Estimate flow τ from batch time and class-level intensification factor."""
    IF = if_override or INTENSIFICATION.get(reaction_class.lower(), INTENSIFICATION["default"])
    tau_center = (batch_time_h * 60.0) / IF
    return {
        "batch_time_min": round(batch_time_h * 60.0, 1),
        "intensification_factor": IF,
        "tau_center_min": round(tau_center, 2),
        "tau_low_min": round(tau_center / 1.5, 2),
        "tau_high_min": round(tau_center * 1.5, 2),
        "source": "override" if if_override else "class_table",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 9 — Design envelope sweep
# ═══════════════════════════════════════════════════════════════════════════════

def compute_design_envelope(
    tau_center_min: float,
    Q_center_mL_min: float,
    ID_center_mm: float,
    T_center_C: float,
    BPR_center_bar: float,
    solvent: str,
    is_gas_liquid: bool = False,
    pump_max_bar: float = PUMP_MAX_DEFAULT,
) -> dict:
    """Sweep ±30% around center values and return the feasible operating window.

    Constraints enforced:
      - Re < 2300 (laminar)
      - ΔP < 80% of pump_max_bar
      - BPR ≥ minimum required
      - ID in [0.25, 4.0] mm
    """
    SWEEP = 0.30

    def _clamp_Q(Q, ID):
        """Pull Q down until laminar and ΔP within limits."""
        L = max(tau_center_min * Q * 1e-3, 0.01)  # rough length estimate
        while Q > Q_center_mL_min * 0.5:
            re = calculate_reynolds(Q, ID, solvent, T_center_C)
            dP = calculate_pressure_drop(Q, ID, L, solvent)
            if re["laminar"] and dP["delta_P_bar"] < 0.80 * pump_max_bar:
                break
            Q *= 0.95
        return round(Q, 4)

    # τ range
    tau_lo = round(tau_center_min * (1 - SWEEP), 2)
    tau_hi = round(tau_center_min * (1 + SWEEP), 2)

    # Q range — constrain by Re and ΔP at high end
    Q_lo = round(Q_center_mL_min * (1 - SWEEP), 4)
    Q_hi = _clamp_Q(Q_center_mL_min * (1 + SWEEP), ID_center_mm)

    # ID range — keep laminar at high Q
    ID_lo = max(round(ID_center_mm * (1 - SWEEP), 3), 0.25)
    ID_hi_raw = ID_center_mm * (1 + SWEEP)
    ID_hi = min(round(ID_hi_raw, 3), 4.0)

    # BPR range — floor is the hard minimum
    bpr_tool = calculate_bpr_required(T_center_C, solvent, 0.0, is_gas_liquid)
    BPR_lo = max(round(BPR_center_bar * (1 - 0.10), 1), bpr_tool["P_min_bar"])
    BPR_hi = round(BPR_center_bar * (1 + 0.30), 1)

    # V_R range derived from τ × Q
    V_lo = round(tau_lo * Q_lo, 3)
    V_hi = round(tau_hi * Q_hi, 3)

    return {
        "tau_min": {"min": tau_lo, "center": round(tau_center_min, 2), "max": tau_hi, "unit": "min"},
        "Q_mL_min": {"min": Q_lo, "center": round(Q_center_mL_min, 4), "max": Q_hi, "unit": "mL/min"},
        "ID_mm": {"min": ID_lo, "center": round(ID_center_mm, 3), "max": ID_hi, "unit": "mm"},
        "T_C": {"min": round(T_center_C * (1 - 0.10), 1), "center": T_center_C,
                "max": round(T_center_C * (1 + 0.10), 1), "unit": "°C"},
        "BPR_bar": {"min": BPR_lo, "center": round(BPR_center_bar, 1), "max": BPR_hi,
                    "hard_minimum": bpr_tool["P_min_bar"], "unit": "bar"},
        "V_R_mL": {"min": V_lo, "center": round(tau_center_min * Q_center_mL_min, 3),
                   "max": V_hi, "unit": "mL"},
        "constraints_applied": [
            "Re < 2300 (laminar) at Q_max",
            "ΔP < 80% pump_max at Q_max",
            f"BPR ≥ {bpr_tool['P_min_bar']:.2f} bar (hard minimum)",
            "ID in [0.25, 4.0] mm",
        ],
    }
