"""Shared residence-time basis helpers for gas-liquid flow designs."""

from __future__ import annotations


P_STP_BAR = 1.01325
T_STP_K = 273.15
R_GAS_L_BAR = 0.08314
STP_MOLAR_VOLUME_ML_PER_MMOL = R_GAS_L_BAR * T_STP_K / P_STP_BAR

INLET_STP_BASIS = "inlet_stp"
IN_CHANNEL_BASIS = "in_channel"
LIQUID_ONLY_BASIS = "liquid_only"
UNKNOWN_BASIS = "unknown"


def requested_residence_time_basis(constraints: object) -> str:
    """Read an explicit time-basis request, not a gas-metering instruction."""
    import re
    text = str(constraints or "").lower()
    if re.search(r"(?:nominal\s+)?empty[- ]bed\s+(?:liquid\s+)?space time", text):
        return LIQUID_ONLY_BASIS
    if re.search(r"use liquid flow.{0,100}(?:residence|space) time|(?:residence|space) time.{0,60}(?:based on|using|from) (?:the )?liquid flow", text):
        return LIQUID_ONLY_BASIS
    return UNKNOWN_BASIS


def gas_time_basis_consistent(proposal: object) -> bool:
    """Gas is metered at STP; the declared reactor-time basis is independent."""
    data = proposal if isinstance(proposal, dict) else proposal.model_dump()
    gases = [s for s in data.get("streams", []) if s.get("phase") == "gas"]
    if not gases:
        return True
    basis = normalize_residence_time_basis(data.get("residence_time_basis"))
    if basis == UNKNOWN_BASIS:
        return False
    if data.get("stage_parameters"):
        times = []
        for stage in data["stage_parameters"]:
            stage_basis = normalize_residence_time_basis(stage.get("residence_time_basis"))
            flow = float(stage.get("Q_liquid_mL_min") or stage.get("flow_rate_mL_min") or 0)
            if stage_basis == INLET_STP_BASIS:
                flow += float(stage.get("Q_gas_sccm") or stage.get("gas_flow_sccm") or 0)
            elif stage_basis == IN_CHANNEL_BASIS:
                flow += float(stage.get("Q_gas_actual_mL_min") or stage.get("gas_flow_actual_mL_min") or 0)
            elif stage_basis != LIQUID_ONLY_BASIS:
                return False
            volume = float(stage.get("reactor_volume_mL") or stage.get("V_R_mL") or 0)
            tau = float(stage.get("residence_time_min") or 0)
            if min(volume, flow, tau) <= 0 or abs(volume - flow*tau) > max(.02, .02*volume):
                return False
            times.append(tau)
        return abs(sum(times) - float(data.get("residence_time_min") or 0)) <= max(1e-5, .02*sum(times))
    q = float(data.get("flow_rate_mL_min") or 0)
    v = float(data.get("reactor_volume_mL") or 0)
    if basis == INLET_STP_BASIS:
        q += sum(float(s.get("gas_flow_sccm") or 0) for s in gases)
    elif basis == IN_CHANNEL_BASIS:
        q += sum(float(s.get("gas_flow_actual_mL_min") or 0) for s in gases)
    tau = float(data.get("residence_time_min") or 0)
    return v > 0 and q > 0 and tau > 0 and abs(v - q * tau) <= max(1e-6, .02 * v)


def gas_reagent_fraction(species: str, declared: float | None = None) -> float:
    """Resolve the active fraction without treating named air as pure oxygen.

    Custom mixtures must be identified as mixtures, not as ordinary air.
    Their explicit fraction remains authoritative.
    """
    import math

    if str(species).strip().lower() == "air":
        return 0.21
    value = 1.0 if declared is None else float(declared)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Active gas mole fraction must be finite and in (0, 1].")
    return value


def normalize_residence_time_basis(value: object) -> str:
    """Normalize free-text residence-time basis labels."""

    text = str(value or "").strip().lower()
    if not text:
        return UNKNOWN_BASIS
    normalized = (
        text.replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )
    if any(token in normalized for token in ("inlet", "stp", "mfc", "metered")):
        return INLET_STP_BASIS
    if any(token in normalized for token in ("channel", "actual", "pressure_corrected", "reactor_total")):
        return IN_CHANNEL_BASIS
    if "liquid" in normalized or "empty_bed" in normalized:
        return LIQUID_ONLY_BASIS
    return UNKNOWN_BASIS


def residence_time_basis_label(basis: object) -> str:
    """Human-readable basis label for JSON output and prompts."""

    normalized = normalize_residence_time_basis(basis)
    if normalized == INLET_STP_BASIS:
        return "inlet/STP apparent residence time"
    if normalized == IN_CHANNEL_BASIS:
        return "in-channel pressure-corrected total residence time"
    if normalized == LIQUID_ONLY_BASIS:
        return "liquid-only reactor volume / liquid flow"
    return "unknown"


def actual_gas_flow_from_stp(
    gas_stp_mL_min: float,
    temperature_C: float,
    pressure_gauge_bar: float,
    *,
    min_abs_bar: float = P_STP_BAR,
) -> float:
    """Convert an MFC/STP gas flow to reactor actual volume flow."""

    if gas_stp_mL_min <= 0:
        return 0.0
    t_k = float(temperature_C) + 273.15
    p_abs = max(float(pressure_gauge_bar or 0.0) + P_STP_BAR, min_abs_bar)
    return float(gas_stp_mL_min) * (t_k / T_STP_K) * (P_STP_BAR / p_abs)


def stp_gas_flow_from_actual(
    gas_actual_mL_min: float,
    temperature_C: float,
    pressure_gauge_bar: float,
    *,
    min_abs_bar: float = P_STP_BAR,
) -> float:
    """Convert reactor actual gas volume flow to an MFC/STP setpoint."""

    if gas_actual_mL_min <= 0:
        return 0.0
    t_k = float(temperature_C) + 273.15
    p_abs = max(float(pressure_gauge_bar or 0.0) + P_STP_BAR, min_abs_bar)
    return float(gas_actual_mL_min) * (T_STP_K / t_k) * (p_abs / P_STP_BAR)


def stp_gas_flow_for_equiv(
    liquid_flow_mL_min: float,
    concentration_M: float,
    gas_equiv: float,
    gas_reagent_fraction: float = 1.0,
) -> float:
    """Return STP gas flow needed for the requested molar equivalents.

    ``concentration_M * liquid_flow_mL_min`` gives substrate mmol/min.
    Multiplying by the requested equivalents and STP molar volume gives the
    gas MFC setpoint in mL/min STP, i.e. sccm. ``gas_reagent_fraction`` handles
    diluted gases such as air where O2 is only about 21%.
    """

    try:
        q_liq = float(liquid_flow_mL_min)
        conc = float(concentration_M)
        equiv = float(gas_equiv)
        fraction = float(gas_reagent_fraction)
    except (TypeError, ValueError):
        return 0.0
    if q_liq <= 0 or conc <= 0 or equiv <= 0 or fraction <= 0:
        return 0.0
    substrate_mmol_min = q_liq * conc
    gas_mmol_min = substrate_mmol_min * equiv / fraction
    return gas_mmol_min * STP_MOLAR_VOLUME_ML_PER_MMOL


def gas_equiv_from_stp_flow(
    gas_stp_mL_min: float,
    liquid_flow_mL_min: float,
    concentration_M: float,
    gas_reagent_fraction: float = 1.0,
) -> float:
    """Return reagent equivalents delivered by an STP gas MFC setpoint."""

    try:
        gas_stp = float(gas_stp_mL_min)
        q_liq = float(liquid_flow_mL_min)
        conc = float(concentration_M)
        fraction = float(gas_reagent_fraction)
    except (TypeError, ValueError):
        return 0.0
    if gas_stp <= 0 or q_liq <= 0 or conc <= 0 or fraction <= 0:
        return 0.0
    gas_mmol_min = gas_stp / STP_MOLAR_VOLUME_ML_PER_MMOL
    substrate_mmol_min = q_liq * conc
    return gas_mmol_min * fraction / substrate_mmol_min
