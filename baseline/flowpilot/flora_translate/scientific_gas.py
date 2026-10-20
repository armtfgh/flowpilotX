"""Gas delivery arithmetic for unvalidated screens, without a kinetic model."""
import math
import re

from flora_translate.residence_time_basis import (
    P_STP_BAR, INLET_STP_BASIS, actual_gas_flow_from_stp,
    stp_gas_flow_for_equiv, gas_equiv_from_stp_flow,
)


def gas_context(calc, liquid_flow, proposal, plan, frozen=False):
    feeds = [s for s in (proposal.streams if proposal else plan.stream_logic) if s.phase == "gas"]
    if len(feeds) != 1:
        raise ValueError("Scientific gas screens require exactly one metered gas feed; mixed-gas models are not supported")
    feed = feeds[0]
    names = feed.contents if proposal else feed.reagents
    label = " ".join(names).lower().replace("₂", "2")
    species = "air" if re.search(r"\bair\b", label) else "O2" if re.search(r"\b(?:o2|oxygen)\b", label) else next(iter(names), "")
    fraction = feed.gas_reagent_mole_fraction
    if fraction is None:
        fraction = 0.21 if species == "air" else 1.0
    equiv = feed.molar_equiv
    if not equiv or not math.isfinite(equiv) or equiv <= 0 or not 0 < fraction <= 1:
        raise ValueError("Positive proposed gas equivalents and a valid feed fraction are required")
    if not math.isfinite(liquid_flow) or liquid_flow <= 0 or not calc.concentration_M or calc.concentration_M <= 0:
        raise ValueError("Positive liquid flow and limiting-feed concentration are required")
    rate = feed.gas_flow_sccm if proposal else None
    if frozen and not rate:
        raise ValueError("Frozen gas screen requires an explicit inlet/STP flow")
    if not rate:
        rate = stp_gas_flow_for_equiv(liquid_flow, calc.concentration_M, equiv, fraction)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid inlet gas flow")
    pressure = float(proposal.BPR_bar or 0) if proposal else 0.0
    actual = actual_gas_flow_from_stp(rate, calc.temperature_C, pressure)
    supplied = gas_equiv_from_stp_flow(rate, liquid_flow, calc.concentration_M, fraction)
    epsilon = actual / (actual + liquid_flow)
    return {"species": species, "reagent_fraction": fraction,
        "oxygen_fraction": fraction if species.lower() in {"o2", "oxygen", "air"} else 0.0,
        "P_abs_bar": pressure + P_STP_BAR, "gas_sccm": rate, "gas_sccm_uncapped": rate,
        "gas_actual_mL_min": actual, "gas_actual_uncapped_mL_min": actual,
        "gas_flow_capped_by_holdup": False, "gas_liquid_ratio": actual / liquid_flow,
        "gas_holdup": epsilon,
        "two_phase_multiplier": min(12.0, 1 + 12 * epsilon + 25 * epsilon ** 2),
        "residence_time_basis": INLET_STP_BASIS,
        "residence_time_inlet_min": calc.residence_time_min,
        "residence_time_in_channel_min": calc.residence_time_min * (liquid_flow + rate) / (liquid_flow + actual),
        "gas_supply_mmol_min": supplied * liquid_flow * calc.concentration_M,
        "target_gas_equiv_inlet": equiv, "explicit_gas_equiv_inlet": equiv,
        "gas_equiv_supplied": supplied,
        "o2_equiv_supplied": supplied if species.lower() in {"o2", "oxygen", "air"} else 0.0}
