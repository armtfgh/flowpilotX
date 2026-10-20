"""Experimental lumped pressure-network tool, not a validated two-phase CFD model.

All flows exposed to the caller use mL/min; gas feed uses 273.15 K, 1.01325 bar.
The solver uses seconds, mL and absolute bar. Missing dynamics are never inferred
from equipment ratings. An explicit assumption profile is required.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


VERSION = "flowpilot_lumped_transient_v0.1"
P_STP = 1.01325
T_STP = 273.15
SCENARIOS = ("steady", "gas_first_start", "pump_interruption", "gas_overshoot")
LIMITATIONS = [
    "Uncalibrated screening model: no real-world backflow probability or safety certification.",
    "Isothermal ideal gas; one upstream liquid branch and one gas branch at one junction.",
    "Downstream lumped gas/liquid storage with homogeneous outlet; no slug, dissolution, reaction, capillary or gas-front model.",
    "Negative liquid-branch flow means predicted liquid displacement, not proof that oxygen reaches Reactor 1.",
    "Liquid compliance, pump response, MFC response, gas conductance and multiphase resistance require measurements.",
    "Ideal algebraic BPR opening law; BPR hysteresis, valve closing dynamics and pressure-wave propagation are not resolved.",
    "All alternative equipment and controls remain unapproved; no inventory or final design is changed by this tool.",
]


class TransientProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, revalidate_instances="always")
    provenance: Literal["illustrative_assumptions", "operator_supplied_unvalidated"]
    source_note: str = Field(min_length=8)
    viscosity_mPa_s: float = Field(gt=0, le=1000)
    liquid_density_kg_m3: float = Field(gt=0, le=3000)
    liquid_compliance_mL_bar: float = Field(gt=0, le=10)
    gas_plenum_mL: float = Field(gt=0, le=100)
    gas_supply_bar_g: float = Field(gt=0, le=200)
    mfc_required_dp_bar: float = Field(gt=0, le=50)
    gas_conductance_STP_mL_min_bar: float = Field(gt=0, le=1000)
    pump_response_s: float = Field(gt=0, le=60)
    mfc_response_s: float = Field(gt=0, le=60)
    pump_pressure_limit_bar_g: float = Field(gt=0, le=200)
    pump_droop_bar: float = Field(gt=0, le=20)
    stopped_pump_leak_mL_min_bar: float = Field(ge=0, le=10)
    two_phase_resistance_multiplier: float = Field(ge=1, le=100)
    outlet_resistance_bar_min_mL: float = Field(gt=0, le=100)
    startup_gas_fraction: float = Field(gt=0.001, lt=0.999)
    startup_liquid_delay_s: float = Field(gt=0, le=20)
    hypothetical_liquid_valve_cracking_bar: float = Field(gt=0, le=10)
    hypothetical_liquid_valve_reverse_leak_mL_min_bar: float = Field(ge=0, le=1)


def illustrative_profile():
    """Opt-in demonstration assumptions, deliberately NOT KHU specifications."""
    return TransientProfile(provenance="illustrative_assumptions",
        source_note="No measured KHU pressure traces or device dynamics available; illustrative assumptions only.",
        viscosity_mPa_s=1, liquid_density_kg_m3=1000, liquid_compliance_mL_bar=0.01, gas_plenum_mL=0.2,
        gas_supply_bar_g=12, mfc_required_dp_bar=1, gas_conductance_STP_mL_min_bar=0.5,
        pump_response_s=1, mfc_response_s=0.5, pump_pressure_limit_bar_g=10,
        pump_droop_bar=0.2, stopped_pump_leak_mL_min_bar=0,
        two_phase_resistance_multiplier=3, outlet_resistance_bar_min_mL=0.1,
        startup_gas_fraction=0.05, startup_liquid_delay_s=10,
        hypothetical_liquid_valve_cracking_bar=0.1,
        hypothetical_liquid_valve_reverse_leak_mL_min_bar=0)


class HydraulicNetwork(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, revalidate_instances="always")
    liquid_feed_mL_min: float = Field(gt=0)
    gas_feed_STP_mL_min: float = Field(gt=0)
    upstream_volume_mL: float = Field(gt=0)
    upstream_id_mm: float = Field(gt=0)
    downstream_volume_mL: float = Field(gt=0)
    downstream_id_mm: float = Field(gt=0)
    temperature_C: float = Field(gt=-100, lt=300)
    BPR_bar_g: float = Field(ge=0)
    gas_valve_cracking_bar: float = Field(ge=0)
    liquid_valve_cracking_bar: float | None = Field(default=None, ge=0)
    liquid_valve_reverse_leak_mL_min_bar: float = Field(default=0, ge=0)


def tube_resistance(volume_mL, id_mm, viscosity_mPa_s):
    """Hagen-Poiseuille resistance in bar / (mL/min) for a circular tube."""
    if not all(math.isfinite(v) and v > 0 for v in (volume_mL, id_mm, viscosity_mPa_s)):
        raise ValueError("Positive finite volume, ID and viscosity required")
    diameter = id_mm * 1e-3
    length = 4 * volume_mL * 1e-6 / (math.pi * diameter**2)
    return 128 * viscosity_mPa_s * 1e-3 * length / (math.pi * diameter**4) * (1e-6 / 60) / 1e5


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def simulate(network, profile, scenario, *, max_step_s=0.5, rtol=1e-7, duration_s=90):
    """Return conditional predictions plus conservation residuals and full traces.

    State: liquid pressure, gas-plenum pressure, junction pressure, downstream
    liquid inventory, pump/MFC actuator flows, reverse-volume integral, net liquid
    and gas input integrals. The reverse branch remains liquid-filled in this
    approximation; loss of its inventory terminates the model's validity.
    """
    n = HydraulicNetwork.model_validate(network)
    p = TransientProfile.model_validate(profile)
    if scenario not in SCENARIOS:
        raise ValueError("Unknown transient scenario")
    if not (0.01 <= max_step_s <= 2 and 1e-10 <= rtol <= 1e-5 and 70 <= duration_s <= 180):
        raise ValueError("Solver budget outside supported bounds")
    a = P_STP * (n.temperature_C + T_STP) / T_STP
    r1 = tube_resistance(n.upstream_volume_mL, n.upstream_id_mm, p.viscosity_mPa_s)
    r2 = (tube_resistance(n.downstream_volume_mL, n.downstream_id_mm, p.viscosity_mPa_s)
          * p.two_phase_resistance_multiplier + p.outlet_resistance_bar_min_mL)
    bpr, supply = P_STP + n.BPR_bar_g, P_STP + p.gas_supply_bar_g
    cap = P_STP + p.pump_pressure_limit_bar_g
    nominal_l, nominal_g = n.liquid_feed_mL_min, n.gas_feed_STP_mL_min
    v = n.downstream_volume_mL

    def flows(t, y):
        pl, pg, pj, vl, qp, qg = y[:6]
        delta = pl - pj
        if n.liquid_valve_cracking_bar is None:
            ql = delta / r1
        else:
            ql = (max(delta - n.liquid_valve_cracking_bar, 0) / r1
                  - n.liquid_valve_reverse_leak_mL_min_bar * max(-delta, 0))
        gas = p.gas_conductance_STP_mL_min_bar * max(pg - pj - n.gas_valve_cracking_bar, 0)
        pump = max(qp, 0) * min(max((cap - pl) / p.pump_droop_bar, 0), 1)
        mfc = max(qg, 0) * min(max((supply - pg) / p.mfc_required_dp_bar, 0), 1)
        out = max(pj - bpr, 0) / r2
        pump_off = scenario == "pump_interruption" and 30 <= t < 60
        leak = p.stopped_pump_leak_mL_min_bar * max(pl - P_STP, 0) if pump_off else 0
        return ql, gas, pump, mfc, out, leak

    def commands(t):
        if scenario == "gas_first_start":
            return (0 if t < p.startup_liquid_delay_s else nominal_l), nominal_g
        if scenario == "pump_interruption":
            return (0 if 30 <= t < 60 else nominal_l), nominal_g
        return nominal_l, nominal_g * (2 if scenario == "gas_overshoot" and 30 <= t < 40 else 1)

    def rhs(t, y):
        pl, pg, pj, vl, qp, qg = y[:6]
        ql, gas, pump, mfc, out, leak = flows(t, y)
        cmd_l, cmd_g = commands(t)
        vg = v - vl
        # Guards permit event root finding, not acceptance of an invalid state.
        vg_safe = max(vg, v * 1e-8)
        liquid_out, gas_out_stp = vl / v * out, vg / v * out * pj / a
        return [(pump - leak - ql) / (60 * p.liquid_compliance_mL_bar),
                a * (mfc - gas) / (60 * p.gas_plenum_mL),
                (a * gas + pj * (ql - out)) / (60 * vg_safe),
                (ql - liquid_out) / 60,
                (cmd_l - qp) / p.pump_response_s,
                (cmd_g - qg) / p.mfc_response_s,
                max(-ql, 0) / 60,
                (pump - leak - liquid_out) / 60,
                (mfc - gas_out_stp) / 60]

    pj0 = brentq(lambda pj: pj - bpr - r2 * (nominal_l + nominal_g * a / pj), bpr, bpr + r2 * (nominal_l + nominal_g * a / bpr) + 1)
    pl0 = pj0 + r1 * nominal_l + (n.liquid_valve_cracking_bar or 0)
    pg0 = pj0 + n.gas_valve_cracking_bar + nominal_g / p.gas_conductance_STP_mL_min_bar
    alpha = nominal_g * a / pj0 / (nominal_l + nominal_g * a / pj0)
    y0 = np.array([pl0, pg0, pj0, v * (1 - alpha), nominal_l, nominal_g, 0, 0, 0])
    if scenario == "gas_first_start":
        y0[:6] = [P_STP, P_STP, P_STP, v * (1 - p.startup_gas_fraction), 0, 0]
    elif pg0 + p.mfc_required_dp_bar > supply or pl0 > cap - p.pump_droop_bar:
        return {"schema_version": VERSION, "status": "not_assessable", "scenario": scenario,
                "reason": "Assumed supply/pump capability cannot sustain the initial steady state; change the explicit profile, not equipment ratings.",
                "backflow_probability": None, "laboratory_execution_status": "review_required"}

    def validity(t, y):
        return min(y[0] - 0.1, y[1] - 0.1, y[2] - 0.1,
                   y[3] - v * 1e-6, v * (1 - 1e-6) - y[3],
                   n.upstream_volume_mL - y[6])
    validity.terminal = True
    validity.direction = -1

    # Split known forcing discontinuities so no solver step skips a perturbation.
    cuts = sorted({0., float(duration_s), p.startup_liquid_delay_s, 30., 40., 60.})
    times, states, initial = [], [], y0
    completed, reason = True, ""
    for start, end in zip(cuts, cuts[1:]):
        # Use the left limit at the end of each segment for discontinuous commands.
        def segment_rhs(t, y):
            return rhs(min(t, np.nextafter(end, start)), y)
        sol = solve_ivp(segment_rhs, (start, end), initial, method="Radau", rtol=rtol,
                        atol=rtol * 1e-3, max_step=max_step_s, events=validity)
        times.extend(sol.t[1:] if times else sol.t)
        states.extend(sol.y.T[1:] if states else sol.y.T)
        initial = sol.y[:, -1]
        if not sol.success or sol.status == 1:
            completed = False
            reason = sol.message if not sol.success else "Pressure, phase-inventory or upstream liquid-displacement validity boundary reached."
            break
    t, y = np.asarray(times), np.asarray(states)
    f = np.asarray([flows(ti, yi) for ti, yi in zip(t, y)])
    l_store = p.liquid_compliance_mL_bar * y[:, 0] + y[:, 3]
    g_store = (p.gas_plenum_mL * y[:, 1] + (v - y[:, 3]) * y[:, 2]) / a
    l_error = float(np.max(np.abs(l_store - l_store[0] - y[:, 7])))
    g_error = float(np.max(np.abs(g_store - g_store[0] - y[:, 8])))
    reverse_threshold = max(1e-7, nominal_l * 1e-5)
    numerical_ok = l_error < 1e-5 and g_error < 1e-5 and np.all(np.isfinite(y))
    peak_liquid_re = (4 * p.liquid_density_kg_m3 * float(np.max(np.abs(f[:, 0]))) * 1e-6 / 60
                      / (math.pi * n.upstream_id_mm * 1e-3 * p.viscosity_mPa_s * 1e-3))
    laminar_valid = peak_liquid_re < 2000
    if not laminar_valid:
        reason = "Upstream Reynolds number exceeds this tool's laminar resistance domain."
    peak_reverse = max(0., -float(np.min(f[:, 0])))
    trace = [{"time_s": float(ti), "liquid_upstream_bar_g": float(yi[0] - P_STP),
              "gas_plenum_bar_g": float(yi[1] - P_STP), "junction_bar_g": float(yi[2] - P_STP),
              "liquid_branch_mL_min": float(fi[0]), "gas_to_junction_STP_mL_min": float(fi[1]),
              "gas_MFC_STP_mL_min": float(fi[3]), "reverse_displacement_mL": float(yi[6])}
             for ti, yi, fi in zip(t, y, f)]
    return {"schema_version": VERSION, "status": "simulated_unvalidated" if completed and numerical_ok and laminar_valid else "invalid_simulation",
        "scenario": scenario, "reason": reason, "profile_provenance": p.provenance,
        "input_sha256": fingerprint({"network": n.model_dump(), "profile": p.model_dump(), "scenario": scenario,
                                     "max_step_s": max_step_s, "rtol": rtol, "duration_s": duration_s}),
        "reverse_flow_predicted": bool(peak_reverse > reverse_threshold),
        "reverse_flow_reporting_threshold_mL_min": reverse_threshold,
        "peak_reverse_liquid_mL_min": peak_reverse, "reverse_displacement_uL": max(0., float(y[-1, 6] * 1000)),
        "peak_junction_bar_g": float(np.max(y[:, 2]) - P_STP),
        "peak_gas_plenum_bar_g": float(np.max(y[:, 1]) - P_STP),
        "peak_liquid_upstream_bar_g": float(np.max(y[:, 0]) - P_STP),
        "peak_upstream_liquid_Re": peak_liquid_re,
        "BPR_opening_pressure_reached": bool(np.max(y[:, 2]) >= bpr),
        "window_note": "Only the first startup interval was simulated; operating pressure was not reached."
            if scenario == "gas_first_start" and np.max(y[:, 2]) < bpr else "Finite scenario window, not a lifetime or reliability assessment.",
        "min_MFC_differential_bar": float(np.min(supply - y[:, 1])),
        "min_MFC_margin_above_required_bar": float(np.min(supply - y[:, 1]) - p.mfc_required_dp_bar),
        "simulated_until_s": float(t[-1]), "backflow_probability": None,
        "laboratory_execution_status": "review_required", "gas_reaching_upstream_reactor": "not_modeled",
        "numerical_checks": {"solver_completed": completed, "conservation_passed": bool(numerical_ok), "laminar_domain": laminar_valid,
            "max_liquid_balance_error_mL": l_error, "max_gas_balance_error_STP_mL": g_error},
        "resistances_bar_min_mL": {"upstream_liquid": r1, "downstream_effective": r2},
        "limitations": LIMITATIONS, "trace": trace}
