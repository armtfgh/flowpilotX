"""
FLORA ENGINE — Design Space Grid Search.

Runs BEFORE the council. Enumerates (τ, d, Q) combinations and computes metrics
using the SAME deterministic functions the Designer/Council use downstream
(`flora_translate.engine.sampling.compute_metrics` and `hard_filter`). This
guarantees Design Space and Council Designer agree about feasibility for the
same point — without this unification, design space could pass a candidate
that the Council's hard-gate then rejects (and silently bypasses the council).

τ values : [τ_lit/2, τ_center×0.75, τ_center, τ_center×1.25, τ_center×1.5]
           plus τ_lit itself if τ_lit > τ_center
d values : commercial FEP/PFA sizes from `choose_d_set` (photochem + gas-liquid
           aware — gas-liquid photochem allows up to 1.6 mm because gas holdup
           makes sub-mm IDs impractical at the required gas flow).
Q per (τ,d): at L fractions [0.5, 0.65, 0.8, 0.9] × L_MAX_BENCH_M

Hard filter logic lives in sampling.py::hard_filter — DO NOT inline duplicated
checks here. If a check belongs at design-space stage, add it there too.

Soft scoring (0–1, weighted sum):
  productivity_score = Productivity_mg_h / max(Productivity_mg_h) — normalized.
                       Zeroed below the X_FLOOR (conversion floor) so very-short-τ
                       low-conversion candidates can't win on productivity alone.
  L_score   = full credit ≤15 m, linear penalty to 0 at 30 m (matches sampling L cap)
  dP_score  = full credit ≤2 bar, linear penalty to 0 at 5 bar (matches sampling ΔP cap)
  mixing_score = 1 − min(r_mix / 0.20, 1.0)
  re_score = 1 − Re / 2300
  conversion_score = expected_conversion (X = 1 − exp(−τ/τ_kinetics))

Weights by reaction class:
  photoredox/photocatalysis/photochem: productivity=0.50, L=0.20, mixing=0.10, re=0.05, conversion=0.10, dP=0.05
  thermal:                              productivity=0.55, L=0.15, mixing=0.15, re=0.05, conversion=0.10
  default:                              productivity=0.50, L=0.15, mixing=0.20, re=0.05, conversion=0.10
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import flora_translate.config as cfg
from flora_translate.inventory_constraints import reactor_is_compatible
from flora_translate.engine.sampling import (
    compute_metrics,
    hard_filter,
    choose_d_set,
    L_MAX_BENCH_M,
    V_MAX_SINGLE_REACTOR_ML,
    Q_MIN_ML_MIN,
    GAS_LIQUID_MIN_ID_MM,
)

PI = math.pi
# L fractions: stay in the 10–20 m range — avoids both extremes
# 0.5 → 10 m (compact, easy to handle), 0.9 → 18 m (near limit, more volume)
L_FRACTIONS = [0.50, 0.65, 0.80, 0.90]

# Score weights: short-τ-with-bounded-L-and-ΔP design philosophy.
# Productivity (1/τ) is the primary objective. L and ΔP get explicit soft
# penalties; the corresponding hard caps live in sampling.py::hard_filter.
# The X_FLOOR mechanism (below) prevents the productivity reward from picking
# very-short-τ candidates with non-credible conversion per pass.
SCORE_WEIGHTS = {
    "photoredox":     {"productivity": 0.50, "L": 0.20, "mixing": 0.10, "re": 0.05, "conversion": 0.10, "dP": 0.05},
    "photocatalysis": {"productivity": 0.50, "L": 0.20, "mixing": 0.10, "re": 0.05, "conversion": 0.10, "dP": 0.05},
    "photochem":      {"productivity": 0.50, "L": 0.20, "mixing": 0.10, "re": 0.05, "conversion": 0.10, "dP": 0.05},
    "thermal":        {"productivity": 0.55, "L": 0.15, "mixing": 0.15, "re": 0.05, "conversion": 0.10},
    "default":        {"productivity": 0.50, "L": 0.15, "mixing": 0.20, "re": 0.05, "conversion": 0.10},
}

# Conversion floor: below this single-pass X, productivity reward is zeroed.
# This prevents the design space from picking a τ so short the chemistry can't
# do useful work per pass just because (1/τ) is huge.
X_FLOOR = 0.30


def _l_score(L_m: float) -> float:
    """L soft penalty: full credit ≤15 m, linear penalty to 0 at 30 m.

    Matches the L_MAX_BENCH_M=30 m hard cap in sampling.py. The 15-m soft
    knee corresponds to a single bench-cassette coil; 15–30 m needs a
    multi-coil arrangement and is increasingly impractical.
    """
    if L_m <= 15.0:
        return 1.0
    return max(0.0, 1.0 - (L_m - 15.0) / 15.0)


def _dp_score(dP_bar: float) -> float:
    """ΔP soft penalty: full credit ≤2 bar, linear penalty to 0 at 5 bar.

    Matches the DELTA_P_MAX_BAR=5 bar hard cap in sampling.py. The 2-bar soft
    knee keeps ΔP well below the gas-liquid BPR floor (5 bar) so the BPR can
    independently regulate two-phase pressure.
    """
    if dP_bar <= 2.0:
        return 1.0
    return max(0.0, 1.0 - (dP_bar - 2.0) / 3.0)


@dataclass
class DesignPoint:
    # Inputs
    tau_min: float
    d_mm: float
    Q_mL_min: float
    L_fraction: float

    # Derived geometry
    V_R_mL: float = 0.0
    L_m: float = 0.0
    liquid_holdup_volume_mL: float = 0.0   # tau*Q for gas-liquid; equals V_R otherwise

    # Fluid dynamics
    Re: float = 0.0
    delta_P_bar: float = 0.0

    # Mass transfer
    t_mix_s: float = 0.0
    r_mix: float = 0.0
    Da_mass: float = 0.0

    # Kinetics
    expected_conversion: float = 0.0
    tau_kinetics_min: float = 0.0
    IF_used: float = 1.0

    # Process metrics
    STY_mol_L_h: float = 0.0
    productivity_mg_h: float = 0.0
    assumed_MW: float = 250.0

    # Gas-liquid bookkeeping (zero for liquid-only)
    is_gas_liquid: bool = False
    gas_holdup: float = 0.0
    gas_flow_actual_mL_min: float = 0.0
    two_phase_multiplier: float = 1.0
    required_bpr_bar: float = 0.0

    # Score
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)

    # Status
    feasible: bool = True
    violations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # Metadata
    tau_source: str = ""
    is_council_candidate: bool = False
    inventory_reactor_name: str = ""
    inventory_system: str = ""
    inventory_light_source: str = ""
    inventory_wavelength_nm: Optional[float] = None
    inventory_intensity_mW_cm2: Optional[float] = None


def _metrics_to_point(
    m: dict,
    *,
    tau_source: str,
    L_fraction: float,
    is_gas_liquid: bool,
    feasible: bool,
    violations: list,
    warnings: list,
    inventory_reactor=None,
) -> DesignPoint:
    """Map a metric dict from sampling.compute_metrics into a DesignPoint."""
    return DesignPoint(
        tau_min=float(m["tau_min"]),
        d_mm=float(m["d_mm"]),
        Q_mL_min=float(m["Q_mL_min"]),
        L_fraction=L_fraction,
        V_R_mL=float(m["V_R_mL"]),
        L_m=float(m["L_m"]),
        liquid_holdup_volume_mL=float(m.get("liquid_holdup_volume_mL", m["V_R_mL"])),
        Re=float(m["Re"]),
        delta_P_bar=float(m["delta_P_bar"]),
        t_mix_s=float(m["t_mix_s"]),
        r_mix=float(m["r_mix"]),
        Da_mass=float(m["Da_mass"]),
        expected_conversion=float(m["expected_conversion"]),
        tau_kinetics_min=float(m["tau_kinetics_min"]),
        IF_used=float(m["IF_used"]),
        STY_mol_L_h=float(m["STY_mol_L_h"]),
        productivity_mg_h=float(m["productivity_mg_h"]),
        assumed_MW=float(m["assumed_MW"]),
        is_gas_liquid=is_gas_liquid,
        gas_holdup=float(m.get("gas_holdup", 0.0) or 0.0),
        gas_flow_actual_mL_min=float(m.get("gas_flow_actual_mL_min", 0.0) or 0.0),
        two_phase_multiplier=float(m.get("two_phase_multiplier", 1.0) or 1.0),
        required_bpr_bar=float(m.get("required_bpr_bar", 0.0) or 0.0),
        feasible=feasible,
        violations=list(violations),
        warnings=list(warnings),
        inventory_reactor_name=getattr(inventory_reactor, "name", "") if inventory_reactor else "",
        inventory_system=getattr(inventory_reactor, "system", "") if inventory_reactor else "",
        inventory_light_source=getattr(inventory_reactor, "light_source", "") if inventory_reactor else "",
        inventory_wavelength_nm=getattr(inventory_reactor, "wavelength_nm", None) if inventory_reactor else None,
        inventory_intensity_mW_cm2=getattr(inventory_reactor, "intensity_mW_cm2", None) if inventory_reactor else None,
        tau_source=tau_source,
    )


def _inventory_reactors(inventory) -> list:
    try:
        return [r for r in (getattr(inventory, "reactors", []) or []) if r.volume_mL and r.ID_mm]
    except AttributeError:
        return []


def _flow_for_target_volume(
    *,
    target_volume_mL: float,
    tau_min: float,
    d_mm: float,
    solvent: str,
    temperature_C: float,
    concentration_M: float,
    assumed_MW: float,
    IF_used: float,
    tau_kinetics_min: float,
    pump_max: float,
    is_photochem: bool,
    is_gas_liquid: bool,
    BPR_bar: float,
    ext_coeff: Optional[float],
    tau_source: str,
) -> tuple[float, dict]:
    """Find liquid Q that makes compute_metrics hit a target tube volume."""

    q = max(target_volume_mL / max(tau_min, 1e-9), Q_MIN_ML_MIN)
    metrics: dict = {}
    for _ in range(8):
        metrics = compute_metrics(
            tau_min=tau_min,
            d_mm=d_mm,
            Q_mL_min=q,
            solvent=solvent,
            temperature_C=temperature_C,
            concentration_M=concentration_M,
            assumed_MW=assumed_MW,
            IF_used=IF_used,
            tau_kinetics_min=tau_kinetics_min,
            pump_max_bar=pump_max,
            is_photochem=is_photochem,
            is_gas_liquid=is_gas_liquid,
            BPR_bar=BPR_bar,
            extinction_coeff_M_cm=ext_coeff,
            tau_source=tau_source,
        )
        observed = float(metrics.get("V_R_mL") or 0.0)
        if observed <= 0:
            break
        ratio = target_volume_mL / observed
        if abs(ratio - 1.0) < 0.001:
            break
        q *= ratio
    return q, metrics


class DesignSpaceSearch:
    """Grid search over (τ, d, Q) combinations before the council.

    Uses `flora_translate.engine.sampling.compute_metrics` and `hard_filter`
    as the single source of truth for metrics and feasibility. This ensures
    Design Space and the Council's Designer cannot disagree about whether
    a candidate is feasible.
    """

    def run(
        self,
        batch_record,
        chemistry_plan=None,
        calculations=None,
        inventory=None,
        reaction_class: str = "default",
    ) -> list[DesignPoint]:
        """Enumerate and score all (τ, d, Q) design points.

        Returns the full list (feasible + infeasible), sorted by score desc.
        """
        # ── τ_center and τ_lit ────────────────────────────────────────────
        tau_center = 30.0
        if calculations is not None:
            try:
                tau_center = float(calculations.residence_time_min or 30.0)
            except (TypeError, AttributeError):
                tau_center = 30.0

        tau_lit: Optional[float] = None
        if calculations is not None:
            tau_lit = getattr(calculations, "tau_analogy_min", None) or \
                      getattr(calculations, "tau_class_min", None)

        # ── Build τ_values ────────────────────────────────────────────────
        tau_candidates = [
            tau_center * 0.75,
            tau_center,
            tau_center * 1.25,
            tau_center * 1.50,
        ]
        if tau_lit and tau_lit > 0:
            tau_candidates.append(tau_lit / 2.0)
            if tau_lit > tau_center:
                tau_candidates.append(tau_lit)

        seen: set[float] = set()
        tau_values: list[float] = []
        tau_sources: dict[float, str] = {}
        for t in tau_candidates:
            t_r = round(t, 1)
            if t_r < 5.0:
                t_r = 5.0
            if t_r in seen:
                continue
            seen.add(t_r)
            tau_values.append(t_r)
            if abs(t - tau_center) < 0.01:
                tau_sources[t_r] = "center"
            elif abs(t - tau_center * 0.75) < 0.01:
                tau_sources[t_r] = "center×0.75"
            elif abs(t - tau_center * 1.25) < 0.01:
                tau_sources[t_r] = "center×1.25"
            elif abs(t - tau_center * 1.50) < 0.01:
                tau_sources[t_r] = "center×1.50"
            elif tau_lit and abs(t - tau_lit / 2.0) < 0.01:
                tau_sources[t_r] = "τ_lit/2"
            elif tau_lit and abs(t - tau_lit) < 0.01:
                tau_sources[t_r] = "τ_lit"
            else:
                tau_sources[t_r] = "derived"

        tau_values.sort()
        batch_time_min = 0.0
        if batch_record is not None:
            try:
                batch_time_min = float((batch_record.reaction_time_h or 0.0) * 60.0)
            except AttributeError:
                batch_time_min = 0.0
        max_tau_min: Optional[float] = None
        if (
            (getattr(cfg, "FLOW_TRANSLATION_POLICY", "intensify") or "intensify").lower() == "intensify"
            and batch_time_min > 0
        ):
            max_tau_min = batch_time_min * float(getattr(cfg, "FLOW_MAX_TAU_TO_BATCH_RATIO", 1.0) or 1.0)
            tau_values = [t for t in tau_values if t <= max_tau_min]
            if not tau_values and max_tau_min >= 5.0:
                tau_values = [round(max_tau_min, 1)]
                tau_sources[tau_values[0]] = "batch_ceiling"

        # ── Photochem detection ───────────────────────────────────────────
        is_photochem = False
        if chemistry_plan is not None:
            rc = (getattr(chemistry_plan, "reaction_class", "") or "").lower()
            if any(x in rc for x in ("photo", "redox", "photocatalysis")):
                is_photochem = True
            wl = getattr(chemistry_plan, "recommended_wavelength_nm", None)
            if wl and wl > 0:
                is_photochem = True
        if batch_record is not None and not is_photochem:
            wl_br = getattr(batch_record, "wavelength_nm", None)
            if wl_br and wl_br > 0:
                is_photochem = True

        # ── Gas-liquid detection ──────────────────────────────────────────
        # Critical: design_space MUST be gas-liquid-aware so it shares the
        # same two-phase ΔP and BPR floor logic as the Council Designer.
        # Otherwise design_space passes points the Council immediately rejects,
        # causing the council to be silently skipped via the "no survivors"
        # fallback path.
        is_gas_liquid = bool(getattr(calculations, "is_gas_liquid", False))
        if not is_gas_liquid and chemistry_plan is not None:
            for stream in getattr(chemistry_plan, "stream_logic", []) or []:
                phase = (getattr(stream, "phase", "") or "").lower()
                if phase == "gas":
                    is_gas_liquid = True
                    break
                # Stream contains pure gas reagent (O2, H2, etc.)
                reagents = [str(r).lower() for r in (getattr(stream, "reagents", []) or [])]
                if any(g in r for g in ("o2", "o₂", "h2", "h₂", "co2", "co₂", "cl2", "cl₂", "ozone", "o3", "o₃") for r in reagents):
                    is_gas_liquid = True
                    break

        # ── d_values (photochem + gas-liquid aware) ───────────────────────
        # choose_d_set returns [0.5,0.75,1.0] for photochem alone, but
        # [0.75,1.0,1.6] for photochem+gas-liquid (gas holdup makes 0.5 mm
        # impractical), and [0.75,1.0,1.6] for gas-liquid alone.
        d_values = choose_d_set(
            is_photochem=is_photochem,
            is_gas_liquid=is_gas_liquid,
        )

        # ── pump_max from inventory ───────────────────────────────────────
        pump_max = 20.0
        if inventory is not None:
            try:
                pumps = getattr(inventory, "pumps", []) or []
                if pumps:
                    pump_max = max((float(p.max_pressure_bar or 0.0) for p in pumps), default=20.0)
                    if pump_max <= 0:
                        pump_max = 20.0
                else:
                    pump_max = float(getattr(inventory, "pump_max_bar", 20.0) or 20.0)
            except (AttributeError, TypeError, ValueError):
                pump_max = 20.0

        # ── Solvent ───────────────────────────────────────────────────────
        solvent = "MeCN"
        if batch_record is not None:
            solvent = getattr(batch_record, "solvent", None) or solvent
        if chemistry_plan is not None:
            sp = getattr(chemistry_plan, "stages", None)
            if sp and sp[0].solvent:
                solvent = sp[0].solvent

        # ── Temperature, concentration, MW ────────────────────────────────
        temperature_C = float(getattr(batch_record, "temperature_C", None) or 25.0)
        concentration_M = 0.1
        if calculations is not None:
            try:
                concentration_M = float(calculations.concentration_M or 0.1)
            except AttributeError:
                pass
        if batch_record is not None:
            c = getattr(batch_record, "concentration_M", None)
            if c and c > 0:
                concentration_M = float(c)

        assumed_MW = 250.0
        if batch_record is not None:
            mw = getattr(batch_record, "product_MW", None)
            if mw and mw > 0:
                assumed_MW = float(mw)

        # ── τ_kinetics ────────────────────────────────────────────────────
        tau_kinetics_min = tau_center
        if calculations is not None:
            k_flow = getattr(calculations, "rate_constant", None)
            if k_flow and k_flow > 0:
                tau_kinetics_min = (-math.log(0.10) / k_flow) / 60.0
            tau_k_calc = getattr(calculations, "tau_kinetics_min", None)
            if tau_k_calc and tau_k_calc > 0:
                tau_kinetics_min = float(tau_k_calc)

        # ── IF ────────────────────────────────────────────────────────────
        IF_used = 1.0
        if calculations is not None:
            try:
                IF_used = float(calculations.intensification_factor or 1.0)
            except AttributeError:
                pass

        # ── BPR (from upstream calculator) ────────────────────────────────
        BPR_bar = 0.0
        if calculations is not None:
            try:
                BPR_bar = float(getattr(calculations, "bpr_pressure_bar", 0.0) or 0.0)
            except (TypeError, ValueError):
                BPR_bar = 0.0

        # ── Extinction coefficient (for Beer-Lambert if photochem) ────────
        ext_coeff: Optional[float] = None
        if calculations is not None:
            ext_coeff = getattr(calculations, "extinction_coefficient_M_cm", None)

        reactor_options = _inventory_reactors(inventory)

        # ── Enumerate ─────────────────────────────────────────────────────
        candidates: list[DesignPoint] = []

        if reactor_options:
            for tau_min in tau_values:
                tau_source = tau_sources.get(tau_min, "derived")
                for reactor in reactor_options:
                    d_mm = float(reactor.ID_mm)
                    if d_mm not in d_values:
                        # Inventory is authoritative, but still reject IDs that
                        # violate chemistry class constraints such as dry
                        # photochemistry Beer-Lambert limits.
                        if not (is_gas_liquid and d_mm >= GAS_LIQUID_MIN_ID_MM):
                            continue
                    if not reactor_is_compatible(
                        reactor,
                        temperature_C=temperature_C,
                        concentration_M=concentration_M,
                        pressure_bar=BPR_bar,
                    ):
                        continue

                    Q_mL_min, metrics = _flow_for_target_volume(
                        target_volume_mL=float(reactor.volume_mL),
                        tau_min=tau_min,
                        d_mm=d_mm,
                        solvent=solvent,
                        temperature_C=temperature_C,
                        concentration_M=concentration_M,
                        assumed_MW=assumed_MW,
                        IF_used=IF_used,
                        tau_kinetics_min=tau_kinetics_min,
                        pump_max=pump_max,
                        is_photochem=is_photochem,
                        is_gas_liquid=is_gas_liquid,
                        BPR_bar=BPR_bar,
                        ext_coeff=ext_coeff,
                        tau_source=tau_source,
                    )
                    if Q_mL_min < Q_MIN_ML_MIN:
                        continue
                    if max_tau_min is not None and tau_min > max_tau_min:
                        continue

                    ok, violations, warnings = hard_filter(
                        metrics,
                        is_photochem=is_photochem,
                        is_gas_liquid=is_gas_liquid,
                        pump_max_bar=pump_max,
                        BPR_bar=BPR_bar,
                        max_tau_min=max_tau_min,
                    )
                    if abs(float(metrics.get("V_R_mL") or 0.0) - float(reactor.volume_mL)) > 0.05:
                        violations.append(
                            f"inventory volume closure failed: {metrics.get('V_R_mL')} mL "
                            f"vs {reactor.volume_mL} mL"
                        )
                        ok = False

                    point = _metrics_to_point(
                        metrics,
                        tau_source=tau_source,
                        L_fraction=0.0,
                        is_gas_liquid=is_gas_liquid,
                        feasible=ok,
                        violations=violations,
                        warnings=warnings,
                        inventory_reactor=reactor,
                    )
                    candidates.append(point)
        else:
            for tau_min in tau_values:
                tau_source = tau_sources.get(tau_min, "derived")
                for d_mm in d_values:
                    d_m = d_mm * 1e-3
                    for L_frac in L_FRACTIONS:
                        # Derive Q from desired physical tube volume.
                        # For gas-liquid the metric calc internally inflates
                        # V_total = V_liquid / (1 - ε_gas), so we treat the
                        # target volume here as physical-tube and let
                        # compute_metrics reconcile. For consistency with the
                        # original behavior, use physical-tube here and let
                        # the engine compute liquid holdup downstream.
                        L_target_m = L_frac * L_MAX_BENCH_M
                        V_R_target_mL = (PI * d_m ** 2 / 4.0) * L_target_m * 1e6
                        # Liquid flow rate: Q_liquid × τ = liquid holdup volume.
                        # For liquid-only this equals V_R. For gas-liquid the
                        # physical V_R is larger; we size Q_liquid for the
                        # liquid holdup target so τ_liquid = τ_min.
                        if is_gas_liquid:
                            # Liquid holdup ≈ 18% of physical V_R for design (worst-case)
                            V_liquid_target_mL = V_R_target_mL * 0.18
                        else:
                            V_liquid_target_mL = V_R_target_mL
                        Q_mL_min = V_liquid_target_mL / tau_min

                        if Q_mL_min < Q_MIN_ML_MIN:
                            continue
                        if max_tau_min is not None and tau_min > max_tau_min:
                            continue

                        metrics = compute_metrics(
                            tau_min=tau_min,
                            d_mm=d_mm,
                            Q_mL_min=Q_mL_min,
                            solvent=solvent,
                            temperature_C=temperature_C,
                            concentration_M=concentration_M,
                            assumed_MW=assumed_MW,
                            IF_used=IF_used,
                            tau_kinetics_min=tau_kinetics_min,
                            pump_max_bar=pump_max,
                            is_photochem=is_photochem,
                            is_gas_liquid=is_gas_liquid,
                            BPR_bar=BPR_bar,
                            extinction_coeff_M_cm=ext_coeff,
                            tau_source=tau_source,
                        )

                        ok, violations, warnings = hard_filter(
                            metrics,
                            is_photochem=is_photochem,
                            is_gas_liquid=is_gas_liquid,
                            pump_max_bar=pump_max,
                            BPR_bar=BPR_bar,
                            max_tau_min=max_tau_min,
                        )

                        point = _metrics_to_point(
                            metrics,
                            tau_source=tau_source,
                            L_fraction=L_frac,
                            is_gas_liquid=is_gas_liquid,
                            feasible=ok,
                            violations=violations,
                            warnings=warnings,
                        )
                        candidates.append(point)

        # ── Normalize productivity and score ──────────────────────────────
        feasible_candidates = [c for c in candidates if c.feasible]
        max_prod = max((c.productivity_mg_h for c in feasible_candidates), default=1.0)
        if max_prod <= 0:
            max_prod = 1.0

        weights = SCORE_WEIGHTS.get(
            (reaction_class or "default").lower(),
            SCORE_WEIGHTS["default"],
        )
        if (reaction_class or "").lower() not in SCORE_WEIGHTS:
            for key in SCORE_WEIGHTS:
                if key in (reaction_class or "").lower():
                    weights = SCORE_WEIGHTS[key]
                    break

        for c in feasible_candidates:
            prod_score = c.productivity_mg_h / max_prod
            # Conversion floor: a candidate that does <30% conversion per pass
            # gets no productivity reward. This is the gate that stops the
            # productivity-led scoring from chasing τ → 0 at the expense of
            # ever producing meaningful product.
            if c.expected_conversion < X_FLOOR:
                prod_score = 0.0
            L_score = _l_score(c.L_m)
            dP_score = _dp_score(c.delta_P_bar)
            mixing_score = 1.0 - min(c.r_mix / 0.20, 1.0)
            re_score = max(0.0, 1.0 - (c.Re / 2300.0))
            conv_score = c.expected_conversion

            score = (
                weights["productivity"] * prod_score
                + weights["L"] * L_score
                + weights["mixing"] * mixing_score
                + weights["re"] * re_score
                + weights["conversion"] * conv_score
                + weights.get("dP", 0.0) * dP_score
            )
            c.score = round(score, 5)
            c.score_breakdown = {
                "productivity": round(prod_score, 4),
                "L": round(L_score, 4),
                "dP": round(dP_score, 4),
                "mixing": round(mixing_score, 4),
                "re": round(re_score, 4),
                "conversion": round(conv_score, 4),
            }

        feasible_candidates.sort(key=lambda c: c.score, reverse=True)
        infeasible = [c for c in candidates if not c.feasible]

        candidates = feasible_candidates + infeasible

        if feasible_candidates:
            feasible_candidates[0].is_council_candidate = True

        return candidates


def get_council_starting_point(candidates: list[DesignPoint]) -> Optional[DesignPoint]:
    """Return the top feasible candidate to use as the council starting proposal."""
    feasible = [c for c in candidates if c.feasible]
    return feasible[0] if feasible else None


def candidates_to_dicts(candidates: list[DesignPoint]) -> list[dict]:
    """Convert to JSON-serializable list of dicts for storage in result dict."""
    import dataclasses
    return [dataclasses.asdict(c) for c in candidates]


def feasible_candidates_as_council_seeds(
    candidates: list[DesignPoint],
    *,
    BPR_bar: float = 0.0,
    tubing_material: str = "FEP",
    concentration_M: float = 0.1,
    temperature_C: float = 25.0,
    batch_time_min: Optional[float] = None,
    translation_policy: str = "intensify",
    n_max: int = 6,
) -> list[dict]:
    """Return top-N feasible Design Space candidates in the dict shape the
    Council Designer's hard-gate filter expects.

    Used as the fallback survivor pool when the Council's own Designer
    sampling returns zero feasible candidates. Each seed is annotated with
    `source="design_space"` so the council log can show where it came from.
    """
    feasible = [c for c in candidates if c.feasible][:max(1, n_max)]
    out: list[dict] = []
    for idx, c in enumerate(feasible, start=1):
        d = {
            "id": idx,
            "tau_min": round(c.tau_min, 3),
            "d_mm": round(c.d_mm, 3),
            "Q_mL_min": round(c.Q_mL_min, 5),
            "tau_source": c.tau_source or "design_space",
            "V_R_mL": round(c.V_R_mL, 4),
            "liquid_holdup_volume_mL": round(c.liquid_holdup_volume_mL, 4),
            "L_m": round(c.L_m, 3),
            "Re": round(c.Re, 2),
            "flow_regime": "laminar" if c.Re < 2300 else "turbulent",
            "velocity_m_s": 0.0,
            "delta_P_bar": round(c.delta_P_bar, 5),
            "delta_P_headroom_pct": 0.0,
            "gas_holdup": round(c.gas_holdup, 4),
            "gas_flow_actual_mL_min": round(c.gas_flow_actual_mL_min, 4),
            "gas_liquid_ratio": 0.0,
            "two_phase_multiplier": round(c.two_phase_multiplier, 4),
            "required_bpr_bar": round(c.required_bpr_bar, 2),
            "t_mix_s": round(c.t_mix_s, 3),
            "r_mix": round(c.r_mix, 5),
            "Da_mass": round(c.Da_mass, 4),
            "expected_conversion": round(c.expected_conversion, 4),
            "tau_kinetics_min": round(c.tau_kinetics_min, 2),
            "IF_used": round(c.IF_used, 2),
            "STY_mol_L_h": round(c.STY_mol_L_h, 4),
            "productivity_mg_h": round(c.productivity_mg_h, 2),
            "assumed_MW": c.assumed_MW,
            "is_photochem": False,  # not authoritative; council recomputes
            "BPR_bar": round(BPR_bar or 0.0, 2),
            "tubing_material": tubing_material,
            "concentration_M": concentration_M,
            "temperature_C": temperature_C,
            "batch_time_min": batch_time_min,
            "translation_policy": translation_policy,
            "feasible": True,
            "violations": list(c.violations),
            "warnings": list(c.warnings),
            "pareto_front": idx == 1,
            "source": "design_space_fallback",
            "design_space_score": round(c.score, 5),
            "inventory_reactor_name": c.inventory_reactor_name,
            "inventory_system": c.inventory_system,
            "inventory_light_source": c.inventory_light_source,
            "inventory_wavelength_nm": c.inventory_wavelength_nm,
            "inventory_intensity_mW_cm2": c.inventory_intensity_mW_cm2,
        }
        out.append(d)
    return out
