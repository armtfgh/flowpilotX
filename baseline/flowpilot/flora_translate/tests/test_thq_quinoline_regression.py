"""Regression test for the tetrahydroquinoline -> quinoline aerobic
photo-oxidation case study (May 27 2026).

In the original log this protocol triggered:
  - calculator τ = 187.5 min, ΔP = 64 bar (way above BPR ceiling)
  - design_space top candidate τ = 93.8 min, d = 1.0 mm (feasible)
  - council Designer v4 sampling: 0 feasible / 4 sampling-infeasible
  - council fully SKIPPED → no domain agent input, no Skeptic audit
  - final confidence: MEDIUM

The fixes verified here:
  1. design_space and sampling agree about feasibility (unified function)
  2. gas-liquid photochem allows d up to 1.6 mm (not capped at 1.0 mm)
  3. design space candidates flow into the council as fallback seeds when
     the Designer's own sampling returns zero
  4. O2-is-reagent disambiguation: aerobic oxidation is NOT O2-inhibited
"""

from __future__ import annotations

from flora_translate.engine.design_space import (
    DesignSpaceSearch,
    feasible_candidates_as_council_seeds,
    get_council_starting_point,
)
from flora_translate.engine.sampling import (
    choose_d_set,
    compute_metrics,
    generate_candidates,
    hard_filter,
)
from flora_translate.engine.council_v4.designer import _apply_v4_hard_gates
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    ProcessStage,
    StreamLogic,
)


THQ_PROTOCOL_TEXT = (
    "Photocatalyst-free aerobic oxidative dehydrogenation of "
    "6-methyl-1,2,3,4-tetrahydroquinoline (0.20 mmol) in DMSO (0.40 mL, 0.50 M) "
    "with O2 (1 atm, 2.0 equiv from balloon) under blue LEDs (450 nm), 40 C, 15 h. "
    "Affords 6-methylquinoline in 75% yield."
)


def _thq_batch() -> BatchRecord:
    return BatchRecord(
        reaction_description=THQ_PROTOCOL_TEXT,
        solvent="DMSO",
        temperature_C=40,
        reaction_time_h=15,
        concentration_M=0.5,
        scale_mmol=0.2,
        wavelength_nm=450,
        atmosphere="O2",
    )


def _thq_plan() -> ChemistryPlan:
    return ChemistryPlan(
        reaction_class="photoredox aerobic oxidation",
        mechanism_type="direct photoexcitation, SET to O2",
        oxygen_sensitive=False,
        o2_is_reagent=True,
        recommended_wavelength_nm=450,
        stream_logic=[
            StreamLogic(stream_label="A", reagents=["6-methyl-THQ", "DMSO"], concentration_M=0.5, phase="liquid"),
            StreamLogic(stream_label="B", reagents=["O2"], phase="gas"),
        ],
        stages=[
            ProcessStage(
                stage_number=1,
                stage_name="Aerobic photoredox oxidation",
                requires_light=True,
                wavelength_nm=450,
                solvent="DMSO",
                atmosphere="O2",
                feed_streams=[
                    StreamLogic(stream_label="A", reagents=["6-methyl-THQ", "DMSO"], concentration_M=0.5, phase="liquid"),
                    StreamLogic(stream_label="B", reagents=["O2"], phase="gas"),
                ],
            )
        ],
    )


# ─────────────────────────────────────────────────────────────────────────
#  1. Photochem + gas-liquid d ceiling is 1.6 mm, not 1.0 mm
# ─────────────────────────────────────────────────────────────────────────

def test_photochem_gas_liquid_d_set_includes_1p6_mm():
    """The legacy ceiling of 1.0 mm for any photochem killed every candidate
    when O2 holdup pushed liquid into a small effective cross-section.
    Gas-liquid photochem must allow up to 1.6 mm."""
    d_set = choose_d_set(is_photochem=True, is_gas_liquid=True)
    assert 1.6 in d_set, f"d=1.6 mm missing from photochem+gas_liquid d_set: {d_set}"
    # Dry photochem stays at 1.0 mm
    dry = choose_d_set(is_photochem=True, is_gas_liquid=False)
    assert 1.6 not in dry, f"dry photochem must NOT include 1.6 mm: {dry}"
    assert max(dry) == 1.0


def test_hard_filter_allows_d_1p6_for_gas_liquid_photochem():
    """sampling.hard_filter must not reject d=1.6 mm for gas-liquid photochem."""
    metrics = compute_metrics(
        tau_min=5.0, d_mm=1.6, Q_mL_min=2.0,
        solvent="DMSO", temperature_C=40, concentration_M=0.5,
        assumed_MW=143.0, IF_used=6.0, tau_kinetics_min=5.0,
        pump_max_bar=20.0, is_photochem=True, is_gas_liquid=True,
        BPR_bar=7.0,
    )
    ok, violations, _ = hard_filter(
        metrics, is_photochem=True, is_gas_liquid=True,
        pump_max_bar=20.0, BPR_bar=7.0,
    )
    # The Beer-Lambert d>1.0 mm violation should NOT fire at d=1.6 mm for gas-liquid
    inner_filter_violations = [v for v in violations if "Beer-Lambert" in v]
    assert not inner_filter_violations, (
        f"d=1.6 mm gas-liquid photochem was rejected with Beer-Lambert: {inner_filter_violations}"
    )


def test_hard_filter_still_rejects_d_1p6_for_dry_photochem():
    """Dry (liquid-only) photochem must still cap at 1.0 mm."""
    metrics = compute_metrics(
        tau_min=5.0, d_mm=1.6, Q_mL_min=2.0,
        solvent="EtOH", temperature_C=25, concentration_M=0.1,
        assumed_MW=200.0, IF_used=4.0, tau_kinetics_min=5.0,
        pump_max_bar=20.0, is_photochem=True, is_gas_liquid=False,
    )
    ok, violations, _ = hard_filter(
        metrics, is_photochem=True, is_gas_liquid=False,
        pump_max_bar=20.0, BPR_bar=0.0,
    )
    assert any("Beer-Lambert" in v for v in violations), (
        "Dry photochem at d=1.6 mm must still violate Beer-Lambert"
    )


# ─────────────────────────────────────────────────────────────────────────
#  2. Design Space → Council fallback path produces seeds
# ─────────────────────────────────────────────────────────────────────────

class _StubCalc:
    """Minimal stand-in for DesignCalculations used in pre-council seeding."""
    is_gas_liquid = True
    residence_time_min = 90.0
    tau_analogy_min = 180.0
    tau_class_min = None
    tau_kinetics_min = 90.0
    rate_constant = None
    intensification_factor = 5.0
    concentration_M = 0.5
    pump_max_bar = 20.0
    bpr_pressure_bar = 7.0
    extinction_coefficient_M_cm = None


def test_design_space_finds_feasible_candidates_for_thq_protocol():
    """The fixed pipeline must find at least one feasible design space candidate
    for the THQ aerobic oxidation. Originally the council got 0 survivors and
    silently skipped — that path is what this test guards against."""
    batch = _thq_batch()
    plan = _thq_plan()
    calc = _StubCalc()
    candidates = DesignSpaceSearch().run(
        batch_record=batch, chemistry_plan=plan, calculations=calc, inventory=None,
        reaction_class="photoredox",
    )
    feasible = [c for c in candidates if c.feasible]
    assert feasible, (
        "Design Space found ZERO feasible candidates for THQ protocol — "
        "this is the regression we fixed. Check d_set / two-phase ΔP / BPR ceiling logic."
    )
    # Top candidate must be gas-liquid-aware (gas_holdup > 0)
    top = get_council_starting_point(candidates)
    assert top is not None
    assert top.is_gas_liquid, "Top candidate must carry is_gas_liquid=True"
    assert top.gas_holdup > 0, f"gas_holdup must be >0, got {top.gas_holdup}"


def test_design_space_seeds_pass_council_v4_hard_gates():
    """Feasible Design Space candidates fed as council fallback seeds must pass
    the council's v4 hard-gate filter (or carry survivable flags). If they
    don't, the fallback is useless — the council still skips."""
    batch = _thq_batch()
    plan = _thq_plan()
    calc = _StubCalc()
    candidates = DesignSpaceSearch().run(
        batch_record=batch, chemistry_plan=plan, calculations=calc, inventory=None,
        reaction_class="photoredox",
    )
    seeds = feasible_candidates_as_council_seeds(
        candidates, BPR_bar=7.0, tubing_material="FEP",
        concentration_M=0.5, temperature_C=40,
        batch_time_min=15 * 60, n_max=6,
    )
    assert seeds, "No seeds produced from feasible candidates"
    # All v4 gates should pass with relaxed X_minimum=0 (consistent with
    # uncertain_kinetics_screen branch the council uses for THQ).
    survivors, flagged = _apply_v4_hard_gates(
        list(seeds), pump_max_bar=20.0,
        is_photochem=True, is_gas_liquid=True,
        BPR_bar=7.0, X_minimum=0.0, tubing_material="FEP",
    )
    assert survivors, (
        f"All {len(seeds)} Design Space seeds were flagged by v4 hard gates: "
        f"{[f.get('reason')[:80] for f in flagged[:3]]}"
    )


# ─────────────────────────────────────────────────────────────────────────
#  3. Designer sampling agrees with design_space (no feasibility divergence)
# ─────────────────────────────────────────────────────────────────────────

def test_sampling_and_design_space_agree_on_feasibility_for_thq():
    """For the THQ regime, sampling.generate_candidates and DesignSpaceSearch
    must agree about whether a given (τ, d, Q) is feasible. Pre-fix they
    disagreed because design_space ignored gas-liquid two-phase ΔP."""
    batch = _thq_batch()
    plan = _thq_plan()
    calc = _StubCalc()
    # Run design_space first
    ds_candidates = DesignSpaceSearch().run(
        batch_record=batch, chemistry_plan=plan, calculations=calc, inventory=None,
        reaction_class="photoredox",
    )
    ds_feasible = [c for c in ds_candidates if c.feasible]
    assert ds_feasible
    # Pick one design space feasible point and re-run through sampling's
    # own metric + filter — they share compute_metrics now, so result MUST match.
    pt = ds_feasible[0]
    metrics = compute_metrics(
        tau_min=pt.tau_min, d_mm=pt.d_mm, Q_mL_min=pt.Q_mL_min,
        solvent="DMSO", temperature_C=40, concentration_M=0.5,
        assumed_MW=143.0, IF_used=5.0, tau_kinetics_min=pt.tau_kinetics_min,
        pump_max_bar=20.0, is_photochem=True, is_gas_liquid=True,
        BPR_bar=7.0,
    )
    ok, _violations, _warnings = hard_filter(
        metrics, is_photochem=True, is_gas_liquid=True,
        pump_max_bar=20.0, BPR_bar=7.0,
    )
    assert ok, (
        "Design Space passed this point but sampling.hard_filter rejected it — "
        "the two modules are out of sync (regression of the unified-feasibility fix)."
    )


# ─────────────────────────────────────────────────────────────────────────
#  4. O2-is-reagent vs O2-inhibits disambiguation
# ─────────────────────────────────────────────────────────────────────────

def test_o2_reagent_and_o2_inhibits_are_distinct_chemistry_plan_fields():
    plan = ChemistryPlan(
        reaction_class="aerobic oxidation",
        oxygen_sensitive=False,
        o2_is_reagent=True,
    )
    assert plan.o2_is_reagent is True
    assert plan.oxygen_sensitive is False
