import flora_translate.config as config
import flora_translate.design_calculator as design_calculator
import flora_translate.engine.council_v4.designer as designer
import flora_translate.engine.council_v4.scoring as scoring
from flora_translate.engine.council_v4.chief import (
    _derive_domain_patch,
    _extract_explicit_patch,
    _resolve_disqualify_ids,
    _run_candidate_refinement_loop,
)
from flora_translate.engine.council_v4.skeptic import _build_weak_pool_report
from flora_translate.schemas import BatchRecord, ChemistryPlan


def _candidate() -> dict:
    return {
        "id": 1,
        "tau_min": 90.0,
        "Q_mL_min": 0.1,
        "d_mm": 1.0,
        "V_R_mL": 9.0,
        "L_m": 10.0,
        "Re": 10.0,
        "delta_P_bar": 0.1,
        "r_mix": 0.1,
        "expected_conversion": 0.9,
        "productivity_mg_h": 100.0,
        "flow_sense_report": {
            "tau_ratio": 0.9,
            "process_value_score": 0.1,
            "boundary_hugging": True,
            "primary_advantage_proxy_score": 0.2,
        },
    }


def _run_designer(monkeypatch, policy: str) -> dict:
    monkeypatch.setattr(designer, "call_llm", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(
        designer,
        "generate_candidates",
        lambda **kwargs: ([_candidate()], []),
    )
    monkeypatch.setattr(
        designer,
        "attach_flow_sense_reports",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        designer,
        "_apply_v4_hard_gates",
        lambda candidates, **kwargs: (candidates, []),
    )
    return designer.run_designer_v4(
        reaction_class="thermal",
        is_photochem=False,
        is_gas_liquid=False,
        is_O2_sensitive=False,
        tau_center_min=90.0,
        tau_lit_min=90.0,
        tau_kinetics_min=90.0,
        d_center_mm=1.0,
        Q_center_mL_min=0.1,
        solvent="ethanol",
        temperature_C=25.0,
        concentration_M=0.1,
        assumed_MW=100.0,
        IF_used=1.0,
        pump_max_bar=20.0,
        batch_time_min=100.0,
        translation_policy=policy,
        N_target=1,
    )


def test_default_translation_policy_is_evidence_first() -> None:
    assert config.FLOW_TRANSLATION_POLICY == "evidence_first"


def test_evidence_first_does_not_force_limitation_if_target(monkeypatch) -> None:
    monkeypatch.setattr(
        design_calculator, "FLOW_TRANSLATION_POLICY", "evidence_first"
    )
    batch = BatchRecord(
        reaction_description="Photoredox reaction",
        reaction_time_h=10,
        yield_pct=80,
        concentration_M=0.1,
    )
    plan = ChemistryPlan(
        reaction_class="photoredox",
        intensification_mandate={"tau_reduction_target": 20.0},
    )

    result = design_calculator.DesignCalculator().run(
        batch,
        chemistry_plan=plan,
    )

    assert result.intensification_factor < 20.0


def test_evidence_first_does_not_hard_gate_model_predicted_conversion() -> None:
    candidate = _candidate()
    candidate["expected_conversion"] = 0.2

    survivors, flagged = designer._apply_v4_hard_gates(
        [candidate],
        pump_max_bar=20.0,
        is_photochem=False,
        is_gas_liquid=False,
        BPR_bar=0.0,
        X_minimum=0.0,
    )

    assert survivors == [candidate]
    assert flagged == []
    assert candidate["hard_gate_status"] == "PASS"


def test_evidence_first_rejects_model_only_final_tau_reduction(
    monkeypatch,
) -> None:
    winner = _candidate()
    winner.update(
        {
            "tau_min": 2.0,
            "d_mm": 0.75,
            "Q_mL_min": 0.5,
            "BPR_bar": 5.0,
            "tubing_material": "FEP",
            "tau_kinetics_min": 2.0,
            "IF_used": 1.0,
            "assumed_MW": 100.0,
        }
    )
    monkeypatch.setattr(
        scoring,
        "call_llm",
        lambda *args, **kwargs: (
            '{"proposed_changes":{"tau_min":0.2},'
            '"domains_that_triggered_revision":["kinetics"]}'
        ),
    )

    result = scoring.run_revision_stage(
        winner=winner,
        scoring={
            "chemistry_scores": [],
            "kinetics_scores": [
                {"candidate_id": 1, "verdict": "REVISE", "reasoning": "shorten"}
            ],
            "fluidics_scores": [],
            "safety_scores": [],
        },
        chemistry_brief="test",
        is_photochem=False,
        is_gas_liquid=False,
        pump_max_bar=20.0,
        solvent="EtOH",
        temperature_C=25.0,
        concentration_M=0.1,
        translation_policy="evidence_first",
        measured_evidence_available=False,
    )

    assert result is None


def test_evidence_first_preserves_conservative_tau_revision(monkeypatch) -> None:
    winner = _candidate()
    winner.update(
        {
            "tau_min": 2.0,
            "d_mm": 0.75,
            "Q_mL_min": 0.1,
            "BPR_bar": 5.0,
            "tubing_material": "stainless steel",
            "tau_kinetics_min": 0.1,
            "IF_used": 1.0,
            "assumed_MW": 100.0,
        }
    )
    monkeypatch.setattr(
        scoring,
        "call_llm",
        lambda *args, **kwargs: (
            '{"proposed_changes":{"tau_min":3.0},'
            '"domains_that_triggered_revision":["kinetics"]}'
        ),
    )

    result = scoring.run_revision_stage(
        winner=winner,
        scoring={
            "chemistry_scores": [],
            "kinetics_scores": [
                {
                    "candidate_id": 1,
                    "verdict": "REVISE",
                    "reasoning": "increase contact time",
                }
            ],
            "fluidics_scores": [],
            "safety_scores": [],
        },
        chemistry_brief="test",
        is_photochem=False,
        is_gas_liquid=False,
        pump_max_bar=20.0,
        solvent="EtOH",
        temperature_C=25.0,
        concentration_M=0.1,
        batch_yield_fraction=0.94,
        translation_policy="evidence_first",
        measured_evidence_available=False,
    )

    assert result is not None
    assert result["tau_min"] == 3.0


def test_evidence_first_domain_patch_does_not_enforce_intensification_ceiling() -> None:
    candidate = _candidate()
    candidate["batch_time_min"] = 100.0
    candidate["flow_sense_report"] = {"target_reduction_factor": 20.0}

    patch, _ = _derive_domain_patch(
        domain="kinetics",
        entry={
            "candidate_id": 1,
            "verdict": "REVISE",
            "proposed_changes": {"tau_min": 10.0},
        },
        candidate=candidate,
        concentration_M=0.1,
        translation_policy="evidence_first",
    )

    assert patch["tau_min"] == 10.0


def test_evidence_first_keeps_batch_proximate_candidate(monkeypatch) -> None:
    result = _run_designer(monkeypatch, "evidence_first")

    assert result["pool_metadata"]["candidates_dropped"] == 0


def test_explicit_intensify_policy_retains_hard_self_challenge(
    monkeypatch,
) -> None:
    result = _run_designer(monkeypatch, "intensify")

    assert result["pool_metadata"]["candidates_dropped"] == 1


def test_weak_pool_redesign_is_disabled_outside_intensify_policy() -> None:
    report = _build_weak_pool_report(
        candidates=[_candidate()],
        pvs_by_id={1: 0.1},
        batch_time_min=100.0,
        intensification_mandate={
            "tau_reduction_target": 3.0,
            "minimum_flow_advantage": "productivity",
        },
        pool_metadata={"pool_quality": "DEGRADED"},
        translation_policy="evidence_first",
    )

    assert report is None


def test_refinement_rejects_subcommercial_diameter_patch() -> None:
    patch, rationale = _extract_explicit_patch(
        {
            "proposed_changes": {"d_mm": 0.29},
        },
        "fluidics",
        candidate={"d_mm": 0.75},
    )

    assert "d_mm" not in patch
    assert "outside the supported commercial range" in rationale["d_mm_rejected"]


def test_infeasible_refinement_keeps_original_candidate() -> None:
    candidate = {
        "id": 1,
        "tau_min": 2.0,
        "d_mm": 0.75,
        "Q_mL_min": 6.29546,
        "V_R_mL": 12.59092,
        "L_m": 28.5,
        "Re": 180.0,
        "delta_P_bar": 3.54262,
        "r_mix": 0.1,
        "expected_conversion": 0.9,
        "productivity_mg_h": 100.0,
        "concentration_M": 0.1,
        "temperature_C": 50.0,
        "BPR_bar": 5.0,
        "tubing_material": "PFA",
        "feasible": True,
    }
    scoring = {
        "chemistry_scores": [],
        "kinetics_scores": [],
        "fluidics_scores": [
            {
                "candidate_id": 1,
                "verdict": "ACCEPT",
                "proposed_changes": {"d_mm": 0.5},
            }
        ],
        "safety_scores": [],
    }

    revised, summary = _run_candidate_refinement_loop(
        candidates=[candidate],
        scoring=scoring,
        audit={"all_errors": []},
        solvent="DMF",
        temperature_C=50.0,
        concentration_M=0.1,
        assumed_MW=100.0,
        IF_used=3.0,
        pump_max_bar=20.0,
        is_photochem=False,
        is_gas_liquid=False,
        extinction_coeff_M_cm=None,
        strong_revision_mode=True,
        max_total_revised_candidates=1,
    )

    assert len(revised) == 1
    assert revised[0]["d_mm"] == 0.75
    assert revised[0]["variant_mode"] == "original_after_rejected_revision"
    assert summary["changed_count"] == 0
    assert summary["candidate_changes"][0]["rejected_variant_count"] == 1


def test_numerical_audit_cannot_override_all_specialist_blocks() -> None:
    disqualified, overridden = _resolve_disqualify_ids(
        [{"id": 1}, {"id": 2}],
        {"blocked_by_scoring": [1, 2]},
        {"disqualify_ids": []},
    )

    assert disqualified == {1, 2}
    assert overridden == set()


def test_scoring_block_still_removes_candidate_when_alternative_exists() -> None:
    disqualified, overridden = _resolve_disqualify_ids(
        [{"id": 1}, {"id": 2}],
        {"blocked_by_scoring": [1]},
        {"disqualify_ids": []},
    )

    assert disqualified == {1}
    assert overridden == set()


def test_deterministic_audit_disqualification_is_never_overridden() -> None:
    disqualified, overridden = _resolve_disqualify_ids(
        [{"id": 1}, {"id": 2}],
        {"blocked_by_scoring": [2]},
        {"disqualify_ids": [1]},
    )

    assert disqualified == {1, 2}
    assert overridden == set()
