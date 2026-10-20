from copy import deepcopy
import pytest
from flora_translate.translation_llm import TranslationLLM
from flora_translate.schemas import FlowProposal
from flora_translate.design_evidence import capture_reasoning, reconcile_reasoning, execution_state


def gas_proposal(basis="inlet/STP apparent residence time", tau=3 / 1.1):
    return {"reactor_volume_mL": 3., "flow_rate_mL_min": .1, "temperature_C": 60.,
            "BPR_bar": 21., "residence_time_min": tau, "residence_time_basis": basis,
            "residence_time_inlet_min": 3 / 1.1, "residence_time_in_channel_min": 3 / .15614,
            "streams": [{"stream_label": "A", "contents": ["substrate"], "flow_rate_mL_min": .1},
                        {"stream_label": "B", "contents": ["Hydrogen"], "gas_flow_sccm": 1.,
                         "gas_flow_actual_mL_min": .05614}],
            "reasoning_per_field": {"residence_time_min": "Primary time uses inlet/STP gas volume.",
                                    "residence_time_inlet_min": "Same basis as primary time."}}


def test_explicit_metered_gas_without_phase_is_not_lost_in_time_normalization():
    data = TranslationLLM._normalize_proposal_data(gas_proposal())
    assert data["streams"][1]["phase"] == "gas"
    assert data["residence_time_min"] == pytest.approx(3 / 1.1)
    assert data["decision_rationale"]["residence_time_min"]["parameter_snapshot"]["residence_time_min"] == pytest.approx(3 / 1.1)


@pytest.mark.parametrize("basis,tau", [("liquid-only", 30), ("in-channel pressure-corrected", 3 / .15614),
                                      ("nominal empty-bed space time", 30)])
def test_gas_metering_does_not_replace_the_declared_primary_basis(basis, tau):
    data = gas_proposal(basis, tau)
    data["streams"][1]["phase"] = "gas"
    normalized = TranslationLLM._normalize_proposal_data(data)
    assert normalized["residence_time_basis"] == basis
    assert normalized["residence_time_min"] == pytest.approx(tau)


def test_unknown_gas_basis_does_not_invent_a_governing_time():
    data = TranslationLLM._normalize_proposal_data(gas_proposal("unresolved", 8))
    assert data["residence_time_min"] == 8
    assert data["residence_time_basis"] == "unresolved"


def test_missing_gas_rates_do_not_reinterpret_an_explicit_stp_basis_as_liquid():
    data = gas_proposal()
    data["streams"] = data["streams"][:1]
    assert TranslationLLM._normalize_proposal_data(data)["residence_time_min"] == pytest.approx(3 / 1.1)


def test_multiple_gas_feeds_are_summed_instead_of_using_only_the_first():
    data = gas_proposal(tau=20)
    data["streams"].append({"stream_label": "C", "phase": "gas", "contents": ["Nitrogen"],
                            "gas_flow_sccm": 2., "gas_flow_actual_mL_min": .11228})
    normalized = TranslationLLM._normalize_proposal_data(data)
    assert normalized["residence_time_min"] == pytest.approx(3 / 3.1, abs=1e-6)
    assert normalized["decision_rationale"]["residence_time_min"]["parameter_snapshot"]["residence_time_min"] == 20


def test_changed_time_basis_removes_old_claims_but_keeps_the_original_evidence():
    p = FlowProposal(**TranslationLLM._normalize_proposal_data(gas_proposal()))
    p.residence_time_min = 30
    p.residence_time_basis = "nominal empty-bed liquid space time"
    reconcile_reasoning(p)
    assert "residence_time_min" not in p.reasoning_per_field
    assert "residence_time_inlet_min" not in p.reasoning_per_field
    for name in ("residence_time_min", "residence_time_inlet_min"):
        record = p.decision_rationale[name]
        assert record["status"] == "requires_review_after_parameter_change"
        assert "residence_time_basis" in record["changed_dependencies"]
        assert record["text"]


def test_basis_changes_are_reviewed_even_when_the_numeric_time_is_unchanged():
    p = FlowProposal(residence_time_min=10, residence_time_basis="liquid-only",
                     reasoning_per_field={"residence_time_min": "Nominal liquid-only time."})
    capture_reasoning(p)
    original = execution_state(p)
    p.residence_time_basis = "inlet/STP"
    assert original != execution_state(p)
    reconcile_reasoning(p)
    assert p.decision_rationale["residence_time_min"]["status"] == "requires_review_after_parameter_change"


def test_semantically_equivalent_basis_labels_do_not_invalidate_reasoning():
    p = FlowProposal(residence_time_min=10, residence_time_basis="liquid-only",
                     reasoning_per_field={"residence_time_min": "Nominal liquid-only time."})
    capture_reasoning(p)
    p.residence_time_basis = "nominal empty-bed liquid space time"
    reconcile_reasoning(p)
    assert p.decision_rationale["residence_time_min"]["status"] == "conditions_preserved"
