import json
import pytest
from flora_translate.component_identity import has_stationary_catalytic_role, has_output_only_role
from flora_translate.chemistry_contract import reconcile_chemistry_plan
from flora_translate.schemas import BatchRecord, ChemistryPlan
from flora_translate.engine.council_v4 import chief


@pytest.mark.parametrize("role", ["Target product", "Reaction byproduct", "Expected intermediate"])
def test_output_identity_is_not_an_initial_charge(role):
    batch = BatchRecord(raw_text="Substrate: feed A. Reaction gives output B.")
    plan = ChemistryPlan.model_validate({"reagents": [
        {"name": "feed A", "role": "substrate"}, {"name": "output B", "role": role}],
        "stream_logic": [{"stream_label": "A", "reagents": ["feed A"]}]})
    fixed, _ = reconcile_chemistry_plan(batch, plan)
    assert all("output B" not in f.reagents for f in fixed.stream_logic)
    assert any(r.name == "output B" for r in fixed.reagents)


@pytest.mark.parametrize("role", ["Product seed additive", "Product analytical standard", "Intermediate substrate"])
def test_explicit_input_use_survives(role):
    assert not has_output_only_role(role)


@pytest.mark.parametrize("role,loading", [
    ("Heterogeneous precursor and stationary phase", "Flow: use the installed metal/support bed"),
    ("Supported reaction material", "Flow: use the installed metal/support bed"),
    ("Immobilized source", "Installed cartridge")])
def test_physical_placement_does_not_need_catalyst_keyword(role, loading):
    assert has_stationary_catalytic_role(role, declared_loading=loading)
    assert not has_stationary_catalytic_role(role, declared_loading="Flow: feed a slurry through the bed")


@pytest.mark.parametrize("role", ["Heterogeneous catalyst pumped as a slurry",
                                 "Supported catalyst suspended in the pumped feed"])
def test_explicit_pumped_role_is_not_silently_reassigned_to_a_bed(role):
    assert not has_stationary_catalytic_role(role)


def test_batch_slurry_loading_does_not_override_explicit_flow_bed():
    assert has_stationary_catalytic_role("Heterogeneous catalyst",
        declared_loading="Batch: stirred slurry; Flow: retained in installed catalyst bed")


@pytest.mark.parametrize("reply", [{"selected_candidate_id": 1, "selection_flag": "SCREEN"},
    {"selected_candidate_id": None, "selection_rationale": "Unresolved constraint"}, {}])
def test_ambiguous_screen_requires_an_explicit_second_decision(monkeypatch, reply):
    calls = []
    def fake(system, user, **kwargs):
        calls.append(user)
        result = {"selected_candidate_id": None, "first_screen_candidate_id": 1, "selection_flag": "SCREEN"} if len(calls) == 1 else reply
        return json.dumps(result), []
    monkeypatch.setattr(chief, "call_llm_with_tools", fake)
    result, _ = chief._run_chief_llm(candidates=[{"id": 1}], weighted_scores=[], scoring={}, audit={},
        table_markdown="", chemistry_brief="", objectives="first screen", disqualify_ids=set())
    assert len(calls) == 2
    assert "You may reject all candidates" in calls[1]
    assert result["selected_candidate_id"] == reply.get("selected_candidate_id")


def test_clarification_failure_preserves_rejection(monkeypatch):
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("provider interrupted")
        return json.dumps({"selected_candidate_id": None, "first_screen_candidate_id": 1, "selection_flag": "SCREEN"}), []
    monkeypatch.setattr(chief, "call_llm_with_tools", fake)
    result, _ = chief._run_chief_llm(candidates=[{"id": 1}], weighted_scores=[], scoring={}, audit={},
        table_markdown="", chemistry_brief="", objectives="first screen", disqualify_ids=set())
    assert result["selected_candidate_id"] is None
