from copy import deepcopy
from flora_translate.design_evidence import capture_reasoning, reconcile_reasoning
from flora_translate.schemas import FlowProposal
from flora_translate.engine.council_v4 import chief
from flora_translate.tests.test_inventory_council_contract import candidates


def test_repeated_reasoning_reconciliation_does_not_duplicate_text():
    p = FlowProposal(temperature_C=40, BPR_bar=20,
        reasoning_per_field={"temperature_C": "Use 40 C because no cooler is available."},
        chemistry_notes="Proposed first screen; conversion is unmeasured.")
    capture_reasoning(p)
    reconcile_reasoning(p)
    once = deepcopy(p.model_dump())
    reconcile_reasoning(p)
    reconcile_reasoning(p)
    assert p.model_dump() == once
    p.temperature_C = 100
    reconcile_reasoning(p)
    assert "Use 40 C" not in p.reasoning_per_field.get("temperature_C", "")
    assert p.decision_rationale["temperature_C"]["status"] == "requires_review_after_parameter_change"
    assert "Use 40 C" in p.decision_rationale["temperature_C"]["text"]


def test_selected_operating_point_keeps_the_chiefs_actual_justification():
    row = candidates()["survivors"][0]
    p, _ = chief._apply_winner(FlowProposal(), row, {}, chief_data={
        "selection_rationale": "Candidate 1 ranks first.",
        "selection_justification": "Longer nominal exposure is a conservative screen, not measured conversion."})
    reconcile_reasoning(p)
    record = p.decision_rationale["council_selection_justification"]
    assert record["status"] == "conditions_preserved"
    assert "not measured conversion" in record["text"]
    assert record["text"] in p.reasoning_per_field["council_selection_justification"]
