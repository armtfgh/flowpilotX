import pytest
from flora_translate.engine.council_v4 import chief


def args():
    return dict(winner_id=1, winner={"id": 1, "batch_time_min": 30, "tau_min": 60,
                "d_mm": 1, "Q_mL_min": .1, "L_m": 2},
                weighted_scores=[{"candidate_id": 1, "PVS": 2.1}],
                intensification_mandate={"minimum_flow_advantage": "higher yield"})


def test_specific_rationale_does_not_need_advantage_keyword():
    decision = {"selection_rationale": "Retain catalyst in the installed bed; test conversion at the supplied pump minimum.",
                "selection_flag": "SCREEN"}
    chief._ensure_selection_justification(decision, **args())
    assert decision["selection_justification"] == decision["selection_rationale"]
    assert decision["selection_flag"] == "SCREEN"


@pytest.mark.parametrize("flag", ["SCREEN", "REQUIRES_HUMAN_REVIEW", "BLOCK"])
def test_revision_keeps_renewed_review_and_does_not_promote_its_flag(flag):
    decision = {"selection_justification": "Longer exposure is a hypothesis requiring conversion measurement.",
                "selection_flag": flag}
    chief._refresh_selection_justification_from_winner(decision, **args())
    assert decision["selection_flag"] == flag
    assert decision["selection_justification"].startswith("Longer exposure is a hypothesis")
    assert "nominal exposure=60.00 min" in decision["selection_justification"]
    assert "recorded batch duration=30.00 min" in decision["selection_justification"]
    assert "residence-time reduction" not in decision["selection_justification"]
    assert "higher yield" not in decision["selection_justification"]


def test_missing_rationale_does_not_invent_superiority_from_pvs():
    decision = {}
    chief._ensure_selection_justification(decision, **args())
    assert decision["selection_flag"] == "SCREEN"
    assert "require experimental measurement" in decision["selection_justification"]
    assert "higher yield" not in decision["selection_justification"]
    assert "risks acceptable" not in decision["selection_justification"]
