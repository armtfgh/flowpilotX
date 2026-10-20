import math
import pytest
from flora_translate.design_realizer import _superficial_velocity_estimates, _replace_stale_reasoning
from flora_translate.schemas import FlowProposal, StreamAssignment


def test_velocity_uses_actual_gas_flow_and_empty_cross_section():
    p = FlowProposal(flow_rate_mL_min=.1, tubing_ID_mm=4.35,
                     streams=[StreamAssignment(phase="gas", gas_flow_sccm=1, gas_flow_actual_mL_min=.05614)])
    row = _superficial_velocity_estimates(p)[0]
    area = math.pi * 4.35**2 / 4
    assert row["liquid_mm_s"] == pytest.approx(.1 * 1000 / 60 / area)
    assert row["gas_mm_s"] == pytest.approx(.05614 * 1000 / 60 / area)
    assert row["gas_mm_s"] != pytest.approx(1 * 1000 / 60 / area)
    _replace_stale_reasoning(p, [], {})
    assert "do not establish bed wetting" in p.reasoning_per_field["superficial_velocity"]
    assert "not a physical contact time" in p.reasoning_per_field["gas_time_interpretation"]


def test_missing_actual_gas_flow_stays_unresolved():
    p = FlowProposal(flow_rate_mL_min=.1, tubing_ID_mm=1,
                     streams=[StreamAssignment(phase="gas", gas_flow_sccm=1)])
    assert _superficial_velocity_estimates(p)[0]["gas_mm_s"] is None


def test_multistage_velocity_uses_local_diameter_and_flow():
    p = FlowProposal(tubing_ID_mm=10, flow_rate_mL_min=20,
        stage_parameters=[{"stage_number": 1, "d_mm": 1, "Q_liquid_mL_min": .1, "Q_gas_actual_mL_min": 0},
                          {"stage_number": 2, "d_mm": 2, "Q_liquid_mL_min": .4, "Q_gas_actual_mL_min": .2}])
    rows = _superficial_velocity_estimates(p)
    assert rows[0]["liquid_mm_s"] == pytest.approx(rows[1]["liquid_mm_s"])
    assert rows[1]["gas_mm_s"] == pytest.approx(rows[1]["liquid_mm_s"] / 2)


@pytest.mark.parametrize("diameter", [0, -1, float("nan")])
def test_invalid_geometry_does_not_emit_a_velocity(diameter):
    assert _superficial_velocity_estimates(FlowProposal(tubing_ID_mm=diameter, flow_rate_mL_min=.1)) == []
