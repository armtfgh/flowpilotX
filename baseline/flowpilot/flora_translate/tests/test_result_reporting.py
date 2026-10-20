from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from flora_translate.result_reporting import build_result_report
from flora_translate.final_engineering import calculate_final_stages
from flora_translate.residence_time_basis import stp_gas_flow_for_equiv
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory, StreamAssignment, UnitOperation
from flora_design.visualizer.flowsheet_builder import _pump_label, _mfc_label, _reactor_label, _wrap


def result_fixture():
    return {"final_design": {"status": "executable", "parameters": {"BPR_bar": 3, "BPR_basis": "gauge"},
        "streams": [{"stream_label": "A", "phase": "liquid", "flow_rate_mL_min": 0.1, "introduction_stage": 1},
                    {"stream_label": "G", "phase": "gas", "gas_flow_sccm": 0.4, "flow_rate_mL_min": 0.04, "gas_flow_actual_mL_min": 0.04, "introduction_stage": 2}],
        "stages": [{"stage_number": 1, "reactor_volume_mL": 10, "Q_liquid_mL_min": 0.1, "Q_gas_sccm": 0, "residence_time_inlet_min": 100, "temperature_C": 40},
                   {"stage_number": 2, "reactor_volume_mL": 20, "Q_liquid_mL_min": 0.1, "Q_gas_sccm": 0.4, "residence_time_inlet_min": 40, "residence_time_in_channel_min": 142.857, "temperature_C": 50}]}}


def test_stage_summary_uses_only_final_inlet_flows_and_is_read_only():
    result = result_fixture()
    result["proposal"] = {"residence_time_min": 2, "flow_rate_mL_min": 999}
    before = deepcopy(result)
    report = build_result_report(result)
    assert [s["residence_time_min"] for s in report["stages"]] == [100, 40]
    assert [s["temperature_C"] for s in report["stages"]] == [40, 50]
    assert report["streams"][1]["flow_mL_min"] == 0.4
    assert report["issues"] == []
    assert "in_channel" not in json.dumps(report)
    assert "sccm" not in json.dumps(report)
    assert result == before


def test_pressure_corrections_cannot_change_reported_gas_or_time():
    result = result_fixture()
    before = build_result_report(result)
    result["final_design"]["streams"][1]["gas_flow_actual_mL_min"] = 900
    result["final_design"]["stages"][1]["residence_time_in_channel_min"] = 900
    assert build_result_report(result) == before


def test_missing_stp_never_falls_back_to_compressed_flow():
    result = result_fixture()
    del result["final_design"]["streams"][1]["gas_flow_sccm"]
    assert build_result_report(result)["streams"][1]["flow_mL_min"] is None


def test_unknown_gas_flow_does_not_get_labeled_liquid_only():
    result = result_fixture()
    del result["final_design"]["stages"][1]["Q_gas_sccm"]
    row = build_result_report(result)["stages"][1]
    assert row["residence_basis"] == "not verified"
    assert row["closure"] is None


def test_three_stages_with_additional_liquid_and_gas_feeds():
    result = result_fixture()
    final = result["final_design"]
    final["streams"].append({"stream_label": "B", "phase": "liquid", "flow_rate_mL_min": 0.2, "introduction_stage": 2})
    final["stages"][1].update(Q_liquid_mL_min=0.3, reactor_volume_mL=14, residence_time_inlet_min=20)
    final["stages"].append({"stage_number": 3, "reactor_volume_mL": 7, "Q_liquid_mL_min": 0.3,
                           "Q_gas_sccm": 0.4, "residence_time_inlet_min": 10, "temperature_C": 60})
    report = build_result_report(result)
    assert not report["issues"]
    assert [s["residence_time_min"] for s in report["stages"]] == [100, 20, 10]
    assert report["streams"][2]["introduction_stage"] == 2
    assert report["streams"][2]["flow_mL_min"] == 0.2


def test_single_stage_multiple_gases_use_sum_of_stp_flows():
    result = result_fixture()
    final = result["final_design"]
    final["stages"] = []
    final["parameters"].update(reactor_volume_mL=10, flow_rate_mL_min=0.1, residence_time_inlet_min=10)
    final["streams"].append({"stream_label": "G2", "phase": "gas", "gas_flow_stp_mL_min": 0.5, "introduction_stage": 1})
    report = build_result_report(result)
    assert report["stages"][0]["gas_flow_stp_mL_min"] == 0.9
    assert not report["issues"]


def test_withheld_result_does_not_publish_diagnostic_stage_values():
    result = result_fixture()
    result["final_design"]["status"] = "blocked"
    report = build_result_report(result)
    assert report["stages"] == report["streams"] == []


def test_inconsistent_stage_times_are_flagged_not_silently_rewritten():
    result = result_fixture()
    result["final_design"]["stages"][1]["residence_time_inlet_min"] = 2
    report = build_result_report(result)
    assert report["stages"][1]["residence_time_min"] == 2
    assert report["stages"][1]["closure"] is False
    assert "inconsistent" in report["issues"][0]


def test_single_stage_parameters_supported_without_aggregate_guessing():
    result = result_fixture()
    result["final_design"]["stages"] = []
    result["final_design"]["streams"] = result["final_design"]["streams"][:1]
    result["final_design"]["parameters"].update(reactor_volume_mL=10, flow_rate_mL_min=0.1, residence_time_min=100)
    report = build_result_report(result)
    assert len(report["stages"]) == 1
    assert report["stages"][0]["closure"] is True


def test_answer_history_and_unavailable_are_not_claimed_as_verified_effects():
    result = result_fixture()
    result["intake_package"] = {"question_log": [{"question_id": "Q-HYP-001", "question": "Hypothesis?"}, {"question_id": "Q-HIST-001", "question": "History?"}],
        "hypotheses": ["longer time"], "answers": [{"question_id": "Q-HYP-001", "answer": "shorter", "status": "answered"},
        {"question_id": "Q-HYP-001", "answer": "longer", "status": "answered"}, {"question_id": "Q-HIST-001", "answer": None, "status": "unavailable"}]}
    rows = build_result_report(result)["responses"]
    assert rows[0]["answer"] == "longer"
    assert len(rows[0]["answer_history"]) == 2
    assert rows[0]["assessment"] == "context only"
    assert rows[1]["assessment"] == "explicitly unavailable"


def test_photo_multistep_and_inventory_answers_have_traceable_final_evidence():
    result = result_fixture()
    result["intake_package"] = {"question_log": [{"question_id": q, "question": q} for q in
                                               ["Q-PHOTO-001", "Q-MULTI-001", "Q-INV-001"]]}
    result["final_design"]["stages"][0]["wavelength_nm"] = 450
    result["inventory_allocation"] = {"status": "complete", "checks": {"all_required_operations_assigned": True}}
    photo, multi, inv = build_result_report(result)["responses"]
    assert "450 nm" in str(photo["evidence"])
    assert "Stream G enters stage 2" in str(multi["evidence"])
    assert inv["assessment"] == "assignment checked"


def test_long_caption_keeps_parenthetical_chemistry_and_every_component():
    name = "(((4-methoxyphenyl)thio)methyl)trimethylsilane (1 equiv)"
    contents = [name, "catalyst (0.5 mol%)", "third", "fourth", "fifth"]
    op = UnitOperation(op_id="a", op_type="pump", label="Feed", parameters={"contents": contents,
        "instrument_name": "Very long laboratory peristaltic pump equipment name", "flow_rate_mL_min": 0.010123})
    root = ET.fromstring("<root>" + _pump_label(op) + "</root>")
    text = "".join(root.itertext()).replace(" ", "")
    for item in contents:
        assert item.replace(" ", "") in text
    assert "0.010123" in text
    assert "..." not in text and "\u2026" not in text
    assert _wrap(name).replace("\n", "").replace(" ", "") == name.replace(" ", "")


def test_renderer_inlet_units_and_no_channel_time():
    op = UnitOperation(op_id="g", op_type="mfc", label="O2", parameters={"contents": ["O2"], "gas_flow_sccm": 0.4,
        "gas_flow_actual_mL_min": 0.04, "molar_equiv": 2})
    assert "0.4 mL/min at STP" in _mfc_label(op)
    assert "sccm" not in _mfc_label(op)
    assert "2 equiv" in _mfc_label(op)
    reactor = UnitOperation(op_id="r", op_type="photoreactor", label="Reactor", parameters={"residence_time_inlet_min": 40,
        "residence_time_in_channel_min": 130, "residence_time_min": 130})
    assert "130" not in _reactor_label(reactor)
    assert "40" in _reactor_label(reactor)


@pytest.mark.parametrize("gas_solvent", ["", "none (gas phase)", "N/A"])
def test_final_calculator_recomputes_each_stage_not_lumped_total(gas_solvent):
    q = 0.1
    gas = stp_gas_flow_for_equiv(q, 0.1, 2)
    tau2 = 20 / (q + gas)
    proposal = FlowProposal(residence_time_min=100 + tau2, flow_rate_mL_min=q, concentration_M=0.1, temperature_C=40,
        BPR_bar=3, reactor_volume_mL=30, tubing_ID_mm=1.016, streams=[
        StreamAssignment(stream_label="A", phase="liquid", concentration_M=0.1, molar_equiv=1, flow_rate_mL_min=q, introduction_stage=1),
        StreamAssignment(stream_label="G", phase="gas", contents=["O2"], solvent=gas_solvent, molar_equiv=2, gas_flow_sccm=gas, introduction_stage=2)],
        stage_parameters=[{"stage_number": 1, "reactor_volume_mL": 10, "Q_liquid_mL_min": q, "residence_time_inlet_min": 100, "temperature_C": 40, "d_mm": 1.016},
                          {"stage_number": 2, "reactor_volume_mL": 20, "Q_liquid_mL_min": q, "residence_time_inlet_min": tau2, "temperature_C": 50, "d_mm": 1.016}])
    before = proposal.model_dump()
    calc = calculate_final_stages(proposal, BatchRecord(reaction_description="Two-stage oxidation with oxygen", temperature_C=25,
        concentration_M=0.1, reaction_time_h=10, solvent="ethanol"), ChemistryPlan(), LabInventory())
    assert calc["complete"], calc
    records = [x["calculations"] for x in calc["stages"]]
    assert [x["reactor_volume_mL"] for x in records] == [10, 20]
    assert [x["temperature_C"] for x in records] == [40, 50]
    assert [x["is_gas_liquid"] for x in records] == [False, True]
    assert records[0]["gas_flow_sccm"] == 0
    assert proposal.model_dump() == before


@pytest.mark.parametrize("run", ["20260907_111829_deterministic_replay", "20260907_105438_deterministic_replay"])
def test_actual_saved_multistage_runs(run):
    path = Path("outputs/gui_runs") / run / "result.json"
    if not path.exists():
        pytest.skip("Local regression archive not installed")
    result = json.loads(path.read_text())
    report = build_result_report(result)
    assert len(report["stages"]) == 2
    assert not report["issues"]
    assert report["stages"][0]["gas_flow_stp_mL_min"] == 0
    assert report["streams"][-1]["equiv"] == 2
    calcs = calculate_final_stages(FlowProposal.model_validate(result["proposal"]), BatchRecord.model_validate(result["batch_record"]),
                                   ChemistryPlan.model_validate(result["chemistry_plan"]), LabInventory.model_validate(result["inventory_snapshot"]))
    assert calcs["complete"], calcs
    for stage, calculation in zip(report["stages"], calcs["stages"]):
        assert calculation["calculations"]["reactor_volume_mL"] == stage["volume_mL"]
        assert calculation["calculations"]["residence_time_min"] == stage["residence_time_min"]
