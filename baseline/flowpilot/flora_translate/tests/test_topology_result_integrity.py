from types import SimpleNamespace

from flora_translate.main import (
    _store_blocked_topology,
    _store_process_topology,
    _topology_matches_serialized_proposal,
    _uses_offline_deoxygenation,
)
from flora_translate.schemas import FlowProposal, ProcessTopology, UnitOperation


def test_storing_topology_never_replaces_validated_proposal_volume():
    proposal = FlowProposal(
        residence_time_min=20,
        flow_rate_mL_min=0.2,
        reactor_volume_mL=10,
        tubing_ID_mm=1.016,
        tubing_material="PFA",
    )
    topology = ProcessTopology(
        topology_id="test",
        total_flow_rate_mL_min=0.2,
        residence_time_min=20,
        reactor_volume_mL=10,
        unit_operations=[
            UnitOperation(
                op_id="reactor_1",
                op_type="coil_reactor",
                label="KHU PFA coil 10 mL",
                parameters={"volume_mL": 10.0, "residence_time_min": 20.0},
            )
        ],
    )
    result = {
        "proposal": proposal.model_dump(),
        "design_calculations": {},
    }

    _store_process_topology(result, topology)

    assert result["proposal"]["reactor_volume_mL"] == 10
    assert result["design_calculations"]["topology_total_reactor_volume_mL"] == 10
    assert _topology_matches_serialized_proposal(topology, proposal)


def test_topology_rejects_matching_global_fields_with_wrong_reactor_times():
    proposal = FlowProposal(
        residence_time_min=25,
        flow_rate_mL_min=0.1,
        reactor_volume_mL=10,
    )
    topology = ProcessTopology(
        topology_id="mismatch",
        total_flow_rate_mL_min=0.1,
        residence_time_min=25,
        reactor_volume_mL=10,
        unit_operations=[
            UnitOperation(
                op_id="reactor_1",
                op_type="coil_reactor",
                parameters={"volume_mL": 5, "residence_time_min": 162},
            ),
            UnitOperation(
                op_id="reactor_2",
                op_type="coil_reactor",
                parameters={"volume_mL": 5, "residence_time_min": 162},
            ),
        ],
    )

    assert not _topology_matches_serialized_proposal(topology, proposal)


def test_blocked_candidate_has_no_executable_topology():
    disposition = SimpleNamespace(
        hard_failures=(
            SimpleNamespace(
                finding_id="INVENTORY-MULTISTAGE-TOPOLOGY",
                message="No declared serial reactor train.",
            ),
        )
    )
    result = {"svg_path": "old.svg", "png_path": "old.png"}

    _store_blocked_topology(result, disposition)

    assert result["svg_path"] == ""
    assert result["png_path"] == ""
    assert result["process_topology"]["generation_status"] == "blocked"
    assert result["process_topology"]["unit_operations"] == []


def test_negated_deoxygenation_text_cannot_create_inline_hardware_requirement():
    proposal = FlowProposal(
        deoxygenation_method=(
            "N2 blanket is optional; deoxygenation is not required for this reaction."
        )
    )

    assert _uses_offline_deoxygenation(proposal)
