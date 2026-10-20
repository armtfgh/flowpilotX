from copy import deepcopy

from flora_translate.final_design_contract import (
    _gas_composition_and_equivalent_issues,
    build_final_design_contract,
    publish_final_design_artifacts,
)
from flora_translate.design_realizer import _solve_gas_stream_rates
from flora_translate.schemas import FlowProposal, GasHardwareSpec, LabInventory
from flora_translate.schemas import ProcessTopology
from flora_translate.topology_semantics import (
    normalize_topology_semantics,
    topology_semantic_issues,
)


def test_final_contract_rejects_pure_oxygen_with_air_fraction():
    issues = _gas_composition_and_equivalent_issues(
        {
            "concentration_M": 0.1,
            "multiphase_metrics": {
                "limiting_liquid_flow_mL_min": 0.2,
                "limiting_reagent_concentration_M": 0.1,
            },
            "streams": [
                {
                    "stream_label": "G",
                    "phase": "gas",
                    "contents": ["O2"],
                    "pump_role": "Pure O2 feed",
                    "gas_flow_sccm": 2.0,
                    "gas_reagent_mole_fraction": 0.21,
                    "molar_equiv": 2.0,
                }
            ],
        }
    )

    codes = {item["code"] for item in issues}
    assert "FINAL-GAS-COMPOSITION-INCONSISTENT" in codes
    assert "FINAL-GAS-EQUIVALENT-CLOSURE" in codes


def _hydrogen_result() -> dict:
    topology = ProcessTopology.model_validate(
        {
            "topology_id": "hydrogenolysis",
            "compilation_status": "inventory_assigned",
            "total_flow_rate_mL_min": 0.15,
            "residence_time_min": 0.6452,
            "reactor_volume_mL": 3.0,
            "unit_operations": [
                {
                    "op_id": "pump_a",
                    "op_type": "pump",
                    "label": "DMAOL pump",
                    "parameters": {
                        "stream": "A",
                        "phase": "liquid",
                        "contents": ["DMAOL (0.126 M, 1.0 equiv)", "Methanol"],
                        "solvent": "Methanol",
                        "flow_rate_mL_min": 0.15,
                    },
                    "inventory_item_id": "pump_1",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "mfc_h2",
                    "op_type": "mfc",
                    "label": "Hydrogen MFC",
                    "parameters": {
                        "stream": "G",
                        "phase": "gas",
                        "contents": ["H2"],
                        "gas_flow_sccm": 4.5,
                        "gas_flow_actual_mL_min": 0.25,
                    },
                    "inventory_item_id": "mfc_h2_100",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "mixer_1",
                    "op_type": "mixer",
                    "label": "Gas-liquid mixer",
                    "parameters": {},
                    "inventory_item_id": "mixer_1",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "reactor_1",
                    "op_type": "packed_bed_reactor",
                    "label": "Packed bed",
                    "parameters": {
                        "volume_mL": 3.0,
                        "Q_liquid_mL_min": 0.15,
                        "temperature_C": 60.0,
                        "Q_gas_sccm": 4.5,
                        "Q_gas_actual_mL_min": 0.25,
                        "residence_time_min": 0.6452,
                        "residence_time_inlet_min": 0.6452,
                        "residence_time_in_channel_min": 7.5,
                        "residence_time_basis": "inlet/STP apparent residence time",
                    },
                    "inventory_item_id": "reactor_3",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "heater_1",
                    "op_type": "heater",
                    "label": "External temperature controller",
                    "parameters": {"temperature_C": 60.0},
                    "inventory_item_id": "bath_60",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "bpr_1",
                    "op_type": "bpr",
                    "label": "21 bar BPR",
                    "parameters": {"pressure_bar": 21.0},
                    "inventory_item_id": "bpr_21",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "separator_1",
                    "op_type": "phase_separator",
                    "label": "Vented gas-liquid separator",
                    "parameters": {},
                    "inventory_item_id": "separator_vented",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "collector_1",
                    "op_type": "collector",
                    "label": "Product collector",
                    "parameters": {},
                    "inventory_item_id": "collector_1",
                    "assignment_status": "assigned",
                },
            ],
            "streams": [
                {"stream_id": "s1", "from_op": "pump_a", "to_op": "mixer_1"},
                {"stream_id": "s2", "from_op": "mfc_h2", "to_op": "mixer_1"},
                {"stream_id": "s3", "from_op": "mixer_1", "to_op": "reactor_1"},
                {"stream_id": "s4", "from_op": "reactor_1", "to_op": "bpr_1"},
                {"stream_id": "s5", "from_op": "bpr_1", "to_op": "separator_1"},
                {
                    "stream_id": "s6",
                    "from_op": "separator_1",
                    "to_op": "collector_1",
                    "label": "liquid product",
                },
            ],
        }
    )
    topology = normalize_topology_semantics(topology)
    inventory = {
        "pumps": [{"equipment_id": "pump_1", "name": "Liquid pump"}],
        "gas_hardware": [
            {"equipment_id": "mfc_h2_100", "name": "Hydrogen MFC", "gas": "H2"},
            {"equipment_id": "n2_purge", "name": "Nitrogen purge line", "gas": "N2"},
            {"equipment_id": "h2_check", "name": "Hydrogen non-return check valve"},
        ],
        "mixers": [{"equipment_id": "mixer_1", "name": "Gas-liquid mixer"}],
        "reactors": [{"equipment_id": "reactor_3", "name": "Packed bed"}],
        "temperature_controllers": [{"equipment_id": "bath_60", "name": "Water bath"}],
        "pressure_controllers": [{"equipment_id": "bpr_21", "name": "21 bar BPR"}],
        "separators": [{"equipment_id": "separator_vented", "name": "Vented separator"}],
        "collectors": [{"equipment_id": "collector_1", "name": "Collector"}],
    }
    return {
        "recommended_disposition": "SCREEN",
        "batch_record": {
            "reaction_description": "Hydrogenolysis/debenzylation of DMAOL to 3-azetidinol under H2.",
            "raw_text": "Hydrogenolysis of N-diphenylmethylazetidin-3-ol under H2.",
        },
        "chemistry_plan": {
            "reaction_name": "Hydrogenolysis of DMAOL to 3-azetidinol",
            "reaction_class": "hydrogenolysis/debenzylation",
            "bond_broken": "N-C(benzyl)",
            "reagents": [
                {"name": "DMAOL", "role": "substrate", "equiv_or_loading": "1.0 equiv"},
                {"name": "Methanol", "role": "solvent", "equiv_or_loading": ""},
                {"name": "H2", "role": "reductant", "equiv_or_loading": "10 equiv"},
            ],
        },
        "proposal": {
            "residence_time_min": 0.6452,
            "residence_time_inlet_min": 0.6452,
            "residence_time_in_channel_min": 7.5,
            "residence_time_basis": "inlet/STP apparent residence time",
            "flow_rate_mL_min": 0.15,
            "reactor_volume_mL": 3.0,
            "temperature_C": 60.0,
            "concentration_M": 0.126,
            "BPR_bar": 21.0,
            "reactor_type": "packed-bed",
            "tubing_material": "stainless steel",
            "tubing_ID_mm": 4.35,
            "streams": [
                {
                    "stream_label": "A",
                    "contents": ["DMAOL (0.126 M, 1.0 equiv)", "Methanol"],
                    "solvent": "Methanol",
                    "concentration_M": 0.126,
                    "flow_rate_mL_min": 0.15,
                    "phase": "liquid",
                    "molar_equiv": 1.0,
                },
                {
                    "stream_label": "G",
                    "contents": ["H2"],
                    "phase": "gas",
                    "flow_rate_mL_min": 0.25,
                    "gas_flow_sccm": 4.5,
                    "gas_flow_actual_mL_min": 0.25,
                    "molar_equiv": 10.0,
                },
            ],
        },
        "design_calculations": {
            "residence_time_min": 0.6452,
            "residence_time_inlet_min": 0.6452,
            "residence_time_in_channel_min": 7.5,
            "residence_time_basis": "inlet/STP apparent residence time",
            "reactor_volume_mL": 3.0,
            "flow_rate_mL_min": 0.15,
        },
        "inventory_snapshot": inventory,
        "inventory_allocation": {"status": "complete", "instrument_manifest": []},
        "instrument_manifest": [],
        "process_topology": topology.model_dump(mode="json"),
        "final_validation": {
            "status": "ready",
            "checks": {
                "geometry_closure": True,
                "inventory_topology_assignment_complete": True,
                "topology_matches_serialized_design": True,
                "gas_primary_basis_inlet_stp": True,
            },
        },
        "safety_report": {
            "total_checks": 0,
            "validation_experiments": ["Use stale Q=0.138 mL/min"],
        },
    }


def _issue_codes(contract: dict) -> set[str]:
    return {
        item["code"] for item in contract["consistency"]["issues"]
    }


def test_phase_compiler_propagates_gas_and_mixed_phase_and_excludes_controller():
    result = _hydrogen_result()
    topology = ProcessTopology.model_validate(result["process_topology"])
    phases = {item.stream_id: item.stream_type for item in topology.streams}

    assert phases["s1"] == "liquid"
    assert phases["s2"] == "gas"
    assert phases["s3"] == "gas_liquid"
    assert phases["s4"] == "gas_liquid"
    assert "External temperature controller" not in topology.pid_description
    assert topology_semantic_issues(topology) == []


def test_valid_design_compiles_one_hashed_executable_artifact_bundle():
    contract = build_final_design_contract(_hydrogen_result())

    assert contract["status"] == "executable"
    assert len(contract["canonical_sha256"]) == 64
    assert contract["safety"]["complete"] is True
    assert contract["operating_procedure"]
    assert contract["validation_experiments"]
    assert all(contract["consistency"]["semantic_checks"].values())
    preparation = next(
        item for item in contract["operating_procedure"]
        if item["step_id"] == "PREP-A"
    )
    assert "mL final-volume basis" in preparation["instruction"]
    assert "mmol" in preparation["instruction"]
    check_valve = next(
        item for item in contract["operating_procedure"]
        if item["step_id"] == "H2-BACKFLOW-PREVENTION"
    )
    assert check_valve["equipment_ids"] == ["h2_check"]


def test_chemistry_identity_mutation_blocks_release():
    result = _hydrogen_result()
    result["chemistry_plan"]["reaction_name"] = "Hydrogenation of DMAOL"
    result["chemistry_plan"]["reaction_class"] = "catalytic hydrogenation"

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-CHEMISTRY-IDENTITY-DRIFT" in _issue_codes(contract)


def test_phase_mutation_blocks_release():
    result = _hydrogen_result()
    result["process_topology"]["streams"][1]["stream_type"] = "liquid"

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-TOPOLOGY-PHASE-INCONSISTENT" in _issue_codes(contract)


def test_missing_hydrogen_backflow_hardware_blocks_release():
    result = _hydrogen_result()
    result["inventory_snapshot"]["gas_hardware"] = [
        item
        for item in result["inventory_snapshot"]["gas_hardware"]
        if item["equipment_id"] != "h2_check"
    ]

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-SAFETY-CONTROLS-INCOMPLETE" in _issue_codes(contract)
    safety = contract["diagnostic"]["semantic_artifacts"]["safety"]
    assert "H2-BACKFLOW-PREVENTION" in safety["missing_control_ids"]


def test_unquantified_multicomponent_stream_blocks_release():
    result = _hydrogen_result()
    result["proposal"]["streams"][0]["contents"] = ["DMAOL", "TBHP", "Methanol"]
    result["process_topology"]["unit_operations"][0]["parameters"]["contents"] = [
        "DMAOL", "TBHP", "Methanol"
    ]
    result["chemistry_plan"]["reagents"].append(
        {"name": "TBHP", "role": "oxidant", "equiv_or_loading": ""}
    )

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-COMPONENT-STOICHIOMETRY-INCOMPLETE" in _issue_codes(contract)


def test_publishing_replaces_stale_council_artifacts():
    result = _hydrogen_result()
    contract = build_final_design_contract(result)

    publish_final_design_artifacts(result, contract)

    assert result["safety_report"]["source"] == "final_design"
    assert result["safety_report"]["total_checks"] > 0
    assert all("0.138" not in item for item in result["safety_report"]["validation_experiments"])
    assert result["council_safety_report_diagnostic"]["total_checks"] == 0
    assert result["canonical_design_sha256"] == contract["canonical_sha256"]
    assert "h2_check" in {
        item["equipment_id"] for item in result["instrument_manifest"]
    }


def test_council_text_cannot_change_canonical_design_hash():
    first_result = _hydrogen_result()
    first = build_final_design_contract(first_result)
    second_result = deepcopy(first_result)
    second_result["safety_report"] = {
        "total_checks": 999,
        "validation_experiments": ["Conflicting Q=999 mL/min"],
        "free_text": "This text is deliberately untrusted.",
    }

    second = build_final_design_contract(second_result)

    assert first["status"] == second["status"] == "executable"
    assert first["canonical_sha256"] == second["canonical_sha256"]


def test_rendered_topology_hash_mutation_blocks_release():
    result = _hydrogen_result()
    result["diagram_render_manifest"] = {"topology_sha256": "0" * 64}

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-DIAGRAM-TOPOLOGY-HASH-MISMATCH" in _issue_codes(contract)


def test_hypothesis_only_segmented_flow_claim_does_not_block_release():
    result = _hydrogen_result()
    result["proposal"]["streams"] = [result["proposal"]["streams"][0]]
    result["proposal"].update({
        "residence_time_min": 20.0,
        "residence_time_inlet_min": 20.0,
        "residence_time_in_channel_min": 20.0,
        "residence_time_basis": "liquid-only reactor volume / liquid flow",
    })
    result["design_calculations"].update({
        "residence_time_min": 20.0,
        "residence_time_inlet_min": 20.0,
        "residence_time_in_channel_min": 20.0,
        "residence_time_basis": "liquid-only reactor volume / liquid flow",
    })
    result["process_topology"]["residence_time_min"] = 20.0
    reactor = next(
        item for item in result["process_topology"]["unit_operations"]
        if item["op_id"] == "reactor_1"
    )
    reactor["parameters"].update({
        "residence_time_min": 20.0,
        "residence_time_inlet_min": 20.0,
        "residence_time_in_channel_min": 20.0,
        "residence_time_basis": "liquid-only reactor volume / liquid flow",
    })
    reactor["parameters"].pop("Q_gas_sccm", None)
    reactor["parameters"].pop("Q_gas_actual_mL_min", None)
    result["process_topology"]["unit_operations"] = [
        item
        for item in result["process_topology"]["unit_operations"]
        if item["op_id"] != "mfc_h2"
    ]
    result["process_topology"]["streams"] = [
        item
        for item in result["process_topology"]["streams"]
        if item["from_op"] != "mfc_h2"
    ]
    result["process_topology"] = normalize_topology_semantics(
        ProcessTopology.model_validate(result["process_topology"])
    ).model_dump(mode="json")
    result["chemistry_plan"]["intensification_mandate"] = {
        "required_mixing_regime": "slug_flow"
    }
    result["batch_record"] = {
        "raw_text": "Hydrogenolysis of DMAOL to product without a gas feed.",
        "reaction_description": "Hydrogenolysis",
    }

    contract = build_final_design_contract(result)

    assert "FINAL-MIXING-REGIME-TOPOLOGY-CONFLICT" not in _issue_codes(contract)


def test_binding_segmented_flow_claim_without_multiphase_topology_blocks_release(monkeypatch):
    from flora_translate import config

    monkeypatch.setattr(config, "FLOW_TRANSLATION_POLICY", "intensify")
    result = _hydrogen_result()
    result["proposal"]["streams"] = [result["proposal"]["streams"][0]]
    result["proposal"].update({
        "residence_time_min": 20.0,
        "residence_time_inlet_min": 20.0,
        "residence_time_in_channel_min": 20.0,
        "residence_time_basis": "liquid-only reactor volume / liquid flow",
    })
    result["design_calculations"].update({
        "residence_time_min": 20.0,
        "residence_time_inlet_min": 20.0,
        "residence_time_in_channel_min": 20.0,
        "residence_time_basis": "liquid-only reactor volume / liquid flow",
    })
    result["process_topology"]["residence_time_min"] = 20.0
    reactor = next(
        item for item in result["process_topology"]["unit_operations"]
        if item["op_id"] == "reactor_1"
    )
    reactor["parameters"].update({
        "residence_time_min": 20.0,
        "residence_time_inlet_min": 20.0,
        "residence_time_in_channel_min": 20.0,
        "residence_time_basis": "liquid-only reactor volume / liquid flow",
    })
    reactor["parameters"].pop("Q_gas_sccm", None)
    reactor["parameters"].pop("Q_gas_actual_mL_min", None)
    result["process_topology"]["unit_operations"] = [
        item
        for item in result["process_topology"]["unit_operations"]
        if item["op_id"] != "mfc_h2"
    ]
    result["process_topology"]["streams"] = [
        item
        for item in result["process_topology"]["streams"]
        if item["from_op"] != "mfc_h2"
    ]
    result["process_topology"] = normalize_topology_semantics(
        ProcessTopology.model_validate(result["process_topology"])
    ).model_dump(mode="json")
    result["chemistry_plan"]["intensification_mandate"] = {
        "required_mixing_regime": "slug_flow"
    }
    result["batch_record"] = {
        "raw_text": "Hydrogenolysis of DMAOL to product without a gas feed.",
        "reaction_description": "Hydrogenolysis",
    }

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-MIXING-REGIME-TOPOLOGY-CONFLICT" in _issue_codes(contract)


def test_explicit_solvent_annotation_overrides_incorrect_upstream_role():
    result = _hydrogen_result()
    result["proposal"]["streams"][0]["contents"] = [
        "DMAOL (0.126 M, 1.0 equiv)",
        "Water (solvent/co-solvent)",
    ]
    result["proposal"]["streams"][0]["solvent"] = "Methanol/Water"
    result["chemistry_plan"]["reagents"].append(
        {"name": "Water", "role": "substrate", "equiv_or_loading": ""}
    )

    contract = build_final_design_contract(result)

    water = next(item for item in contract["stream_components"] if item["name"] == "Water")
    assert water["role"] == "solvent"
    assert water["quantification_required"] is False
    assert "FINAL-COMPONENT-STOICHIOMETRY-INCOMPLETE" not in _issue_codes(contract)


def test_conflicting_component_quantities_cannot_be_released():
    result = _hydrogen_result()
    result["proposal"]["streams"][0]["contents"].append("DMAOL (0.5 M, 1.0 equiv)")
    contract = build_final_design_contract(result)
    assert contract["status"] == "blocked"
    assert "FINAL-COMPONENT-DESCRIPTIONS-CONFLICT" in _issue_codes(contract)


def test_hydrogen_realization_preserves_target_equiv_except_for_mfc_minimum():
    proposal = FlowProposal(
        residence_time_min=30.0,
        flow_rate_mL_min=0.1,
        temperature_C=60.0,
        concentration_M=0.126,
        BPR_bar=21.0,
        reactor_type="packed-bed",
        tubing_material="stainless steel",
        tubing_ID_mm=4.35,
        reactor_volume_mL=3.0,
        residence_time_basis="liquid-only",
        streams=[
            {
                "stream_label": "A",
                "pump_role": "liquid feed",
                "contents": ["DMAOL"],
                "solvent": "methanol",
                "concentration_M": 0.126,
                "flow_rate_mL_min": 0.1,
                "molar_equiv": 1.0,
                "phase": "liquid",
            },
            {
                "stream_label": "B",
                "pump_role": "H2 MFC",
                "contents": ["H2"],
                "flow_rate_mL_min": 0.1,
                "molar_equiv": 1.0,
                "phase": "gas",
            },
        ],
    )
    mfc = GasHardwareSpec(
        equipment_id="mfc_h2",
        name="H2 MFC",
        type="MFC",
        gas="H2",
        min_flow_sccm=1.0,
        max_flow_sccm=100.0,
    )
    decisions = []
    issues = []

    _solve_gas_stream_rates(
        proposal,
        {"B": mfc},
        LabInventory(gas_hardware=[mfc]),
        decisions,
        issues,
    )

    gas = next(stream for stream in proposal.streams if stream.phase == "gas")
    decision = next(item for item in decisions if item["decision"] == "gas_flow_solution")
    assert decision["target_equiv"] == 1.0
    assert gas.gas_flow_sccm == 1.0
    assert 3.5 < gas.molar_equiv < 3.6
    assert issues == []
