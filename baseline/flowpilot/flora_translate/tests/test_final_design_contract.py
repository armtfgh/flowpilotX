from flora_translate.final_design_contract import (
    build_final_design_contract,
    publish_final_design_artifacts,
)


def _ready_result():
    stages = [
        {
            "stage_number": 1,
            "stage_name": "Stage 1",
            "inventory_resolved": True,
            "reactor_equipment_id": "reactor_10",
            "reactor_volume_mL": 10.0,
            "Q_liquid_mL_min": 0.5,
            "residence_time_inlet_min": 20.0,
            "residence_time_in_channel_min": 20.0,
        },
        {
            "stage_number": 2,
            "stage_name": "Stage 2",
            "inventory_resolved": True,
            "reactor_equipment_id": "reactor_20",
            "reactor_volume_mL": 20.0,
            "Q_liquid_mL_min": 0.5,
            "residence_time_inlet_min": 40.0,
            "residence_time_in_channel_min": 55.0,
        },
    ]
    return {
        "recommended_disposition": "SCREEN",
        "proposal": {
            "residence_time_min": 60.0,
            "residence_time_inlet_min": 60.0,
            "residence_time_in_channel_min": 75.0,
            "residence_time_basis": "sum of reconciled per-stage residence times",
            "flow_rate_mL_min": 0.5,
            "reactor_volume_mL": 30.0,
            "temperature_C": 40.0,
            "BPR_bar": 5.0,
            "streams": [{"stream_label": "A", "flow_rate_mL_min": 0.5}],
            "stage_parameters": stages,
        },
        "design_calculations": {
            "residence_time_min": 60.0,
            "batch_time_s": 36000.0,
        },
        "chemistry_plan": {
            "intensification_mandate": {"tau_reduction_target": 20.0}
        },
        "multistage_inventory_plan": {
            "applied": True,
            "status": "complete",
            "total_reactor_volume_mL": 30.0,
            "total_residence_time_inlet_min": 60.0,
            "stage_parameters": stages,
        },
        "inventory_allocation": {
            "status": "complete",
            "instrument_manifest": [{"equipment_id": "reactor_10"}],
        },
        "process_topology": {
            "topology_id": "ready_two_stage",
            "unit_operations": [
                {
                    "op_id": "pump_a",
                    "op_type": "pump",
                    "label": "Pump A",
                    "parameters": {
                        "stream": "A",
                        "phase": "liquid",
                        "flow_rate_mL_min": 0.5,
                    },
                    "inventory_item_id": "pump_1",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "mixer_1",
                    "op_type": "mixer",
                    "label": "Mixer 1",
                    "parameters": {},
                    "inventory_item_id": "mixer_1",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "reactor_1",
                    "op_type": "coil_reactor",
                    "label": "Reactor 1",
                    "parameters": {
                        "volume_mL": 10.0,
                        "Q_liquid_mL_min": 0.5,
                        "temperature_C": 40.0,
                        "residence_time_min": 20.0,
                        "residence_time_inlet_min": 20.0,
                        "residence_time_in_channel_min": 20.0,
                        "residence_time_basis": "liquid-only",
                    },
                    "inventory_item_id": "reactor_10",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "mixer_2",
                    "op_type": "mixer",
                    "label": "Mixer 2",
                    "parameters": {},
                    "inventory_item_id": "mixer_2",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "reactor_2",
                    "op_type": "coil_reactor",
                    "label": "Reactor 2",
                    "parameters": {
                        "volume_mL": 20.0,
                        "Q_liquid_mL_min": 0.5,
                        "temperature_C": 40.0,
                        "residence_time_min": 40.0,
                        "residence_time_inlet_min": 40.0,
                        "residence_time_in_channel_min": 55.0,
                        "residence_time_basis": "liquid-only",
                    },
                    "inventory_item_id": "reactor_20",
                    "assignment_status": "assigned",
                },
                {
                    "op_id": "collector_1",
                    "op_type": "collector",
                    "label": "Collector",
                    "parameters": {},
                },
            ],
            "streams": [
                {"stream_id": "s1", "from_op": "pump_a", "to_op": "mixer_1"},
                {"stream_id": "s2", "from_op": "mixer_1", "to_op": "reactor_1"},
                {"stream_id": "s3", "from_op": "reactor_1", "to_op": "mixer_2"},
                {"stream_id": "s4", "from_op": "mixer_2", "to_op": "reactor_2"},
                {"stream_id": "s5", "from_op": "reactor_2", "to_op": "collector_1"},
            ],
            "total_flow_rate_mL_min": 0.5,
            "residence_time_min": 60.0,
            "reactor_volume_mL": 30.0,
            "compilation_status": "inventory_assigned",
        },
        "final_validation": {
            "status": "ready",
            "checks": {
                "geometry_closure": True,
                "inventory_topology_assignment_complete": True,
                "topology_matches_serialized_design": True,
            },
        },
    }


def test_contract_exposes_one_executable_stagewise_design():
    contract = build_final_design_contract(_ready_result())

    assert contract["status"] == "executable"
    assert contract["parameters"]["reactor_volume_mL"] == 30.0
    assert contract["parameters"]["residence_time_inlet_min"] == 60.0
    assert contract["parameters"]["residence_time_in_channel_min"] == 75.0
    assert contract["parameters"]["flow_rate_mL_min"] == 0.5
    assert contract["parameters"]["flow_rate_basis"] == "final liquid outlet flow"
    assert contract["parameters"]["stage_specific_flow_rates"] is False
    assert len(contract["stages"]) == 2
    assert contract["streams"][0]["stream_label"] == "A"
    assert contract["intensification"]["realized_factor"] == 10.0
    assert contract["intensification"]["target_factor"] is None
    assert contract["intensification"]["hypothesis_factor"] == 20.0
    assert contract["intensification"]["target_role"] == "not applied under evidence-first policy"
    assert contract["process_graph"]["schema_version"] == "flowpilot_executable_process_v1.0"
    assert len(contract["process_graph"]["stages"]) == 2


def test_contract_preserves_stationary_components_without_adding_them_to_feeds():
    result = _ready_result()
    result["proposal"]["inventory_constraints"] = {
        "stationary_components": [
            {
                "name": "Cu/C heterogeneous catalyst",
                "placement": "stationary_reactor_phase",
            }
        ]
    }

    contract = build_final_design_contract(result)

    assert contract["status"] == "executable"
    assert contract["stationary_components"][0]["placement"] == "stationary_reactor_phase"
    assert all(
        "Cu/C" not in str(stream.get("contents") or [])
        for stream in contract["streams"]
    )


def test_contract_blocks_stationary_component_serialized_in_a_pumped_feed():
    result = _ready_result()
    result["proposal"]["inventory_constraints"] = {
        "stationary_components": [
            {
                "name": "Cu/C heterogeneous catalyst",
                "placement": "stationary_reactor_phase",
            }
        ]
    }
    result["proposal"]["streams"][0]["contents"] = [
        "substrate",
        "Cu/C heterogeneous catalyst (pre-packed)",
    ]

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-STATIONARY-COMPONENT-IN-FEED" in {
        item["code"] for item in contract["consistency"]["issues"]
    }
    assert not contract["consistency"]["semantic_checks"][
        "stationary_components_not_in_feeds"
    ]


def test_contract_infers_packed_bed_catalyst_and_blocks_feed_duplication():
    result = _ready_result()
    result["proposal"]["reactor_type"] = "packed-bed"
    result["proposal"]["inventory_selection"] = {
        "name": "Cu/C CatCart packed-bed cartridge",
        "type": "packed_bed",
    }
    result["proposal"]["streams"][0]["contents"] = [
        "benzyl azide (0.25 M, 1.0 equiv)",
        "phenylacetylene (0.275 M, 1.1 equiv)",
        "copper on carbon (0.025 M screening assumption)",
    ]
    result["chemistry_plan"]["reagents"] = [
        {
            "name": "copper on carbon",
            "role": "heterogeneous copper catalyst packed in cartridge",
            "notes": "Use the installed CatCart as the catalytic zone.",
        }
    ]

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-STATIONARY-COMPONENT-IN-FEED" in {
        item["code"] for item in contract["consistency"]["issues"]
    }


def test_contract_blocks_air_and_pure_oxygen_as_parallel_stage_feeds():
    result = _ready_result()
    result["proposal"]["streams"].extend(
        [
            {
                "stream_label": "B",
                "phase": "gas",
                "contents": ["air"],
                "introduction_stage": 1,
                "gas_flow_sccm": 1.0,
                "gas_flow_actual_mL_min": 0.3,
                "molar_equiv": 1.0,
            },
            {
                "stream_label": "G",
                "phase": "gas",
                "contents": ["O2"],
                "introduction_stage": 1,
                "gas_flow_sccm": 0.21,
                "gas_flow_actual_mL_min": 0.06,
                "molar_equiv": 1.0,
            },
        ]
    )

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-CONFLICTING-OXIDANT-DELIVERY" in {
        item["code"] for item in contract["consistency"]["issues"]
    }
    assert not contract["consistency"]["semantic_checks"][
        "reagent_gas_delivery_unique"
    ]


def test_contract_compares_multistage_total_using_declared_residence_basis():
    result = _ready_result()
    result["proposal"].update(
        {
            "residence_time_min": 60.0,
            "residence_time_inlet_min": 30.0,
            "residence_time_in_channel_min": 50.0,
            "residence_time_basis": (
                "sum of nominal per-stage liquid contact times"
            ),
        }
    )
    result["multistage_inventory_plan"].update(
        {
            "total_residence_time_liquid_min": 60.0,
            "total_residence_time_inlet_min": 30.0,
            "total_residence_time_in_channel_min": 50.0,
        }
    )

    contract = build_final_design_contract(result)

    assert contract["status"] == "executable"
    assert "FINAL-TAU-STAGE-TOTAL-MISMATCH" not in {
        issue["code"] for issue in contract["consistency"]["issues"]
    }


def test_multistage_contract_reports_outlet_flow_when_stage_flows_differ():
    result = _ready_result()
    result["proposal"].update(
        {"flow_rate_mL_min": 1.0, "reactor_volume_mL": 50.0}
    )
    result["proposal"]["stage_parameters"][0]["Q_liquid_mL_min"] = 0.5
    result["proposal"]["stage_parameters"][1].update(
        {"Q_liquid_mL_min": 1.0, "reactor_volume_mL": 40.0}
    )
    result["multistage_inventory_plan"]["stage_parameters"] = result["proposal"][
        "stage_parameters"
    ]
    result["multistage_inventory_plan"]["total_reactor_volume_mL"] = 50.0
    result["process_topology"]["total_flow_rate_mL_min"] = 1.0
    result["process_topology"]["reactor_volume_mL"] = 50.0
    result["process_topology"]["unit_operations"][4]["parameters"].update(
        {"volume_mL": 40.0, "Q_liquid_mL_min": 1.0}
    )

    contract = build_final_design_contract(result)

    assert contract["parameters"]["flow_rate_mL_min"] == 1.0
    assert contract["parameters"]["stage_specific_flow_rates"] is True
    assert contract["intensification"]["applied_as_hard_constraint"] is False


def test_multistage_summary_is_derived_from_final_stages_not_stale_proposal():
    result = _ready_result()
    result["proposal"].update(
        {
            "tubing_ID_mm": 0.75,
            "tubing_material": "PFA",
            "temperature_C": 35.0,
            "wavelength_nm": 450.0,
        }
    )
    first, second = result["multistage_inventory_plan"]["stage_parameters"]
    first.update(
        {"d_mm": 1.0, "material": "PFA", "temperature_C": 35.0, "wavelength_nm": 450.0}
    )
    second.update(
        {"d_mm": 1.0, "material": "FEP", "temperature_C": 40.0, "wavelength_nm": 448.0}
    )

    contract = build_final_design_contract(result)

    assert contract["parameters"]["tubing_ID_mm"] == 1.0
    assert contract["parameters"]["tubing_material"] is None
    assert contract["parameters"]["temperature_C"] is None
    assert contract["parameters"]["wavelength_nm"] is None
    assert contract["parameters"]["stage_specific_conditions"] is True


def test_standard_accessory_assumption_keeps_final_contract_executable():
    result = _ready_result()
    result["inventory_allocation"].update(
        {
            "status": "complete_with_assumptions",
            "assumed_standard_accessories": [
                {
                    "operation_id": "st2_mixer",
                    "category": "mixers",
                    "name": "Generic compatible T-mixer - verify before run",
                    "requires_pre_run_verification": True,
                }
            ],
        }
    )

    contract = build_final_design_contract(result)

    assert contract["status"] == "executable"
    assert contract["parameters"] is not None


def test_contract_withholds_parameters_when_stage_inventory_is_incomplete():
    result = _ready_result()
    result["recommended_disposition"] = "BLOCK"
    result["multistage_inventory_plan"]["status"] = "incomplete"
    result["multistage_inventory_plan"]["unresolved_requirements"] = [
        {"stage_number": 2, "category": "reactors"}
    ]
    result["multistage_inventory_plan"]["stage_parameters"][1][
        "inventory_resolved"
    ] = False

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert contract["parameters"] is None
    assert contract["stages"] == []
    assert contract["streams"] == []
    assert "FINAL-MULTISTAGE-INCOMPLETE" in {
        issue["code"] for issue in contract["consistency"]["issues"]
    }
    assert contract["diagnostic"]["stage_requirements"][1][
        "residence_time_inlet_min"
    ] is None


def test_contract_blocks_cross_section_residence_time_mismatch():
    result = _ready_result()
    result["design_calculations"]["residence_time_min"] = 25.0

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert contract["parameters"] is None
    assert "FINAL-TAU-PROPOSAL-CALCULATION-MISMATCH" in {
        issue["code"] for issue in contract["consistency"]["issues"]
    }


def test_contract_blocks_legacy_topology_with_hidden_per_reactor_tau_mismatch():
    result = _ready_result()
    result["proposal"]["stage_parameters"] = []
    result["multistage_inventory_plan"] = {"applied": False}
    result["process_topology"] = {
        "reactor_volume_mL": 30.0,
        "residence_time_min": 60.0,
        "unit_operations": [
            {
                "op_type": "coil_reactor",
                "parameters": {"volume_mL": 10.0, "residence_time_min": 162.0},
            },
            {
                "op_type": "coil_reactor",
                "parameters": {"volume_mL": 20.0, "residence_time_min": 162.0},
            },
        ],
    }

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert "FINAL-TOPOLOGY-REACTOR-TAU-MISMATCH" in {
        issue["code"] for issue in contract["consistency"]["issues"]
    }


def test_contract_blocks_process_graph_with_unassigned_feed_device():
    result = _ready_result()
    result["process_topology"]["unit_operations"][0]["inventory_item_id"] = None

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert contract["process_graph"] is None
    assert "FINAL-EXECUTABLE-PROCESS-GRAPH-INVALID" in {
        issue["code"] for issue in contract["consistency"]["issues"]
    }


def test_contract_blocks_process_graph_when_stage_v_q_tau_does_not_close():
    result = _ready_result()
    result["process_topology"]["unit_operations"][2]["parameters"][
        "Q_liquid_mL_min"
    ] = 0.25

    contract = build_final_design_contract(result)

    assert contract["status"] == "blocked"
    assert contract["process_graph"] is None
    assert "FINAL-EXECUTABLE-PROCESS-GRAPH-INVALID" in {
        issue["code"] for issue in contract["consistency"]["issues"]
    }


def test_published_canonical_summary_does_not_repeat_stale_model_values():
    result = _ready_result()
    result["explanation"] = "Use a stale candidate BPR of 3.0 bar."
    contract = build_final_design_contract(result)

    publish_final_design_artifacts(result, contract)

    assert "**BPR:** 5 bar" in result["canonical_explanation"]
    assert "3.0 bar" not in result["canonical_explanation"]
    assert result["explanation_status"] == "pre_realization_audit_only"
