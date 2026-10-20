"""Self-contained synthetic example; no historical or laboratory data.

The numerical values illustrate the software contract only. They are not a
reaction recipe, an equipment specification, or an experimentally tested design.
"""
from __future__ import annotations


def build_demo_profile():
    question = {
        "id": "DEMO-Q1",
        "question": "How would separate product, starting-material and byproduct measurements change across this synthetic control grid?",
        "competing_explanations": ["Product formation changes", "Byproduct formation changes", "Operational variation dominates"],
        "expected_observations": ["Separate component assays may differ across the grid; no response is predicted"],
        "measurements": ["Actual liquid flow and temperature", "Separate component amounts and analytical method"],
        "confounders": ["Unmeasured temperature, residence distribution and analytical uncertainty"],
        "decision_rules": ["Review qualified measurements before revising the candidate pool"],
        "evidence_ids": [], "status": "unresolved",
    }
    candidates = []
    for temperature in (30, 40, 50, 60, 70, 80):
        for flow in (0.5, 1.0):
            candidates.append({
                "id": f"DEMO{len(candidates) + 1:02d}",
                "parameters": {"temperature_C": {"value": temperature, "unit": "degC"},
                               "liquid_flow_mL_min": {"value": flow, "unit": "mL/min"}},
                "inventory_ids": ["demo-pump", "demo-reactor"],
                "feasible": True, "rejection_reasons": [], "engineering_status": "conditional",
                "question_ids": [question["id"]], "evidence_ids": [],
                "curiosity": "Explore complementary settings in an invented software example",
                "rationale": "Covers a synthetic control grid; replace with a scientifically reviewed pool for actual chemistry.",
                "diagram_note": "Synthetic software example. No installed equipment or laboratory design is claimed.",
            })
    return {
        "schema_version": "flowpilotx-demo-v1", "profile_id": "synthetic-demo-v1",
        "name": "Synthetic demonstration — no experimental data",
        "synthetic_fixture": True, "experimental_validation": False,
        "chemistry": {"transformation": "Unspecified liquid-phase reaction for software demonstration", "synthetic_fixture": True},
        "protocol_text": "Synthetic demonstration only. Enter your chemistry protocol, inventory, evidence and reviewed candidate pool in a new study. These invented settings are not laboratory recommendations.",
        "objective": {"mode": "explore", "metric": "product_yield_pct", "direction": "maximize", "unit": "%",
                      "separate_components": ["starting_material_pct", "product_yield_pct", "byproduct_pct"]},
        "inventory": {"schema_version": "flowpilotx-inventory-v1", "items": [
            {"id": "demo-pump", "kind": "pump", "name": "Invented demonstration pump", "quantity": 1,
             "availability": "available", "capabilities": {"synthetic_fixture": True},
             "limits": [{"parameter": "liquid_flow_mL_min", "operator": ">=", "value": 0.1, "unit": "mL/min", "source_reference": "synthetic-demo"},
                        {"parameter": "liquid_flow_mL_min", "operator": "<=", "value": 2.0, "unit": "mL/min", "source_reference": "synthetic-demo"}],
             "source_references": ["Invented software fixture; no actual instrument"]},
            {"id": "demo-reactor", "kind": "reactor", "name": "Invented demonstration reactor", "quantity": 1,
             "availability": "available", "capabilities": {"synthetic_fixture": True},
             "limits": [{"parameter": "temperature_C", "operator": ">=", "value": 20, "unit": "degC", "source_reference": "synthetic-demo"},
                        {"parameter": "temperature_C", "operator": "<=", "value": 90, "unit": "degC", "source_reference": "synthetic-demo"}],
             "source_references": ["Invented software fixture; no actual instrument"]},
        ]},
        "requirements": [{"id": "demo-not-released", "description": "Replace synthetic settings and confirm actual laboratory apparatus before use",
                          "scope": "release", "status": "unresolved", "severity": "hard", "source_reference": "synthetic-demo"}],
        "questions": [question], "candidates": candidates, "evidence": [],
        "default_selected_candidate_ids": ["DEMO01", "DEMO02", "DEMO05", "DEMO06", "DEMO09", "DEMO10"],
        "optional_repeat": {"repeat_of": "DEMO06"},
        "observation_contract": {"schema_version": "flowpilotx-observation-contract-v1",
            "required_actual_conditions": ["temperature_C", "liquid_flow_mL_min"],
            "parameter_mapping": {"temperature_C": "temperature_C", "liquid_flow_mL_min": "liquid_flow_mL_min"},
            "units": {"temperature_C": "degC", "liquid_flow_mL_min": "mL/min"},
            "types": {"temperature_C": "number", "liquid_flow_mL_min": "number"},
            "actual_conditions_template": {"temperature_C": None, "liquid_flow_mL_min": None}},
        "source_references": ["Self-contained synthetic demonstration in flowpilotx.demo; no KHU source data"],
        "limitations": ["No experimental history, model output, fitted kinetics or predicted yield is supplied",
                        "Candidate generation for new chemistry requires a reviewed pool or a chemistry adapter"],
    }


def empty_demo_history():
    return {"id": "synthetic-demo-empty-history", "rows": [], "sources": [], "known_sources": {},
            "warnings": ["Synthetic demonstration: no historical or experimental records are included."],
            "summary": {"record_count": 0, "synthetic_fixture": True,
                        "scope": "Empty evidence; the candidate settings are invented software examples."}}


def portable_requirements():
    rows = [
        ("portable-demo", "Standalone synthetic demonstration", "No private historical data or model service is needed", "implemented"),
        ("count", "Requested design count", "Select N unique feasible settings from an explicit reviewed candidate pool", "implemented"),
        ("inventory", "Inventory constraints", "Apply structured equipment limits and preserve unresolved release requirements", "implemented"),
        ("history", "Evidence provenance", "Preserve uploaded original representations, missing values and conflicts", "implemented"),
        ("feedback", "Experimental feedback", "Record separate analytes and use only qualified comparable flow observations", "implemented"),
        ("questions", "Question tracking", "Archive hypotheses, amendments, observations and next decisions", "implemented"),
        ("audit", "Versioned decisions", "Preserve candidates, rejections, frozen rounds and result corrections", "implemented"),
        ("model", "Explicit model routing", "Model service must be configured explicitly; no paid fallback", "configuration_required"),
        ("optimization", "Validated optimization", "BO, fitted kinetics and calibrated information gain are not implemented", "not_implemented"),
    ]
    return {"schema_version": "flowpilotx-requirements-v1", "source": "Portable software capabilities; no study-specific completion claim",
            "requirements": [{"id": key, "title": title, "request": description, "status": status, "evidence": []}
                             for key, title, description, status in rows]}
