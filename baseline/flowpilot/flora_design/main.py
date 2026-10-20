"""FLORA-Design — Main entry point.

Pipeline:
  Goal text → ChemistryClassifier → ChemFeatures
                                        ↓
                                 UnitOpSelector (rule-based)
                                        ↓
                                 TopologyAgent (RAG + LLM)
                                        ↓
                                 ParameterAgent (fill numbers)
                                        ↓
                                 ENGINE validation
                                        ↓
                                 FlowsheetBuilder (SVG/PNG)
                                        ↓
                                 DesignResult
"""

import json
import logging
import re
import sys
from pathlib import Path

import anthropic

from flora_design.chemistry_classifier import ChemistryClassifier
from flora_design.parameter_agent import ParameterAgent
from flora_design.topology_agent import TopologyAgent
from flora_design.unit_op_selector import UnitOpSelector
from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder
from flora_translate.config import LAB_INVENTORY_PATH, TRANSLATION_MODEL
from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.topology_compiler import compile_inventory_topology
from flora_translate.engine.council_v3 import CouncilV3
from flora_translate.schemas import (
    BatchRecord,
    DesignResult,
    FlowProposal,
    LabInventory,
)
from flora_translate.vector_store import VectorStore

logger = logging.getLogger("flora.design")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = Path("outputs")


def _get_client():
    return anthropic.Anthropic()


def design(
    goal_text: str,
    inventory_path: str = str(LAB_INVENTORY_PATH),
    output_dir: str = str(OUTPUT_DIR),
) -> DesignResult:
    """Full FLORA-Design pipeline: chemistry goal → validated flow process design."""

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Extract chemistry features
    logger.info("Step 1: Classifying chemistry goal")
    features = ChemistryClassifier().classify(goal_text)
    logger.info(f"  Class: {features.reaction_class}, Catalyst: {features.photocatalyst}")

    # 2. Select unit operations (rule-based, no LLM)
    logger.info("Step 2: Selecting unit operations (rule-based)")
    ops = UnitOpSelector().select(features)

    # 3. RAG-based topology generation
    logger.info("Step 3: Generating topology (RAG + LLM)")
    store = VectorStore()
    topo_agent = TopologyAgent(store=store)
    primary_topology, alt_topology = topo_agent.run(features, ops)

    # 4. Fill parameters
    logger.info("Step 4: Filling parameters")
    records = topo_agent.retrieve(features)
    primary_topology = ParameterAgent().run(primary_topology, features, records)

    # 5. ENGINE validation
    logger.info("Step 5: ENGINE validation")
    inventory = LabInventory.from_json(inventory_path)
    proposal = _topology_to_proposal(primary_topology, features)
    batch_record = _features_to_batch_record(features, goal_text)
    design_candidate, _ = CouncilV3().run(proposal, batch_record, [], inventory)

    primary_topology, inventory_allocation = compile_inventory_topology(
        primary_topology,
        proposal=design_candidate.proposal,
        inventory=inventory,
    )

    # 6. Generate flowsheet diagram
    logger.info("Step 6: Generating flowsheet diagram")
    artifacts = render_topology_artifacts(
        primary_topology,
        title=goal_text[:60],
        base_dir=output_dir_path / "design_diagram_runs",
        builder=FlowsheetBuilder(),
    )

    # 7. Generate explanation
    logger.info("Step 7: Generating explanation")
    explanation = _generate_explanation(
        goal_text, primary_topology, design_candidate, records
    )

    # 8. Collect warnings
    warnings = [
        m.concern for m in design_candidate.council_messages
        if m.status in ("WARNING", "REJECT") and m.concern
    ]

    result = DesignResult(
        goal_text=goal_text,
        chem_features=features,
        topology=primary_topology,
        design_candidate=design_candidate,
        svg_path=artifacts["svg_path"],
        png_path=artifacts["png_path"],
        diagram_artifacts={
            key: value for key, value in artifacts.items() if key != "manifest"
        },
        diagram_render_manifest=artifacts["manifest"],
        inventory_allocation=inventory_allocation,
        instrument_manifest=inventory_allocation.get("instrument_manifest", []),
        explanation=explanation,
        retrieved_records=primary_topology.literature_support,
        alternatives=[alt_topology] if alt_topology else [],
        warnings=warnings,
    )

    logger.info(f"Done — {len(primary_topology.unit_operations)} unit ops, "
                f"confidence={primary_topology.topology_confidence}")
    return result


def _topology_to_proposal(topology, features) -> FlowProposal:
    """Convert ProcessTopology to FlowProposal for ENGINE validation."""
    reactor_temp = features.temperature_C or 25
    tubing_material = "FEP"
    tubing_id = 1.0
    bpr_bar = 0
    light_setup = ""
    wavelength = features.wavelength_nm
    deoxy = None

    for op in topology.unit_operations:
        if op.op_type in ("coil_reactor", "chip_reactor"):
            tubing_material = op.parameters.get("material", "FEP")
            tubing_id = op.parameters.get("ID_mm", 1.0)
            reactor_temp = op.parameters.get("temperature_C", reactor_temp)
        elif op.op_type == "bpr":
            bpr_bar = op.parameters.get("pressure_bar", 0)
        elif op.op_type == "led_module":
            wl = op.parameters.get("wavelength_nm")
            if wl:
                wavelength = wl
                light_setup = f"LED {wl}nm"
        elif op.op_type == "deoxygenation_unit":
            deoxy = op.parameters.get("method", "N2 sparging")

    return FlowProposal(
        residence_time_min=topology.residence_time_min,
        flow_rate_mL_min=topology.total_flow_rate_mL_min,
        temperature_C=reactor_temp,
        concentration_M=features.concentration_M or 0.1,
        BPR_bar=bpr_bar,
        reactor_type="coil",
        tubing_material=tubing_material,
        tubing_ID_mm=tubing_id,
        reactor_volume_mL=topology.reactor_volume_mL,
        light_setup=light_setup,
        wavelength_nm=wavelength,
        deoxygenation_method=deoxy,
        literature_analogies=topology.literature_support,
        confidence="MEDIUM",
    )


def _features_to_batch_record(features, goal_text: str) -> BatchRecord:
    return BatchRecord(
        reaction_description=goal_text,
        photocatalyst=features.photocatalyst,
        solvent=features.solvent,
        temperature_C=features.temperature_C,
        concentration_M=features.concentration_M,
        wavelength_nm=features.wavelength_nm,
        atmosphere="N2" if features.O2_sensitive else "air",
    )


def _generate_explanation(goal_text, topology, design_candidate, records) -> str:
    """Generate human-readable explanation via Claude."""
    try:
        dois = topology.literature_support[:3]
        resp = _get_client().messages.create(
            model=TRANSLATION_MODEL,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": (
                    f"You are FLORA-Design. Write a concise explanation (4-6 sentences) "
                    f"of this flow process design.\n\n"
                    f"Goal: {goal_text}\n"
                    f"Process: {topology.pid_description}\n"
                    f"Conditions: tau={topology.residence_time_min}min, "
                    f"Q={topology.total_flow_rate_mL_min}mL/min, "
                    f"V={topology.reactor_volume_mL}mL\n"
                    f"Literature: {dois}\n"
                    f"Council rounds: {design_candidate.council_rounds}\n\n"
                    f"Write for a synthetic chemist. Be specific with numbers. "
                    f"No marketing language."
                ),
            }],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Explanation generation failed: {e}")
        return topology.pid_description


# ---------------------------------------------------------------------------
# Tests init
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m flora_design.main '<chemistry goal>'")
        sys.exit(1)

    goal = sys.argv[1]
    result = design(goal)
    print(json.dumps(result.model_dump(), indent=2, default=str))
