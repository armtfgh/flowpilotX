"""
FLORA-Translate — Revision Agent.

Applies targeted revisions to an existing flow design WITHOUT re-running
the full 7-step translate pipeline.  Uses a single LLM call to produce a
revised FlowProposal (and optionally an updated ChemistryPlan), then
re-validates via DesignCalculator + ENGINE council + topology/diagram rebuild.

Typical revision latency: ~10-15 s  vs  ~60 s for a full translate().
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
import flora_translate.config as cfg
from flora_translate.config import LAB_INVENTORY_PATH
from flora_translate.design_calculator import DesignCalculator
from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.engine.council_v4 import CouncilV4 as CouncilV3
from flora_translate.engine.llm_agents import call_model_text
from flora_translate.input_parser import InputParser
from flora_translate.output_formatter import OutputFormatter
from flora_translate.topology_compiler import compile_inventory_topology
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    LabInventory,
)

logger = logging.getLogger("flora.revision")

# ── System prompt for the revision LLM ────────────────────────────────────────

_REVISION_SYSTEM = """\
You are FLORA's revision engine — an expert flow chemistry AI.

You receive an existing, validated flow chemistry design (FlowProposal JSON
+ ChemistryPlan JSON) together with a user's revision request.  Your task
is to apply the requested changes **precisely** while preserving everything
the user did NOT ask to change.

## Rules
1. Output a COMPLETE revised FlowProposal — every field, not just the delta.
2. Only modify fields that the revision demands.  Copy all other values verbatim.
3. If the revision affects chemistry (different catalyst, reagents, streams,
   mechanism, sensitivities), ALSO output a revised ChemistryPlan.
   Otherwise set `revised_chemistry_plan` to null.
4. Maintain physical consistency:
   - reactor_volume_mL = residence_time_min × flow_rate_mL_min
   - If one changes, update the others.
5. Update `reasoning_per_field` for every parameter you changed.
6. Update `streams` if the revision adds, removes, or modifies a stream.
7. Update `chemistry_notes` when the design rationale shifts.
8. If the user asks to add a unit operation (quench, filter, BPR, degasser…),
   add it to `pre_reactor_steps` or `post_reactor_steps` as appropriate.

## Output — JSON only, no prose outside the JSON block
```json
{
  "revised_proposal": { <full FlowProposal object> },
  "revised_chemistry_plan": { <full ChemistryPlan object> } | null,
  "changes_summary": "One-to-two sentence plain-language summary of what changed."
}
```
"""

# ── Helper: parse JSON from LLM output ───────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """Extract JSON from a response that may be wrapped in markdown fences."""
    text = raw.strip()
    # Strip leading ```json or ```
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    # Strip trailing ```
    if "```" in text:
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


# ═══════════════════════════════════════════════════════════════════════════════

class RevisionAgent:
    """Apply targeted revisions to an existing FLORA flow design."""

    # ── Public entry point ─────────────────────────────────────────────────

    def revise(
        self,
        current_result: dict,
        revision_instructions: str,
        original_query: str,
    ) -> dict:
        """Revise an existing design and return an updated result dict.

        Parameters
        ----------
        current_result : dict
            The full result dict from a previous translate() or revise() call.
        revision_instructions : str
            Natural-language description of what to change (from the classifier
            or directly from the user).
        original_query : str
            The original batch protocol text (needed to rebuild the BatchRecord).

        Returns
        -------
        dict
            Updated result in the same format as translate() — drop-in
            replacement for the Streamlit UI.
        """
        logger.info("Revision Agent — starting")
        logger.info("  Instructions: %s", revision_instructions[:120])

        # ── 1. Reconstruct Pydantic objects from the result dict ───────────
        proposal = FlowProposal(**current_result["proposal"])
        chem_plan_data = current_result.get("chemistry_plan", {})
        chemistry_plan = ChemistryPlan(**chem_plan_data) if chem_plan_data else ChemistryPlan()
        batch_record = InputParser().parse(original_query)
        inventory = _inventory_from_result(current_result)

        # ── 2. LLM revision call — single shot ────────────────────────────
        logger.info("  Step 1/5: LLM revision call")
        revised_proposal, revised_chem, changes_summary = self._llm_revise(
            proposal, chemistry_plan, revision_instructions,
        )

        # ── 3. Re-run DesignCalculator (pure math — instant) ──────────────
        logger.info("  Step 2/5: Design calculator")
        calculations = DesignCalculator().run(
            batch_record,
            chemistry_plan=revised_chem,
            proposal=revised_proposal,
            inventory=inventory,
            target_flow_rate_mL_min=revised_proposal.flow_rate_mL_min or None,
            target_tubing_ID_mm=revised_proposal.tubing_ID_mm or None,
        )
        logger.info(
            "    τ = %.1f min, Re = %.0f, ΔP = %.4f bar",
            calculations.residence_time_min,
            calculations.reynolds_number,
            calculations.pressure_drop_bar,
        )

        # ── 4. ENGINE council validation ───────────────────────────────────
        #   Uses the same Moderator — ensures physics consistency (τ×Q=V),
        #   checks safety, fluidics, kinetics.  Typically converges in ≤2
        #   rounds since the LLM already produced a coherent proposal.
        logger.info("  Step 3/5: ENGINE validation")
        original_analogies = current_result.get("_analogies", [])
        design_candidate, calculations = CouncilV3().run(
            revised_proposal,
            batch_record,
            analogies=original_analogies,
            inventory=inventory,
            chemistry_plan=revised_chem,
            calculations=calculations,
        )

        # ── 5. Format output + generate explanation ────────────────────────
        logger.info("  Step 4/5: Formatting output")
        result = OutputFormatter().format(design_candidate, original_analogies)
        result["chemistry_plan"] = revised_chem.model_dump(exclude_none=True)
        result["design_calculations"] = asdict(calculations)
        result["revision_summary"] = changes_summary
        # Preserve analogies for future revisions
        result["_analogies"] = original_analogies
        # Attach deliberation log
        if design_candidate.deliberation_log:
            result["deliberation_log"] = design_candidate.deliberation_log.model_dump()

        # ── 6. Rebuild topology + diagram ──────────────────────────────────
        logger.info("  Step 5/5: Topology & diagram")
        self._rebuild_diagram(
            result, design_candidate, revised_chem, batch_record, inventory
        )

        logger.info("  Revision complete — Confidence: %s", result["confidence"])
        return result

    # ── LLM revision call ──────────────────────────────────────────────────

    def _llm_revise(
        self,
        proposal: FlowProposal,
        chemistry_plan: ChemistryPlan,
        revision_instructions: str,
    ) -> tuple[FlowProposal, ChemistryPlan, str]:
        """Single LLM call → (revised_proposal, revised_chem_plan, summary)."""

        user_content = (
            "## Current FlowProposal\n"
            "```json\n"
            f"{json.dumps(proposal.model_dump(), indent=2, default=str)}\n"
            "```\n\n"
            "## Current ChemistryPlan\n"
            "```json\n"
            f"{json.dumps(chemistry_plan.model_dump(exclude_none=True), indent=2, default=str)}\n"
            "```\n\n"
            "## Revision Request\n"
            f"{revision_instructions}\n"
        )

        raw = ""
        try:
            result = call_model_text(
                model=cfg.MODEL_REVISION_AGENT,
                api_name="revision_agent",
                max_tokens=6144,
                system=_REVISION_SYSTEM,
                user_content=user_content,
            )
            raw = result.text.strip()
            data = _parse_llm_json(raw)

            revised_proposal = FlowProposal(**data["revised_proposal"])

            if data.get("revised_chemistry_plan"):
                revised_chem = ChemistryPlan(**data["revised_chemistry_plan"])
            else:
                revised_chem = chemistry_plan

            changes = data.get("changes_summary", "Revision applied.")
            logger.info("    Changes: %s", changes[:120])
            return revised_proposal, revised_chem, changes

        except json.JSONDecodeError as e:
            logger.error("Revision LLM returned invalid JSON: %s", e)
            logger.debug("Raw response (first 500 chars): %s", raw[:500])
            raise RuntimeError(
                "The revision model returned invalid JSON. "
                "Please try rephrasing your revision request."
            ) from e
        except KeyError as e:
            logger.error("Revision LLM response missing key: %s", e)
            raise RuntimeError(
                f"The revision response was missing expected field: {e}. "
                f"Please try again."
            ) from e
        except anthropic.APIError as e:
            logger.error("Anthropic API error during revision: %s", e)
            raise RuntimeError(f"API error during revision: {e}") from e
        except Exception as e:
            logger.error("Revision LLM call failed: %s", e, exc_info=True)
            raise RuntimeError(f"Revision failed: {e}") from e

    # ── Topology + diagram rebuild ─────────────────────────────────────────

    def _rebuild_diagram(
        self,
        result: dict,
        design_candidate,
        chemistry_plan: ChemistryPlan,
        batch_record: BatchRecord,
        inventory: LabInventory,
    ) -> None:
        """Rebuild the process topology and SVG/PNG diagram in-place."""
        try:
            from flora_translate.main import _build_translate_topology
            from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder

            topology = _build_translate_topology(
                design_candidate.proposal, chemistry_plan, batch_record,
            )
            result["process_requirements_topology"] = topology.model_dump()
            topology, allocation = compile_inventory_topology(
                topology,
                proposal=design_candidate.proposal,
                inventory=inventory,
            )
            result["inventory_allocation"] = allocation
            result["instrument_manifest"] = allocation.get("instrument_manifest", [])
            if not allocation.get("checks", {}).get("all_required_operations_assigned"):
                result["svg_path"] = ""
                result["png_path"] = ""
                result["recommended_disposition"] = "BLOCK"
                result["reported_disposition"] = "BLOCK"
                result["disposition_rationale"] = (
                    "Revision blocked because required physical equipment could not be assigned."
                )
                result["process_topology"] = {
                    "topology_id": "blocked",
                    "generation_status": "blocked",
                    "unit_operations": [],
                    "streams": [],
                    "blocking_reasons": allocation.get("unresolved_requirements", []),
                }
                return
            artifacts = render_topology_artifacts(
                topology,
                title=batch_record.reaction_description[:70],
                builder=FlowsheetBuilder(),
            )
            result["svg_path"] = artifacts["svg_path"]
            result["png_path"] = artifacts["png_path"]
            result["diagram_artifacts"] = {
                key: value for key, value in artifacts.items() if key != "manifest"
            }
            result["diagram_render_manifest"] = artifacts["manifest"]
            result["process_topology"] = topology.model_dump()
            logger.info("    Diagram saved: %s", artifacts["svg_path"])
        except Exception as e:
            logger.warning("    Diagram rebuild failed: %s", e)
            # Keep previous diagram paths if rebuild fails
            if "svg_path" not in result:
                result["svg_path"] = ""
            if "png_path" not in result:
                result["png_path"] = ""


def _inventory_from_result(result: dict) -> LabInventory:
    intake = result.get("intake_package") or {}
    snapshot = intake.get("inventory_profile_snapshot") or {}
    payload = snapshot.get("lab_inventory") if isinstance(snapshot, dict) else None
    if not payload:
        payload = intake.get("inventory_constraints") if isinstance(intake, dict) else None
    if isinstance(payload, dict):
        try:
            return LabInventory.model_validate(payload)
        except Exception:
            pass
    return LabInventory.from_json(str(LAB_INVENTORY_PATH))
