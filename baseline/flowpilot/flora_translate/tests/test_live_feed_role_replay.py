"""Replay actual raw model replies through the normal production parsers."""
import json
import os
from pathlib import Path
import pytest
from flora_translate.chemistry_agent import _normalize_plan_data
from flora_translate.chemistry_contract import reconcile_chemistry_plan
from flora_translate.translation_llm import TranslationLLM
from flora_translate.design_realizer import realize_executable_design
from flora_translate.component_identity import component_key, has_output_only_role, has_stationary_catalytic_role
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory


@pytest.mark.parametrize("case", ["cuaac", "hydrogenolysis", "photochemical_oxidation"])
def test_raw_live_proposals_survive_feed_reconciliation(case):
    root = Path(os.environ.get("FLOW_REPLAY_ROOT", Path(__file__).resolve().parents[2]))
    base = root / "ablation_results/manuscript_benchmark/astra_verified_repairs_dev01_20261012/generation" / case / "astra_flowpilot/repeat_01"
    if not (base / "result.json").exists():
        pytest.skip("Optional live development archive absent")
    result = json.loads((base / "result.json").read_text())
    events = [json.loads(line) for line in (base / "llm_events.jsonl").read_text().splitlines()]
    def raw(api):
        text = next(e["response_text"] for e in events if e["api_name"] == api)
        return json.loads(text[text.find("{"):text.rfind("}") + 1])
    plan = ChemistryPlan.model_validate(_normalize_plan_data(raw("chemistry_agent")))
    batch = BatchRecord.model_validate(result["batch_record"])
    constraints = result["design_disposition"]["requirements"]["source_text"]
    plan, audit = reconcile_chemistry_plan(batch, plan, hard_constraints=constraints)
    parsed = raw("translation_llm")
    proposal = FlowProposal.model_validate(TranslationLLM._normalize_proposal_data(parsed.get("proposal", parsed)))
    inventory = LabInventory.model_validate(result["inventory_snapshot"])
    final, realization, validation = realize_executable_design(proposal, batch_record=batch,
        chemistry_plan=plan, inventory=inventory, hard_constraints=constraints)
    feeds = {component_key(x) for s in final.streams if s.phase == "liquid" for x in s.contents}
    output = {component_key(x.name) for x in plan.reagents if has_output_only_role(x.role)}
    stationary = {component_key(x.name) for x in plan.reagents
        if has_stationary_catalytic_role(x.role + " " + x.notes, declared_loading=x.equiv_or_loading)}
    assert not (feeds & output)
    assert not (feeds & stationary)
    assert stationary <= {component_key(x["name"]) for x in final.inventory_constraints.get("stationary_components", [])}
    assert realization["status"] == "complete", realization["issues"]
    destination = os.environ.get("FLOW_REPLAY_OUTPUT")
    if destination:
        Path(destination).mkdir(exist_ok=True, parents=True)
        (Path(destination) / (case + ".json")).write_text(json.dumps({
            "scope": "Offline replay from actual raw model outputs; no new model selection or score",
            "case": case, "passed": True, "proposal": final.model_dump(),
            "validation": validation, "reconciliation": audit}, indent=2))
