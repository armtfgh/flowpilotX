"""Failure-chain regressions, independent of a generator or target benchmark score."""
from copy import deepcopy
import json

import pytest

from flora_translate.design_calculator import DesignCalculator
from flora_translate.design_evidence import capture_reasoning, reconcile_reasoning
from flora_translate.design_realizer import realize_executable_design
from flora_translate.engine.council_v4 import designer, chief
from flora_translate.engine.council_v4.inventory_candidates import materialize
from flora_translate.engine.council_v4.review_contract import candidate_fingerprint
from flora_translate.engine.sampling import compute_metrics
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
from flora_translate.tests.test_inventory_council_contract import candidates


@pytest.mark.parametrize("t90", [.001, 10, 100])
def test_kinetic_conversion_uses_the_same_t90_as_damkohler(t90):
    row = compute_metrics(t90, 1, .1, "methanol", 25, .1, 200, 1, t90, 40, False)
    assert row["expected_conversion"] == pytest.approx(.9)


def test_unused_packed_bed_does_not_remove_a_coil_feed_component():
    from flora_translate.design_realizer import _enforce_stationary_component_placement
    inventory = LabInventory.model_validate({"reactors": [
        {"equipment_id": "coil", "name": "selected coil", "type": "coil", "material": "FEP", "volume_mL": 1, "ID_mm": 1},
        {"equipment_id": "bed", "name": "unused packed bed", "type": "packed-bed", "material": "steel", "volume_mL": 20, "ID_mm": 10}]})
    proposal = FlowProposal.model_validate({"reactor_type": "coil", "reactor_volume_mL": 1,
        "tubing_ID_mm": 1, "streams": [{"stream_label": "A", "contents": ["substrate", "supported catalyst"]}]})
    _enforce_stationary_component_placement(proposal, None, inventory, [])
    assert "supported catalyst" in proposal.streams[0].contents


def test_unvalidated_transport_cannot_veto_but_validated_failure_can():
    row = candidates()["survivors"][0]
    row.update(Da_mass=100, r_mix=10, expected_conversion=.001)
    args = dict(pump_max_bar=40, is_photochem=False, is_gas_liquid=False, BPR_bar=3)
    _, flagged = designer._apply_v4_hard_gates([row], **args)
    assert not flagged
    assert any("mixing fail" in w for w in row["warnings"])
    row["model_validity"]["mixing"] = {"applicable": True, "validated": True, "evidence": ["transport experiment at this operating point"]}
    _, flagged = designer._apply_v4_hard_gates([row], **args)
    assert flagged
    row["model_validity"]["mixing"]["validated"] = False
    row["delta_P_bar"] = 50
    _, flagged = designer._apply_v4_hard_gates([row], **args)
    assert flagged  # uncertainty does not waive actual pressure-limit checks


def test_refinement_preserves_distinct_original_and_rechecks_hard_gates():
    pool = candidates()["survivors"]
    scoring = {"kinetics_scores": [{"candidate_id": pool[1]["id"], "verdict": "REVISE",
               "proposed_changes": {"tau_min": pool[0]["tau_min"]}}]}
    revised, summary = chief._run_candidate_refinement_loop(candidates=pool, scoring=scoring,
        audit={"all_errors": []}, solvent="methanol", temperature_C=25, concentration_M=.1,
        assumed_MW=200, IF_used=1, pump_max_bar=40, is_photochem=False,
        is_gas_liquid=False, extinction_coeff_M_cm=None, strong_revision_mode=True,
        max_total_revised_candidates=2)
    assert len({candidate_fingerprint(row) for row in revised}) == 2
    assert summary["diversity"]["restored_original_ids"]
    assert not any(row["hard_gate_flags"] for row in revised)


def test_structural_repair_is_checked_and_out_of_scope_requests_fail():
    context = candidates()["survivors"][0]["_realization_context"]
    context["plan"] = {"reagents": [{"name": "supported material", "role": "Supported metal source",
                       "equiv_or_loading": "Flow: retained in supplied packed bed, mass unknown"}]}
    context["proposal"]["streams"][0]["contents"].append("supported material")
    request = {"component": "supported material", "placement": "stationary_reactor_phase",
               "equipment_id": "bed", "reason": "Retain the declared stationary phase"}
    patch, _ = chief._extract_explicit_patch({"proposed_changes": {"component_placements": [request]}},
                                          "chemistry", candidate={"inventory_realized": True})
    repaired = materialize(context, patch=patch)
    assert "supported material" not in repaired["streams"][0]["contents"]
    assert repaired["realized_proposal"]["inventory_constraints"]["council_component_placements"]
    assert any(r["name"] == "supported material" for r in repaired["stationary_components"])
    for invalid in [{**request, "component": "invented catalyst"}, {**request, "equipment_id": "unlisted bed"}]:
        with pytest.raises(ValueError):
            materialize(context, patch={"component_placements": [invalid]})
    context["plan"]["reagents"][0]["role"] = "Soluble homogeneous catalyst"
    context["plan"]["reagents"][0]["equiv_or_loading"] = "1 mol% in liquid feed"
    with pytest.raises(ValueError, match="not supported"):
        materialize(context, patch=patch)


def test_reasoning_survives_unchanged_conditions_but_stale_conditions_are_labeled():
    p = FlowProposal(temperature_C=40, BPR_bar=20,
        reasoning_per_field={"temperature_C": "Use 40 C because no outlet cooler is available."})
    capture_reasoning(p)
    p.reasoning_per_field = {"temperature_C": "Within hardware limits."}
    reconcile_reasoning(p)
    assert "no outlet cooler" in p.reasoning_per_field["temperature_C"]
    assert p.decision_rationale["temperature_C"]["status"] == "conditions_preserved"
    p.temperature_C = 100
    p.reasoning_per_field = {"temperature_C": "Final temperature 100 C."}
    capture_reasoning(p)
    reconcile_reasoning(p)
    assert "40 C" not in p.reasoning_per_field["temperature_C"]
    assert p.decision_rationale["temperature_C"]["status"] == "requires_review_after_parameter_change"
    assert "40 C" in p.decision_rationale["temperature_C"]["text"]


def test_reviewed_feed_identity_drift_blocks_delivery():
    row = candidates()["survivors"][0]
    p, _ = chief._apply_winner(FlowProposal(), row, {}, chief_data={"selection_rationale": "Retain a conservative first screen."})
    p.streams[0].contents = ["different substrate (0.1 M)"]
    _, audit, _ = realize_executable_design(p, batch_record=BatchRecord(solvent="methanol"),
        chemistry_plan=None, inventory=LabInventory.model_validate(row["_realization_context"]["inventory"]))
    assert any(issue.get("code") == "COUNCIL-DELIVERY-STATE-DRIFT" for issue in audit["issues"])


def test_stationary_material_and_rationale_are_idempotent_across_realization():
    context = candidates()["survivors"][0]["_realization_context"]
    context["plan"] = {"reagents": [{"name": "bed material", "role": "Supported catalyst",
        "equiv_or_loading": "Flow: retained in supplied packed bed"}]}
    context["proposal"]["streams"][0]["contents"].append("bed material")
    row = materialize(context)
    again = materialize(row["_realization_context"])
    assert row["stationary_components"] == again["stationary_components"]
    assert row["realized_proposal"]["reasoning_per_field"] == again["realized_proposal"]["reasoning_per_field"]


def test_problem_framing_receives_protocol_inventory_and_real_batch_time(monkeypatch):
    captured = {}
    def fake(system, user, **kwargs):
        captured["user"] = user
        return json.dumps({"reaction_class": "thermal", "ambiguities": []})
    monkeypatch.setattr(designer, "call_llm", fake)
    designer.run_problem_framing("thermal", False, False, False, 16, 13.3,
        "methanol", 60, .1, "first screen", batch_time_min=300,
        protocol="Substrate in methanol for 5 h", inventory={"reactors": [{"equipment_id": "bed"}]},
        intake_context="Only the supplied packed bed may be used.")
    text = captured["user"]
    assert "actual_batch_time_min=300" in text and "estimated_flow_time_min=16" in text
    assert "Substrate in methanol for 5 h" in text and '"equipment_id": "bed"' in text
    assert "tau_batch_equiv" not in text


@pytest.mark.parametrize("metadata", [{"score": .1}, {"score": .99, "retrieval_method": "lexical"},
                                       {"score": .99, "quantitative_transfer": {"eligible": False}}])
def test_remote_or_low_relevance_analogy_cannot_reenter_as_unweighted_median(metadata):
    batch = BatchRecord(reaction_time_h=1, concentration_M=.1, yield_pct=80, solvent="methanol")
    plan = ChemistryPlan(reaction_class="thermal")
    unrelated = {**metadata, "full_record": {"translation_logic": {"time_reduction_factor": 1000}}}
    calc = DesignCalculator().run(batch, chemistry_plan=plan, analogies=[unrelated])
    assert calc.if_analogy is None
    assert calc.intensification_factor < 1000


@pytest.mark.parametrize("mode,mechanism,eligible", [("semantic", "thermal", True),
    ("semantic", "photoredox", False), ("lexical", "thermal", False)])
def test_retrieval_transfer_uses_indexed_class_metadata(mode, mechanism, eligible):
    from flora_translate.retriever import VectorRetriever
    class Store:
        def query(self, **kwargs):
            return {"ids": [["a", "b", "c"]], "distances": [[.1, .2, .3]],
                "documents": [["Matched precedent"] * 3], "metadatas": [[{
                    "chemistry_class": "coupling", "mechanism_type": mechanism}] * 3]}
        query_lexical = query
    class Engine:
        def embed(self, text):
            return [1, 0]
    retriever = VectorRetriever(store=Store())
    retriever.engine = Engine()
    records = retriever.retrieve(BatchRecord(solvent="methanol"),
        chemistry_plan=ChemistryPlan(reaction_class="coupling", mechanism_type="thermal"), retrieval_mode=mode)
    assert all(r["quantitative_transfer"]["eligible"] is eligible for r in records)


def test_council_rejection_is_independent_of_a_stale_ready_validation():
    from flora_translate.tests.test_executable_artifact_contract import _hydrogen_result
    from flora_translate.final_design_contract import build_final_design_contract
    result = _hydrogen_result()
    result["proposal"].setdefault("inventory_constraints", {})["council_selection_block"] = "Unresolved feed conflict"
    result["final_validation"] = {"status": "ready", "checks": {"all_good": True}}
    final = build_final_design_contract(result)
    assert final["status"] == "blocked"
    assert any(issue["code"] == "FINAL-COUNCIL-NO-APPROVED-CANDIDATE" for issue in final["consistency"]["issues"])
