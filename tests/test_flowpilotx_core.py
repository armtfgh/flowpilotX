"""Synthetic thermal esterification fixture; never laboratory settings."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from flowpilotx import CampaignStore, CampaignError
from flowpilotx.evidence import digest, import_evidence
from flowpilotx.inventory import validate_inventory
from flowpilotx.selection import feedback_summary


def synthetic_spec(mode="explore"):
    return {"name": "Synthetic thermal esterification software fixture", "synthetic_fixture": True,
        "chemistry": {"transformation": "carboxylic acid + alcohol to ester", "substrate": "benzoic acid",
                      "reagent": "methanol", "catalyst": "generic acid", "photochemistry": False},
        "objective": {"mode": mode, "metric": "ester_yield", "direction": "maximize", "unit": "%"},
        "inventory": {"schema_version": "flowpilotx-inventory-v1", "items": [{
            "id": "thermal-coil", "kind": "heated reactor", "quantity": 1, "availability": "available",
            "capabilities": {"heating": True}, "limits": [{"parameter": "temperature", "operator": "<", "value": 100,
                "unit": "C", "source_reference": "synthetic-only"}], "source_references": ["synthetic-only"]}]},
        "questions": [{"id": "Q-temperature", "question": "Does component distribution change across the proposed temperature range?",
            "competing_explanations": ["Product formation increases", "Side-product formation becomes material"],
            "expected_observations": ["Compare ester, residual acid and side-product assay values"],
            "measurements": ["Calibrated GC for separate components"], "confounders": ["Actual temperature and contact time"],
            "decision_rules": ["Use qualified component measurements before changing the range"], "evidence_ids": [], "status": "unresolved"}],
        "candidates": [{"id": f"E{i}", "parameters": {"temperature": {"value": t, "unit": "C"}, "feed_mL_min": 1.0},
            "inventory_ids": ["thermal-coil"], "feasible": True, "rejection_reasons": [],
            "engineering_status": "conditional", "question_ids": ["Q-temperature"], "evidence_ids": [],
            "substrate_id": "acid-A", "costs": {"material_mg": 5}, "curiosity": "Compare measured component distribution"}
            for i, t in enumerate([30, 50, 70, 90], 1)],
        "evidence": [{"id": "batch:one", "experiment_mode": "batch", "outcomes": {"ester_yield": 103, "byproduct": None},
                      "source_representations": [{"source_id": "synthetic", "raw": "103"}]}],
        "requirements": [], "source_references": ["Synthetic test fixture; no experimental validation"]}


def result(round_, ident="obs1", design_id="E4", actual_temperature=50, value=87, status="qualified", mode="flow"):
    return {"id": ident, "round_id": round_["id"], "design_id": design_id, "experiment_mode": mode,
        "actual_conditions": {"temperature": {"value": actual_temperature, "unit": "C"}, "feed_mL_min": 1.0},
        "analytical_method": "Synthetic calibrated GC", "analytes": {"ester_yield": {"value": value, "unit": "%"},
            "unidentified": {"value": None, "unit": "%"}}, "outcome_status": status,
        "quality": {"accepted": status == "qualified", "basis": "Synthetic accepted result for software test"}}


class CoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = CampaignStore(self.temp.name)

    def test_persistent_generic_campaign_and_immutable_round(self):
        c = self.store.create(synthetic_spec())
        r = self.store.generate(c["id"], {"requested_count": 3, "repeat_count": 1, "repeat_of": "E2"})
        self.assertEqual((len(r["selected"]), len(r["repeats"]), r["total_proposed_runs"]), (3, 1, 4))
        self.assertEqual(len(r["candidate_pool"]), 4)
        self.assertTrue(all(x["question_cards"] for x in r["selected"]))
        self.assertFalse(r["laboratory_released"])
        self.assertIsNone(r["predicted_outcomes"])
        reload = CampaignStore(self.temp.name).get(c["id"])
        self.assertEqual(reload["last_round"]["id"], r["id"])
        self.assertEqual(self.store.list()[0]["round_count"], 1)
        self.assertEqual(reload["spec"]["evidence"][0]["outcomes"], {"ester_yield": 103, "byproduct": None})
        self.assertEqual(r["questions_snapshot"][0]["status"], "unresolved")

    def test_generation_rejections_archived_with_complete_pool(self):
        c = self.store.create(synthetic_spec())
        for request in [{"requested_count": 5}, {"requested_count": True},
                        {"requested_count": 1, "allow_conditional": False}, {"requested_count": 1, "mode": "model"}]:
            with self.assertRaises(CampaignError):
                self.store.generate(c["id"], request)
        rounds = self.store.get(c["id"])["rounds"]
        self.assertEqual(len(rounds), 4)
        self.assertTrue(all(r["status"] == "rejected" and r["selected"] == [] for r in rounds))
        self.assertTrue(all(len(r["candidate_pool"]) == 4 for r in rounds))

    def test_no_pool_is_explicitly_unsupported(self):
        spec = synthetic_spec()
        spec["candidates"] = []
        c = self.store.create(spec)
        with self.assertRaisesRegex(CampaignError, "No candidate pool"):
            self.store.generate(c["id"], {"requested_count": 1})

    def test_no_silent_model_fallback_or_ignored_operating_override(self):
        c = self.store.create(synthetic_spec())
        for request in [{"requested_count": 1, "model_route": "qwen3.6-27b"},
                        {"requested_count": 1, "temperature_C": 1000},
                        {"requested_count": 1, "objective_mode": "optimize"}]:
            with self.assertRaises(CampaignError):
                self.store.generate(c["id"], request)
        self.assertEqual(len(self.store.get(c["id"])["rounds"]), 3)

    def test_researcher_instructions_are_frozen_context_not_operating_override(self):
        c = self.store.create(synthetic_spec())
        instructions = "Consider 500 C and revise the temperature cap."
        r = self.store.generate(c["id"], {"requested_count": 2, "researcher_instructions": instructions})
        self.assertEqual(r["request"]["researcher_instructions"], instructions)
        self.assertTrue(all(x["parameters"]["temperature"]["value"] < 100 for x in r["selected"]))
        self.assertIn("no implicit", r["researcher_instructions_role"])
        with self.assertRaises(CampaignError):
            self.store.generate(c["id"], {"requested_count": 1, "researcher_instructions": {"temperature": 500}})

    def test_duplicate_realizable_settings_rejected_even_with_different_ids(self):
        spec = synthetic_spec()
        duplicate = deepcopy(spec["candidates"][0])
        duplicate["id"] = "different-label"
        spec["candidates"].append(duplicate)
        c = self.store.create(spec)
        r = self.store.generate(c["id"], {"requested_count": 4})
        rejected = next(x for x in r["candidate_pool"] if x["id"] == "different-label")
        self.assertEqual(rejected["selection_disposition"], "rejected")
        self.assertIn("Duplicate", " ".join(rejected["selection_reasons"]))

    def test_model_cannot_bypass_pool_hard_constraints_or_unique_count(self):
        spec = synthetic_spec()
        spec["requirements"] = [{"id": "temperature-cap", "description": "Synthetic strict limit", "scope": "candidate",
            "status": "confirmed", "source_reference": "test", "parameter": "temperature", "operator": "<", "value": 90, "unit": "C"}]
        c = self.store.create(spec)
        for selected in [["E4"], ["invented"], ["E1", "E1"]]:
            with self.assertRaises(CampaignError):
                self.store.generate(c["id"], {"requested_count": len(selected), "selected_candidate_ids": selected})
        r = self.store.generate(c["id"], {"requested_count": 2, "selected_candidate_ids": ["E2", "E1"],
            "model_review": {"review": "Synthetic archived test"}, "model_route": "test-local"})
        self.assertEqual([x["id"] for x in r["selected"]], ["E2", "E1"])
        self.assertEqual(next(x for x in r["candidate_pool"] if x["id"] == "E4")["selection_disposition"], "rejected")

    def test_recipe_edits_invalidate_old_candidates_and_preserve_round(self):
        c = self.store.create(synthetic_spec())
        r = self.store.generate(c["id"], {"requested_count": 1})
        file = Path(self.temp.name) / c["id"] / "rounds" / (r["id"] + ".json")
        original = file.read_bytes()
        c = self.store.revise(c["id"], {"chemistry": {"substrate": "new acid"}})
        self.assertTrue(all(not x["feasible"] for x in c["spec"]["candidates"]))
        self.assertEqual(c["rounds"][0]["current_contract_status"], "superseded_contract_preserved")
        self.assertEqual(file.read_bytes(), original)
        with self.assertRaises(CampaignError):
            self.store.generate(c["id"], {"requested_count": 1})
        self.assertEqual(c["version"], 2)

    def test_actual_qualified_flow_changes_ranking_not_proposed_or_batch(self):
        c = self.store.create(synthetic_spec("optimize"))
        r = self.store.generate(c["id"], {"requested_count": 4})
        c = self.store.add_results(c["id"], [result(r), result(r, "failed", "E1", 30, 999, "assay_failed"),
                                            result(r, "batch", "E1", 30, 200, mode="batch")])
        new = self.store.generate(c["id"], {"requested_count": 1})
        self.assertEqual(new["selected"][0]["id"], "E2")  # measured 50 C, not proposed E4 90 C
        self.assertEqual(new["feedback_used"], ["obs1"])
        self.assertEqual(len(new["feedback_exclusions"]), 2)
        self.assertEqual(c["results"][1]["analytes"]["ester_yield"]["value"], 999)
        self.assertEqual(c["results"][0]["proposed_conditions"]["temperature"]["value"], 90)
        self.assertEqual(c["spec"]["questions"][0]["status"], "unresolved")

    def test_result_validation_actual_method_link_quality(self):
        c = self.store.create(synthetic_spec())
        r = self.store.generate(c["id"], {"requested_count": 4})
        for field, value in [("actual_conditions", {}), ("analytical_method", None), ("design_id", "missing"),
                             ("quality", {}), ("outcome_status", "predicted")]:
            bad = result(r)
            bad[field] = value
            with self.assertRaises(ValueError):
                self.store.add_results(c["id"], [bad])
        self.assertEqual(self.store.get(c["id"])["results"], [])

    def test_results_preserve_missing_above_100_and_corrections(self):
        c = self.store.create(synthetic_spec("optimize"))
        r = self.store.generate(c["id"], {"requested_count": 4})
        self.store.add_results(c["id"], [result(r, value=104)])
        corrected = result(r, "correction", actual_temperature=90, value=20)
        corrected.update(supersedes="obs1", correction_reason="Synthetic calibration correction")
        c = self.store.add_results(c["id"], [corrected])
        self.assertEqual(c["results"][0]["analytes"]["ester_yield"]["value"], 104)
        self.assertIsNone(c["results"][0]["analytes"]["unidentified"]["value"])
        new = self.store.generate(c["id"], {"requested_count": 1})
        self.assertEqual(new["feedback_used"], ["correction"])
        self.assertEqual(new["superseded_result_ids_excluded"], ["obs1"])
        with self.assertRaises(CampaignError):
            self.store.add_results(c["id"], [corrected])

    def test_unknown_assay_cannot_qualify_but_unqualified_and_legacy_records_survive(self):
        c = self.store.create(synthetic_spec("optimize"))
        r = self.store.generate(c["id"], {"requested_count": 4})
        for method in ["unknown", {"status": "unknown", "name": None}, {"name": None},
                       {"calibration": "reported", "status": "known"}, {"name": "GC", "status": "unconfirmed"}]:
            with self.subTest(method=method):
                bad = result(r)
                bad["analytical_method"] = method
                with self.assertRaisesRegex(ValueError, "[Aa]nalytical|method"):
                    self.store.add_results(c["id"], [bad])
        unknown = result(r, "unknown-unqualified", status="unqualified")
        unknown["analytical_method"] = {"status": "unknown", "name": None}
        c = self.store.add_results(c["id"], [unknown, result(r)])
        self.assertEqual(c["results"][0]["analytical_method"], unknown["analytical_method"])
        self.assertEqual(feedback_summary(c)["numerical_optimization_feedback_eligible_ids"], ["obs1"])
        # Emulate an old archived record admitted before this guard. Auditing
        # must exclude it without rewriting or deleting the original record.
        legacy = deepcopy(c)
        legacy["results"][1]["analytical_method"] = {"status": "unknown", "name": None}
        before = deepcopy(legacy["results"])
        ledger = feedback_summary(legacy)
        self.assertEqual(ledger["numerical_optimization_feedback_eligible_ids"], [])
        self.assertIn("Analytical method", next(x["reason"] for x in ledger["excluded_from_numerical_feedback"] if x["result_id"] == "obs1"))
        self.assertEqual(legacy["results"], before)

    def test_generalize_prioritizes_different_substrates(self):
        spec = synthetic_spec("generalize")
        spec["candidates"][2]["substrate_id"] = "acid-B"
        spec["candidates"][3]["substrate_id"] = "acid-C"
        c = self.store.create(spec)
        r = self.store.generate(c["id"], {"requested_count": 3})
        self.assertEqual({x["substrate_id"] for x in r["selected"]}, {"acid-A", "acid-B", "acid-C"})
        self.assertFalse(r["experimental_validation"])

    def test_budget_counts_repeats_and_rejects_unknown_costs(self):
        c = self.store.create(synthetic_spec())
        r = self.store.generate(c["id"], {"requested_count": 2, "repeat_count": 1, "budget": {"material_mg": 15}})
        self.assertEqual(r["cost_totals"]["material_mg"], 15)
        for budget in [{"material_mg": 10}, {"unassigned_collection_mL": 20}]:
            with self.assertRaises(CampaignError):
                self.store.generate(c["id"], {"requested_count": 2, "repeat_count": 1, "budget": budget})

    def test_question_changes_require_evidence_not_fake_posterior(self):
        c = self.store.create(synthetic_spec())
        q = deepcopy(c["spec"]["questions"][0])
        q["status"] = "supported"
        with self.assertRaises(ValueError):
            self.store.add_question(c["id"], q)
        q["amendment"] = {"author": "test researcher", "reason": "Synthetic inspection", "evidence_ids": ["batch:one"]}
        q["posterior_probability"] = .9
        with self.assertRaises(ValueError):
            self.store.add_question(c["id"], q)
        del q["posterior_probability"]
        updated = self.store.add_question(c["id"], q)
        self.assertEqual(updated["spec"]["questions"][0]["status"], "supported")
        original = self.store._read(Path(self.temp.name) / c["id"] / "revisions" / "000001.json")
        self.assertEqual(original["spec"]["questions"][0]["status"], "unresolved")

    def test_question_amendment_cannot_cite_invented_evidence(self):
        c = self.store.create(synthetic_spec())
        q = deepcopy(c["spec"]["questions"][0])
        q.update(status="supported", amendment={"author": "researcher", "reason": "Test", "evidence_ids": ["nonexistent"]})
        with self.assertRaisesRegex(ValueError, "unarchived evidence"):
            self.store.add_question(c["id"], q)
        with self.assertRaisesRegex(ValueError, "unarchived evidence"):
            self.store.revise(c["id"], {"questions": [q]})
        self.assertEqual(self.store.get(c["id"])["version"], 1)

    def test_feedback_context_identifies_failed_and_superseded_results(self):
        c = self.store.create(synthetic_spec("optimize"))
        r = self.store.generate(c["id"], {"requested_count": 4})
        self.store.add_results(c["id"], [result(r), result(r, "failed", status="assay_failed", value=None)])
        correction = result(r, "corrected", value=75)
        correction.update(supersedes="obs1", correction_reason="Synthetic calibration update")
        c = self.store.add_results(c["id"], [correction])
        ledger = feedback_summary(c)
        self.assertEqual(ledger["numerical_optimization_feedback_eligible_ids"], ["corrected"])
        self.assertEqual({e["result_id"] for e in ledger["excluded_from_numerical_feedback"]}, {"obs1", "failed"})

    def test_explicit_measured_mapping_excludes_targets_and_uses_actual_controls(self):
        spec = synthetic_spec("optimize")
        spec["observation_contract"] = {"schema_version": "flowpilotx-observation-contract-v1",
            "required_actual_conditions": ["observed_temperature", "observed_feed"],
            "parameter_mapping": {"temperature": "observed_temperature", "feed_mL_min": "observed_feed"},
            "types": {"observed_temperature": "number", "observed_feed": "number"},
            "units": {"observed_temperature": "C", "observed_feed": "mL/min"},
            "comparability_requirements": [{"parameter": "observed_feed", "operator": ">", "value": 0, "unit": "mL/min"}]}
        for i, candidate in enumerate(spec["candidates"]):
            candidate["parameters"].update(target_time_min=10 * i, calculated_index=100 - i,
                                            pressure_assumed=2, unmeasured_metadata=None)
        c = self.store.create(spec)
        r = self.store.generate(c["id"], {"requested_count": 4})
        observed = result(r)
        observed["actual_conditions"] = {"observed_temperature": 50, "observed_feed": 1.0}
        c = self.store.add_results(c["id"], [observed])
        new = self.store.generate(c["id"], {"requested_count": 1})
        self.assertEqual(new["feedback_used"], ["obs1"])
        self.assertEqual(new["selected"][0]["id"], "E2")
        self.assertEqual(c["results"][0]["source_observation_contract_sha256"], digest(spec["observation_contract"]))
        self.assertEqual(c["results"][0]["actual_conditions"], observed["actual_conditions"])
        self.assertEqual(c["results"][0]["proposed_conditions"]["temperature"]["value"], 90)
        self.assertNotIn("target_time_min", c["results"][0]["actual_conditions"])

    def test_observation_mapping_requires_existing_controls_not_guessed_aliases(self):
        spec = synthetic_spec()
        spec["observation_contract"] = {"schema_version": "flowpilotx-observation-contract-v1",
            "required_actual_conditions": ["measured_temperature"],
            "parameter_mapping": {"invented_control": "measured_temperature"}}
        with self.assertRaisesRegex(ValueError, "missing candidate control"):
            self.store.create(spec)
        spec["observation_contract"]["parameter_mapping"] = {"temperature": "measured_temperature"}
        spec["observation_contract"]["units"] = {"measured_temperature": "K"}
        with self.assertRaisesRegex(ValueError, "units differ"):
            self.store.create(spec)


    def test_inventory_unknowns_and_operators_explicit(self):
        inv = synthetic_spec()["inventory"]
        for key in ["availability", "quantity"]:
            bad = deepcopy(inv)
            del bad["items"][0][key]
            with self.assertRaises(ValueError):
                validate_inventory(bad)
        bad = deepcopy(inv)
        del bad["items"][0]["limits"][0]["operator"]
        with self.assertRaises(ValueError):
            validate_inventory(bad)
        inv["items"][0].update(quantity=None, availability="unknown")
        self.assertIsNone(validate_inventory(inv)["items"][0]["quantity"])

    def test_generic_import_preserves_unknown_columns_and_exact_ids(self):
        raw = b'dataset_id,source_entry,experiment_mode,ester_yield_pct,unknown_peak,note\nother,9007199254740993,batch,104,,keep this\n'
        row = import_evidence("ester.csv", raw)["records"][0]
        self.assertEqual((row["source_entry"], row["ester_yield_pct"], row["unknown_peak"], row["note"]), ("9007199254740993", "104", "", "keep this"))
        self.assertFalse(row["feedback_eligible"])
        self.assertNotIn("sulfoxide_pct", row)
        self.assertEqual(row["id"], import_evidence("renamed.csv", raw)["records"][0]["id"])
        original = {"records": [{"source_entry": "p", "status": "conditional_proposal", "conditions": {"T": 80}, "actual_conditions": {"T": None}}]}
        loaded = import_evidence("proposal.json", json.dumps(original).encode())["records"][0]
        self.assertEqual(loaded["actual_conditions"], {"T": None})
        self.assertEqual(loaded["conditions"], {"T": 80})

    def test_import_rejects_duplicate_keys_and_ids(self):
        for raw in [b'{"records":[],"records":[{}]}', b'{"records":[{"source_entry":1},{"source_entry":1}]}']:
            with self.assertRaises(ValueError):
                import_evidence("bad.json", raw)

    def test_evidence_changes_invalidate_compact_packet_not_physics(self):
        spec = synthetic_spec()
        spec["model_evidence"] = {"compact": "original"}
        c = self.store.create(spec)
        c = self.store.revise(c["id"], {"evidence": [{"id": "new-record", "value": None}]})
        self.assertNotIn("model_evidence", c["spec"])
        self.assertEqual(c["spec"]["evidence_sha256"], digest(c["spec"]["evidence"]))
        self.assertTrue(all(x["feasible"] for x in c["spec"]["candidates"]))

    def test_compact_model_views_replace_atomically_without_obsolete_fields(self):
        spec = synthetic_spec()
        spec["model_inventory"] = {"catalog_index": ["old large archive"], "obsolete": True}
        spec["model_evidence"] = {"old_records": ["old"]}
        c = self.store.create(spec)
        c = self.store.revise(c["id"], {"model_inventory": {"items": ["current compact"]},
                                      "model_evidence": {"records": ["current"]}})
        self.assertEqual(c["spec"]["model_inventory"], {"items": ["current compact"]})
        self.assertEqual(c["spec"]["model_evidence"], {"records": ["current"]})
        self.assertTrue(all(x["feasible"] for x in c["spec"]["candidates"]))

    def test_revision_guard_rejects_plan_from_stale_context_and_archives_it(self):
        c = self.store.create(synthetic_spec())
        request = {"requested_count": 1, "expected_campaign_version": c["version"],
                   "expected_revision_sha256": c["content_sha256"]}
        self.store.generate(c["id"], request)
        changed = self.store.revise(c["id"], {"research_notes": "New information available"})
        with self.assertRaisesRegex(CampaignError, "version changed"):
            self.store.generate(c["id"], request)
        request["expected_campaign_version"] = changed["version"]
        with self.assertRaisesRegex(CampaignError, "revision hash changed"):
            self.store.generate(c["id"], request)
        request["expected_revision_sha256"] = changed["content_sha256"]
        self.assertEqual(self.store.generate(c["id"], request)["unique_design_count"], 1)
        self.assertEqual(len(self.store.get(c["id"])["rounds"]), 4)

    def test_context_staleness_is_derived_without_rewriting_round(self):
        c = self.store.create(synthetic_spec())
        r = self.store.generate(c["id"], {"requested_count": 1})
        path = Path(self.temp.name) / c["id"] / "rounds" / (r["id"] + ".json")
        original = path.read_bytes()
        c = self.store.revise(c["id"], {"evidence": [{"id": "new-observation"}]})
        prior_round = c["rounds"][0]
        self.assertEqual(prior_round["current_contract_status"], "matches_current_contract")
        self.assertEqual(prior_round["scientific_context_changes_since_freeze"], ["evidence"])
        self.assertEqual(prior_round["current_scientific_context_status"], "changed_context_preserved_prospective_snapshot")
        self.assertEqual(path.read_bytes(), original)

    def test_detects_modified_frozen_snapshot(self):
        c = self.store.create(synthetic_spec())
        path = Path(self.temp.name) / c["id"] / "revisions" / "000001.json"
        data = json.loads(path.read_text())
        data["name"] = "tampered"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(CampaignError, "integrity"):
            self.store.get(c["id"])


if __name__ == "__main__":
    unittest.main()
