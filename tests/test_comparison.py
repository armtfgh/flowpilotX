"""Deterministic synthetic fixtures; none are KHU laboratory recommendations."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from khu_sulfoxidation.comparison import (
    ARMS, build_comparison_plan, build_matched_inputs, canonical_json,
    freeze_comparison, sha256, validate_matched_inputs,
)
from khu_sulfoxidation.review import (
    archive_review_attempt, build_review_request, disabled_campaign,
    validate_paid_campaign, validate_review_response,
)

DIGEST = "a" * 64


def common_fixture(count=1):
    return {
        "study_id": "synthetic_test_only",
        "configuration_version": "fixture-v1",
        "nomination_count": count,
        "shared_recipe": {"protocol_id": "standard-batch-protocol", "protocol_sha256": DIGEST, "substrate": "fixture-sulfide", "catalyst": "fixture-catalyst", "solvent": "fixture-solvent", "substrate_concentration_M": 0.25, "catalyst_loading_mol_percent": 3.0, "batch_temperature_C": 24.0, "batch_time_h": 10.0, "batch_oxidant": "air", "selection_disclosure": "Shared protocol chosen from the historical anchor; its outcomes are withheld."},
        "apparatus": {"profile_id": "fixture-apparatus", "profile_sha256": DIGEST, "confirmed": True, "eligible_equipment_ids": ["fixture-coil"]},
        "physical_candidates": [
            {"candidate_id": "c1", "reactor_id": "fixture-coil", "liquid_flow_mL_min": 0.1, "gas_reference_flow_mL_min": 1.0, "time_basis": "liquid_only_space_time_min", "time_coordinate_min": 10.0, "oxygen_equivalents": 1.0, "feasible": True, "calculation_sha256": DIGEST},
            {"candidate_id": "c2", "reactor_id": "fixture-coil", "liquid_flow_mL_min": 0.2, "gas_reference_flow_mL_min": 2.0, "time_basis": "liquid_only_space_time_min", "time_coordinate_min": 5.0, "oxygen_equivalents": 1.0, "feasible": True, "calculation_sha256": DIGEST},
        ],
        "pool_origin": {"basis": "apparatus_envelope", "independent_of_history": True, "declaration": "Synthetic physical pool created before loading evidence; not study settings.", "configuration_sha256": DIGEST},
        "endpoints": {"primary": "measured_sulfoxide_yield_percent", "sulfone_limit_percent": 5.0, "secondary": ["residual_sulfide_percent", "sulfone_percent"]},
        "run_order": {"method": "randomized", "seed": 12, "sample_blinding": True, "measurement_reuse": "one_shared_condition_not_independent_evidence"},
        "generation": {"generator_id": "offline-fixture", "provider": "offline", "model": "fixture", "temperature": 0.0, "seed": 12, "candidate_budget": 2, "max_calls_per_arm": 6, "max_output_tokens_per_call": 1000, "revision_budget": 0, "max_cost_USD_per_arm": 0.0, "routing_config_sha256": sha256(disabled_campaign())},
        "instructions": "Choose informative experiments within the physical pool; no invented flow outcomes.",
        "shared_sources": [],
    }


def history_fixture():
    return [{"source_id": "homogeneous-batch-112", "source_sha256": DIGEST, "kind": "batch_record", "payload": {"experiment_mode": "batch", "residual_sulfide_percent": 80, "sulfoxide_percent": 1, "sulfone_percent": 0}}]


def audit_fixture():
    return {"reviewer": "synthetic-test-reviewer", "reviewed_at": "2026-10-13T12:00:00+09:00", "no_history_in_common_text": True, "candidate_pool_independent_of_history": True, "no_external_rag_or_cache": True, "notes": "Synthetic fixture only; no model or laboratory execution."}


def nominations_fixture(count=1):
    return {arm: [{"candidate_id": f"c{index+1}", "rationale": "Exploratory hypothesis within the shared apparatus.", "evidence_citations": ["fixture-apparatus"]} for index in range(count)] for arm in ARMS}


def response_fixture(role="Chief"):
    return {"role": role, "candidate_reviews": [{"candidate_id": candidate, "verdict": "eligible", "statements": [{"kind": "hypothesis", "text": "Screen this feasible exposure; performance remains unknown.", "citations": ["fixture-apparatus"]}], "rejection_reasons": []} for candidate in ("c1", "c2")], "nominated_candidate_ids": ["c1"] if role == "Chief" else [], "limitations": ["Synthetic fixture does not establish flow performance."]}


def metadata_fixture():
    return {"attempt_id": "fixture-attempt-1", "started_at": "2026-10-13T12:00:00+09:00", "completed_at": "2026-10-13T12:00:01+09:00", "role": "Chief", "provider": "offline", "model": "fixture", "status": "completed", "input_tokens": None, "output_tokens": None, "cost_USD": None, "source_hashes": {"standard-batch-protocol": DIGEST, "fixture-apparatus": DIGEST}}


def paid_fixture():
    return {**disabled_campaign(), "enabled": True, "campaign_id": "fixture-campaign", "authorization": {"authorized": True, "reference": "Synthetic gate test, not actual authorization", "established_at": "2026-10-13T12:00:00+09:00", "study_id": "khu_sulfoxidation_v1"}, "routing": [{"role": "Chief", "provider": "fixture-provider", "model": "fixture-model-1"}], "maximum_cost_USD": 2.0, "max_calls": 2, "max_output_tokens_per_call": 1000, "revision_budget": 0, "decoding": {"temperature": 0.0, "seed": 1}}


class ComparisonTests(unittest.TestCase):
    def test_unconfirmed_plan_has_slots_and_no_nominations(self):
        plan = build_comparison_plan(2)
        self.assertEqual(plan["nomination_count_per_arm"], 2)
        self.assertFalse(plan["paid_calls_enabled"])
        self.assertTrue(all(not data["nominations"] for data in plan["arms"].values()))
        with self.assertRaises(ValueError):
            build_comparison_plan(3)

    def test_zero_optional_nominations_disables_comparison_but_not_freeze_requirements(self):
        plan = build_comparison_plan(0)
        self.assertFalse(plan["enabled"])
        self.assertEqual(plan["status"], "disabled_optional_comparison")
        self.assertEqual(plan["nomination_count_per_arm"], 0)
        self.assertTrue(all(not data["nominations"] for data in plan["arms"].values()))
        common = common_fixture()
        common["nomination_count"] = 0
        with self.assertRaises(ValueError):
            build_matched_inputs(common, history_fixture())

    def test_complete_inputs_isolate_history_and_share_physical_pool(self):
        inputs = build_matched_inputs(common_fixture(), history_fixture())
        validate_matched_inputs(inputs)
        informed = inputs["arms"][ARMS[0]]["visible_input"]
        omitted = inputs["arms"][ARMS[1]]["visible_input"]
        self.assertEqual(informed["common"], omitted["common"])
        self.assertNotIn("homogeneous-batch-112", canonical_json(omitted))
        self.assertFalse(omitted["batch_optimization_evidence"])
        self.assertIn("historical anchor", omitted["common"]["shared_recipe"]["selection_disclosure"])
        self.assertEqual(sha256(omitted), inputs["arms"][ARMS[1]]["visible_input_sha256"])
        informed["common"]["instructions"] = "modified"
        self.assertNotEqual(informed["common"], omitted["common"])
        with self.assertRaises(ValueError):
            validate_matched_inputs(inputs)

    def test_unknown_history_fields_rejected_in_shared_inputs(self):
        for part, field in (("shared_recipe", "reported_sulfoxide_percent"), ("apparatus", "history_summary")):
            common = common_fixture()
            common[part][field] = 98
            with self.subTest(part=part), self.assertRaises(ValueError):
                build_matched_inputs(common, history_fixture())
        common = common_fixture()
        common["physical_candidates"][0]["historical_rationale"] = "best prior yield"
        with self.assertRaises(ValueError):
            build_matched_inputs(common, history_fixture())

    def test_retrieval_cache_injection_rejected(self):
        inputs = build_matched_inputs(common_fixture(), history_fixture())
        inputs["arms"][ARMS[1]]["visible_input"]["access"]["shared_rag_enabled"] = True
        inputs["arms"][ARMS[1]]["visible_input_sha256"] = sha256(inputs["arms"][ARMS[1]]["visible_input"])
        with self.assertRaises(ValueError):
            validate_matched_inputs(inputs)

    def test_historically_selected_pool_and_unconfirmed_apparatus_rejected(self):
        for part, field, value in (("pool_origin", "independent_of_history", False), ("apparatus", "confirmed", False)):
            common = common_fixture()
            common[part][field] = value
            with self.subTest(part=part), self.assertRaises(ValueError):
                build_matched_inputs(common, history_fixture())

    def test_duplicate_history_shared_identity_and_missing_endpoint_rejected(self):
        history = history_fixture()
        history[0]["source_id"] = "standard-batch-protocol"
        with self.assertRaises(ValueError):
            build_matched_inputs(common_fixture(), history)
        common = common_fixture()
        common["endpoints"]["sulfone_limit_percent"] = None
        with self.assertRaises(ValueError):
            build_matched_inputs(common, history_fixture())

    def test_freeze_both_arms_allows_convergence_and_is_immutable(self):
        inputs = build_matched_inputs(common_fixture(2), history_fixture())
        with tempfile.TemporaryDirectory() as temporary:
            result = freeze_comparison(temporary, inputs, nominations_fixture(2), frozen_at="2026-10-13T12:00:00+09:00", boundary_audit=audit_fixture())
            self.assertEqual(result["shared_condition_ids"], ["c1", "c2"])
            stored = json.loads(Path(result["path"]).read_text())
            self.assertEqual(sha256(stored["snapshot"]), stored["snapshot_sha256"])
            self.assertEqual(stored["snapshot"]["matched_inputs"], inputs)
            self.assertIn("not independent evidence", stored["snapshot"]["measurement_policy"])
            with self.assertRaises(FileExistsError):
                freeze_comparison(temporary, inputs, nominations_fixture(2), frozen_at="2026-10-13T12:00:00+09:00", boundary_audit=audit_fixture())

    def test_postoutcome_and_unreviewed_freezes_rejected(self):
        inputs = build_matched_inputs(common_fixture(), history_fixture())
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                freeze_comparison(temporary, inputs, nominations_fixture(), frozen_at="fixture", boundary_audit=audit_fixture(), outcomes_seen=True)
            audit = audit_fixture()
            audit["no_history_in_common_text"] = False
            with self.assertRaises(ValueError):
                freeze_comparison(temporary, inputs, nominations_fixture(), frozen_at="fixture", boundary_audit=audit)
            self.assertFalse(list(Path(temporary).iterdir()))

    def test_hidden_history_citation_and_asymmetric_nominations_rejected(self):
        inputs = build_matched_inputs(common_fixture(), history_fixture())
        with tempfile.TemporaryDirectory() as temporary:
            nominations = nominations_fixture()
            nominations[ARMS[1]][0]["evidence_citations"] = ["homogeneous-batch-112"]
            with self.assertRaises(ValueError):
                freeze_comparison(temporary, inputs, nominations, frozen_at="fixture", boundary_audit=audit_fixture())
            nominations = nominations_fixture()
            nominations[ARMS[1]] = []
            with self.assertRaises(ValueError):
                freeze_comparison(temporary, inputs, nominations, frozen_at="fixture", boundary_audit=audit_fixture())


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.inputs = build_matched_inputs(common_fixture(), history_fixture())
        self.omitted = self.inputs["arms"]["history_omitted"]["visible_input"]
        self.informed = self.inputs["arms"]["history_informed"]["visible_input"]

    def test_review_requests_preserve_common_system_and_exact_hashes(self):
        informed = build_review_request(self.informed, "Chief")
        omitted = build_review_request(self.omitted, "Chief")
        self.assertEqual(informed["system_prompt"], omitted["system_prompt"])
        self.assertEqual(sha256({k: omitted[k] for k in ("system_prompt", "user_prompt")}), omitted["prompt_sha256"])
        self.assertNotIn("homogeneous-batch-112", omitted["user_prompt"])
        self.assertTrue(validate_review_response(response_fixture(), self.omitted, "Chief")["valid"])

    def test_hidden_citation_and_measured_flow_claim_type_rejected(self):
        response = response_fixture()
        response["candidate_reviews"][0]["statements"][0]["citations"] = ["homogeneous-batch-112"]
        with self.assertRaises(ValueError):
            validate_review_response(response, self.omitted, "Chief")
        response = response_fixture()
        response["candidate_reviews"][0]["statements"][0]["kind"] = "measured_flow_yield"
        with self.assertRaises(ValueError):
            validate_review_response(response, self.omitted, "Chief")

    def test_batch_observation_requires_visible_historical_record(self):
        response = response_fixture()
        response["candidate_reviews"][0]["statements"] = [{"kind": "batch_observation", "text": "Recorded 1% sulfoxide does not establish flow yield.", "citations": ["homogeneous-batch-112"]}]
        self.assertTrue(validate_review_response(response, self.informed, "Chief")["valid"])
        with self.assertRaises(ValueError):
            validate_review_response(response, self.omitted, "Chief")

    def test_rejections_and_every_candidate_must_remain(self):
        response = response_fixture()
        response["candidate_reviews"].pop()
        with self.assertRaises(ValueError):
            validate_review_response(response, self.omitted, "Chief")
        response = response_fixture()
        response["candidate_reviews"][1]["verdict"] = "reject"
        with self.assertRaises(ValueError):
            validate_review_response(response, self.omitted, "Chief")

    def test_archive_retains_invalid_attempts_and_exact_text_with_unknown_cost(self):
        request = build_review_request(self.omitted, "Chief")
        with tempfile.TemporaryDirectory() as temporary:
            raw = "not JSON\nverbatim unsuccessful response\n"
            first = archive_review_attempt(temporary, request=request, raw_response=raw, configuration=disabled_campaign(), metadata=metadata_fixture())
            second = archive_review_attempt(temporary, request=request, raw_response=canonical_json(response_fixture()), configuration=disabled_campaign(), metadata=metadata_fixture())
            self.assertFalse(first["valid"])
            self.assertTrue(second["valid"], second["errors"])
            self.assertNotEqual(first["path"], second["path"])
            self.assertEqual((Path(first["path"]) / "response.txt").read_text(), raw)
            manifest = json.loads((Path(first["path"]) / "manifest.json").read_text())
            self.assertIsNone(manifest["cost_USD"])
            self.assertEqual(manifest["files_sha256"]["response.txt"], hashlib.sha256(raw.encode()).hexdigest())
            self.assertEqual(len(list(Path(temporary).iterdir())), 2)

    def test_prompt_tampering_and_unapproved_paid_import_are_retained_and_flagged(self):
        request = build_review_request(self.omitted, "Chief")
        request["system_prompt"] += " modified"
        with tempfile.TemporaryDirectory() as temporary:
            result = archive_review_attempt(temporary, request=request, raw_response=canonical_json(response_fixture()), configuration=disabled_campaign(), metadata=metadata_fixture())
            self.assertFalse(result["valid"])
            self.assertTrue(any("hash mismatch" in err for err in result["errors"]))
            metadata = metadata_fixture()
            metadata["provider"] = "paid-provider"
            metadata["cost_USD"] = 0.2
            result = archive_review_attempt(temporary, request=build_review_request(self.omitted, "Chief"), raw_response=canonical_json(response_fixture()), configuration=disabled_campaign(), metadata=metadata)
            self.assertFalse(result["valid"])
            self.assertTrue(Path(result["path"]).is_dir())

    def test_configuration_and_incomplete_source_hashes_are_flagged_after_archiving(self):
        request = build_review_request(self.omitted, "Chief")
        with tempfile.TemporaryDirectory() as temporary:
            metadata = metadata_fixture()
            del metadata["source_hashes"]["fixture-apparatus"]
            result = archive_review_attempt(temporary, request=request, raw_response=canonical_json(response_fixture()), configuration=disabled_campaign(), metadata=metadata)
            self.assertFalse(result["valid"])
            self.assertIn("every visible source", result["errors"][0])
            config = disabled_campaign()
            config["campaign_id"] = "different-config"
            result = archive_review_attempt(temporary, request=request, raw_response=canonical_json(response_fixture()), configuration=config, metadata=metadata_fixture())
            self.assertFalse(result["valid"])
            self.assertIn("frozen routing configuration", result["errors"][0])
            self.assertEqual(len(list(Path(temporary).iterdir())), 2)

    def test_paid_gate_disabled_by_default_and_exhaustion_blocks(self):
        self.assertFalse(validate_paid_campaign(None)["configuration_ready"])
        config = paid_fixture()
        ready = validate_paid_campaign(config, spent_USD=0.2, completed_calls=1, reserved_next_call_USD=0.1)
        self.assertTrue(ready["configuration_ready"])
        self.assertFalse(ready["network_calls_available"])
        self.assertFalse(validate_paid_campaign(config, completed_calls=2)["configuration_ready"])
        self.assertFalse(validate_paid_campaign(config, spent_USD=1.9, reserved_next_call_USD=0.2)["configuration_ready"])

    def test_paid_gate_no_legacy_authorization_fallback_or_unbounded_model(self):
        for field, value in (("fallback_enabled", True), ("automatic_reembedding_enabled", True), ("network_execution_implemented", True)):
            config = paid_fixture()
            config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_paid_campaign(config)
        config = paid_fixture()
        config["authorization"]["study_id"] = "old_astra_campaign"
        with self.assertRaises(ValueError):
            validate_paid_campaign(config)
        config = paid_fixture()
        config["routing"][0]["model"] = "default"
        with self.assertRaises(ValueError):
            validate_paid_campaign(config)


if __name__ == "__main__":
    unittest.main()
