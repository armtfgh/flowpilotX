"""Synthetic frozen nominations exercise report consistency; no model calls."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from khu_sulfoxidation.campaign_report import export_campaign_report
from khu_sulfoxidation.conditional_planning import build_conditional_pool
from khu_sulfoxidation.provenance import digest, file_hash


def synthetic_freezes(directory, *, converge=False):
    """Write synthetic test-only campaign archives, never an actual nomination."""
    directory.mkdir(parents=True, exist_ok=True)
    pool = build_conditional_pool(ROOT)
    informed = pool["grid_groups"][0]["candidate_ids"]
    omitted = informed if converge else pool["grid_groups"][-1]["candidate_ids"]
    def chief(ids):
        return {"role": "Chief", "conclusion": "SYNTHETIC TEST ONLY <not a chemical prediction>",
            "recommended_core_candidate_ids": ids, "comparison_nomination_ids": ids[:2],
            "repeat_candidate_id": ids[2], "candidate_reviews": [{"candidate_id": cid, "verdict": "retain",
                "rationale": "SYNTHETIC test rationale; explore supplied oxygen without claiming a measured outcome.",
                "citations": ["calculation:pool"]} for cid in ids],
            "findings": [{"kind": "unresolved", "statement": "Synthetic fixture has no laboratory release.", "citations": ["apparatus:conditional"]}],
            "limitations": ["SYNTHETIC TEST ONLY; no model call or experiment took place."]}
    snapshot = {"configuration": {"campaign_id": "SYNTHETIC_REPORT_TEST_NO_MODEL_CALLS", "maximum_cost_USD": 0},
        "pool": pool, "inputs": {"history_informed": "synthetic", "history_omitted": "synthetic"},
        "evidence_coverage": {"status": "synthetic_fixture"}, "source_hashes": {}}
    first = {"frozen_at": "SYNTHETIC", "snapshot_sha256": digest(snapshot), "snapshot": snapshot, "flow_outcomes_seen": False}
    second = {"schema_version": "khu-model-nominations-v1", "frozen_at": "SYNTHETIC", "flow_outcomes_seen": False,
        "laboratory_released": False, "input_freeze_sha256": digest(snapshot),
        "arms": {"history_informed": {"Chief": chief(informed)}, "history_omitted": {"Chief": chief(omitted)}},
        "api_audit": {"attempt_count": 0, "known_token_cost_USD": 0, "accounted_cost_USD": 0, "entries": []}}
    for name, data in [("input_freeze.json", first), ("nominations_freeze.json", second)]:
        (directory / name).write_text(json.dumps(data, indent=2))
    return first, second


class CampaignReportTests(unittest.TestCase):
    def test_nine_runs_exact_repeat_and_consistent_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first, second = synthetic_freezes(base / "campaign")
            before = {p.name: p.read_bytes() for p in (base / "campaign").iterdir()}
            result = export_campaign_report(ROOT, base / "campaign", base / "review")
            self.assertEqual(result["condition_count"], 9)
            self.assertEqual(result["unique_condition_count"], 8)
            data = json.loads((base / "review/authoritative_conditions.json").read_text())
            conditions = data["conditions"]
            repeat = conditions[6]
            self.assertEqual(repeat["candidate_id"], second["arms"]["history_informed"]["Chief"]["repeat_candidate_id"])
            original = next(c for c in conditions[:6] if c["candidate_id"] == repeat["candidate_id"])
            self.assertEqual(repeat["repeat_of_condition_id"], original["condition_id"])
            self.assertEqual(repeat["calculated"], original["calculated"])
            self.assertEqual(repeat["proposed_setpoints"], original["proposed_setpoints"])
            with (base / "review/conditions.csv").open() as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 9)
            for row, csv_row in zip(conditions, csv_rows):
                self.assertEqual(row["condition_id"], csv_row["condition_id"])
                self.assertAlmostEqual(float(csv_row["liquid_only_space_time_min"]), row["calculated"]["liquid_only_space_time_min"])
                self.assertAlmostEqual(float(csv_row["proposed_gas_reference_flow_mL_min"]), row["proposed_setpoints"]["gas_reference_flow_mL_min"])
                self.assertEqual(csv_row["selected_BPR"], "")
                self.assertEqual(csv_row["reactor_pressure_bar_abs"], "")
            for name, content in before.items():
                self.assertEqual((base / "campaign" / name).read_bytes(), content)
                self.assertEqual((base / "review" / name).read_bytes(), content)

    def test_convergence_reuses_measurement_without_extra_comparison_opportunity(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            synthetic_freezes(base / "campaign", converge=True)
            result = export_campaign_report(ROOT, base / "campaign", base / "review")
            self.assertEqual(result["condition_count"], 7)
            self.assertEqual(result["unique_condition_count"], 6)
            comparison = result["comparison"]
            self.assertEqual(comparison["additional_omitted_runs"], 0)
            self.assertEqual(len(comparison["shared_nominated_candidate_ids"]), 2)
            for arm in comparison["arms"].values():
                self.assertEqual(len(arm), 2)
            self.assertEqual([x["condition_id"] for x in comparison["arms"]["history_informed"]],
                             [x["condition_id"] for x in comparison["arms"]["history_omitted"]])
            self.assertIn("best of the six", comparison["rule"])

    def test_manifest_results_unknowns_and_unselected_are_not_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            synthetic_freezes(base / "campaign")
            export_campaign_report(ROOT, base / "campaign", base / "review")
            manifest = json.loads((base / "review/manifest.json").read_text())
            for name, descriptor in manifest["files"].items():
                self.assertEqual(file_hash(base / "review" / name), descriptor["sha256"])
            template = json.loads((base / "review/results_template.json").read_text())
            for row in template["records"]:
                for field in ["residual_sulfide_pct", "sulfoxide_pct", "sulfone_pct", "leak_observed", "analytical_method", "actual_temperature_C"]:
                    self.assertIsNone(row[field])
                self.assertIsNone(row["actual_independent_preparation"])
            dispositions = json.loads((base / "review/pool_selection_dispositions.json").read_text())
            self.assertTrue(any(r["disposition"] == "eligible_not_selected" for r in dispositions))
            self.assertEqual(sum(r["disposition"] == "rejected_by_pool_constraints" for r in dispositions), 6)
            document = (base / "review/campaign_report.html").read_text()
            self.assertIn("&lt;not a chemical prediction&gt;", document)
            self.assertIn("not BPR settings", document)
            self.assertIn("does not isolate an oxygen-only causal effect", document)
            self.assertIn("stroke-dasharray", (base / "review/apparatus.svg").read_text())
            preparation = json.loads((base / "review/feed_preparation.json").read_text())
            self.assertEqual(preparation["pure_equivalent_mass_mg"]["thioanisole"], 310.50)
            self.assertAlmostEqual(preparation["pure_equivalent_mass_mg"]["acridine"], .075 * 179.22)
            self.assertIsNone(preparation["actual_weighed_masses_mg"])
            self.assertIsNone(preparation["total_campaign_feed_volume_mL"])
            self.assertIn("not part of either arm", preparation["reference_boundary"])
            basis = json.loads((base / "review/calculation_basis.json").read_text())
            self.assertIn("P_ref * Q_ref / (R * T_ref)", basis["equations"]["total_gas_molar_flow"])
            self.assertEqual(basis["R_units"], "mL bar / (mmol K)")
            example = basis["worked_example"]
            self.assertEqual(example["condition_id"], "HI01")
            values, calculated = example["inputs"], example["outputs"]
            substrate = values["C_mol_L"] * values["Q_L_mL_min"]
            gas = values["P_ref_bar_abs"] * values["Q_ref_mL_min"] / (basis["R_value"] * values["T_ref_K"])
            self.assertAlmostEqual(calculated["substrate_mmol_min"], substrate)
            self.assertAlmostEqual(calculated["total_gas_mmol_min"], gas)
            self.assertAlmostEqual(calculated["oxygen_equivalents_supplied"], values["x_O2"] * gas / substrate)
            self.assertAlmostEqual(calculated["liquid_only_space_time_min"], values["V_mL"] / values["Q_L_mL_min"])
            self.assertIn("n_substrate = C * Q_L", document)
            self.assertIn("calculation_basis.json", manifest["files"])

    def test_hash_mismatch_incomplete_grid_and_overwrite_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first, second = synthetic_freezes(base / "campaign")
            bad = deepcopy(first)
            bad["snapshot"]["configuration"]["changed"] = True
            (base / "campaign/input_freeze.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                export_campaign_report(ROOT, base / "campaign", base / "review")
            self.assertFalse((base / "review").exists())
            (base / "campaign/input_freeze.json").write_text(json.dumps(first))
            bad = deepcopy(second)
            bad["arms"]["history_informed"]["Chief"]["recommended_core_candidate_ids"] = ["CP01"] * 6
            (base / "campaign/nominations_freeze.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "3x2 grid"):
                export_campaign_report(ROOT, base / "campaign", base / "review")
            (base / "campaign/nominations_freeze.json").write_text(json.dumps(second))
            export_campaign_report(ROOT, base / "campaign", base / "review")
            with self.assertRaises(FileExistsError):
                export_campaign_report(ROOT, base / "campaign", base / "review")

    @unittest.skipUnless(importlib.util.find_spec("reportlab"), "reportlab optional")
    def test_pdf_exports_from_same_authoritative_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            synthetic_freezes(base / "campaign", converge=True)
            result = export_campaign_report(ROOT, base / "campaign", base / "review", pdf=True)
            pdf = Path(result["pdf_path"])
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            self.assertGreater(pdf.stat().st_size, 5000)
            manifest = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(manifest["files"]["campaign_report.pdf"]["sha256"], file_hash(pdf))
            self.assertEqual(manifest["pdf_visual_QA"], "required_before_delivery")


if __name__ == "__main__":
    unittest.main()
