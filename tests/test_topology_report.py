"""Offline fixtures for six-figure presentation; no model/API calls."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_campaign_report import synthetic_freezes
from khu_sulfoxidation.campaign_report import export_campaign_report
from khu_sulfoxidation.topology_report import export_topology_report


def fixture(base, converge=False):
    synthetic_freezes(base / "campaign", converge=converge)
    export_campaign_report(ROOT, base / "campaign", base / "source", pdf=False)
    data = json.loads((base / "source/authoritative_conditions.json").read_text())
    output = base / "figures"
    (output / "diagrams").mkdir(parents=True)
    cases = []
    for row in data["conditions"][:6]:
        stem = f"{row['condition_id']}_{row['candidate_id']}"
        case = {"condition_id": row["condition_id"], "candidate_id": row["candidate_id"],
            "title": f"{stem} <synthetic fixture>", "png": f"diagrams/{stem}.png", "svg": f"diagrams/{stem}.svg", "topology_json": f"diagrams/{stem}.json"}
        # No raster decoding is needed for focused non-PDF tests.
        (output / case["png"]).write_bytes(b"SYNTHETIC_IMAGE_PLACEHOLDER")
        (output / case["svg"]).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        (output / case["topology_json"]).write_text(json.dumps({"test_only": True, "condition_id": row["condition_id"]}))
        cases.append(case)
    index = {"cases": cases, "source_files_sha256": {}, "renderer_reuse": "synthetic test only"}
    (output / "render_index.json").write_text(json.dumps(index))
    return data, index


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class TopologyReportTests(unittest.TestCase):
    def test_six_core_rows_preserve_exact_authoritative_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, _ = fixture(base)
            result = export_topology_report(base / "source", base / "figures", pdf=False)
            self.assertEqual(result["core_condition_count"], 6)
            self.assertEqual(result["figure_count"], 6)
            rows = csv_rows(base / "figures/core_conditions_summary.csv")
            self.assertEqual(len(rows), 6)
            for actual, source in zip(rows, data["conditions"][:6]):
                self.assertEqual(actual["condition_id"], source["condition_id"])
                self.assertEqual(actual["candidate_id"], source["candidate_id"])
                self.assertEqual(float(actual["liquid_flow_mL_min"]), source["proposed_setpoints"]["liquid_flow_mL_min"])
                self.assertEqual(float(actual["reference_air_flow_mL_min"]), source["proposed_setpoints"]["gas_reference_flow_mL_min"])
                self.assertEqual(float(actual["calculated_supplied_oxygen_equivalents"]), source["calculated"]["oxygen_equivalents_supplied"])
                self.assertEqual(float(actual["liquid_only_space_time_min"]), source["calculated"]["liquid_only_space_time_min"])

    def test_six_figure_links_titles_and_rationales_match_their_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, index = fixture(base)
            export_topology_report(base / "source", base / "figures", pdf=False)
            html = (base / "figures/summary.html").read_text()
            self.assertEqual(html.count('<img class="figure"'), 6)
            self.assertIn("&lt;synthetic fixture&gt;", html)
            for case, row in zip(index["cases"], data["conditions"]):
                for key in ("svg", "png", "topology_json"):
                    self.assertIn(case[key], html)
                self.assertIn(row["review"]["rationale"], html)
            self.assertIn("one proposed apparatus topology", html)
            self.assertIn("not measured liquid contact time", html)

    def test_mismatched_missing_and_unsafe_render_entries_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, index = fixture(base)
            path = base / "figures/render_index.json"
            original = json.loads(json.dumps(index))
            index["cases"][0]["candidate_id"] = "WRONG"
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "pair differs"):
                export_topology_report(base / "source", base / "figures", pdf=False)
            index = json.loads(json.dumps(original))
            index["cases"][0]["svg"] = "../../escape.svg"
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                export_topology_report(base / "source", base / "figures", pdf=False)
            index = json.loads(json.dumps(original))
            index["cases"].pop()
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "exactly one figure"):
                export_topology_report(base / "source", base / "figures", pdf=False)
            self.assertFalse((base / "figures/summary.html").exists())

    def test_convergent_comparison_repeat_unknowns_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, _ = fixture(base, converge=True)
            original = (base / "source/authoritative_conditions.json").read_bytes()
            result = export_topology_report(base / "source", base / "figures", pdf=False)
            self.assertEqual(result["planned_run_count_including_supplementary"], 7)
            supplementary = csv_rows(base / "figures/supplementary_runs.csv")
            self.assertEqual(len(supplementary), 1)
            self.assertEqual(supplementary[0]["repeat_of_condition_id"], data["conditions"][6]["repeat_of_condition_id"])
            comparison = csv_rows(base / "figures/nomination_comparison.csv")
            self.assertEqual(len(comparison), 4)
            self.assertEqual([r["measurement_condition_id"] for r in comparison[:2]], [r["measurement_condition_id"] for r in comparison[2:]])
            shared = {r["setting"]: r for r in csv_rows(base / "figures/shared_settings.csv")}
            self.assertEqual(shared["Pressure / BPR"]["value"], "Unassigned")
            self.assertIn("unknown", shared["Pressure / BPR"]["basis_or_status"])
            self.assertEqual((base / "source/authoritative_conditions.json").read_bytes(), original)
            with self.assertRaises(FileExistsError):
                export_topology_report(base / "source", base / "figures", pdf=False)

    def test_changed_recipe_or_nonblank_actuals_requires_new_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, _ = fixture(base)
            path = base / "source/authoritative_conditions.json"
            data["scenario"]["shared_recipe"]["substrate_concentration_M"] = 1.0
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Changed chemistry"):
                export_topology_report(base / "source", base / "figures", pdf=False)
            data["scenario"]["shared_recipe"]["substrate_concentration_M"] = .25
            data["conditions"][0]["actual_outcomes"] = {"sulfoxide_pct": 90}
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "unknown actual"):
                export_topology_report(base / "source", base / "figures", pdf=False)
            self.assertFalse((base / "figures/summary.html").exists())


if __name__ == "__main__":
    unittest.main()
