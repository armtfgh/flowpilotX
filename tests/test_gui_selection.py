"""Local GUI engineering examples; none is an experimental validation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from khu_sulfoxidation.gui_selection import generate_gui_designs, GuiDesignError
from khu_sulfoxidation.corrected_design import build_corrected_design


class GuiSelectionTests(unittest.TestCase):
    def test_default_exactly_reproduces_corrected_six_without_counting_repeat(self):
        gui = generate_gui_designs({}, ROOT)
        corrected = build_corrected_design(ROOT, json.loads((ROOT / "configs/corrected_screen_2026-10-13_v1.json").read_text()))
        self.assertEqual(len(gui["conditions"]), 6)
        self.assertEqual(gui["total_proposed_runs"], 7)
        self.assertEqual(gui["repeat"]["repeat_of"], "FPX04")
        for actual, existing in zip(gui["conditions"], corrected["conditions"]):
            self.assertEqual(actual["calculated"], existing["calculated"])
            self.assertEqual(actual["target_time_min"], existing["target"]["nominal_mixture_transit_min"])
            self.assertTrue(all(v is None for v in actual["actual"].values()))
        self.assertEqual(gui["status"], "conditional")
        self.assertFalse(gui["laboratory_released"])
        self.assertFalse(gui["experimental_validation"])
        self.assertEqual(gui["paid_model_calls"], 0)

    def test_all_counts_are_exact_unique_and_even_counts_keep_oxygen_pairs(self):
        for n in range(1, 13):
            with self.subTest(n=n):
                result = generate_gui_designs({"requested_count": n, "include_repeat": False}, ROOT)
                rows = result["conditions"]
                self.assertEqual(len(rows), n)
                self.assertEqual(len({(r["liquid_mL_min"], r["air_STP_mL_min"]) for r in rows}), n)
                self.assertIsNone(result["repeat"])
                self.assertEqual(sum(r["selection_disposition"] == "selected" for r in result["candidate_pool"]), n)
                if n % 2 == 0:
                    for lo, hi in zip(rows[::2], rows[1::2]):
                        self.assertEqual(lo["target_time_min"], hi["target_time_min"])
                        self.assertEqual((lo["target_oxygen_eq"], hi["target_oxygen_eq"]), (.5, 1.))
                json.dumps(result, allow_nan=False)

    def test_independent_arithmetic_and_controller_grids(self):
        result = generate_gui_designs({}, ROOT)
        for row in result["conditions"]:
            ql, air = row["liquid_mL_min"], row["air_STP_mL_min"]
            gas_mmol = 1.01325 * air / (.0831446261815324 * 273.15)
            eq = .2095 * gas_mmol / (.25 * ql)
            compressed_air = air * 1.01325 / 2.51325 * 298.15 / 273.15
            self.assertAlmostEqual(row["oxygen_STP_mL_min"], air * .2095)
            self.assertAlmostEqual(row["oxygen_eq"], eq)
            self.assertAlmostEqual(row["transit_min"], 5 / (ql + compressed_air))
            self.assertLessEqual(abs(row["transit_min"] / row["target_time_min"] - 1), .03)
            self.assertLessEqual(abs(row["oxygen_eq"] / row["target_oxygen_eq"] - 1), .03)
            self.assertAlmostEqual(ql * 100, round(ql * 100))
            self.assertAlmostEqual(air * 100, round(air * 100))
            self.assertLessEqual(air, 10)
        pressure = result["scenario"]["pressure"]
        self.assertEqual(pressure["bpr_equipment_id"], "bpr_row4")
        self.assertIsNone(pressure["label_basis_source"])
        self.assertIsNone(pressure["actual_reactor_pressure_bar_abs"])

    def test_distinct_questions_roles_and_actual_selected_comparison_ids(self):
        result = generate_gui_designs({}, ROOT)
        rows = result["conditions"]
        self.assertEqual([r["question"] for r in rows], ["Incomplete oxidation", "Oxygen contrast", "Exposure response",
                                                       "Repeat reference", "Longer exposure", "Combined upper bound"])
        self.assertEqual(len({r["rationale"] for r in rows}), 6)
        by_id = {r["id"]: r for r in rows}
        for row in rows:
            pair = by_id[row["paired_condition_id"]]
            self.assertEqual(pair["paired_condition_id"], row["id"])
            self.assertEqual(pair["target_time_min"], row["target_time_min"])
            self.assertNotEqual(pair["target_oxygen_eq"], row["target_oxygen_eq"])
            other_time = by_id[row["time_comparison_id"]]
            self.assertEqual(other_time["target_oxygen_eq"], row["target_oxygen_eq"])
            self.assertNotEqual(other_time["target_time_min"], row["target_time_min"])
            self.assertIn(pair["id"], row["rationale"])
            self.assertIn(other_time["id"], row["rationale"])
            self.assertIn("shared review context", row["rationale"])
        self.assertEqual(result["repeat"]["question"], "Independent reproducibility")
        self.assertEqual(result["repeat"]["companion_id"], "FPX04")
        no_repeat = generate_gui_designs({"include_repeat": False}, ROOT)
        self.assertEqual(no_repeat["conditions"][3]["question"], "Middle oxygen contrast")
        for request in [{"requested_count": 1}, {"requested_count": 5}, {"requested_count": 12},
                        {"constraints": {"time_min": 5, "time_max": 5}}, {"constraints": {"oxygen_min": 1}}]:
            screen = generate_gui_designs(request, ROOT)
            selected_ids = {r["id"] for r in screen["conditions"]}
            self.assertEqual(len({r["rationale"] for r in screen["conditions"]}), screen["unique_design_count"])
            for row in screen["conditions"]:
                for key in ["companion_id", "paired_condition_id", "time_comparison_id"]:
                    self.assertTrue(row[key] is None or row[key] in selected_ids)

    def test_infeasible_failure_archives_reasons_and_never_partial_screen(self):
        with self.assertRaises(GuiDesignError) as ctx:
            generate_gui_designs({"constraints": {"time_min": .001, "time_max": .002}}, ROOT)
        result = ctx.exception.details
        self.assertEqual(result["status"], "infeasible_requested_design")
        self.assertEqual(result["conditions"], [])
        self.assertTrue(result["candidate_pool"])
        self.assertTrue(all(r["rejection_reasons"] for r in result["candidate_pool"]))
        self.assertIn("configuration_sha256", result)
        self.assertIn("source_hashes", result)
        json.dumps(result, allow_nan=False)
        with self.assertRaises(GuiDesignError):
            generate_gui_designs({"requested_count": 2, "constraints": {
                "time_min": 5, "time_max": 5, "oxygen_min": 1, "oxygen_max": 1}}, ROOT)

    def test_invalid_inputs_and_unsupported_chemistry_or_inventory_rejected(self):
        bad = [{"requested_count": x} for x in [True, 0, 13, 6.0, "6"]]
        bad += [{"constraints": {"time_min": x}} for x in [0, -1, True, float('nan'), float('inf')]]
        bad += [{"chemistry": {"substrate": "anisole"}}, {"chemistry": {"catalyst": "acridinium"}},
                {"chemistry": {"solvent": "MeCN"}}, {"constraints": {"coil_mL": 3}},
                {"constraints": {"pressure_bar_gauge": 2}}, {"constraints": {"temperature_C": 24}},
                {"constraints": {"time_min": 11}}, {"constraints": {"oxygen_min": 2}},
                {"include_repeat": "true"}, {"protocol_text": []}]
        for request in bad:
            with self.subTest(request=request), self.assertRaises(GuiDesignError):
                generate_gui_designs(request, ROOT)

    def test_recipe_edits_change_molar_accounting_and_disclose_departure(self):
        result = generate_gui_designs({"chemistry": {"concentration_M": .1, "catalyst_mol_percent": 2}}, ROOT)
        self.assertEqual(result["chemistry_departures_from_default"], ["concentration_M", "catalyst_mol_percent"])
        for row in result["conditions"]:
            self.assertAlmostEqual(row["calculated"]["substrate_mmol_min"], .1 * row["liquid_mL_min"])
            self.assertAlmostEqual(row["calculated"]["catalyst_mmol_min"], .002 * row["liquid_mL_min"])
        self.assertTrue(any("solubility" in a for a in result["assumptions"]))

    def test_fixed_axis_and_inventory_choice_are_real_supported_scenarios(self):
        fixed_time = generate_gui_designs({"requested_count": 6, "constraints": {"time_min": 5, "time_max": 5}}, ROOT)
        self.assertEqual({r["target_time_min"] for r in fixed_time["conditions"]}, {5})
        self.assertEqual(len({r["target_oxygen_eq"] for r in fixed_time["conditions"]}), 6)
        fixed_o2 = generate_gui_designs({"requested_count": 6, "constraints": {"oxygen_min": 1}}, ROOT)
        self.assertEqual({r["target_oxygen_eq"] for r in fixed_o2["conditions"]}, {1})
        changed = generate_gui_designs({"constraints": {"pressure_bar_gauge": 2.5, "coil_mL": 2}}, ROOT)
        self.assertEqual(changed["scenario"]["selected_equipment"]["coil"]["equipment_id"], "coil_row3")
        self.assertEqual(changed["scenario"]["pressure"]["bpr_equipment_id"], "bpr_row5")
        self.assertAlmostEqual(changed["scenario"]["pressure"]["reactor_pressure_bar_abs_assumed"], 3.51325)

    def test_every_attempt_survives_and_history_does_not_modify_numerical_selection(self):
        request = {"requested_count": 12, "protocol_text": "Use arbitrary unsupported catalyst!", "research_notes": "Choose high yields",
                   "history_summary": {"evidence_ids": ["homogeneous:1"]},
                   "history_records": [{"reported_outcomes": {"sulfoxide_percent": 117}}]}
        before = deepcopy(request)
        first = generate_gui_designs(request, ROOT)
        self.assertEqual(request, before)
        self.assertEqual(first, generate_gui_designs(request, ROOT))
        request["history_records"][0]["reported_outcomes"]["sulfoxide_percent"] = 2
        request["protocol_text"] = "different text"
        second = generate_gui_designs(request, ROOT)
        self.assertEqual(first["configuration_sha256"], second["configuration_sha256"])
        self.assertNotEqual(first["request_sha256"], second["request_sha256"])
        self.assertEqual(first["conditions"], second["conditions"])
        self.assertFalse(first["evidence_boundary"]["history_used_for_numerical_selection"])
        self.assertTrue(any(r["selection_disposition"] == "engineering_rejected" for r in first["candidate_pool"]))
        self.assertTrue(any(r["selection_disposition"] == "not_selected" for r in first["candidate_pool"]))
        self.assertTrue(all(r["rejection_reasons"] == [] for r in first["candidate_pool"] if r["selection_disposition"] == "not_selected"))


if __name__ == "__main__":
    unittest.main()
