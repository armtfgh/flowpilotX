"""Checks for conditional planning, never validation of laboratory performance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from khu_sulfoxidation.conditional_planning import build_conditional_pool


class ConditionalPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = build_conditional_pool(ROOT)

    def test_common_pool_retains_infeasible_targets_and_never_clamps(self):
        pool = self.pool
        self.assertEqual(pool["summary"], {"requested_candidates": 28, "eligible_conditional_candidates": 22,
            "rejected_candidates": 6, "complete_three_by_two_groups": 52})
        self.assertEqual(pool["generation_policy"]["liquid_flow_coordinates_mL_min"], [.74, .36, .18, .09, .04, .02, .01])
        eligible = [c for c in pool["candidates"] if c["status"] == "eligible_under_declared_assumptions"]
        for c in eligible:
            settings = c["proposed_setpoints"]
            self.assertTrue(.01 <= settings["liquid_flow_mL_min"] <= 10.)
            self.assertTrue(.01 <= settings["gas_reference_flow_mL_min"] <= 10.)
            for field in ["liquid_flow_mL_min", "gas_reference_flow_mL_min"]:
                self.assertAlmostEqual(settings[field] / .01, round(settings[field] / .01))
            self.assertLessEqual(c["calculated"]["oxygen_relative_quantization_error"], .03)
        rejected = [c for c in pool["candidates"] if c["status"] == "rejected"]
        self.assertTrue(all(c["proposed_setpoints"]["gas_reference_flow_mL_min"] > 10 for c in rejected))
        self.assertTrue(all(c["rejection_reasons"] for c in rejected))

    def test_oxygen_delivery_and_feed_rates_recomputed_after_quantization(self):
        c = self.pool["candidates"][0]
        self.assertEqual(c["proposed_setpoints"]["gas_reference_flow_mL_min"], 9.9)
        self.assertNotEqual(c["ideal_unquantized_gas_reference_flow_mL_min"], 9.9)
        values = c["calculated"]
        # Independent arithmetic with the supplied reference conditions.
        gas_mmol = 1.01325 * 9.9 / (.0831446261815324 * 273.15)
        self.assertAlmostEqual(values["total_gas_mmol_min"], gas_mmol)
        self.assertAlmostEqual(values["oxygen_mmol_min_supplied"], gas_mmol * .2095)
        self.assertAlmostEqual(values["substrate_mmol_min"], .74 * .25)
        self.assertAlmostEqual(values["oxygen_equivalents_supplied"], gas_mmol * .2095 / (.74 * .25))
        self.assertAlmostEqual(values["catalyst_mmol_min"] / values["substrate_mmol_min"], .03)
        self.assertAlmostEqual(values["feed_mL_per_elapsed_hour_at_continuous_setting"], 44.4)
        self.assertEqual(self.pool["scenario"]["feed_calculations"]["catalyst_concentration_mM"], 7.5)

    def test_space_time_parametric_transit_and_actual_remain_distinct(self):
        scenario = self.pool["scenario"]
        self.assertIsNone(scenario["bpr_assignment"])
        self.assertIsNone(scenario["reactor_pressure_bar_abs"])
        self.assertIsNone(scenario["pressure_basis_confirmed"])
        self.assertTrue(all(v["actual_assignment"] is None for v in scenario["hardware_choices"].values()))
        self.assertEqual(scenario["proposed_temperature_C"], 25.)
        for candidate in self.pool["candidates"]:
            values = candidate["calculated"]
            ql = candidate["proposed_setpoints"]["liquid_flow_mL_min"]
            self.assertAlmostEqual(values["liquid_only_space_time_min"], 10 / ql)
            self.assertIsNone(values["actual_liquid_contact_time_min"])
            illustrations = values["mixture_transit"]["illustrations"]
            self.assertEqual([i["pressure_bar_abs"] for i in illustrations], [2., 4.])
            self.assertAlmostEqual(illustrations[0]["gas_reactor_flow_mL_min_estimate"],
                                   2 * illustrations[1]["gas_reactor_flow_mL_min_estimate"])
            for illustration in illustrations:
                self.assertIn("not_selected_BPR", illustration["pressure_status"])
                self.assertAlmostEqual(illustration["no_slip_mixture_transit_min_estimate"],
                    10 / (ql + illustration["gas_reactor_flow_mL_min_estimate"]))
                self.assertLess(illustration["no_slip_mixture_transit_min_estimate"], values["liquid_only_space_time_min"])
        self.assertEqual(self.pool["generation_policy"]["liquid_only_space_time_coordinates_min"][-1], 1000.)
        self.assertFalse(self.pool["experimental_validation"])

    def test_every_group_is_complete_three_by_two_with_same_hardware(self):
        by_id = {c["candidate_id"]: c for c in self.pool["candidates"]}
        for group in self.pool["grid_groups"]:
            candidates = [by_id[i] for i in group["candidate_ids"]]
            self.assertEqual(len(candidates), 6)
            self.assertEqual(len({c["candidate_id"] for c in candidates}), 6)
            self.assertEqual(len({c["time_coordinate"] for c in candidates}), 3)
            self.assertEqual(len({c["oxygen_coordinate"] for c in candidates}), 2)
            self.assertEqual(len({(c["time_coordinate"], c["oxygen_coordinate"]) for c in candidates}), 6)
            self.assertEqual({c["scenario_id"] for c in candidates}, {group["scenario_id"]})
            self.assertTrue(all(c["status"] == "eligible_under_declared_assumptions" for c in candidates))
            self.assertEqual(sorted(group["time_coordinate_values"]), sorted({c["calculated"]["liquid_only_space_time_min"] for c in candidates}))
            self.assertEqual(sorted(group["oxygen_coordinate_values"]), sorted({c["nominal_oxygen_equivalents"] for c in candidates}))

    def test_no_history_needed_and_changed_history_cannot_change_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in self.pool["sources"]:
                dest = root / source["path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / source["path"], dest)
                self.assertEqual(hashlib.sha256(dest.read_bytes()).hexdigest(), source["sha256"])
            # No historical rows or derived history exist in this test workspace.
            baseline = build_conditional_pool(root)
            self.assertEqual(baseline, self.pool)
            history = root / "data/derived/homogeneous_history.json"
            history.parent.mkdir(parents=True, exist_ok=True)
            history.write_text('{"yield": 1, "batch_time_h": 1}')
            first = build_conditional_pool(root)
            history.write_text('{"yield": 99, "batch_time_h": 10000}')
            second = build_conditional_pool(root)
            self.assertEqual(first, second)
            self.assertEqual(first["configuration_sha256"], self.pool["configuration_sha256"])

    def test_changed_inventory_cannot_silently_substitute_another_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in self.pool["sources"]:
                dest = root / source["path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / source["path"], dest)
            path = root / "KHU_inventory_complete_20260918_v7.json"
            original = json.loads(path.read_text())
            changes = [("reactors", "coil_row5", "volume_mL", 20.),
                       ("gas_hardware", "mfc_ffc00", "gas", "O2"),
                       ("gas_hardware", "mfc_ffc00", "max_flow_sccm", .1)]
            for category, equipment_id, field, value in changes:
                changed = deepcopy(original)
                item = next(x for x in changed["lab_inventory"][category] if x["equipment_id"] == equipment_id)
                item[field] = value
                path.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    build_conditional_pool(root)


if __name__ == "__main__":
    unittest.main()
