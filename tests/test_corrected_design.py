"""Offline engineering examples; no fixture is a laboratory confirmation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from khu_sulfoxidation.corrected_design import build_corrected_design, calculate_coupled_values


def fixture_config():
    """Explicit engineering-test assumptions using catalog-supported choices."""
    return {"design_id": "TEST_CONDITIONAL_NOT_LAB_CONFIRMED", "version": 1,
            "selected_equipment": {"pump_id": "pump_r_series_hplc", "mfc_id": "mfc_ffc00",
                "coil_id": "coil_row4", "light_id": "light_uv150_row10", "mixer_id": "p-713",
                "check_valve_id": "cv-3301", "bpr_id": "bpr_row4"},
            "target_mixture_transit_min": [2., 5., 10.], "target_inlet_oxygen_equivalents": [.5, 1.],
            "repeat_condition_id": "KD04", "chemistry": {"substrate": "thioanisole", "solvent": "EtOH",
                "substrate_concentration_M": .25, "catalyst": "acridine", "catalyst_loading_mol_percent": 3.},
            "operating": {"temperature_C": 25.},
            "assumptions": {"gas_identity": "air", "oxygen_mole_fraction": .2095,
                "gas_reference_temperature_K": 273.15, "gas_reference_pressure_bar_abs": 1.01325,
                "bpr_label_basis": "gauge", "ambient_pressure_bar_abs": 1.01325,
                "pressure_drop_model": "uniform_BPR_setpoint_approximation"},
            "tolerances": {"time_relative": .03, "oxygen_relative": .03},
            "max_grid_pair_evaluations": 100000,
            "rationale": {key: "Explicit engineering test assumption, not experimental evidence"
                          for key in ["time_window", "coil_choice", "bpr_choice", "oxygen_levels"]}}


class CorrectedDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = fixture_config()
        cls.design = build_corrected_design(ROOT, cls.config)

    def test_six_joint_targets_after_quantization_and_independent_repeat(self):
        design = self.design
        self.assertEqual(design["status"], "conditional_proposals_ready")
        self.assertEqual([c["condition_id"] for c in design["conditions"]], [f"KD{i:02d}" for i in range(1, 7)])
        self.assertEqual(design["independent_repeat"]["repeat_of"], "KD04")
        self.assertEqual(design["independent_repeat"]["condition_id"], "KD-R01")
        for row in design["conditions"]:
            c = row["calculated"]
            self.assertLessEqual(row["realized_time_relative_error"], .03)
            self.assertLessEqual(row["realized_oxygen_relative_error"], .03)
            self.assertAlmostEqual(c["no_slip_mixture_transit_min"], 5 / (c["liquid_flow_mL_min"] + c["gas_reactor_flow_mL_min_estimate"]))
            self.assertEqual(c, calculate_coupled_values(design["scenario"], c["liquid_flow_mL_min"], c["gas_reference_flow_mL_min"]))
            for key in ["liquid_flow_mL_min", "gas_reference_flow_mL_min"]:
                self.assertAlmostEqual(c[key] * 100, round(c[key] * 100))
            self.assertTrue(all(v is None for v in row["actual"].values()))
        for low, high in zip(design["conditions"][::2], design["conditions"][1::2]):
            self.assertEqual(low["target"]["nominal_mixture_transit_min"], high["target"]["nominal_mixture_transit_min"])
            self.assertLess(high["calculated"]["liquid_flow_mL_min"], low["calculated"]["liquid_flow_mL_min"])
        self.assertFalse(design["laboratory_released"])
        self.assertFalse(design["experimental_validation"])

    def test_inlet_equivalents_independent_of_reactor_pressure(self):
        scenario = deepcopy(self.design["scenario"])
        first = calculate_coupled_values(scenario, .36, 4.84)
        scenario["pressure"]["reactor_pressure_bar_abs_assumed"] *= 2
        second = calculate_coupled_values(scenario, .36, 4.84)
        for key in ["total_gas_mmol_min", "oxygen_mmol_min_supplied", "oxygen_equivalents_supplied",
                    "oxygen_reference_volume_flow_mL_min"]:
            self.assertEqual(first[key], second[key])
        self.assertAlmostEqual(second["gas_reactor_flow_mL_min_estimate"], first["gas_reactor_flow_mL_min_estimate"] / 2)
        self.assertGreater(second["no_slip_mixture_transit_min"], first["no_slip_mixture_transit_min"])

    def test_gauge_vs_absolute_assumptions_change_transit_not_fixed_inlet_equivalents(self):
        config = fixture_config()
        config["assumptions"]["bpr_label_basis"] = "absolute"
        absolute = build_corrected_design(ROOT, config)
        self.assertEqual(absolute["scenario"]["pressure"]["reactor_pressure_bar_abs_assumed"], 1.5)
        gauge = self.design["scenario"]
        self.assertAlmostEqual(gauge["pressure"]["reactor_pressure_bar_abs_assumed"], 2.51325)
        a = calculate_coupled_values(absolute["scenario"], .36, 4.84)
        g = calculate_coupled_values(gauge, .36, 4.84)
        self.assertEqual(a["oxygen_equivalents_supplied"], g["oxygen_equivalents_supplied"])
        self.assertLess(a["no_slip_mixture_transit_min"], g["no_slip_mixture_transit_min"])
        self.assertIsNone(gauge["pressure"]["label_basis_source"])
        self.assertNotEqual(self.design["configuration_sha256"], absolute["configuration_sha256"])

    def test_reference_conversion_and_component_volume_are_independent(self):
        s = deepcopy(self.design["scenario"])
        c = calculate_coupled_values(s, .36, 4.84)
        self.assertAlmostEqual(c["total_gas_mmol_min"], 1.01325 * 4.84 / (.0831446261815324 * 273.15))
        self.assertAlmostEqual(c["oxygen_reference_volume_flow_mL_min"], 4.84 * .2095)
        self.assertAlmostEqual(c["oxygen_equivalents_supplied"], c["total_gas_mmol_min"] * .2095 / (.25 * .36))
        s["gas_reference"]["temperature_K"] *= 2
        changed = calculate_coupled_values(s, .36, 4.84)
        self.assertAlmostEqual(changed["total_gas_mmol_min"], c["total_gas_mmol_min"] / 2)
        self.assertAlmostEqual(changed["oxygen_equivalents_supplied"], c["oxygen_equivalents_supplied"] / 2)

    def test_feasible_interval_search_matches_independent_full_grid(self):
        # Probe one target across the entire 0.01..10 x 0.01..10 catalog grid.
        row = self.design["conditions"][0]
        pressure, temp, pref, tref, x, concentration, volume = 2.51325, 298.15, 1.01325, 273.15, .2095, .25, 5.
        best = None
        for i in range(1, 1001):
            ql = i / 100
            for j in range(1, 1001):
                qg = j / 100
                tau = volume / (ql + qg * pref / pressure * temp / tref)
                eq = x * pref * qg / (.0831446261815324 * tref) / (concentration * ql)
                terr, eerr = abs(tau / 2 - 1), abs(eq / .5 - 1)
                if terr <= .03 + 1e-12 and eerr <= .03 + 1e-12:
                    score = (max(terr, eerr), terr + eerr, ql, qg)
                    if best is None or score < best:
                        best = score
        self.assertEqual((row["calculated"]["liquid_flow_mL_min"], row["calculated"]["gas_reference_flow_mL_min"]), best[2:])
        self.assertEqual(row["required_pair_evaluations"], sum(t["tested_pair_count"] for t in row["quantization_search"]))

    def test_alternatives_are_retained_with_specific_rejections(self):
        alternatives = {a["equipment_id"]: a for a in self.design["alternative_evaluations"]}
        two = [c for c in alternatives["coil_row3"]["conditions"] if c["status"] == "rejected"]
        ten = [c for c in alternatives["coil_row5"]["conditions"] if c["status"] == "rejected"]
        self.assertEqual([c["target"] for c in two], [{"nominal_mixture_transit_min": 10., "inlet_oxygen_equivalents": 1.}])
        self.assertTrue(any("grid" in reason for reason in two[0]["rejection_reasons"]))
        self.assertEqual(ten[0]["target"], {"nominal_mixture_transit_min": 2., "inlet_oxygen_equivalents": 1.})
        self.assertTrue(any("mfc" in reason for reason in ten[0]["rejection_reasons"]))
        review = self.design["inventory_review"]["categories"]
        self.assertIn("coil_row8", {c["equipment_id"] for c in review["reactors"]})
        self.assertIn("bpr_row8", {c["equipment_id"] for c in review["pressure_controllers"]})
        self.assertEqual(len(review["pumps"]), 15)

    def test_invalid_assumptions_hardware_and_budget_fail_explicitly(self):
        for path, value in [(('assumptions', 'bpr_label_basis'), 'unknown'),
                            (('assumptions', 'gas_reference_temperature_K'), None),
                            (('assumptions', 'oxygen_mole_fraction'), 1.),
                            (('operating', 'temperature_C'), 24.),
                            (('selected_equipment', 'coil_id'), 'coil_row7'),
                            (('selected_equipment', 'bpr_id'), 'bpr_row8')]:
            config = fixture_config()
            config[path[0]][path[1]] = value
            with self.subTest(path=path), self.assertRaises((ValueError, TypeError)):
                build_corrected_design(ROOT, config)
        config = fixture_config()
        config["max_grid_pair_evaluations"] = 1
        design = build_corrected_design(ROOT, config)
        self.assertEqual(design["status"], "infeasible_requested_design")
        self.assertTrue(any("budget" in r for r in design["conditions"][0]["rejection_reasons"]))
        self.assertIsNone(design["conditions"][0]["calculated"])
        self.assertEqual(sum(t["tested_pair_count"] for t in design["conditions"][0]["quantization_search"]), 0)
        for bad in [float('nan'), float('inf'), True, 0.]:
            config = fixture_config()
            config["target_mixture_transit_min"][0] = bad
            with self.assertRaises((ValueError, TypeError)):
                build_corrected_design(ROOT, config)

    def test_deterministic_provenance_and_input_immutability(self):
        config = fixture_config()
        before = deepcopy(config)
        result = build_corrected_design(ROOT, config)
        self.assertEqual(config, before)
        self.assertEqual(result, self.design)
        self.assertIn("KHU_inventory_complete_20260915.xlsx", result["source_hashes"])
        self.assertIsNone(result["comparison_nominations"])
        self.assertFalse(result["history_read_by_generator"])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
