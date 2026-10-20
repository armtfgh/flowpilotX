"""SYNTHETIC apparatus fixtures exercise arithmetic, never KHU recommendations."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from khu_sulfoxidation.candidate_space import calculate_setpoints, solve_candidate
from khu_sulfoxidation.inventory import absolute_pressure, reconcile_inventory, validate_apparatus
from khu_sulfoxidation.screen_selector import generate_screen


def synthetic_apparatus():
    """Invented equipment envelope for tests only; no actual inventory IDs."""
    def confirmed(**values):
        return {"status": "confirmed", "confirmation_source": "SYNTHETIC_TEST_FIXTURE_ONLY", **values}
    return {
        "profile_id": "SYNTHETIC_TEST_NOT_A_LAB_RECOMMENDATION", "status": "confirmed_for_engineering",
        "confirmation_source": "SYNTHETIC_TEST_FIXTURE_ONLY",
        "scope": "homogeneous_single_premixed_liquid_feed",
        "pump": confirmed(id="synthetic_pump", available=True, min_flow_mL_min=.001,
            max_flow_mL_min=10., increment_mL_min=.001, grid_origin_mL_min=0.,
            max_pressure_bar_abs=20., max_pressure_exclusive=True),
        "mfc": confirmed(id="synthetic_mfc", available=True, min_flow_mL_min=.001,
            max_flow_mL_min=100., increment_mL_min=.001, grid_origin_mL_min=0.,
            display_unit="mL/min_at_reference", calibration_id="synthetic_air_calibration",
            calibration_gas_identity="air", reference_temperature_K=273.15,
            reference_pressure_bar_abs=1.01325, max_outlet_pressure_bar_abs=10.,
            max_inlet_pressure_bar_abs=12., min_pressure_drop_bar=.5),
        "gas": confirmed(identity="air", o2_mole_fraction=.21, supply_pressure_bar_abs=8.),
        "reactor": confirmed(id="synthetic_coil", available=True, material="synthetic_PFA",
            volume_mL=10., volume_status="nominal", ID_mm=1., illuminated_length_m=12.,
            illuminated_length_status="nominal", compatible_light_ids=["synthetic_light"],
            max_pressure_bar_abs=10., max_pressure_exclusive=False,
            min_temperature_C=10., max_temperature_C=80., geometry_relative_tolerance=.01),
        "light": confirmed(id="synthetic_light", available=True, wavelength_nm=450.,
            min_temperature_C=20., max_temperature_C=80., max_reactor_volume_mL=10.,
            power_setting="synthetic_fixed_setting"),
        "pressure": confirmed(bpr_id="synthetic_bpr", available_bpr_ids=["synthetic_bpr"],
            available_bpr_settings=[{"bpr_id": "synthetic_bpr", "value": 2., "unit": "bar", "basis": "gauge"}],
            setpoint_value=2., unit="bar", basis="gauge", ambient_bar_abs=1.,
            reactor_pressure_basis="setpoint_approximation", measured_reactor_pressure_bar_abs=None,
            approved_max_operating_pressure_bar_abs=6., pressure_drop_allowance_bar=.1,
            sensor_id="synthetic_sensor", sensor_location="coil_inlet", sensor_max_pressure_bar_abs=10.),
        "connections": confirmed(mixer_id="synthetic_mixer", check_valve_id="synthetic_check",
            check_valve_cracking_pressure_bar=1., minimum_component_rating_bar_abs=10.,
            minimum_gas_inlet_component_rating_bar_abs=12., minimum_gas_outlet_component_rating_bar_abs=10.,
            regulator_id="synthetic_regulator", collection_arrangement="synthetic_collection"),
        "operating": confirmed(temperature_C=25., feed_mode="continuous", sample_loop_volume_mL=None)}


def synthetic_chemistry():
    return {"substrate_concentration_M": .25, "catalyst_loading_mol_percent": 3., "additional_liquid_streams": []}


def synthetic_target(**changes):
    return {"candidate_id": "SYNTHETIC_C01", "time_min": 2.,
        "time_basis": "no_slip_mixture_transit_min", "oxygen_equivalents": 1.,
        "time_relative_tolerance": .02, "oxygen_relative_tolerance": .02,
        "max_grid_evaluations": 10000, **changes}


def synthetic_screen(**changes):
    return {"screen_id": "SYNTHETIC_SCREEN", "version": 1,
        "design": "three_times_two_oxygen_levels", "times_min": [1., 2., 4.],
        "oxygen_equivalents": [1., 2.], "time_basis": "no_slip_mixture_transit_min",
        "time_relative_tolerance": .02, "oxygen_relative_tolerance": .02,
        "max_grid_evaluations": 10000, "repeat_candidate_id": "C03",
        "comparison_arm": "synthetic_test", **changes}


class EngineeringTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.chemistry = synthetic_apparatus(), synthetic_chemistry()

    def test_actual_evidence_does_not_make_assignments(self):
        p = reconcile_inventory(ROOT)
        self.assertIsNone(p["pump"]["id"])
        self.assertIsNone(p["pressure"]["setpoint_value"])
        self.assertIsNone(p["reactor"]["volume_mL"])
        self.assertTrue(validate_apparatus(p))
        self.assertAlmostEqual(p["new_slide_evidence"]["slide_5"]["bpr_nominal_bar_conversions"][2], 6.894757293168)
        if (ROOT / "khu_inventory.json").exists():
            self.assertEqual(p["planning_preferences"]["pump_id"], "KHU-PUMP-VAPOURTEC-R-SERIES")
            self.assertEqual(p["new_user_inventory_evidence"], json.loads((ROOT / "khu_inventory.json").read_text()))
            self.assertIn("new_inventory_temperature", {i["id"] for i in p["reconciliation_issues"]})

    def test_explicit_pressure_basis(self):
        self.assertAlmostEqual(absolute_pressure(20, "psi", "gauge", 1.), 2.3789514586336)
        self.assertAlmostEqual(absolute_pressure(20, "psi", "absolute"), 1.3789514586336)
        for args in [(20, "psi", "gauge"), (7, "bar", "unknown"), (float("nan"), "bar", "absolute")]:
            with self.assertRaises(ValueError):
                absolute_pressure(*args)

    def test_three_distinct_time_indices_and_oxygen_fraction(self):
        self.assertEqual(validate_apparatus(self.profile), [])
        d = calculate_setpoints(self.profile, self.chemistry, .1, 1.)
        self.assertAlmostEqual(d["substrate_mmol_min"], .025)
        self.assertAlmostEqual(d["catalyst_mmol_min"], .00075)
        self.assertAlmostEqual(d["catalyst_concentration_mM"], 7.5)
        self.assertAlmostEqual(d["liquid_only_space_time_min"], 100.)
        self.assertAlmostEqual(d["inlet_reference_volume_index_min"], 10 / 1.1)
        self.assertAlmostEqual(d["gas_reactor_flow_mL_min_estimate"], 1.01325 / 3 * 298.15 / 273.15)
        self.assertGreater(d["liquid_only_space_time_min"], d["no_slip_mixture_transit_min"])
        self.assertGreater(d["no_slip_mixture_transit_min"], d["inlet_reference_volume_index_min"])
        self.assertIsNone(d["actual_liquid_contact_time_min"])
        self.assertAlmostEqual(d["coil_length_from_volume_and_ID_m"], 40 / math.pi)
        pure = deepcopy(self.profile)
        pure["gas"].update(identity="O2", o2_mole_fraction=1.)
        pure["mfc"]["calibration_gas_identity"] = "O2"
        oxygen = calculate_setpoints(pure, self.chemistry, .1, 1.)
        self.assertAlmostEqual(d["oxygen_equivalents_supplied"] / oxygen["oxygen_equivalents_supplied"], .21)

    def test_reference_state_is_used_not_assumed_stp(self):
        original = calculate_setpoints(self.profile, self.chemistry, .1, 1.)
        self.profile["mfc"]["reference_temperature_K"] *= 2
        changed = calculate_setpoints(self.profile, self.chemistry, .1, 1.)
        self.assertAlmostEqual(changed["oxygen_mmol_min_supplied"], original["oxygen_mmol_min_supplied"] / 2)
        self.assertAlmostEqual(changed["gas_reactor_flow_mL_min_estimate"], original["gas_reactor_flow_mL_min_estimate"] / 2)

    def test_coupled_mixture_target_and_quantization(self):
        low = solve_candidate(self.profile, self.chemistry, synthetic_target())
        high = solve_candidate(self.profile, self.chemistry, synthetic_target(oxygen_equivalents=2.))
        for result in [low, high]:
            self.assertEqual(result["status"], "engineering_feasible_requires_review", result["rejection_reasons"])
            actual = result["calculated"]
            self.assertLessEqual(result["realized_time_relative_error"], .02)
            self.assertLessEqual(result["realized_oxygen_relative_error"], .02)
            recomputed = calculate_setpoints(self.profile, self.chemistry, actual["liquid_flow_mL_min"], actual["gas_reference_flow_mL_min"])
            self.assertEqual(actual, recomputed)
        self.assertLess(high["calculated"]["liquid_flow_mL_min"], low["calculated"]["liquid_flow_mL_min"])
        self.assertNotEqual(low["ideal_unquantized"]["liquid_flow_mL_min"], low["calculated"]["liquid_flow_mL_min"])

    def test_liquid_basis_holds_liquid_flow_while_gas_changes(self):
        values = [solve_candidate(self.profile, self.chemistry, synthetic_target(time_min=100.,
            time_basis="liquid_only_space_time_min", oxygen_equivalents=e)) for e in [1., 2.]]
        self.assertTrue(all(v["status"] == "engineering_feasible_requires_review" for v in values))
        self.assertAlmostEqual(values[0]["calculated"]["liquid_only_space_time_min"], 100.)
        self.assertNotEqual(values[0]["calculated"]["no_slip_mixture_transit_min"], values[1]["calculated"]["no_slip_mixture_transit_min"])

    def test_coarse_grid_and_usable_minimum_are_not_silently_clamped(self):
        self.profile["mfc"]["min_flow_mL_min"] = 50.
        result = solve_candidate(self.profile, self.chemistry, synthetic_target())
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(any("minimum" in r for r in result["rejection_reasons"]))
        coarse = synthetic_apparatus()
        coarse["pump"]["increment_mL_min"] = 1.
        result = solve_candidate(coarse, self.chemistry, synthetic_target(time_relative_tolerance=0., oxygen_relative_tolerance=0.))
        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(result["calculated"])

    def test_unknown_gas_unavailable_coil_and_bad_pressure_rejected(self):
        alterations = [
            ("gas", "identity", "N2"), ("gas", "o2_mole_fraction", 1.),
            ("mfc", "calibration_gas_identity", "O2"), ("mfc", "reference_temperature_K", None),
            ("reactor", "available", False), ("reactor", "volume_mL", 20.),
            ("reactor", "illuminated_length_m", 50.), ("reactor", "max_temperature_C", 24.),
            ("pressure", "setpoint_value", 7.), ("pressure", "sensor_max_pressure_bar_abs", 2.),
            ("connections", "minimum_gas_outlet_component_rating_bar_abs", 4.),
            ("connections", "minimum_gas_inlet_component_rating_bar_abs", 7.),
            ("gas", "supply_pressure_bar_abs", 3.), ("gas", "supply_pressure_bar_abs", 13.),
            ("operating", "temperature_C", 24.), ("pump", "max_pressure_bar_abs", 3.1)]
        # 24 C is permitted by this synthetic module, so explicitly raise its
        # minimum for this one test rather than imply KHU's unresolved minimum.
        for section, key, value in alterations:
            p = deepcopy(self.profile)
            p[section][key] = value
            if section == "operating":
                p["light"]["min_temperature_C"] = 25.
            with self.subTest(section=section, key=key):
                self.assertTrue(validate_apparatus(p))
                self.assertEqual(solve_candidate(p, self.chemistry, synthetic_target())["status"], "blocked")

    def test_mislabelled_pure_oxygen_and_inherited_bpr_blocked(self):
        self.profile["gas"]["o2_mole_fraction"] = .23
        self.assertTrue(validate_apparatus(self.profile))
        self.profile["gas"].update(identity="O2", o2_mole_fraction=.21)
        self.profile["mfc"]["calibration_gas_identity"] = "O2"
        self.assertIn("Pure O2", validate_apparatus(self.profile)[0])
        self.profile = synthetic_apparatus()
        self.profile["pressure"].update(setpoint_value=7.)
        self.assertIn("confirmed setting", validate_apparatus(self.profile)[0])

    def test_extreme_geometry_and_empty_controller_grid_fail_closed(self):
        for diameter in [1e-300, 1e300]:
            p = deepcopy(self.profile)
            p["reactor"]["ID_mm"] = diameter
            self.assertTrue(validate_apparatus(p))
            self.assertEqual(solve_candidate(p, self.chemistry, synthetic_target())["status"], "blocked")
        self.profile["pump"]["increment_mL_min"] = 1000.
        self.assertIn("no controller grid setting", validate_apparatus(self.profile)[0])

    def test_declared_grid_search_budget_fails_before_partial_selection(self):
        result = solve_candidate(self.profile, self.chemistry, synthetic_target(max_grid_evaluations=1))
        self.assertEqual(result["status"], "rejected")
        self.assertGreater(result["required_liquid_grid_points"], 1)
        self.assertEqual(result["maximum_gas_pairs_if_searched"], 2 * result["required_liquid_grid_points"])
        self.assertEqual(result["quantization_search"], [])
        self.assertIsNone(result["calculated"])
        self.assertTrue(any("budget exceeded" in reason for reason in result["rejection_reasons"]))

    def test_measured_pressure_reference_is_not_actual_for_proposed_run(self):
        self.profile["pressure"].update(reactor_pressure_basis="measured", measured_reactor_pressure_bar_abs=3.05)
        self.assertTrue(validate_apparatus(self.profile))
        self.profile["pressure"]["measured_pressure_observation_id"] = "SYNTHETIC_REFERENCE_MEASUREMENT"
        d = calculate_setpoints(self.profile, self.chemistry, .1, 1.)
        self.assertEqual(d["reactor_pressure_bar_abs"], 3.05)
        self.assertIsNone(d["actual_condition_pressure_bar_abs"])
        self.assertEqual(d["pressure_value_status"], "assumed_for_proposed_condition_from_declared_reference")

    def test_nonfinite_targets_and_unknown_basis_fail_closed(self):
        for key, value in [("time_min", float("nan")), ("oxygen_equivalents", float("inf")),
                           ("time_basis", "batch_hours"), ("time_relative_tolerance", -1),
                           ("time_min", True), ("max_grid_evaluations", 0)]:
            with self.subTest(key=key):
                result = solve_candidate(self.profile, self.chemistry, synthetic_target(**{key: value}))
                self.assertEqual(result["status"], "rejected")
                self.assertTrue(result["rejection_reasons"])
        with self.assertRaises(ValueError):
            calculate_setpoints(self.profile, self.chemistry, .1005, 1.)

    def test_six_unique_plus_independent_repeat(self):
        before = deepcopy(self.profile)
        result = generate_screen(self.profile, self.chemistry, synthetic_screen())
        self.assertEqual(result["status"], "engineering_feasible_requires_review")
        self.assertEqual(len(result["candidates"]), 6)
        self.assertEqual(len(result["selected_runs"]), 7)
        self.assertEqual(result["selected_runs"][-1]["repeat_of"], "C03")
        self.assertEqual(result["selected_runs"][-1]["preparation_type"], "independent_preparation_and_operation")
        self.assertEqual(result["selected_runs"][-1]["setpoints"], result["selected_runs"][2]["setpoints"])
        self.assertEqual(len({tuple(c["setpoints"].values()) for c in result["candidates"]}), 6)
        self.assertFalse(result["experimental_validation"])
        self.assertFalse(result["oxygen_sensitivity_tested"])
        self.assertEqual(self.profile, before)
        self.assertEqual(result["configuration_sha256"], generate_screen(self.profile, self.chemistry, synthetic_screen())["configuration_sha256"])
        self.assertNotEqual(result["configuration_sha256"], generate_screen(self.profile, self.chemistry, synthetic_screen(version=2))["configuration_sha256"])

    def test_blocked_actual_profile_preserves_seven_slots_and_all_rejections(self):
        p = reconcile_inventory(ROOT)
        result = generate_screen(p, self.chemistry, synthetic_screen(times_min=[None] * 3,
            oxygen_equivalents=[None] * 2, time_basis=None))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["candidates"]), 6)
        self.assertEqual(len(result["planned_slots"]), 7)
        self.assertEqual(len(result["rejection_reasons"]), 6)
        self.assertFalse(result["selected_runs"])

    def test_explicit_fallback_and_failed_grid_are_not_silently_reduced(self):
        result = generate_screen(self.profile, self.chemistry, synthetic_screen(
            design="one_oxygen_level_fallback", times_min=[1., 2., 3., 4.], oxygen_equivalents=[1.]))
        self.assertEqual(len(result["selected_runs"]), 5)
        self.assertFalse(result["oxygen_sensitivity_planned"])
        result = generate_screen(self.profile, self.chemistry, synthetic_screen(times_min=[1., 2., 1e6]))
        self.assertEqual(result["status"], "infeasible_requested_grid")
        self.assertEqual(len(result["candidates"]), 6)
        self.assertEqual(result["selected_runs"], [])
        self.assertTrue(result["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
