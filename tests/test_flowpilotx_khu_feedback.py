"""Private adapter integration coverage; fixtures remain synthetic measurements."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from flowpilotx import CampaignStore
from flowpilotx.selection import feedback_summary


class KhuFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = CampaignStore(self.temp.name)

    def test_completed_khu_template_eligible_incomplete_and_incompatible_preserved_excluded(self):
        # Real adapter/template contract, entirely synthetic measurements. This
        # does not add results to any KHU campaign or validate laboratory values.
        from khu_sulfoxidation.flowpilotx_profile import build_profile
        from khu_sulfoxidation.flowpilotx_report import results_template
        spec = build_profile(Path(__file__).resolve().parents[1])
        spec["objective"]["mode"] = "optimize"
        c = self.store.create(spec)
        r = self.store.generate(c["id"], {"requested_count": 6,
            "selected_candidate_ids": spec["default_selected_candidate_ids"]})
        template = results_template(spec, spec["default_selected_candidate_ids"])
        actual = deepcopy(next(x for x in template["experiments"] if x["proposal_id"] == "KX04")["actual_conditions"])
        self.assertTrue(all(v is None for v in actual.values()))
        controls = next(x for x in spec["candidates"] if x["id"] == "KX04")["parameters"]
        actual.update(carrier_flow_mL_min=controls["liquid_mL_min"],
            oxygen_flow_reference_mL_min=controls["oxygen_STP_mL_min"],
            gas_reference_temperature_K=273.15, gas_reference_pressure_bar_abs=1.01325,
            reactor_pressure_bar_abs=2.47, temperature_C=25., concentration_M=.25,
            catalyst_loading_mol_percent=3., loop_volume_mL=1., injected_sample_volume_mL=1.,
            fraction_start_min=10., fraction_end_min=12., fraction_volume_mL=.5, oxygen_purity=1.,
            MFC_calibration="Synthetic O2 calibration fixture", gas_identity="O2", feed_mode="sample_loop")
        measured = {"id": "synthetic-khu-complete", "round_id": r["id"], "design_id": "KX04", "experiment_mode": "flow",
            "actual_conditions": actual, "analytical_method": "Synthetic calibrated NMR software fixture",
            "analytes": {"sulfoxide_pct": {"value": 81., "unit": "%"}, "residual_sulfide_pct": {"value": 12., "unit": "%"},
                         "sulfone_pct": {"value": 3., "unit": "%"}, "other_products_pct": {"value": None, "unit": "%"}},
            "outcome_status": "qualified", "quality": {"accepted": True, "basis": "Synthetic software test; no laboratory result"}}
        records = [measured]
        cases = [("carrier_flow_mL_min", None), ("MFC_calibration", "unknown"),
                 ("gas_reference_temperature_K", 298.15), ("gas_identity", "air"),
                 ("concentration_M", .5), ("catalyst_loading_mol_percent", 1.),
                 ("reactor_pressure_bar_abs", {"value": 2.51325, "unit": "bar absolute", "status": "assumed"}),
                 ("oxygen_flow_reference_mL_min", {"value": .1, "unit": "L/min"})]
        for i, (field, value) in enumerate(cases):
            row = deepcopy(measured)
            row["id"] = "synthetic-incompatible-" + str(i)
            row["actual_conditions"][field] = value
            records.append(row)
        c = self.store.add_results(c["id"], records)
        ledger = feedback_summary(c)
        self.assertEqual(ledger["numerical_optimization_feedback_eligible_ids"], ["synthetic-khu-complete"])
        excluded = {x["result_id"]: x["reason"] for x in ledger["excluded_from_numerical_feedback"]}
        self.assertEqual(set(excluded), {x["id"] for x in records[1:]})
        for i, (field, _) in enumerate(cases):
            self.assertIn(field, excluded["synthetic-incompatible-" + str(i)])
        self.assertEqual(len(c["results"]), len(records))
        self.assertEqual(c["results"][0]["actual_conditions"]["reactor_pressure_bar_abs"], 2.47)
        self.assertNotIn("pressure_bar_abs_assumed", c["results"][0]["actual_conditions"])
        self.assertIn("reactor_pressure_bar_abs", ledger["recorded_context_not_distance_dimensions"])
        self.assertTrue(all(p not in actual for p in spec["observation_contract"]["excluded_from_measurement_requirements"]))
        new = self.store.generate(c["id"], {"requested_count": 1})
        self.assertEqual(new["feedback_used"], ["synthetic-khu-complete"])
        self.assertEqual(new["selected"][0]["id"], "KX04")

