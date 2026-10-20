import unittest

from khu_sulfoxidation.materials import feed_preparation, run_consumption


class MaterialTests(unittest.TestCase):
    def test_feed_amounts_do_not_guess_masses(self):
        p = feed_preparation(10, .25, 3)
        self.assertAlmostEqual(p["substrate_mmol"], 2.5)
        self.assertAlmostEqual(p["catalyst_mmol"], .075)
        self.assertAlmostEqual(p["catalyst_concentration_M"], .0075)
        self.assertIsNone(p["substrate_mass_mg"])

    def test_purity_and_density_are_explicit(self):
        p = feed_preparation(10, .25, 3, {"substrate": {"source": "synthetic calculation fixture", "molecular_mass_g_mol": 100, "purity_fraction": .8, "density_g_mL": 1.25}})
        self.assertAlmostEqual(p["substrate_mass_mg"], 312.5)
        self.assertAlmostEqual(p["substrate_aliquot_mL"], .25)

    def test_continuous_budget_includes_nonproductive_feed(self):
        p = run_consumption(liquid_flow_mL_min=.1, gas_reference_flow_mL_min=1,
            substrate_concentration_M=.25, catalyst_loading_mol_percent=3, feed_mode="continuous",
            sampling={"priming_feed_mL": 2, "equilibration_min": 10, "collection_min": 5,
                      "flush_solvent_mL": 3, "flush_min": 4, "handling_min": 6, "gas_on_min": 15})
        self.assertAlmostEqual(p["feed_volume_mL"], 3.5)
        self.assertAlmostEqual(p["substrate_mmol"], .875)
        self.assertAlmostEqual(p["elapsed_min"], 45)
        self.assertAlmostEqual(p["collected_liquid_mL_estimate"], .5)
        self.assertAlmostEqual(p["gas_reference_volume_mL"], 15)

    def test_loop_consumption_does_not_imply_continuous_productivity(self):
        p = run_consumption(liquid_flow_mL_min=.1, gas_reference_flow_mL_min=1,
            substrate_concentration_M=.25, catalyst_loading_mol_percent=3, feed_mode="sample_loop",
            sampling={"injected_feed_mL": .5, "carrier_solvent_mL": 2, "flush_solvent_mL": 1,
                      "elapsed_min": 30, "gas_on_min": 25, "collection_window_definition": "fixture explicit window"})
        self.assertAlmostEqual(p["substrate_mmol"], .125)
        self.assertIsNone(p["collected_liquid_mL_estimate"])
        self.assertNotIn("productivity", p)

    def test_missing_windows_and_nonfinite_numbers_do_not_become_zero(self):
        self.assertIn("final_stock_volume_mL", feed_preparation(None, .25, 3)["missing"])
        p = run_consumption(liquid_flow_mL_min=.1, gas_reference_flow_mL_min=1,
            substrate_concentration_M=.25, catalyst_loading_mol_percent=3,
            feed_mode="continuous", sampling={})
        self.assertEqual(p["status"], "pending")
        self.assertIsNone(p["feed_volume_mL"])
        for bad in [0, -1, float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError): feed_preparation(bad, .25, 3)


if __name__ == "__main__": unittest.main()
