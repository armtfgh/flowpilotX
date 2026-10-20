from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from khu_sulfoxidation.export import build_campaign, document_blocks, export_package, results_csv_rows, verify_package, zip_package
from khu_sulfoxidation.provenance import digest
from test_engineering import synthetic_apparatus, synthetic_screen

ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/"configs/study_v1.json").read_text())

    def test_pending_package_keeps_proposals_unknown_and_deduplicated(self):
        c = build_campaign(ROOT, self.config)
        self.assertEqual(len(c["condition_rows"]), 7)
        self.assertEqual(c["condition_rows"][-1]["repeat_of"], "C03")
        self.assertEqual(len(c["history"]["systems"]["homogeneous"]["records"]), 116)
        self.assertTrue(all(r["actual_conditions"] is None for r in c["results_template"]))
        self.assertTrue(all(r["calculated"] == {} for r in c["condition_rows"]))
        self.assertIsNone(c["core_material_totals"]["elapsed_min"])
        self.assertIn("endpoints.sulfone_limit_pct: pending KHU", c["release_blockers"])
        self.assertEqual(c["configuration_sha256"], digest(c["configuration"]))
        self.assertEqual(c["condition_rows"][0]["screen_configuration_sha256"], c["screen"]["configuration_sha256"])
        self.assertEqual(c["condition_rows"][0]["study_configuration_sha256"], c["configuration_sha256"])
        self.assertFalse(c["methods"]["history_automatically_selects_time_oxygen_coordinates"])
        self.assertEqual(c["methods"]["time_oxygen_coordinate_source"], "caller_supplied_screen_configuration")

    def test_configurable_fallback_and_preparation_reach_document(self):
        cfg = deepcopy(self.config)
        cfg["chemistry"]["substrate_concentration_M"] = .5
        cfg["screen"].update(synthetic_screen(design="one_oxygen_level_fallback", oxygen_equivalents=[1.],repeat_candidate_id="C01"))
        cfg["optional_history_omitted_nominations"] = 1
        cfg["materials"]["final_stock_volume_mL"] = 100
        cfg["order"]["execution_order"] = ["C02", "C03", "C01", "R01"]
        c = build_campaign(ROOT,cfg,synthetic_apparatus())
        text = " ".join(str(v) for _,v in document_blocks(c))
        self.assertEqual(len(c["condition_rows"]),4)
        for expected in ["4 runs", "0.5 M", "C01 reference", "Final stock volume (mL): 100"]:
            self.assertIn(expected,text)
        self.assertNotIn("HO01/HO02",text)
        self.assertNotIn("(7 runs)",text)
        self.assertEqual(c["condition_rows"][0]["proposed_execution_order"],3)
        self.assertEqual(c["condition_rows"][0]["calculated"]["catalyst_concentration_mM"],15)
        actual = results_csv_rows(c)[0]["actual"]
        self.assertTrue(all(value is None for value in actual.values()))

    def test_material_limits_and_independent_repeat_count(self):
        cfg=deepcopy(self.config)
        cfg["screen"].update(synthetic_screen())
        cfg["materials"].update(final_stock_volume_mL=1,extra_stock_reserve_mL=1,
            available_substrate_mmol=.0001,available_catalyst_mmol=.0001)
        cfg["materials"]["sampling"]={"priming_feed_mL":1,"equilibration_min":2,"collection_min":3,
            "flush_solvent_mL":2,"flush_min":2,"handling_min":1,"gas_on_min":5}
        c=build_campaign(ROOT,cfg,synthetic_apparatus())
        expected=sum(1+5*r["calculated"]["liquid_flow_mL_min"] for r in c["condition_rows"])
        self.assertAlmostEqual(c["core_material_totals"]["feed_volume_mL"],expected)
        self.assertEqual(c["core_material_totals"]["flush_solvent_mL"],14)
        self.assertTrue(any("Insufficient substrate" in x for x in c["release_blockers"]))
        self.assertTrue(any("below core feed" in x for x in c["release_blockers"]))

    def test_package_csv_manifest_and_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"package"
            c=export_package(ROOT,self.config,out)
            self.assertEqual(verify_package(out)["status"],"passed")
            with (out/"results_template.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),7)
            self.assertEqual(rows[0]["actual.liquid_flow_mL_min"],"")
            self.assertEqual(rows[-1]["repeat_of"],"C03")
            self.assertTrue(zip_package(out).exists())
            with self.assertRaises(FileExistsError): export_package(ROOT,self.config,out)
            (out/"conditions.csv").write_text("tampered")
            with self.assertRaises(ValueError): verify_package(out)

    def test_invalid_execution_order_and_live_calls_rejected(self):
        for change in [{"live_model_calls_enabled":True},{"order":{"execution_order":["C01"]}}]:
            cfg=deepcopy(self.config);cfg.update(change)
            with self.assertRaises(ValueError):build_campaign(ROOT,cfg)

    def test_optional_comparison_can_be_disabled(self):
        self.config["optional_history_omitted_nominations"]=0
        c=build_campaign(ROOT,self.config)
        self.assertEqual(c["optional_slots"],[])
        self.assertEqual(len(c["condition_rows"]),7)

    def test_preparation_capacity_and_negative_reserve(self):
        self.config["materials"].update(final_stock_volume_mL=100,available_substrate_mmol=1)
        c=build_campaign(ROOT,self.config)
        self.assertTrue(any("Insufficient substrate" in x for x in c["release_blockers"]))
        self.assertEqual(c["core_material_totals"]["minimum_substrate_required_mmol_including_known_preparation_and_reserve"],25)
        for value in [-1,float("nan"),float("inf"),True]:
            self.config["materials"]["extra_stock_reserve_mL"]=value
            with self.assertRaises(ValueError):build_campaign(ROOT,self.config)


if __name__ == "__main__": unittest.main()
