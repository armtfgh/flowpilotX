import csv
import io
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

from khu_sulfoxidation.gui_data import builtin_history, merge_history, parse_history_upload, template_csv

ROOT = Path(__file__).resolve().parents[1]


def upload(rows, **metadata):
    return parse_history_upload("history.json", json.dumps({"dataset_id": "study-a", "catalyst_system": "homogeneous",
        "experiment_mode": "batch", "records": rows, **metadata}).encode())


class GuiDataTests(unittest.TestCase):
    def test_csv_preserves_percentages_and_unknowns(self):
        raw = b"dataset_id,catalyst_system,source_entry,experiment_mode,sulfoxide_pct,sulfone_pct,residual_sulfide_pct\nx,homogeneous,1,batch,103,-,\n"
        data = parse_history_upload("h.csv", raw)
        row = data["rows"][0]
        self.assertEqual(row["reported_outcomes"]["sulfoxide_pct"], 103)
        self.assertIsNone(row["reported_outcomes"]["sulfone_pct"])
        self.assertIsNone(row["reported_outcomes"]["residual_sulfide_pct"])
        self.assertEqual(row["source_representations"][0]["raw_fields"]["sulfone_pct"], "-")
        self.assertEqual(data["summary"]["above_100_count"], 1)

    def test_export_identity_dedup_retains_provenance_and_conflicts(self):
        a = upload([{"source_entry": 1, "sulfoxide_pct": 98}])
        b = upload([{"source_entry": 1, "sulfoxide_pct": 97}, {"source_entry": 2, "sulfoxide_pct": 98}])
        merged = merge_history([a, b, a])
        self.assertEqual(merged["summary"]["canonical_count"], 2)
        self.assertEqual(merged["summary"]["source_representation_count"], 3)
        row = merged["rows"][0]
        self.assertFalse(row["interpretation_eligible"])
        self.assertEqual([r["reported_outcomes"]["sulfoxide_pct"] for r in row["source_representations"]], [98, 97])
        self.assertEqual(merged["summary"]["evidence_ids"], [merged["rows"][1]["record_id"]])

    def test_systems_modes_and_distinct_entries_do_not_pool(self):
        a = upload([{"source_entry": 1, "sulfoxide_pct": 98}, {"source_entry": 2, "sulfoxide_pct": 98}])
        b = upload([{"source_entry": 1, "sulfoxide_pct": 101}], catalyst_system="heterogeneous")
        c = upload([{"source_entry": 1, "sulfoxide_pct": 98}], experiment_mode="flow")
        merged = merge_history([a, b, c])
        self.assertEqual(len(merged["rows"]), 4)
        self.assertEqual(merged["summary"]["heterogeneous_count"], 1)

    def test_undeclared_dataset_does_not_merge_unrelated_numbered_rows(self):
        a = parse_history_upload("one.csv", b"source_entry,sulfoxide_pct\n1,99\n")
        b = parse_history_upload("two.csv", b"source_entry,sulfoxide_pct\n1,98\n")
        merged = merge_history([a, b])
        self.assertEqual(len(merged["rows"]), 2)
        self.assertEqual(merged["summary"]["evidence_ids"], [])
        self.assertTrue(a["warnings"])

    def test_builtin_and_renamed_original_exports_are_not_double_counted(self):
        built = builtin_history(ROOT)
        paths = sorted((ROOT / "data/raw/khu").glob("*raw data_homogeneous.*"))
        imported = [parse_history_upload("renamed" + p.suffix, p.read_bytes(), known_sources=built["known_sources"]) for p in paths]
        merged = merge_history([built, *imported])
        self.assertEqual(merged["summary"]["canonical_count"], 158)
        self.assertEqual(merged["summary"]["homogeneous_count"], 116)
        self.assertEqual(merged["summary"]["heterogeneous_count"], 42)
        self.assertEqual(merged["summary"]["source_representation_count"], 311)
        self.assertEqual(merged["summary"]["conflict_count"], 0)
        self.assertEqual(len(merged["summary"]["evidence_ids"]), 116)

    def test_xlsx_cached_values_and_formulas_preserved(self):
        stream = io.BytesIO()
        with ZipFile(stream, "w") as z:
            z.writestr("xl/workbook.xml", '<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Data" r:id="r1"/></sheets></workbook>')
            z.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
            z.writestr("xl/worksheets/sheet1.xml", '<worksheet><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>source_entry</t></is></c><c r="B1" t="inlineStr"><is><t>sulfoxide_pct</t></is></c></row><row r="2"><c r="A2"><v>1</v></c><c r="B2"><f>100+2</f><v>102</v></c></row></sheetData></worksheet>')
        data = parse_history_upload("test.xlsx", stream.getvalue())
        row = data["rows"][0]
        self.assertEqual(row["reported_outcomes"]["sulfoxide_pct"], 102)
        self.assertEqual(row["source_representations"][0]["formulas"]["B2"], {"formula": "100+2", "cached_value": "102"})

    def test_unknown_columns_retained_template_explicit(self):
        data = upload([{"source_entry": 1, "operator_note": "keep this", "sulfone_pct": None}])
        self.assertEqual(data["rows"][0]["source_representations"][0]["raw_fields"]["operator_note"], "keep this")
        self.assertIn("dataset_id", next(csv.reader(io.StringIO(template_csv()))))

    def test_explicit_reported_actuals_override_generic_groups_and_nulls(self):
        source = {"source_entry": "measured", "status": "completed",
            "reported_actual_conditions": {"liquid_flow_mL_min": .19, "temperature_C": None},
            "reported_outcomes": {"sulfoxide_pct": 102, "sulfone_pct": None},
            "conditions": {"liquid_flow_mL_min": .2, "temperature_C": 25},
            "outcomes": {"sulfoxide_pct": 98, "sulfone_pct": 1}}
        row = upload([source], experiment_mode="flow")["rows"][0]
        self.assertEqual(row["reported_actual_conditions"]["liquid_flow_mL_min"], .19)
        self.assertIsNone(row["reported_actual_conditions"]["temperature_C"])
        self.assertEqual(row["reported_outcomes"]["sulfoxide_pct"], 102)
        self.assertIsNone(row["reported_outcomes"]["sulfone_pct"])
        self.assertEqual(row["source_status"], "completed")
        self.assertTrue(row["interpretation_eligible"])
        self.assertEqual(row["source_representations"][0]["raw_json_record"], source)

    def test_proposal_status_and_null_actuals_cannot_become_measured_history(self):
        proposal = {"source_entry": "x", "status": "conditional_proposal",
            "conditions": {"liquid_flow_mL_min": .2},
            "outcomes": {"sulfoxide_pct": 98}, "actual": {"liquid_flow_mL_min": None}}
        data = upload([proposal], experiment_mode="flow")
        row = data["rows"][0]
        self.assertEqual(row["source_status"], "conditional_proposal")
        self.assertEqual(row["record_kind"], "proposal")
        self.assertEqual(row["proposed_conditions"], {"liquid_flow_mL_min": .2})
        self.assertIsNone(row["reported_actual_conditions"]["liquid_flow_mL_min"])
        self.assertIsNone(row["reported_outcomes"]["sulfoxide_pct"])
        self.assertFalse(row["interpretation_eligible"])
        self.assertEqual(data["summary"]["evidence_ids"], [])
        self.assertEqual(data["summary"]["proposal_count"], 1)
        measured = upload([{"source_entry": "x", "liquid_flow_mL_min": .19, "sulfoxide_pct": 92}], experiment_mode="flow")
        merged = merge_history([data, measured, data])
        self.assertEqual(len(merged["rows"]), 2)
        self.assertEqual(merged["summary"]["interpretation_eligible_count"], 1)
        self.assertEqual(merged["summary"]["evidence_ids"], [measured["rows"][0]["record_id"]])

    def test_null_actual_and_unmeasured_records_stay_ineligible_when_merged(self):
        data = upload([{"source_entry": 1, "actual": None,
            "conditions": {"liquid_flow_mL_min": .2}, "outcomes": {"sulfoxide_pct": 90}}])
        merged = merge_history([data, data])
        self.assertIsNone(merged["rows"][0]["reported_actual_conditions"]["liquid_flow_mL_min"])
        self.assertIsNone(merged["rows"][0]["reported_outcomes"]["sulfoxide_pct"])
        self.assertFalse(merged["rows"][0]["interpretation_eligible"])
        proposal = upload([{"source_entry": 2, "actual": None,
            "proposed_setpoints": {"liquid_flow_mL_min": .2}}])["rows"][0]
        self.assertEqual(proposal["record_kind"], "proposal")
        self.assertFalse(proposal["interpretation_eligible"])

    def test_large_entry_identity_is_exact_across_json_and_csv(self):
        data = upload([{"source_entry": "9007199254740993", "sulfoxide_pct": 91},
            {"source_entry": 9007199254740992, "sulfoxide_pct": 92}])
        self.assertEqual([r["source_entry"] for r in data["rows"]], ["9007199254740993", "9007199254740992"])
        csv_data = parse_history_upload("large.csv", b"dataset_id,catalyst_system,experiment_mode,source_entry,sulfoxide_pct\nstudy-a,homogeneous,batch,9007199254740993.0,91\n")
        merged = merge_history([data, csv_data])
        self.assertEqual(len(merged["rows"]), 2)
        self.assertEqual(len(merged["rows"][0]["source_representations"]), 2)

    def test_invalid_identity_nonfinite_and_ambiguous_columns_rejected(self):
        for rows in [[{"sulfoxide_pct": 20}], [{"source_entry": 1}, {"source_entry": 1}], [{"source_entry": 1, "sulfoxide_pct": float("nan")}]]:
            with self.assertRaises(ValueError):
                upload(rows)
        with self.assertRaises(ValueError):
            parse_history_upload("h.csv", b"Entry,source_entry,sulfoxide_pct\n1,2,99\n")
        with self.assertRaisesRegex(ValueError, "more values"):
            parse_history_upload("h.csv", b"Entry,sulfoxide_pct\n1,99,extra\n")


if __name__ == "__main__":
    unittest.main()
