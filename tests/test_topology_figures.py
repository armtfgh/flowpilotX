"""Study projection contracts, independent of Graphviz/runtime installation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from khu_sulfoxidation.topology_figures import (
    BUILDER, CORE_IDS, EDGES, _configure_captions, _load_file, build_topology, load_proposals,
)
from khu_sulfoxidation.provenance import file_hash

SOURCE = ROOT / "outputs/khu_first_screen_astra_sol_2026-10-13_v2"


class TopologyFigureTests(unittest.TestCase):
    def test_six_core_cases_preserve_frozen_setpoints_and_edges(self):
        proposal, core = load_proposals(SOURCE)
        before = deepcopy(core)
        self.assertEqual(tuple(r["condition_id"] for r in core), CORE_IDS)
        for row in core:
            graph = build_topology(row, proposal["scenario"])
            nodes = {n["op_id"]: n for n in graph["unit_operations"]}
            self.assertEqual(graph["proposed_setpoints"], row["proposed_setpoints"])
            self.assertEqual(nodes["feed"]["parameters"]["flow_rate_mL_min"], row["proposed_setpoints"]["liquid_flow_mL_min"])
            self.assertEqual(nodes["air"]["parameters"]["gas_flow_sccm"], row["proposed_setpoints"]["gas_reference_flow_mL_min"])
            self.assertEqual(tuple((s["from_op"], s["to_op"], s["stream_type"]) for s in graph["streams"]), EDGES)
            self.assertEqual(graph["candidate_id"], row["candidate_id"])
        self.assertEqual(core, before)

    def test_unknowns_never_turn_into_zero_residence_or_pressure(self):
        proposal, core = load_proposals(SOURCE)
        graph = build_topology(core[0], proposal["scenario"])
        nodes = {n["op_id"]: n for n in graph["unit_operations"]}
        self.assertIsNone(graph["actual_liquid_contact_time_min"])
        self.assertIsNone(graph["feed_mode"])
        self.assertIsNone(nodes["bpr"]["parameters"]["pressure_bar"])
        self.assertIsNone(nodes["reactor"]["parameters"]["light_power_setting"])
        self.assertNotIn("residence_time_min", nodes["reactor"]["parameters"])
        self.assertNotIn("residence_time_inlet_min", nodes["reactor"]["parameters"])
        self.assertIsNone(graph["unplaced_requirements"][0]["actual_location"])
        self.assertTrue(all(n["actual_assignment_confirmed"] is None for n in nodes.values()))

    def test_caption_adapter_is_explicit_and_preserves_baseline_bytes(self):
        old_hash = file_hash(ROOT / BUILDER)
        module = _load_file(ROOT / BUILDER, "_test_private_flowpilot_renderer")
        _configure_captions(module)
        proposal, core = load_proposals(SOURCE)
        graph = build_topology(core[0], proposal["scenario"])
        nodes = {n["op_id"]: types.SimpleNamespace(**n) for n in graph["unit_operations"]}
        self.assertIn("reference mL/min", module._mfc_label(nodes["air"]))
        self.assertIn("273.15", module._mfc_label(nodes["air"]))
        self.assertNotIn("at STP", module._mfc_label(nodes["air"]))
        self.assertIn("Liquid-only V/Q_L: 27.78 min", module._reactor_label(nodes["reactor"]))
        self.assertIn("Contact time: unmeasured", module._reactor_label(nodes["reactor"]))
        self.assertEqual(module._main_process_path(list(nodes.values()), [types.SimpleNamespace(**s) for s in graph["streams"]]), ["feed", "mixer", "reactor", "bpr", "collector"])
        self.assertEqual(file_hash(ROOT / BUILDER), old_hash)

    def test_changed_authoritative_source_fails_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            for name in ["manifest.json", "authoritative_conditions.json", "nominations_freeze.json"]:
                shutil.copyfile(SOURCE / name, d / name)
            proposal = json.loads((d / "authoritative_conditions.json").read_text())
            proposal["conditions"][0]["proposed_setpoints"]["liquid_flow_mL_min"] = 9.99
            (d / "authoritative_conditions.json").write_text(json.dumps(proposal))
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_proposals(d)


if __name__ == "__main__":
    unittest.main()
