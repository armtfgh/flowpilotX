"""Offline projection of the six frozen core proposals through FlowPilot icons."""
import argparse
from pathlib import Path
from khu_sulfoxidation.topology_figures import render_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = render_figures(Path(__file__).resolve().parents[1], args.source_report, args.output)
    print(f"Rendered {len(result['cases'])} frozen core proposals with original FlowPilot icons; no model calls.")
