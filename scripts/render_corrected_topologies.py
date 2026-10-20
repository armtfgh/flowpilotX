"""Render six corrected proposals using preserved FlowPilot modules and icons."""
import argparse
from pathlib import Path
from khu_sulfoxidation.corrected_figures import render_corrected_figures

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    args = parser.parse_args()
    index = render_corrected_figures(Path(__file__).resolve().parents[1],
        args.package/'corrected_design.json', args.package/'figure_specs.json', args.package)
    print(f"Rendered {len(index['cases'])} corrected diagrams; original renderer and icons unchanged.")
