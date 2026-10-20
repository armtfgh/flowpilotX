"""FLORA-Design — Streamlit-compatible diagram rendering."""

import base64
from pathlib import Path

from flora_translate.schemas import ProcessTopology


def render_svg_in_streamlit(svg_path: str):
    """Render an SVG file inline in Streamlit."""
    import streamlit as st

    if not svg_path or not Path(svg_path).exists():
        st.warning("No flow diagram available.")
        return

    svg_content = Path(svg_path).read_text()
    # Embed SVG directly in HTML for full resolution
    st.markdown(
        f'<div style="background:white;padding:10px;border-radius:8px;">{svg_content}</div>',
        unsafe_allow_html=True,
    )


def render_topology_summary(topology: ProcessTopology):
    """Render a compact topology summary in Streamlit."""
    import streamlit as st

    col1, col2, col3 = st.columns(3)
    col1.metric("Residence Time", f"{topology.residence_time_min:.1f} min")
    col2.metric("Flow Rate", f"{topology.total_flow_rate_mL_min:.2f} mL/min")
    col3.metric("Reactor Volume", f"{topology.reactor_volume_mL:.1f} mL")
