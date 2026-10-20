"""FLORA — Process Design page."""

import json
from pathlib import Path

import streamlit as st


def render():
    st.title("Process Design")
    st.markdown(
        "Describe a photochemical synthesis goal and FLORA will design "
        "a complete, physically validated flow process from the literature corpus."
    )

    with st.form("design_form"):
        goal_text = st.text_area(
            "Chemistry goal",
            height=130,
            placeholder=(
                "e.g. Design a flow process for Ir(ppy)3-catalyzed photoredox "
                "radical addition of alpha-amino acids to Michael acceptors "
                "in MeCN at room temperature, ~0.5 mL/min scale."
            ),
        )
        c1, c2 = st.columns(2)
        with c1:
            scale = st.selectbox("Scale", ["lab (~1 mL/min)", "analytical", "multigram"])
        with c2:
            show_alt = st.checkbox("Show alternative topology", value=True)
        submitted = st.form_submit_button(
            "Design Flow Process", type="primary", use_container_width=True
        )

    if submitted and goal_text.strip():
        with st.spinner("FLORA-Design is analyzing your chemistry goal..."):
            try:
                from flora_design.main import design as run_design
                result = run_design(goal_text)
                st.session_state["design_result"] = result
            except Exception as e:
                from components.error_card import render_error
                render_error(e, "FLORA-Design")
                return

    if "design_result" not in st.session_state:
        _show_example()
        return

    result = st.session_state["design_result"]
    topo = result.topology

    conf = topo.topology_confidence
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    st.markdown(f"### Confidence: :{conf_color}[{conf}]")

    # Detected features banner
    feats = result.chem_features
    st.info(
        f"Detected: **{feats.reaction_class.replace('_', ' ').title()}** | "
        f"Catalyst: {feats.photocatalyst or 'not specified'} | "
        f"Wavelength: {feats.wavelength_nm or '?'} nm | "
        f"Phase: {feats.phase_regime.replace('_', ' ')}"
    )

    tabs = st.tabs([
        "Process Design",
        "Flow Diagram",
        "Engineering Checks",
        "Equipment & Inventory",
        "Literature",
        "Raw JSON",
    ])

    with tabs[0]:
        st.subheader("Proposed Process")
        st.code(topo.pid_description, language=None)

        c_left, c_right = st.columns([2, 1])
        with c_right:
            st.metric("Residence time", f"{topo.residence_time_min:.1f} min")
            st.metric("Flow rate", f"{topo.total_flow_rate_mL_min:.2f} mL/min")
            st.metric("Reactor volume", f"{topo.reactor_volume_mL:.1f} mL")
        with c_left:
            st.subheader("Unit Operations")
            for i, op in enumerate(topo.unit_operations, 1):
                with st.expander(f"{i}. {op.label} ({op.op_type})"):
                    st.markdown(f"**Rationale:** {op.rationale}")
                    st.json(op.parameters)

        st.subheader("Design Rationale")
        st.markdown(result.explanation)

    with tabs[1]:
        from components.process_diagram import render_process_diagram
        render_process_diagram(
            result.svg_path,
            result.png_path,
            topology=result.topology.model_dump(),
            render_manifest=result.diagram_render_manifest,
        )

        if show_alt and result.alternatives:
            st.divider()
            st.subheader("Alternative Topology")
            alt = result.alternatives[0]
            st.code(alt.pid_description, language=None)

    with tabs[2]:
        dc = result.design_candidate
        if dc:
            from components.council_report import render_council_report
            render_council_report(dc)

            try:
                from flora_design.visualizer.plot_builder import plot_dp_vs_flowrate
                st.subheader("Pressure Drop vs Flow Rate")
                fig = plot_dp_vs_flowrate(topo, feats.solvent)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    with tabs[3]:
        from components.inventory_result import render_inventory_result

        render_inventory_result(result.model_dump())

    with tabs[4]:
        if result.retrieved_records:
            st.markdown(f"**{len(result.retrieved_records)} records** used:")
            for doi in result.retrieved_records:
                st.markdown(f"- {doi}")
        else:
            st.info("No literature records used.")
        if result.warnings:
            st.subheader("Warnings")
            for w in result.warnings:
                st.warning(w)

        with st.expander("Chemistry features (raw)"):
            st.json(feats.model_dump(exclude_none=True))

    with tabs[5]:
        st.json(result.model_dump(exclude_none=True))

    # Feedback
    from components.feedback import render_feedback_widget
    feedback_result = {
        "proposal": {
            "residence_time_min": topo.residence_time_min,
            "flow_rate_mL_min": topo.total_flow_rate_mL_min,
            "temperature_C": feats.temperature_C,
            "concentration_M": feats.concentration_M,
            "reactor_type": "coil",
            "tubing_material": "FEP",
            "confidence": topo.topology_confidence,
        }
    }
    render_feedback_widget(feedback_result, context="design")


def _show_example():
    with st.expander("What can FLORA-Design do?", expanded=False):
        st.markdown("""
**Good input:**
> Design a flow process for a 4CzIPN-catalyzed Giese-type radical addition
> in DMA at 0.1 M, 25°C, ~0.5 mL/min. The reaction generates CO2.

**What you get:**
- Complete unit operation sequence (pumps, mixer, degas, reactor, BPR, collector)
- SVG/PNG process flow diagram with conditions
- Engineering validation (pressure drop, Reynolds number, safety checks)
- Literature references from the corpus
""")
