"""FLORA — Protocol Diagnostics page (standalone ENGINE council)."""

import json
import streamlit as st


def render():
    st.title("Protocol Diagnostics")
    st.markdown(
        "Submit any flow chemistry protocol and the ENGINE council will "
        "evaluate it for fluidics, safety, and hardware compatibility."
    )

    with st.form("diagnose_form"):
        st.markdown("**Describe your flow protocol:**")

        c1, c2 = st.columns(2)
        with c1:
            solvent = st.text_input("Solvent", "MeCN")
            temp = st.number_input("Temperature (°C)", value=25.0)
            flow_rate = st.number_input("Flow rate (mL/min)", value=0.5, step=0.1)
            rt = st.number_input("Residence time (min)", value=10.0, step=1.0)
            conc = st.number_input("Concentration (M)", value=0.1, step=0.01)
        with c2:
            tubing_mat = st.selectbox("Tubing material", ["FEP", "PFA", "SS", "PTFE"])
            tubing_id = st.number_input("Tubing ID (mm)", value=1.0, step=0.5)
            bpr = st.number_input("BPR (bar)", value=5.0, step=1.0)
            wavelength = st.number_input("Wavelength (nm)", value=450, step=10)
            photocatalyst = st.text_input("Photocatalyst", "Ir(ppy)3")
            atmosphere = st.selectbox("Atmosphere", ["N2", "Ar", "air"])

        submitted = st.form_submit_button(
            "Run Diagnostics", type="primary", use_container_width=True
        )

    if not submitted:
        return

    with st.spinner("ENGINE council is deliberating..."):
        try:
            from flora_translate.schemas import BatchRecord, FlowProposal, LabInventory
            from flora_translate.engine.council_v3 import CouncilV3
            from flora_translate.config import LAB_INVENTORY_PATH

            reactor_vol = round(rt * flow_rate, 2)

            proposal = FlowProposal(
                residence_time_min=rt,
                flow_rate_mL_min=flow_rate,
                temperature_C=temp,
                concentration_M=conc,
                BPR_bar=bpr,
                reactor_type="coil",
                tubing_material=tubing_mat,
                tubing_ID_mm=tubing_id,
                reactor_volume_mL=reactor_vol,
                wavelength_nm=wavelength,
                confidence="MEDIUM",
            )

            batch_record = BatchRecord(
                reaction_description=f"{photocatalyst} photocatalysis in {solvent}",
                photocatalyst=photocatalyst,
                solvent=solvent,
                temperature_C=temp,
                concentration_M=conc,
                wavelength_nm=wavelength,
                atmosphere=atmosphere,
            )

            inventory = LabInventory.from_json(str(LAB_INVENTORY_PATH))
            candidate, _ = CouncilV3().run(proposal, batch_record, [], inventory)

            # Overall status
            n_reject = sum(1 for m in candidate.council_messages if m.status == "REJECT")
            n_warn = sum(1 for m in candidate.council_messages if m.status == "WARNING")
            n_accept = sum(1 for m in candidate.council_messages if m.status == "ACCEPT")

            if n_reject > 0:
                overall = "REJECT"
                color = "red"
            elif n_warn > 0:
                overall = "WARNING"
                color = "orange"
            else:
                overall = "ACCEPT"
                color = "green"

            st.markdown(f"### Verdict: :{color}[{overall}]")
            st.caption(
                f"{candidate.council_rounds} round(s) | "
                f"{n_accept} accepted | {n_warn} warnings | {n_reject} rejects"
            )

            from components.council_report import render_council_report
            render_council_report(candidate)

            # Computed values
            import math
            d_m = tubing_id * 1e-3
            area = math.pi * (d_m / 2) ** 2
            length_m = (reactor_vol * 1e-6) / area if area > 0 else 0

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Reactor volume", f"{reactor_vol:.2f} mL")
            m2.metric("Tubing length", f"{length_m:.2f} m")
            m3.metric("V = tau * Q check", f"{reactor_vol:.2f} = {rt} * {flow_rate}")

            st.download_button(
                "Download report (JSON)",
                json.dumps(candidate.model_dump(), indent=2, default=str),
                "engine_report.json", "application/json",
            )

        except Exception as e:
            from components.error_card import render_error
            render_error(e, "Protocol Diagnostics")
