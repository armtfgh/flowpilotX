"""Streamlit views of the same stage-aware report served by the webapp."""

import streamlit as st

from flora_translate.result_reporting import build_result_report


def render_process_summary(result, key="process_summary", streams_only=False):
    report = build_result_report(result)
    if report["status"] != "executable":
        st.info("No accepted process setpoints are available.")
        return
    for issue in report["issues"]:
        st.warning(issue)
    if not streams_only:
        st.markdown("### Reactor conditions")
        st.caption("Gas-stage time = V / (liquid flow + gas flow at inlet/STP). Liquid-stage time = V / liquid flow.")
        st.dataframe([{
            "Stage": s["number"], "Operation": s["name"], "Reactor": s["reactor"],
            "Volume (mL)": s["volume_mL"], "Liquid flow (mL/min)": s["liquid_flow_mL_min"],
            "Gas flow at STP (mL/min)": s["gas_flow_stp_mL_min"], "Time (min)": s["residence_time_min"],
            "Time basis": s["residence_basis"], "Temperature (C)": s["temperature_C"],
            "Pressure (bar)": s["pressure_bar"], "Pressure basis": s["pressure_basis"], "Wavelength (nm)": s["wavelength_nm"],
        } for s in report["stages"]], hide_index=True, use_container_width=True)
    st.markdown("### Feed streams")
    st.dataframe([{
        "Stream": s["label"], "Enters stage": s["introduction_stage"], "Composition": "; ".join(s["contents"]),
        "Phase": s["phase"], "Flow (mL/min)": s["flow_mL_min"], "Flow basis": s["flow_basis"],
        "Concentration (M)": s["concentration_M"], "Reagent equivalents": s["equiv"], "Equipment": s["equipment_name"],
    } for s in report["streams"]], hide_index=True, use_container_width=True)


def render_responses(result):
    rows = build_result_report(result)["responses"]
    if not rows:
        st.info("No standardized question log was saved with this run.")
    for row in rows:
        with st.expander(f"{row['question_id']}: {row['question']} - {row['assessment']}"):
            st.markdown(f"**Answer: {row['status']}**")
            st.write(row["answer"])
            st.markdown("**How it is addressed**")
            for evidence in row["evidence"]:
                st.write(evidence)
            st.caption(row["target_path"])
            st.write(row["bound_value"])
            if row["decisions"]:
                st.json(row["decisions"])
            st.markdown("**Recorded answer history**")
            st.json(row["answer_history"])


def render_engineering_history(result):
    history = result.get("engineering_history") or {}
    for key, title in [("before_council", "Before council"), ("after_council", "Council-selected candidate")]:
        snapshot = history.get(key) or {}
        with st.expander(title):
            st.caption("Historical diagnostic record, not final run instructions.")
            proposal = snapshot.get("proposal") or (result.get("pre_council_proposal") if key == "before_council" else None)
            if proposal:
                st.json(proposal)
            else:
                st.info("This snapshot was not saved in the older run.")
            if snapshot.get("calculations"):
                st.json(snapshot["calculations"])
    st.markdown("### Final realized reactors")
    render_process_summary(result)
    keys = {
        "reactor_volume_mL": "Volume (mL)", "residence_time_min": "Time at inlet basis (min)",
        "temperature_C": "Temperature (C)", "tubing_length_m": "Length (m)",
        "reynolds_number": "Reynolds number", "pressure_drop_bar": "Estimated pressure drop (bar)",
        "UA_W_K": "Estimated UA (W/K)", "intensification_factor": "Implied batch/flow time ratio",
    }
    for row in (result.get("final_stage_engineering") or {}).get("stages", []):
        with st.expander(f"Stage {row['stage_number']} engineering - {row['status']}"):
            if row.get("error"):
                st.warning(row["error"])
            else:
                st.dataframe([{"Quantity": label, "Value": row["calculations"].get(k)} for k, label in keys.items()], hide_index=True)
