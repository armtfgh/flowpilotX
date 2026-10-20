"""Streamlit rendering for the authoritative FlowPilot design disposition."""

from __future__ import annotations

import streamlit as st


def render_design_disposition(result: dict, *, compact: bool = False) -> None:
    disposition = str(
        result.get("recommended_disposition")
        or (result.get("design_disposition") or {}).get("recommended_disposition")
        or ""
    ).upper()
    if disposition not in {"BLOCK", "SCREEN", "EXECUTE"}:
        return

    rationale = str(
        result.get("disposition_rationale")
        or (result.get("design_disposition") or {}).get("rationale")
        or ""
    )
    if disposition == "BLOCK":
        st.error(
            "**BLOCKED - Do not execute this design.**"
            + (f" {rationale}" if rationale else "")
        )
    elif disposition == "SCREEN":
        st.info(
            "**SCREEN - Feasible first experiment, not a validated production design.**"
            + (f" {rationale}" if rationale and not compact else "")
        )
    else:
        st.success(
            "**EXECUTE - Deterministic gates passed and external validation is present.**"
            + (f" {rationale}" if rationale and not compact else "")
        )

    failures = (result.get("design_disposition") or {}).get("hard_failures") or []
    if failures and not compact:
        with st.expander("Blocking reasons", expanded=True):
            for finding in failures:
                finding_id = finding.get("finding_id", "HARD-GATE")
                message = finding.get("message", "Hard feasibility check failed.")
                st.markdown(f"- **{finding_id}:** {message}")
