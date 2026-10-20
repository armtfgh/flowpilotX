"""FLORA — Feedback collection (simple text box)."""

import json
import logging
import time
from pathlib import Path

import streamlit as st

logger = logging.getLogger("flora.feedback")
FEEDBACK_PATH = Path("data/feedback_log.json")


def _load() -> list[dict]:
    if FEEDBACK_PATH.exists():
        try:
            return json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            return []
    return []


def _save(entries: list[dict]):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


def render_feedback_widget(result: dict, context: str = "translate"):
    """Simple approve / free-text correction widget."""
    st.divider()
    st.markdown("**Was this proposal useful?**")

    c1, c2 = st.columns([1, 4])
    with c1:
        approve = st.button("Looks good", key=f"fb_ok_{context}",
                            use_container_width=True)
    with c2:
        needs_fix = st.button("Add correction / suggestion",
                              key=f"fb_fix_{context}",
                              use_container_width=True)

    if approve:
        _save(_load() + [{
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "context": context,
            "rating": "approved",
            "proposal": _key_fields(result),
            "notes": "",
        }])
        st.success("Thanks — feedback saved.")

    if needs_fix:
        st.session_state[f"_fb_open_{context}"] = True

    if st.session_state.get(f"_fb_open_{context}"):
        with st.form(f"fb_form_{context}"):
            notes = st.text_area(
                "Your correction or suggestion",
                height=100,
                placeholder=(
                    "e.g. Residence time should be ~15 min not 5 min for this catalyst. "
                    "BPR not needed at room temperature in MeCN. "
                    "Stream B should also contain the base..."
                ),
            )
            if st.form_submit_button("Submit", type="primary"):
                _save(_load() + [{
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "context": context,
                    "rating": "corrected",
                    "proposal": _key_fields(result),
                    "notes": notes,
                }])
                st.success("Correction saved. Thank you.")
                st.session_state[f"_fb_open_{context}"] = False


def _key_fields(result: dict) -> dict:
    p = result.get("proposal", {})
    return {
        "residence_time_min": p.get("residence_time_min"),
        "flow_rate_mL_min": p.get("flow_rate_mL_min"),
        "temperature_C": p.get("temperature_C"),
        "confidence": p.get("confidence"),
        "reaction": result.get("chemistry_plan", {}).get("reaction_name", ""),
    }


def get_feedback_stats() -> dict:
    entries = _load()
    return {
        "total": len(entries),
        "approved": sum(1 for e in entries if e.get("rating") == "approved"),
        "corrected": sum(1 for e in entries if e.get("rating") == "corrected"),
    }
