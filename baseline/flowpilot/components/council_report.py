"""FLORA — ENGINE council report renderer."""

import streamlit as st


def render_council_report(design_candidate):
    """Render ENGINE council messages grouped by agent."""
    if not design_candidate:
        st.info("No engineering validation data available.")
        return

    msgs = design_candidate.council_messages
    if hasattr(msgs[0] if msgs else None, "model_dump"):
        msgs = [m.model_dump() if hasattr(m, "model_dump") else m for m in msgs]

    st.markdown(f"**Council rounds:** {design_candidate.council_rounds}")

    # Summary metrics
    safety = design_candidate.safety_report
    if isinstance(safety, dict):
        c1, c2, c3 = st.columns(3)
        c1.metric("Passed", safety.get("accepts", 0))
        c2.metric("Warnings", safety.get("warnings", 0))
        c3.metric("Rejects", safety.get("rejects", 0))

    # Group by agent
    agents = {}
    for m in msgs:
        m = m if isinstance(m, dict) else m.model_dump()
        agent = m.get("agent", "Unknown")
        agents.setdefault(agent, []).append(m)

    for agent, agent_msgs in agents.items():
        with st.expander(f"{agent} — {len(agent_msgs)} check(s)", expanded=True):
            for m in agent_msgs:
                status = m.get("status", "")
                field = m.get("field", "")
                concern = m.get("concern", "")
                value = m.get("value", "")
                suggestion = m.get("suggested_revision", "")

                if status == "ACCEPT":
                    if concern:
                        st.success(f"**{field}:** {concern}")
                    else:
                        st.success(f"**{field}:** {value}")
                elif status == "WARNING":
                    st.warning(f"**{field}:** {concern}")
                    if suggestion:
                        st.caption(f"Suggestion: {suggestion}")
                elif status == "REJECT":
                    st.error(f"**{field}:** {concern}")
                    if suggestion:
                        st.caption(f"Suggestion: {suggestion}")
