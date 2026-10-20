"""Availability-aware model routing controls shared by Streamlit pages."""

from __future__ import annotations

import streamlit as st

from flora_translate.model_catalog import (
    available_default_route_ids,
    model_route_statuses,
    model_routes,
    runtime_model_options,
)


@st.cache_data(ttl=30, show_spinner=False)
def _route_statuses() -> dict[str, dict]:
    return model_route_statuses()


def render_model_route_selector(
    key_prefix: str = "flowpilot_model_routes",
) -> tuple[dict | None, list[str]]:
    """Render model selectors and return runtime options plus blockers."""

    routes = model_routes()
    statuses = _route_statuses()
    defaults = available_default_route_ids(statuses)
    route_ids = list(routes)

    def label(route_id: str) -> str:
        route = routes[route_id]
        status = statuses[route_id]
        suffix = "" if status["available"] else " - unavailable"
        return f"{route.label}{suffix}"

    st.markdown("#### Model routing")
    upstream_col, downstream_col = st.columns(2)
    with upstream_col:
        upstream_id = st.selectbox(
            "Upstream chemistry model",
            route_ids,
            index=route_ids.index(defaults["upstream"]),
            format_func=label,
            key=f"{key_prefix}_upstream",
        )
    with downstream_col:
        downstream_id = st.selectbox(
            "Downstream and council model",
            route_ids,
            index=route_ids.index(defaults["downstream"]),
            format_func=label,
            key=f"{key_prefix}_downstream",
        )

    blockers = []
    for role, route_id in (
        ("Upstream", upstream_id),
        ("Downstream", downstream_id),
    ):
        status = statuses[route_id]
        if not status["available"]:
            blockers.append(f"{role}: {routes[route_id].label} - {status['reason']}")

    if blockers:
        st.error("Selected model route is unavailable.\n\n" + "\n\n".join(blockers))
        return None, blockers

    st.caption(
        f"Upstream: {routes[upstream_id].label} | "
        f"downstream/council: {routes[downstream_id].label}"
    )
    return runtime_model_options(upstream_id, downstream_id), []
