"""FlowPilot sidebar navigation."""

import streamlit as st


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## FlowPilot")
        st.caption("Inventory-constrained batch-to-flow design")
        _model_routing_status()
        st.divider()

        st.markdown("##### DESIGN")
        _nav_button("FlowPilot Design", "flora_design", "nav_design")
        _nav_button("Inventory Manager", "inventory", "nav_inventory")

        st.divider()

        st.markdown("##### EVALUATE")
        _nav_button("Protocol Diagnostics", "diagnose", "nav_diagnose")
        _nav_button("Condition Optimization", "optimize", "nav_optimize")
        _nav_button("Benchmark Figure Studio", "figure_studio", "nav_figure_studio")

        st.divider()

        with st.expander("KNOWLEDGE", expanded=False):
            _nav_button("Fundamentals", "fundamentals", "nav_fundamentals")
            _nav_button("Literature Mining", "scout", "nav_scout")
            _nav_button("Knowledge Extraction", "prism", "nav_prism")
            _nav_button("Knowledge Base", "corpus", "nav_corpus")

        st.divider()
        _corpus_status()

    return st.session_state.get("page", "flora_design")


def _nav_button(label: str, page: str, key: str) -> None:
    current = st.session_state.get("page", "flora_design")
    if st.button(
        label,
        use_container_width=True,
        key=key,
        type="primary" if current == page else "secondary",
    ):
        st.session_state.page = page
        st.rerun()


def _model_routing_status():
    try:
        from components.model_route_selector import _route_statuses
        from flora_translate.model_catalog import available_default_route_ids, model_routes

        routes = model_routes()
        defaults = available_default_route_ids(_route_statuses())
        upstream = routes[defaults["upstream"]]
        downstream = routes[defaults["downstream"]]

        with st.expander("Model routing", expanded=False):
            st.caption(f"Configured upstream default: {upstream.label}")
            st.caption(f"Configured downstream/council default: {downstream.label}")
            st.caption("These labels show routing configuration, not a completed model check. Each design can override them; unavailable selected routes are blocked.")
    except Exception as exc:
        st.warning(f"Model routing unavailable: {exc}")


def _corpus_status():
    try:
        import chromadb
        client = chromadb.PersistentClient(path="flora_translate/data/chroma_db")
        col = client.get_or_create_collection("flora_records")
        n = col.count()
        if n > 0:
            st.success(f"Corpus: {n} records")
        else:
            st.warning("Corpus empty — index papers first")
    except Exception:
        st.error("ChromaDB not available")
