"""FLORA — Knowledge Extraction (PRISM) page."""

import json
import glob
import importlib.util
import streamlit as st
from pathlib import Path


def render():
    st.title("Knowledge Extraction")
    st.caption("KNOWLEDGE / ADMIN")
    if importlib.util.find_spec("flora_prism") is None:
        st.info(
            "Knowledge Extraction is unavailable in this installation: the optional "
            "flora_prism package was not included in the supplied source. "
            "Install it and restart the app to enable this feature. "
            "Existing corpus browsing and retrieval remain available."
        )
        return
    st.markdown(
        "Upload PDFs and extract structured records. "
        "Results are saved and indexed into ChromaDB for RAG retrieval."
    )

    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if not uploaded:
        _render_status()
        return

    st.markdown(f"**{len(uploaded)} PDF(s) selected.**")

    with st.form("prism_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            provider = st.radio(
                "Provider",
                ["Anthropic (Claude)", "OpenAI (GPT-4o)"],
                horizontal=True,
            )
        with c2:
            if provider.startswith("Anthropic"):
                st.caption("Scan: Haiku 4.5  |  Extract: Sonnet 4")
            else:
                st.caption("Scan: GPT-4o-mini  |  Extract: GPT-4o")
        with c3:
            auto_index = st.checkbox("Auto-index into ChromaDB", value=True)
        submitted = st.form_submit_button("Run Extraction", type="primary")

    if not submitted:
        return

    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, pdf in enumerate(uploaded):
        status.markdown(f"Processing **{pdf.name}** ({i+1}/{len(uploaded)})...")
        try:
            from flora_prism.prism import Prism
            _provider = "openai" if provider.startswith("OpenAI") else "anthropic"
            record = Prism(provider=_provider).extract(pdf)
            results.append(record)
            st.success(f"{pdf.name} — extracted successfully")
        except Exception as e:
            st.error(f"{pdf.name} — {e}")
        progress.progress((i + 1) / len(uploaded))

    st.markdown(f"### Extraction complete — {len(results)}/{len(uploaded)} succeeded")

    if results:
        with st.expander("Preview extracted records"):
            for rec in results[:3]:
                st.json(rec)

        st.download_button(
            "Download all records (JSON)",
            json.dumps(results, indent=2, ensure_ascii=False),
            "prism_records.json", "application/json",
        )

        if auto_index:
            with st.spinner("Indexing into ChromaDB..."):
                try:
                    from flora_prism.prism import index_records
                    n = index_records(results)
                    st.success(f"Indexed {n} records into ChromaDB.")
                except Exception as e:
                    from components.error_card import render_error
                    render_error(e, "Indexing")


def _render_status():
    """Show existing records when no upload is active."""
    records_dir = Path("extraction_results")
    if records_dir.exists():
        jsons = list(records_dir.glob("*.json"))
        non_meta = [f for f in jsons if not f.name.startswith("_")]
        st.info(f"{len(non_meta)} records in extraction_results/")

        # Cost info
        cost_log = records_dir / "_cost_log.json"
        if cost_log.exists():
            costs = json.loads(cost_log.read_text())
            total = sum(c.get("cost_usd", 0) for c in costs)
            st.caption(f"Total extraction API cost: ${total:.4f}")

        if non_meta:
            with st.expander("Preview a record"):
                sample = json.loads(non_meta[0].read_text())
                st.json(sample)
    else:
        st.warning("No extraction results yet. Upload PDFs to begin.")
