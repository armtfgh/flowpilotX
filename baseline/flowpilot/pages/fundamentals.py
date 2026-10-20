"""FLORA — Flow Chemistry Fundamentals ingestion page."""

import json
import streamlit as st
from pathlib import Path


def render():
    st.title("Flow Chemistry Fundamentals")
    st.caption("KNOWLEDGE / ADMIN")
    st.markdown(
        "Upload handbook or textbook PDFs on flow chemistry. "
        "FLORA will extract structured rules (IF/THEN/BECAUSE) that are "
        "injected into the Chemistry Agent and ENGINE council for every query."
    )

    rules_path = Path("flora_fundamentals/data/rules.json")

    # ── Current status ──
    if rules_path.exists():
        data = json.loads(rules_path.read_text())
        n_rules = len(data.get("rules", []))
        n_handbooks = len(data.get("handbooks", []))
        st.success(f"Knowledge base: **{n_rules} rules** from {n_handbooks} handbook(s)")

        # Browse rules
        with st.expander("Browse extracted rules", expanded=False):
            categories = sorted(set(r.get("category", "") for r in data["rules"]))
            sel_cat = st.selectbox("Filter by category", ["all"] + categories)

            rules = data["rules"]
            if sel_cat != "all":
                rules = [r for r in rules if r.get("category") == sel_cat]

            st.caption(f"Showing {len(rules)} rules")
            for r in rules[:50]:
                severity = r.get("severity", "guideline")
                icon = {"hard_rule": "🔴", "guideline": "🟡", "tip": "🟢"}.get(severity, "⚪")
                with st.expander(f"{icon} [{r.get('category', '?')}] {r.get('recommendation', '?')[:80]}"):
                    st.markdown(f"**When:** {r.get('condition', 'N/A')}")
                    st.markdown(f"**Do:** {r.get('recommendation', 'N/A')}")
                    st.markdown(f"**Why:** {r.get('reasoning', 'N/A')}")
                    if r.get("quantitative"):
                        st.markdown(f"**Numbers:** {r['quantitative']}")
                    if r.get("exceptions"):
                        st.markdown(f"**Exceptions:** {r['exceptions']}")
                    st.caption(f"Source: {r.get('source_handbook', '?')} (pages {r.get('source_page', '?')})")

        # Handbooks list
        with st.expander("Ingested handbooks"):
            for h in data.get("handbooks", []):
                st.markdown(
                    f"- **{h.get('title', h.get('filename', '?'))}** — "
                    f"{h.get('n_rules_extracted', 0)} rules, "
                    f"{h.get('n_pages', 0)} pages"
                )
    else:
        st.info("No fundamentals rules yet. Upload handbook PDFs below to build the knowledge base.")

    # ── Upload new handbooks ──
    st.divider()
    st.subheader("Ingest new handbooks")
    st.caption(
        "PDFs can be uploaded directly or referenced by path on disk "
        "(useful for large files stored outside the project folder)."
    )

    # ── Input mode ──
    input_mode = st.radio(
        "Input method",
        ["Upload files", "File paths (outside project)"],
        horizontal=True,
    )

    pdf_sources = []  # list of (path_str, display_name)

    if input_mode == "Upload files":
        uploaded = st.file_uploader(
            "Upload handbook/textbook PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="handbook_upload",
        )
        if uploaded:
            st.markdown(f"**{len(uploaded)} PDF(s) selected.**")
            # Will be saved to temp files at processing time
            pdf_sources = [("__upload__", u) for u in uploaded]

    else:
        st.markdown(
            "Enter one absolute file path per line. "
            "The files stay on disk — nothing is copied into the project."
        )
        paths_text = st.text_area(
            "PDF file paths",
            height=120,
            placeholder=(
                "/data/handbooks/flow_chemistry_guide.pdf\n"
                "/data/handbooks/microreactor_handbook.pdf\n"
                "/data/references/continuous_manufacturing.pdf"
            ),
        )
        if paths_text.strip():
            for line in paths_text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.exists():
                    st.warning(f"Not found: `{line}`")
                elif not line.lower().endswith(".pdf"):
                    st.warning(f"Not a PDF: `{line}`")
                else:
                    st.success(f"Found: `{p.name}` ({p.stat().st_size // 1024} KB)")
                    pdf_sources.append((str(p), p.name))

    if not pdf_sources:
        return

    # ── Extraction settings ──
    from flora_fundamentals.handbook_reader import MODEL_PROFILES

    with st.form("fundamentals_form"):
        # Model selection (full row)
        profile_names = list(MODEL_PROFILES.keys())
        profile_descriptions = {k: v["description"] for k, v in MODEL_PROFILES.items()}

        model_choice = st.radio(
            "Extraction model",
            profile_names,
            horizontal=True,
            help="Choose the AI backend for extraction.",
        )

        # Show model details for selected profile
        profile = MODEL_PROFILES[model_choice]
        scan_label  = "Scan model"  if model_choice == "Hybrid Claude" else "Scan model"
        ext_label   = "Extract model"
        st.caption(
            f"**{model_choice}** — {profile['description']}  |  "
            f"{scan_label}: `{profile['scan_model']}`  |  "
            f"{ext_label}: `{profile['extract_model']}`"
        )

        st.markdown("")  # spacer

        c1, c2, c3 = st.columns(3)
        with c1:
            chunk_size = st.number_input(
                "Pages per chunk",
                min_value=2, max_value=15, value=5,
                help="Smaller chunks = more API calls but better quality",
            )
        with c2:
            two_pass = st.checkbox(
                "Two-pass mode (recommended)",
                value=True,
                help="Cheap scan model first, expensive extract model only on relevant pages. Saves ~60% cost.",
            )
        with c3:
            threshold = st.number_input(
                "Relevance threshold (0-10)",
                min_value=0, max_value=10, value=4,
                help="Chunks scoring below this in the scan pass are skipped.",
            )
        submitted = st.form_submit_button("Extract Rules", type="primary")

    if not submitted:
        return

    import tempfile
    from flora_fundamentals.handbook_reader import HandbookReader
    from flora_fundamentals.schemas import HandbookIndex, FlowRule

    reader = HandbookReader(model_profile=model_choice)
    all_indices = []
    all_rules = []

    # Load existing rules
    if rules_path.exists():
        existing = json.loads(rules_path.read_text())
        all_indices = [HandbookIndex(**h) for h in existing.get("handbooks", [])]
        all_rules = [FlowRule(**r) for r in existing.get("rules", [])]

    # ── Process each source ──
    for source_key, source_val in pdf_sources:
        if source_key == "__upload__":
            # Uploaded file — save to temp
            pdf = source_val
            display_name = pdf.name
            st.markdown(f"---\n### Processing: {display_name}")
            progress = st.progress(0, text="Starting...")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf.read())
                pdf_path = tmp.name
            cleanup = True
        else:
            # File path — use directly, no copy needed
            pdf_path = source_key
            display_name = source_val
            st.markdown(f"---\n### Processing: {display_name}")
            progress = st.progress(0, text="Starting...")
            cleanup = False

        try:
            idx, rules = reader.read_handbook(
                pdf_path,
                title=display_name.replace(".pdf", ""),
                chunk_pages=chunk_size,
                two_pass=two_pass,
                relevance_threshold=threshold,
                progress_callback=lambda p, msg: progress.progress(min(p, 1.0), text=msg),
            )
            all_indices.append(idx)
            all_rules.extend(rules)
            st.success(
                f"**{display_name}**: {len(rules)} rules extracted "
                f"from {idx.n_pages} pages"
            )
        except Exception as e:
            st.error(f"{display_name}: {e}")
        finally:
            if cleanup:
                Path(pdf_path).unlink(missing_ok=True)

    # Save all rules
    reader.save(all_indices, all_rules)
    st.success(
        f"Knowledge base updated: **{len(all_rules)} total rules** "
        f"from {len(all_indices)} handbooks"
    )
    st.balloons()
