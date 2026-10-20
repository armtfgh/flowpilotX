"""FLORA — Batch-to-Flow translation page."""

import json
from pathlib import Path

import streamlit as st


def render():
    st.title("Batch-to-Flow")
    st.markdown(
        "Paste a batch synthesis protocol and FLORA will propose "
        "a literature-grounded flow chemistry equivalent with full "
        "process design, engineering validation, and flow diagram."
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    input_mode = st.radio("Input mode", ["Free Text", "Structured Form"], horizontal=True)
    batch_input = None

    if input_mode == "Free Text":
        batch_text = st.text_area(
            "Batch protocol",
            height=180,
            placeholder=(
                "Example: fac-Ir(ppy)3 (1 mol%) photocatalyzed decarboxylative "
                "radical addition of N-Boc-proline (1.0 equiv) to methyl vinyl "
                "ketone (2.0 equiv), K2HPO4 (1.5 equiv), DMF, 0.1 M, RT, "
                "450 nm blue LED, N2 atmosphere, 24 hours, 72% yield."
            ),
        )
        if batch_text:
            batch_input = batch_text
    else:
        c1, c2 = st.columns(2)
        with c1:
            reaction_desc = st.text_input("Reaction description")
            photocatalyst = st.text_input("Photocatalyst", placeholder="e.g. Ir(ppy)3")
            cat_loading = st.number_input("Catalyst loading (mol%)", min_value=0.0, value=0.0, step=0.1)
            base = st.text_input("Base", placeholder="e.g. K2HPO4")
            solvent = st.text_input("Solvent", placeholder="e.g. DMF")
            atmosphere = st.selectbox("Atmosphere", ["N2", "Ar", "air", "O2"])
        with c2:
            temperature = st.number_input("Temperature (°C)", value=25.0, step=1.0)
            reaction_time = st.number_input("Reaction time (h)", min_value=0.0, value=0.0, step=0.5)
            concentration = st.number_input("Concentration (M)", min_value=0.0, value=0.0, step=0.01)
            scale = st.number_input("Scale (mmol)", min_value=0.0, value=0.0, step=0.1)
            yield_pct = st.number_input("Yield (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            wavelength = st.number_input("Wavelength (nm)", min_value=0, value=450, step=10)
            light_source = st.text_input("Light source", placeholder="e.g. 450 nm blue LED")

        if reaction_desc:
            batch_input = {
                "reaction_description": reaction_desc,
                "photocatalyst": photocatalyst or None,
                "catalyst_loading_mol_pct": cat_loading or None,
                "base": base or None,
                "solvent": solvent or None,
                "temperature_C": temperature,
                "reaction_time_h": reaction_time or None,
                "concentration_M": concentration or None,
                "scale_mmol": scale or None,
                "yield_pct": yield_pct or None,
                "wavelength_nm": wavelength or None,
                "light_source": light_source or None,
                "atmosphere": atmosphere,
            }

    # ── Run ────────────────────────────────────────────────────────────────────
    if st.button("Translate to Flow", type="primary", disabled=batch_input is None,
                 use_container_width=True):
        with st.spinner("Running FLORA-Translate pipeline..."):
            try:
                from flora_translate.main import translate
                from flora_translate.gui_autosave import autosave_gui_result
                result = translate(batch_input)
                autosave_dir = autosave_gui_result(
                    result,
                    source="legacy_translate",
                    user_input=batch_input if isinstance(batch_input, str) else json.dumps(batch_input, default=str),
                )
                result["autosave_dir"] = str(autosave_dir)
                st.session_state["translate_result"] = result
            except Exception as e:
                from components.error_card import render_error
                render_error(e, "FLORA-Translate")
                return

    # ── Display ────────────────────────────────────────────────────────────────
    if "translate_result" not in st.session_state:
        _show_example()
        return

    result = st.session_state["translate_result"]
    from flora_translate.final_design_contract import canonical_proposal

    final_design = result.get("final_design") or {}
    proposal = canonical_proposal(result)
    is_blocked = final_design.get("status") != "executable"
    if result.get("autosave_dir"):
        st.caption(f"Autosaved run folder: {result['autosave_dir']}")

    from components.design_disposition import render_design_disposition

    render_design_disposition(result)

    # Confidence badge
    conf = result.get("confidence", "LOW")
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    st.markdown(f"### Confidence: :{conf_color}[{conf}]")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Summary",
        "Process Diagram",
        "Chemistry Plan",
        "Stream Assignments",
        "Flow Conditions",
        "Engineering Report",
        "Raw JSON",
        "Equipment & Inventory",
    ])

    # ── TAB 0: Summary ────────────────────────────────────────────────────────
    with tabs[0]:
        if is_blocked:
            st.error("No executable final design was produced. Intermediate values are withheld.")
            st.json((final_design.get("consistency") or {}).get("issues") or [])
        else:
            st.markdown(result.get("canonical_explanation", ""))
            if result.get("explanation"):
                with st.expander("Pre-realization model narrative (audit only)"):
                    st.markdown(result["explanation"])
        chem_notes = proposal.get("chemistry_notes", "")
        if chem_notes:
            st.divider()
            st.markdown(f"**Chemistry Notes:** {chem_notes}")

    # ── TAB 1: Process Diagram ─────────────────────────────────────────────────
    with tabs[1]:
        from components.process_diagram import render_process_diagram
        render_process_diagram(
            result.get("diagnostic_svg_path", "") if is_blocked else result.get("svg_path", ""),
            result.get("diagnostic_png_path", "") if is_blocked else result.get("png_path", ""),
            topology=(result.get("diagnostic_topology") or {}) if is_blocked else (result.get("process_topology") or {}),
            render_manifest=(result.get("diagnostic_diagram_render_manifest") or {}) if is_blocked else (result.get("diagram_render_manifest") or {}),
        )
        # Topology details
        topo = (
            result.get("diagnostic_topology") or result.get("process_requirements_topology") or {}
            if is_blocked
            else result.get("process_topology", {})
        )
        if topo:
            st.divider()
            st.subheader("Unit Operations")
            for i, op in enumerate(topo.get("unit_operations", []), 1):
                if op.get("op_type") == "led_module":
                    continue
                with st.expander(f"{i}. {op.get('label', '?')}"):
                    p = op.get("parameters", {})
                    contents = p.get("contents", [])
                    if contents:
                        for c in contents:
                            st.markdown(f"- {c}")
                    if p.get("solvent"):
                        st.markdown(f"**Solvent:** {p['solvent']}")
                    if p.get("flow_rate_mL_min"):
                        st.markdown(f"**Flow rate:** {p['flow_rate_mL_min']} mL/min")
                    for k, v in p.items():
                        if k not in ("contents", "solvent", "flow_rate_mL_min", "stream", "light_required"):
                            if v is not None:
                                st.markdown(f"**{k.replace('_', ' ')}:** {v}")
                    if op.get("rationale"):
                        st.caption(op["rationale"])
            if topo.get("pid_description"):
                st.code(topo["pid_description"], language=None)

    # ── TAB 2: Chemistry Plan ──────────────────────────────────────────────────
    with tabs[2]:
        _render_chemistry_plan(result.get("chemistry_plan", {}))

    # ── TAB 3: Stream Assignments ──────────────────────────────────────────────
    with tabs[3]:
        if is_blocked:
            st.warning("Final stream assignments are unavailable for a blocked design.")
        else:
            _render_streams(proposal)

    # ── TAB 4: Flow Conditions ─────────────────────────────────────────────────
    with tabs[4]:
        if is_blocked:
            st.warning("Final operating conditions are unavailable for a blocked design.")
        else:
            _render_conditions(proposal)

    # ── TAB 5: Engineering Report ──────────────────────────────────────────────
    with tabs[5]:
        from components.council_report import render_council_report

        class _Candidate:
            def __init__(self, r):
                self.council_rounds = r.get("council_rounds", 0)
                self.safety_report = r.get("safety_report", {})
                self.council_messages = r.get("council_messages", [])

        render_council_report(_Candidate(result))

    # ── TAB 6: Raw JSON ───────────────────────────────────────────────────────
    with tabs[6]:
        st.json(result)

    with tabs[7]:
        from components.inventory_result import render_inventory_result

        render_inventory_result(result)

    # ── Feedback widget ────────────────────────────────────────────────────────
    from components.feedback import render_feedback_widget
    render_feedback_widget(result, context="translate")


# ── Helper renderers ──────────────────────────────────────────────────────────

def _render_chemistry_plan(plan: dict):
    if not plan:
        st.info("No chemistry plan available.")
        return

    st.markdown(f"### {plan.get('reaction_name', 'Unknown Reaction')}")
    st.markdown(
        f"**Class:** {plan.get('reaction_class', 'N/A')} | "
        f"**Mechanism:** {plan.get('mechanism_type', 'N/A')} | "
        f"**Bond formed:** {plan.get('bond_formed', 'N/A')}"
    )

    # Reagents
    reagents = plan.get("reagents", [])
    if reagents:
        st.markdown("#### Species Inventory")
        for r in reagents:
            role = r.get("role", "?")
            name = r.get("name", "?")
            amt = r.get("equiv_or_loading", "")
            notes = f" — *{r['notes']}*" if r.get("notes") else ""
            st.markdown(f"- **{name}** ({role}, {amt}){notes}")

    # Mechanism
    steps = plan.get("mechanism_steps", [])
    if steps:
        st.markdown("#### Reaction Mechanism")
        for step in steps:
            prefix = "**[hv]** " if step.get("is_photon_dependent") else ""
            rls = " *(rate-limiting)*" if step.get("is_rate_limiting") else ""
            st.markdown(f"{step.get('step_number', '?')}. {prefix}{step.get('description', '')}{rls}")
        if plan.get("key_intermediate"):
            st.info(f"Key intermediate: **{plan['key_intermediate']}**")

    # Sensitivities
    st.markdown("#### Sensitivities")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("O2 sensitive", "Yes" if plan.get("oxygen_sensitive") else "No",
              help="Reaction is INHIBITED by ambient O2 — requires degassing")
    c2.metric("O2 is reagent", "Yes" if plan.get("o2_is_reagent") else "No",
              help="O2 is consumed stoichiometrically — needs MFC + BPR + gas-liquid")
    c3.metric("Moisture sensitive", "Yes" if plan.get("moisture_sensitive") else "No")
    c4.metric("Temp sensitive", "Yes" if plan.get("temperature_sensitive") else "No")

    if plan.get("deoxygenation_required"):
        st.warning(f"Deoxygenation required: {plan.get('deoxygenation_reasoning', '')}")

    # Batch limitations + intensification strategy ──────────────────────
    limitations = plan.get("batch_limitations") or []
    mandate = plan.get("intensification_mandate") or {}
    if limitations or mandate:
        st.markdown("#### Intensification Strategy")
        from flora_translate.config import FLOW_TRANSLATION_POLICY

        target_is_hard = FLOW_TRANSLATION_POLICY == "intensify"
        # Top-line metrics
        m_target = (mandate.get("tau_reduction_target") if isinstance(mandate, dict) else 0) or 0
        m_advantage = (mandate.get("minimum_flow_advantage") if isinstance(mandate, dict) else "") or "—"
        m_regime = (mandate.get("required_mixing_regime") if isinstance(mandate, dict) else "") or "—"
        i1, i2, i3 = st.columns(3)
        i1.metric(
            "Target IF" if target_is_hard else "Enforced IF target",
            f"{float(m_target):.1f}×" if target_is_hard and m_target else "None",
            help=(
                "Active residence-time objective in explicit intensify mode."
                if target_is_hard
                else "Not enforced under evidence-first policy; measured evidence and final geometry control residence time."
            ),
        )
        i2.metric("Primary flow advantage", str(m_advantage).replace("_", " "))
        i3.metric("Mixing regime", str(m_regime).replace("_", " "))

        if limitations:
            st.markdown(
                "**Batch limitations identified** "
                + ("(used by explicit intensification policy):" if target_is_hard else "(screening hypotheses):")
            )
            # Render as chips with the empirical IF ceiling next to each
            _LIM_LABELS = {
                "mass_transfer_gas_liquid": ("Gas–liquid mass transfer", 20),
                "photon_penetration":       ("Photon penetration",       15),
                "heat_removal":             ("Heat removal",             10),
                "stirring_diffusion":       ("Stirring / diffusion",     12),
                "thermodynamic_equilibrium": ("Equilibrium (NOT removable)", 1.5),
                "kinetic":                  ("Intrinsic kinetics",       3),
            }
            chip_cols = st.columns(min(len(limitations), 4))
            for idx, lim in enumerate(limitations):
                label, if_ceiling = _LIM_LABELS.get(lim, (lim, None))
                chip_cols[idx % len(chip_cols)].markdown(
                    f"- **{label}**"
                    + (
                        f" — up to **{if_ceiling}×** in flow"
                        if target_is_hard and if_ceiling
                        else ""
                    )
                )
        reasoning = plan.get("batch_limitations_reasoning") or ""
        if reasoning:
            st.caption(f"How limitations were detected: {reasoning}")
        if not target_is_hard:
            st.caption(
                "Evidence-first policy is active. The limitation factor is explanatory and does not force the final residence time."
            )

        basis = mandate.get("flow_justification_basis") if isinstance(mandate, dict) else None
        if basis:
            with st.expander(
                "Why this target IF was chosen"
                if target_is_hard
                else "Flow-translation rationale"
            ):
                st.write(basis)

    # Stream logic
    slogic = plan.get("stream_logic", [])
    if slogic:
        st.markdown("#### Stream Separation Logic")
        for sl in slogic:
            st.markdown(f"**Stream {sl.get('stream_label', '?')}:** {', '.join(sl.get('reagents', []))}")
            if sl.get("reasoning"):
                st.caption(sl["reasoning"])

    incompat = plan.get("incompatible_pairs", [])
    if incompat:
        st.error(f"Incompatible pairs: {incompat}")

    if plan.get("recommended_wavelength_nm"):
        st.markdown(
            f"**Wavelength:** {plan['recommended_wavelength_nm']} nm "
            f"— {plan.get('wavelength_reasoning', '')}"
        )

    keywords = plan.get("retrieval_keywords", [])
    if keywords:
        st.caption(f"Retrieval keywords: {', '.join(keywords)}")


def _render_streams(proposal: dict, design_calc: dict | None = None):
    streams = proposal.get("streams", [])
    if not streams:
        st.info("No stream assignments available.")
        return

    # ── Engineering context from design calculator ─────────────────────────
    if design_calc:
        n_lim = design_calc.get("n_molar_flow_mmol_min")
        q_proposal = proposal.get("flow_rate_mL_min")

        # Classify streams: reactor feeds go into the main reactor,
        # quench streams are injected downstream.
        _QUENCH_KW = ("quench", "neutraliz", "workup", "post-reactor")

        def _is_quench(s):
            role = (s.get("pump_role") or "").lower()
            return any(kw in role for kw in _QUENCH_KW)

        def _norm_words(value):
            import re
            return set(re.sub(r"[^a-z0-9₂]+", " ", str(value).lower()).split())

        def _is_gas(s):
            phase_words = _norm_words(s.get("phase") or s.get("state") or "")
            role_words = _norm_words(s.get("pump_role") or "")
            text_words = set()
            for item in (s.get("contents") or []):
                text_words |= _norm_words(item)
            if role_words & {"quench", "neutralization", "neutralisation", "workup"}:
                return False
            if phase_words & {"gas", "gaseous", "vapor", "vapour"}:
                return True
            if phase_words & {"liquid", "solution"}:
                return False
            gas_words = {
                "air", "oxygen", "o2", "o₂", "hydrogen", "h2", "h₂", "co2", "co₂",
                "co", "monoxide", "syngas", "chlorine", "cl2", "cl₂", "ammonia",
                "nh3", "nh₃", "so2", "so₂", "ozone", "o3", "o₃", "mfc",
            }
            liquid_words = {"solution", "solvent", "aqueous", "dissolved", "substrate"}
            return bool((role_words | text_words) & gas_words) and not bool((role_words | text_words) & liquid_words)

        feed_rates = [
            s.get("flow_rate_mL_min") for s in streams
            if s.get("flow_rate_mL_min") and not _is_quench(s) and not _is_gas(s)
        ]
        quench_rates = [
            s.get("flow_rate_mL_min") for s in streams
            if s.get("flow_rate_mL_min") and _is_quench(s) and not _is_gas(s)
        ]
        q_reactor = round(sum(feed_rates), 4) if feed_rates else q_proposal
        q_quench_total = round(sum(quench_rates), 4)
        q_outlet = round((q_reactor or 0.0) + q_quench_total, 4) if q_quench_total else q_reactor

        n_metric_cols = 3 + (1 if q_quench_total else 0)
        ctx_cols = st.columns(n_metric_cols)
        col_i = 0
        if n_lim:
            ctx_cols[col_i].metric(
                "ṅ_limiting", f"{n_lim:.4f} mmol/min",
                help="Molar flow of limiting reagent",
            )
            col_i += 1
        if q_reactor:
            ctx_cols[col_i].metric(
                "Q_reactor_inlet", f"{q_reactor} mL/min",
                help="Σ feed pumps entering the main reactor (excludes quench streams)",
            )
            col_i += 1
        if q_quench_total:
            ctx_cols[col_i].metric(
                "Q_outlet (after quench)", f"{q_outlet} mL/min",
                help="Reactor outlet + quench stream(s)",
            )
            col_i += 1
        C_rxr = design_calc.get("C_reactor_M")
        if C_rxr and col_i < n_metric_cols:
            ctx_cols[col_i].metric(
                "C_reactor", f"{C_rxr:.3f} M",
                help="[limiting reagent] inside reactor after stream mixing",
            )
        st.divider()

    # ── Summary table ─────────────────────────────────────────────────────
    st.markdown("#### Pump / Stream Assignments")
    chief_derived = any(s.get("reasoning") for s in streams)
    if chief_derived:
        st.caption("Flowrates derived by the Chief Engineer from ṅ_limiting and feed concentrations. "
                   "Expand each pump below for the derivation formula.")

    for s in streams:
        label = s.get("stream_label", "?")
        role = s.get("pump_role", "")
        contents = s.get("contents", [])
        solvent = s.get("solvent", "")
        conc = s.get("concentration_M")
        rate = s.get("flow_rate_mL_min")
        gas_sccm = s.get("gas_flow_sccm")
        gas_actual = s.get("gas_flow_actual_mL_min")
        reasoning = s.get("reasoning", "")

        # Compact header with flowrate badge
        if gas_sccm:
            rate_badge = f" — **{gas_sccm} sccm**"
            hardware = "MFC"
        else:
            rate_badge = f" — **{rate} mL/min**" if rate else ""
            hardware = "Pump"
        with st.expander(f"{hardware} {label}: {role}{rate_badge}", expanded=True):
            if contents:
                cols = st.columns([1, 2])
                with cols[0]:
                    st.markdown("**Contents:**")
                    for item in contents:
                        st.markdown(f"- {item}")
                with cols[1]:
                    detail_rows = []
                    if solvent:
                        detail_rows.append(("Solvent", solvent))
                    if conc:
                        detail_rows.append(("Feed conc.", f"{conc} M"))
                    if rate:
                        detail_rows.append(("Flow rate", f"{rate} mL/min"))
                    if gas_sccm:
                        detail_rows.append(("Gas setpoint", f"{gas_sccm} sccm"))
                    if gas_actual:
                        detail_rows.append(("Gas actual", f"{gas_actual} mL/min at reactor"))
                    for k, v in detail_rows:
                        st.markdown(f"**{k}:** {v}")
            else:
                details = []
                if solvent:
                    details.append(f"Solvent: {solvent}")
                if conc:
                    details.append(f"{conc} M")
                if rate:
                    details.append(f"**{rate} mL/min**")
                if details:
                    st.markdown(" | ".join(details))

            if reasoning:
                st.markdown("**Chief derivation:**")
                st.code(reasoning, language=None)

    mixing = proposal.get("mixing_order_reasoning", "")
    if mixing:
        st.divider()
        st.markdown(f"**Mixing ({proposal.get('mixer_type', 'T-mixer')}):** {mixing}")

    pre = proposal.get("pre_reactor_steps", [])
    post = proposal.get("post_reactor_steps", [])
    if pre:
        st.divider()
        st.markdown("**Pre-reactor steps:**")
        for s in pre:
            st.markdown(f"- {s}")
    if post:
        st.markdown("**Post-reactor steps:**")
        for s in post:
            st.markdown(f"- {s}")


def _render_conditions(proposal: dict):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Residence Time", f"{proposal.get('residence_time_min', 0):.1f} min")
        st.metric("Flow Rate", f"{proposal.get('flow_rate_mL_min', 0):.2f} mL/min")
        st.metric("Temperature", f"{proposal.get('temperature_C', 25):.0f} °C")
        st.metric("Concentration", f"{proposal.get('concentration_M', 0):.3f} M")
    with c2:
        st.metric("Reactor Type", proposal.get("reactor_type", "N/A"))
        st.metric("Tubing Material", proposal.get("tubing_material", "N/A"))
        st.metric("Tubing ID", f"{proposal.get('tubing_ID_mm', 0):.1f} mm")
        st.metric("Reactor Volume", f"{proposal.get('reactor_volume_mL', 0):.1f} mL")
    with c3:
        st.metric("BPR", f"{proposal.get('BPR_bar', 0):.0f} bar")
        st.metric("Light Setup", proposal.get("light_setup", "N/A"))
        st.metric("Wavelength", f"{proposal.get('wavelength_nm', 'N/A')} nm")
        st.metric("Deoxygenation", proposal.get("deoxygenation_method", "N/A"))

    reasoning = proposal.get("reasoning_per_field", {})
    if reasoning:
        st.divider()
        st.markdown("#### Reasoning per Field")
        for field, reason in reasoning.items():
            st.markdown(f"- **{field}:** {reason}")


def _show_example():
    with st.expander("See an example", expanded=False):
        st.markdown("""
**Example input:**

> fac-Ir(ppy)3 (1 mol%) photocatalyzed decarboxylative radical addition
> of N-Boc-proline (1.0 equiv) to methyl vinyl ketone (2.0 equiv),
> K2HPO4 (1.5 equiv), DMF, 0.1 M, RT, 450 nm blue LED, N2, 24h, 72% yield.

**Expected output:**
- FEP coil reactor (1.0 mm ID, ~5 mL), inline N2 deoxygenation
- Pump A: proline + Ir(ppy)3 in DMF | Pump B: MVK + K2HPO4 in DMF
- Kessil 450 nm LED, BPR 5 bar
- Residence time ~8-12 min, flow rate ~0.5 mL/min
""")
