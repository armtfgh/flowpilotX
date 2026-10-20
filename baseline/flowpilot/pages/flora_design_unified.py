"""FlowPilot unified batch-to-flow translation and process design page."""

from pathlib import Path
import streamlit as st


def render():
    st.title("FlowPilot Design")
    st.markdown(
        "Describe your chemistry. FlowPilot will design a validated flow process "
        "grounded in literature. Works from a batch protocol or a free-text goal."
    )

    # ── Mode selector ──────────────────────────────────────────────────────
    mode = st.radio(
        "What do you have?",
        ["I have a batch protocol to convert",
         "I have a chemistry goal (no batch data)"],
        horizontal=True,
        key="design_mode",
    )

    if mode.startswith("I have a batch"):
        _render_translate()
    else:
        _render_design()


# ─────────────────────────────────────────────────────────────────────────────
# BATCH-TO-FLOW mode  — conversational chat interface
# ─────────────────────────────────────────────────────────────────────────────

def _init_chat_state():
    """Initialise session-state keys for the translate chat."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_agent" not in st.session_state:
        from flora_translate.conversation_agent import ConversationAgent
        st.session_state.chat_agent = ConversationAgent()
    if "active_result" not in st.session_state:
        st.session_state.active_result = None


def _render_translate():
    _init_chat_state()

    st.markdown("---")

    # ── Control bar ────────────────────────────────────────────────────────
    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.subheader("Batch → Flow Chat")
    with col_reset:
        if st.button("🔄 Reset", key="chat_reset", help="Start a new conversation"):
            st.session_state.chat_messages = []
            st.session_state.active_result = None
            st.session_state.chat_agent.reset()
            for key in list(st.session_state.keys()):
                if str(key).startswith("flowpilot_intake"):
                    del st.session_state[key]
            st.rerun()

    from components.intake_wizard import render_intake_wizard

    intake_package = render_intake_wizard("flowpilot_intake")
    if st.session_state.get("active_result") is None:
        if not intake_package or not intake_package.ready_for_design:
            st.info("Complete the standardized intake package before running FlowPilot design.")
            return

        from components.model_route_selector import render_model_route_selector

        runtime_options, route_blockers = render_model_route_selector(
            "flowpilot_design_models"
        )

        if st.button(
            "Run FlowPilot Design",
            type="primary",
            use_container_width=True,
            disabled=bool(route_blockers),
        ):
            with st.spinner("Running FlowPilot pipeline from standardized intake..."):
                try:
                    from flora_translate.main import translate

                    result = translate(
                        intake_package.raw_protocol,
                        intake_package=intake_package,
                        runtime_options=runtime_options,
                    )
                    from flora_translate.gui_autosave import autosave_gui_result

                    autosave_dir = autosave_gui_result(
                        result,
                        intake_package=intake_package,
                        source="standardized_intake",
                        user_input=intake_package.raw_protocol,
                    )
                    result["autosave_dir"] = str(autosave_dir)
                    st.session_state["active_result"] = result
                    st.session_state.chat_agent.current_result = result
                    st.session_state.chat_agent.original_query = intake_package.raw_protocol
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": "FlowPilot design generated from standardized intake.",
                        "result": result,
                        "questions": [],
                    })
                    st.rerun()
                except Exception as e:
                    from components.error_card import render_error

                    render_error(e, "FlowPilot standardized intake design")
                    return
        st.info("Intake is ready. Run FlowPilot Design to generate the first flow proposal.")
        return

    # ── Render chat history ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Design Chat")
    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show a compact result badge in the chat — no tabs inside chat bubble
            if msg.get("result"):
                _render_result_compact(msg["result"], i)
            if msg.get("questions"):
                st.info("**Questions for you:**\n" +
                        "\n".join(f"{j+1}. {q}" for j, q in enumerate(msg["questions"])))

    # ── Example prompts (only when chat is empty) ──────────────────────────
    if not st.session_state.chat_messages and st.session_state.get("active_result") is None:
        st.markdown(
            """
            <div style="color:#888; font-size:0.88em; margin-bottom:8px;">
            <b>Try one of these to get started:</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        examples = [
            "fac-Ir(ppy)₃ (1 mol%) photoredox, N-Boc-proline (1 equiv) + MVK (2 equiv), K₂HPO₄, DMF, 0.1 M, RT, 450 nm, N₂, 24 h, 72% yield",
            "Pd-catalyzed Suzuki coupling, ArBr + PhB(OH)₂, K₂CO₃, EtOH/H₂O 4:1, 80°C, 2h, 89% yield",
            "NaBH₄ reduction of ketone to alcohol, MeOH, 0°C, 30 min, quant. yield",
        ]
        for ex in examples:
            if st.button(ex[:80] + "…" if len(ex) > 80 else ex,
                         key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state["_prefill_input"] = ex
                st.rerun()

    # ── Chat input ─────────────────────────────────────────────────────────
    prefill = st.session_state.pop("_prefill_input", "")
    prompt  = st.chat_input(
        "Describe your batch protocol, ask a question, or request a revision…",
        key="chat_input",
    ) or (prefill if prefill else None)

    if prompt:
        # Show user message immediately
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with agent
        with st.chat_message("assistant"):
            agent = st.session_state.chat_agent

            # Determine if this will trigger a translation run
            has_result = agent.current_result is not None
            is_likely_translation = not has_result or any(
                kw in prompt.lower()
                for kw in ["translate", "convert", "new reaction", "fresh", "start over"]
            )

            spinner_msg = (
                "Running FlowPilot pipeline. This takes about 30-60 seconds..."
                if is_likely_translation or any(
                    kw in prompt.lower()
                    for kw in ["add", "remove", "change", "revise", "modify",
                                "instead", "try", "use", "switch"]
                )
                else "Thinking…"
            )

            with st.spinner(spinner_msg):
                try:
                    response = agent.process(prompt)
                except Exception as e:
                    response_msg = f"Something went wrong: {e}"
                    st.error(response_msg)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": response_msg}
                    )
                    st.stop()

            # Display assistant response
            st.markdown(response.message)

            if response.result:
                from flora_translate.gui_autosave import autosave_gui_result

                autosave_dir = autosave_gui_result(
                    response.result,
                    source="chat",
                    user_input=prompt,
                )
                response.result["autosave_dir"] = str(autosave_dir)
                st.session_state["active_result"] = response.result
                _render_result_compact(response.result, "new")

            if response.questions:
                st.info("**I have a few questions to improve the design:**\n" +
                        "\n".join(f"{j+1}. {q}" for j, q in enumerate(response.questions)))

            if response.error:
                from components.error_card import render_error
                render_error(Exception(response.error), "FlowPilot translation")

        # Persist to history
        st.session_state.chat_messages.append({
            "role":      "assistant",
            "content":   response.message,
            "result":    response.result,
            "questions": response.questions,
        })

    # ── Full result view — OUTSIDE chat bubbles (no nested tabs issue) ─────
    active = st.session_state.get("active_result")
    if active:
        st.markdown("---")
        _render_result(active, key_prefix="active")


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN-FROM-GOAL mode
# ─────────────────────────────────────────────────────────────────────────────

def _render_design():
    st.markdown("---")
    st.subheader("Chemistry Goal")

    goal = st.text_area(
        "Describe what you want to do",
        height=140,
        key="d_goal",
        placeholder=(
            "e.g. Design a flow process for Ir(ppy)3-catalyzed photoredox "
            "radical addition of alpha-amino acids to electron-poor alkenes "
            "in MeCN at room temperature, ~0.5 mL/min scale."
        ),
    )

    if st.button("Design Flow Process", type="primary",
                 disabled=not (goal or "").strip(),
                 use_container_width=True, key="d_run"):
        with st.spinner("Designing your flow process…"):
            try:
                from flora_design.main import design
                result = design(goal)
                st.session_state["flora_design_result"] = result
                st.session_state["flora_result_type"] = "design"
            except Exception as e:
                from components.error_card import render_error
                render_error(e, "FlowPilot design")
                return

    if st.session_state.get("flora_result_type") == "design":
        _render_design_result(st.session_state["flora_design_result"])


# ─────────────────────────────────────────────────────────────────────────────
# Compact result badge (shown INSIDE chat bubble — no tabs, no nested widgets)
# ─────────────────────────────────────────────────────────────────────────────

def _result_with_contract_disposition(result: dict, final_design: dict) -> dict:
    """Return a display view whose status cannot disagree with final closure."""

    if final_design.get("status") == "executable":
        return result
    issues = (final_design.get("consistency") or {}).get("issues") or []
    view = dict(result)
    view["recommended_disposition"] = "BLOCK"
    view["reported_disposition"] = "BLOCK"
    view["disposition_rationale"] = (
        "The post-validation final-design contract did not close; run parameters "
        "and the executable diagram are withheld."
    )
    view["design_disposition"] = {
        **dict(result.get("design_disposition") or {}),
        "recommended_disposition": "BLOCK",
        "rationale": view["disposition_rationale"],
        "hard_failures": [
            {
                "finding_id": issue.get("code", "FINAL-CHECK"),
                "message": issue.get("message", "Final consistency check failed."),
            }
            for issue in issues
        ],
    }
    return view


def _stored_or_current_final_design(result: dict) -> dict:
    """Preserve a frozen v2 contract; rebuild only legacy/uncontracted results."""

    stored = result.get("final_design") or {}
    if stored.get("schema_version") == "flowpilot_final_design_v2.0":
        return stored
    from flora_translate.final_design_contract import build_final_design_contract

    return build_final_design_contract(result)


def _render_result_compact(result: dict, key_suffix):
    """Show a small summary card inside a chat message — no tabs, no downloads."""
    from components.design_disposition import render_design_disposition
    final_design = _stored_or_current_final_design(result)
    render_design_disposition(
        _result_with_contract_disposition(result, final_design), compact=True
    )
    if final_design.get("status") != "executable":
        confirmation_required = (
            result.get("design_status") == "inventory_confirmation_required"
        )
        st.caption(
            "Inventory confirmation is required before numerical design. A "
            "requirements topology is available below."
            if confirmation_required
            else "No executable parameters were produced. A diagnostic "
            "requirements topology is available below."
        )
        unresolved = (
            (result.get("inventory_allocation") or {}).get("unresolved_requirements")
            or []
        )
        if unresolved:
            st.markdown("**Unresolved inventory requirements**")
            for item in unresolved:
                st.warning(
                    f"{item.get('operation_id', 'process')}: {item.get('reason', 'Missing equipment')}"
                )
        if result.get("autosave_dir"):
            st.caption(f"Autosaved: {result['autosave_dir']}")
        return
    proposal = final_design.get("parameters") or {}
    conf     = result.get("confidence", "?")
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    rt  = proposal.get("residence_time_min", "?")
    rxt = proposal.get("reactor_type", "?")
    fr  = proposal.get("flow_rate_mL_min", "?")
    st.markdown(
        f"**Confidence:** :{conf_color}[{conf}] &nbsp;|&nbsp; "
        f"**Reactor:** {rxt} &nbsp;|&nbsp; "
        f"**τ =** {rt} min &nbsp;|&nbsp; "
        f"**Q =** {fr} mL/min"
    )
    st.caption("↓ Full design with process diagram, chemistry plan, and conditions shown below")
    if result.get("autosave_dir"):
        st.caption(f"Autosaved: {result['autosave_dir']}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared result renderer (translate output) — shown OUTSIDE chat bubbles
# ─────────────────────────────────────────────────────────────────────────────

def _render_result(result: dict, key_prefix: str = ""):
    from components.design_disposition import render_design_disposition
    from flora_translate.final_design_contract import (
        canonical_proposal,
    )

    final_design = _stored_or_current_final_design(result)
    result["final_design"] = final_design
    proposal = canonical_proposal(result)
    is_blocked = final_design["status"] != "executable"
    confirmation_required = (
        result.get("design_status") == "inventory_confirmation_required"
    )
    conf = result.get("confidence", "LOW")
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    render_design_disposition(_result_with_contract_disposition(result, final_design))
    if is_blocked:
        if confirmation_required:
            st.markdown("### Design status: :orange[INVENTORY CONFIRMATION REQUIRED]")
        else:
            st.markdown("### Candidate status: :red[REJECTED]")
        st.caption(
            f"Model confidence was {conf}, but deterministic feasibility gates "
            "override model confidence."
        )
    else:
        st.markdown(f"### Confidence: :{conf_color}[{conf}]")
    if result.get("autosave_dir"):
        st.caption(f"Autosaved run folder: {result['autosave_dir']}")

    tabs = st.tabs([
        "Summary",
        "Engineering Design",
        "Process Diagram",
        "Chemistry Plan & Recipe",
        "Stream Assignments",
        "Council Deliberation",
        "Council Report",
        "Experiment Loop",
        "Raw JSON",
        "Equipment & Inventory",
        "Process summary",
        "Responses",
    ])

    # ── Tab 0: Summary ────────────────────────────────────────────────────────
    with tabs[0]:
        _render_summary(result, final_design)

    # ── Tab 1: Engineering Design ─────────────────────────────────────────────
    with tabs[1]:
        if is_blocked:
            _render_reconciliation(final_design)
        else:
            from components.result_reporting import render_engineering_history
            render_engineering_history(result)
            design_calc = result.get("design_calculations")
            if design_calc and st.checkbox(
                "Show pre-final calculation audit trail",
                value=False,
                key=f"{key_prefix}_show_calc_audit",
            ):
                st.caption(
                    "Supporting calculator trace. The final values above are authoritative."
                )
                from components.design_steps import render_design_steps
                render_design_steps(design_calc, key_prefix=f"{key_prefix}_ds")

        # Before vs After council comparison table
        pre = result.get("pre_council_proposal")
        if not is_blocked and pre and proposal:
            _render_before_after_table(pre, proposal, result.get("deliberation_log"))

    # ── Tab 2: Process Diagram ────────────────────────────────────────────────
    with tabs[2]:
        if is_blocked:
            st.warning(
                "REQUIREMENTS TOPOLOGY - NOT EXECUTABLE. This diagram shows the "
                "required process structure and unresolved inventory assignments; "
                "it contains no approved run instructions."
            )
            diagnostic_topology = (
                result.get("diagnostic_topology")
                or result.get("process_requirements_topology")
                or {}
            )
            from components.process_diagram import render_process_diagram
            render_process_diagram(
                result.get("diagnostic_svg_path", ""),
                result.get("diagnostic_png_path", ""),
                key_prefix=f"{key_prefix}_diagnostic",
                topology=diagnostic_topology,
                render_manifest=(
                    result.get("diagnostic_diagram_render_manifest") or {}
                ),
            )
            _render_reconciliation(final_design)
        else:
            from components.process_diagram import render_process_diagram
            from flora_translate.diagram_views import current_diagram
            current_svg = current_diagram(result, "process-svg")
            current_png = current_diagram(result, "process-png")
            render_process_diagram(
                str(current_svg) if current_svg else result.get("svg_path", ""),
                str(current_png) if current_png else result.get("png_path", ""),
                key_prefix=key_prefix,
                topology=result.get("process_topology") or {},
                render_manifest=result.get("diagram_render_manifest") or {},
            )
        topo = (
            result.get("diagnostic_topology")
            or result.get("process_requirements_topology")
            or {}
            if is_blocked
            else result.get("process_topology", {})
        )
        if topo and topo.get("unit_operations"):
            st.divider()
            st.markdown(
                "#### Required Unit Operations"
                if is_blocked
                else "#### Unit Operations"
            )
            for i, op in enumerate(topo.get("unit_operations", []), 1):
                if op.get("op_type") == "led_module":
                    continue
                rationale = op.get("rationale") or ""
                label = op.get("label", "?")
                # Flag separator/filter/special nodes that lack a rationale
                missing_rationale = (
                    not rationale
                    and op.get("op_type") not in ("pump", "mfc", "mixer", "collector",
                                                   "coil_reactor", "photoreactor",
                                                   "reactor", "bpr")
                )
                expander_label = f"{i}. {label}"
                if missing_rationale:
                    expander_label += " ⚠️"
                with st.expander(expander_label, expanded=missing_rationale):
                    # Why it's here
                    if rationale:
                        st.info(f"**Why:** {rationale}")
                    elif missing_rationale:
                        st.warning(
                            "No rationale provided for this module. "
                            "If you did not request this step, it may have been "
                            "inferred incorrectly — check the chemistry plan."
                        )
                    # Parameters
                    p = op.get("parameters", {})
                    param_items = [(k.replace("sccm", "inlet_STP_mL_min"), v) for k, v in p.items()
                                   if v is not None and k not in ("light_required", "gas_holdup", "liquid_holdup_volume_mL")
                                   and "in_channel" not in k and "actual" not in k
                                   and not (op.get("op_type") == "mfc" and k == "flow_rate_mL_min")]
                    if param_items:
                        for k, v in param_items:
                            st.markdown(f"**{k.replace('_', ' ')}:** {v}")
            if topo.get("pid_description"):
                st.code(topo["pid_description"])

    # ── Tab 3: Chemistry Plan & Recipe ────────────────────────────────────────
    with tabs[3]:
        from pages.translate import _render_chemistry_plan
        st.info(
            "Upstream chemistry analysis is retained for audit and hypothesis "
            "review. It is not an executable authority; the canonical procedure "
            "below is compiled from the final validated design."
        )
        _render_chemistry_plan(result.get("chemistry_plan", {}))
        if is_blocked:
            st.warning(
                "Experimental recipe withheld because the current candidate is blocked."
            )
        else:
            st.divider()
            _render_canonical_procedure(final_design)

    # ── Tab 4: Stream Assignments ─────────────────────────────────────────────
    with tabs[4]:
        if is_blocked:
            st.warning(
                "No final stream assignments are available because the design is blocked."
            )
            st.caption("Resolve consistency and inventory failures before using pump setpoints.")
        else:
            st.caption(
                "These assignments belong to the validated final design shown in "
                "Summary and Engineering Design."
            )
            from components.result_reporting import render_process_summary
            render_process_summary(result, streams_only=True)

    # ── Tab 5: Council Deliberation ───────────────────────────────────────────
    with tabs[5]:
        _render_council_deliberation(result)

    # ── Tab 6: Council Report ─────────────────────────────────────────────────
    with tabs[6]:
        from components.council_report import render_council_report
        class _C:
            def __init__(self, r):
                self.council_rounds = r.get("council_rounds", 0)
                self.safety_report  = r.get("safety_report", {})
                self.council_messages = r.get("council_messages", [])
        render_council_report(_C(result))

    # ── Tab 7: Experiment Loop ────────────────────────────────────────────────
    with tabs[7]:
        if is_blocked:
            st.warning(
                "Resolve the inventory and engineering blockers before starting "
                "an experimental feedback cycle."
            )
        else:
            _render_experiment_loop(result, proposal=proposal, key_prefix=key_prefix)

    # ── Tab 8: Raw JSON ───────────────────────────────────────────────────────
    with tabs[8]:
        st.markdown("### Authoritative Final Design")
        st.json(final_design)
        with st.expander("Complete pipeline audit JSON", expanded=False):
            st.json(result)

    # ── Tab 9: Equipment & Inventory ─────────────────────────────────────────
    with tabs[9]:
        from components.inventory_result import render_inventory_result

        render_inventory_result(result)

    with tabs[10]:
        from components.result_reporting import render_process_summary
        render_process_summary(result)
    with tabs[11]:
        from components.result_reporting import render_responses
        render_responses(result)

    from components.feedback import render_feedback_widget
    render_feedback_widget(result, context="flora_design_translate")


def _render_experiment_loop(
    result: dict,
    proposal: dict | None = None,
    key_prefix: str = "",
):
    """Closed-loop experiment entry and deterministic next-design refinement."""
    from flora_translate.experiment_loop import (
        ActualConditions,
        ExperimentalOutcomes,
        ExperimentResult,
        refine_from_experiment,
    )

    proposal = proposal or result.get("proposal", {})
    loop_key = f"{key_prefix or 'result'}_experiment_loop"
    campaign = st.session_state.setdefault(
        loop_key,
        {"cycles": [], "design_versions": [result]},
    )

    design_version = int(result.get("design_version", proposal.get("design_version", 1)) or 1)
    st.markdown("### Experimental Feedback Loop")
    st.caption(
        "Enter what actually happened in the lab. FlowPilot will diagnose the gap "
        "and create the next design version while keeping previous cycles in history."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current version", f"v{design_version}")
    tau_label = "Target tau (inlet/STP)" if _proposal_has_gas(proposal) else "Target tau"
    c2.metric(tau_label, f"{proposal.get('residence_time_min', '?')} min")
    c3.metric("Target Q", f"{proposal.get('flow_rate_mL_min', '?')} mL/min")
    c4.metric("Target C", f"{proposal.get('concentration_M', '?')} M")

    form_key = f"{loop_key}_form_v{design_version}_{len(campaign.get('cycles', []))}"
    with st.form(form_key):
        st.markdown("#### Actual Run Conditions")
        a1, a2, a3, a4 = st.columns(4)
        residence_time = a1.number_input(
            "Residence time at inlet/STP (min)" if _proposal_has_gas(proposal) else "Residence time (min)",
            min_value=0.0,
            value=_float_default(proposal.get("residence_time_min"), 10.0),
            step=0.5,
            key=f"{form_key}_tau",
        )
        flow_rate = a2.number_input(
            "Flow rate (mL/min)",
            min_value=0.0,
            value=_float_default(proposal.get("flow_rate_mL_min"), 0.1),
            step=0.01,
            format="%.5f",
            key=f"{form_key}_q",
        )
        temperature = a3.number_input(
            "Temperature (C)",
            value=_float_default(proposal.get("temperature_C"), 25.0),
            step=1.0,
            key=f"{form_key}_temp",
        )
        concentration = a4.number_input(
            "Concentration (M)",
            min_value=0.0,
            value=_float_default(proposal.get("concentration_M"), 0.1),
            step=0.01,
            format="%.4f",
            key=f"{form_key}_conc",
        )

        b1, b2, b3, b4 = st.columns(4)
        reactor_volume = b1.number_input(
            "Reactor volume (mL)",
            min_value=0.0,
            value=_float_default(proposal.get("reactor_volume_mL"), residence_time * flow_rate),
            step=0.1,
            key=f"{form_key}_vol",
        )
        tubing_id = b2.number_input(
            "Tubing ID (mm)",
            min_value=0.0,
            value=_float_default(proposal.get("tubing_ID_mm"), 1.0),
            step=0.1,
            key=f"{form_key}_id",
        )
        bpr = b3.number_input(
            "BPR (bar)",
            min_value=0.0,
            value=_float_default(proposal.get("BPR_bar"), 0.0),
            step=1.0,
            key=f"{form_key}_bpr",
        )
        wavelength = b4.number_input(
            "Wavelength (nm)",
            min_value=0.0,
            value=_float_default(proposal.get("wavelength_nm"), 0.0),
            step=10.0,
            key=f"{form_key}_wl",
        )

        st.markdown("#### Gas-Liquid Timing")
        gas_stream = _first_gas_stream(proposal)
        g1, g3, g4 = st.columns(3)
        substrate_flow = g1.number_input(
            "Substrate flow (mL/min)",
            min_value=0.0,
            value=_float_default(proposal.get("flow_rate_mL_min"), flow_rate),
            step=0.001,
            format="%.5f",
            key=f"{form_key}_substrate_q",
        )
        gas_stp = g3.number_input(
            "Gas inlet/STP (mL/min)",
            min_value=0.0,
            value=_float_default(gas_stream.get("gas_flow_sccm"), 0.0),
            step=0.001,
            format="%.5f",
            key=f"{form_key}_gas_stp",
        )
        gas_equiv = g4.number_input(
            "Active gas equiv at inlet/STP",
            min_value=0.0,
            value=_float_default(gas_stream.get("molar_equiv"), 0.0),
            step=0.1,
            format="%.3f",
            key=f"{form_key}_gas_equiv",
        )

        t_inlet = st.number_input(
            "t inlet (min)",
            min_value=0.0,
            value=_time_default(reactor_volume, substrate_flow, gas_stp),
            step=1.0,
            format="%.3f",
            key=f"{form_key}_t_inlet",
        )
        default_basis_index = 0 if gas_stp > 0 else 1
        basis_choice = st.selectbox(
            "Primary calibration basis",
            [
                "inlet/STP apparent residence time",
                "liquid-only reactor volume / liquid flow",
            ],
            index=default_basis_index,
            key=f"{form_key}_basis",
        )

        st.markdown("#### Experimental Outcome")
        o1, o2, o3, o4, o5 = st.columns(5)
        yield_pct = o1.number_input("Yield (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"{form_key}_yield")
        product_pct = o2.number_input("Product (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"{form_key}_product")
        starting_material_pct = o3.number_input("Starting material (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"{form_key}_sm")
        conversion_pct = o4.number_input("Conversion (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"{form_key}_conv")
        selectivity_pct = o5.number_input("Selectivity (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"{form_key}_sel")

        pressure_bar = st.number_input("Observed pressure (bar)", min_value=0.0, value=0.0, step=0.5, key=f"{form_key}_pressure")

        p1, p2, p3, p4 = st.columns(4)
        pressure_drift = p1.number_input("Pressure drift (bar)", min_value=0.0, value=0.0, step=0.5, key=f"{form_key}_drift")
        clogging = p2.checkbox("Clogging observed", key=f"{form_key}_clog")
        precipitation = p3.checkbox("Precipitation observed", key=f"{form_key}_ppt")
        gas_state = p4.selectbox(
            "Gas-liquid stability",
            ["stable", "unknown", "slugging", "flooding"],
            key=f"{form_key}_gas",
        )

        impurity_notes = st.text_input("Impurity / analytical notes", key=f"{form_key}_impurity")
        free_text = st.text_area("Additional observations", height=90, key=f"{form_key}_notes")

        submitted = st.form_submit_button("Analyze Result and Create Next Design", type="primary")

    if submitted:
        run_index = len(campaign.get("cycles", [])) + 1
        if basis_choice.startswith("inlet"):
            submitted_residence_time = t_inlet or residence_time
        else:
            submitted_residence_time = residence_time
        experiment = ExperimentResult(
            run_id=f"run_{run_index:02d}",
            design_version=design_version,
            actual_conditions=ActualConditions(
                residence_time_min=submitted_residence_time,
                residence_time_inlet_min=t_inlet or None,
                residence_time_basis=basis_choice,
                flow_rate_mL_min=flow_rate,
                substrate_flow_mL_min=substrate_flow or None,
                gas_flow_stp_mL_min=gas_stp or None,
                gas_equiv_inlet=gas_equiv or None,
                temperature_C=temperature,
                concentration_M=concentration,
                reactor_volume_mL=reactor_volume,
                tubing_ID_mm=tubing_id,
                BPR_bar=bpr,
                wavelength_nm=wavelength or None,
            ),
            outcomes=ExperimentalOutcomes(
                yield_pct=yield_pct or None,
                product_pct=product_pct or None,
                starting_material_pct=starting_material_pct or None,
                conversion_pct=conversion_pct or None,
                selectivity_pct=selectivity_pct or None,
                pressure_bar=pressure_bar or None,
                pressure_drift_bar=pressure_drift or None,
                clogging_observed=clogging,
                precipitation_observed=precipitation,
                gas_liquid_stability=gas_state,
                impurity_notes=impurity_notes,
                notes=free_text,
            ),
            free_text_observations=free_text,
        )
        closed_loop = refine_from_experiment(
            result,
            experiment,
            campaign_history=campaign.get("cycles", []),
        )
        campaign.setdefault("cycles", []).append(closed_loop.model_dump())
        campaign.setdefault("design_versions", []).append(closed_loop.refined_result)
        st.session_state[loop_key] = campaign
        st.session_state["active_result"] = closed_loop.refined_result
        st.success(f"Created design v{closed_loop.decision.design_version_out}.")
        st.rerun()

    cycles = campaign.get("cycles", [])
    if not cycles:
        st.info("No experimental cycles entered yet.")
        return

    st.divider()
    st.markdown("### Closed-Loop History")
    rows = []
    for item in cycles:
        exp = item.get("experiment", {})
        decision = item.get("decision", {})
        outcomes = exp.get("outcomes", {})
        rows.append({
            "Run": exp.get("run_id"),
            "Design In": decision.get("design_version_in"),
            "Design Out": decision.get("design_version_out"),
            "Yield": outcomes.get("yield_pct"),
            "Conversion": outcomes.get("conversion_pct"),
            "Selectivity": outcomes.get("selectivity_pct"),
            "Score": decision.get("score"),
            "Status": decision.get("status"),
            "Failure Modes": ", ".join(decision.get("failure_modes", [])),
        })
    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    last = cycles[-1]
    decision = last.get("decision", {})
    st.markdown("### Latest Diagnosis")
    st.info(decision.get("diagnosis", "No diagnosis available."))
    for action in decision.get("recommended_actions", []):
        st.markdown(f"- {action}")

    changes = decision.get("parameter_changes", {})
    changed_rows = [
        {"Parameter": key, "Before": value.get("old"), "Next": value.get("new")}
        for key, value in changes.items()
        if value.get("changed")
    ]
    if changed_rows:
        import pandas as pd
        st.markdown("#### Parameter Changes")
        st.dataframe(pd.DataFrame(changed_rows), hide_index=True, use_container_width=True)

    st.markdown("#### Next Experiment Package")
    st.json(decision.get("next_experiment", {}))
    calibration = (decision.get("next_experiment", {}) or {}).get("evidence_calibration")
    if calibration:
        st.markdown("#### Evidence-Calibrated Design Ladder")
        st.caption(
            "When multiple experimental cycles are available, measured response "
            "at the recorded calibration basis can revise the original residence-time estimate."
        )
        c1, c2, c3, c4 = st.columns(4)
        st.caption(f"Recorded basis: {calibration.get('residence_time_basis', 'not recorded')}")
        c1.metric("Best observed tau", f"{calibration.get('best_tau_inlet_min')} min at inlet/STP")
        c2.metric("Best response", f"{calibration.get('best_response_pct')}%")
        c3.metric("Next tau", f"{calibration.get('recommended_tau_inlet_min')} min at inlet/STP")
        c4.metric("Target estimate", f"{calibration.get('target_tau_inlet_min')} min at inlet/STP")
        ladder = calibration.get("design_ladder") or []
        if ladder:
            import pandas as pd
            st.dataframe(pd.DataFrame(ladder), hide_index=True, use_container_width=True)

    import json
    st.download_button(
        "Download closed-loop campaign JSON",
        json.dumps(campaign, indent=2, default=str),
        "flora_closed_loop_campaign.json",
        "application/json",
        key=f"{loop_key}_download",
    )


def _float_default(value, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_gas_stream(proposal: dict) -> dict:
    for stream in proposal.get("streams") or []:
        if str(stream.get("phase", "")).lower() == "gas":
            return stream
    return {}


def _time_default(volume_mL: float, liquid_q: float, gas_q: float) -> float:
    total_q = _float_default(liquid_q, 0.0) + _float_default(gas_q, 0.0)
    if total_q <= 0:
        return 0.0
    return float(volume_mL) / total_q


def _render_parameter_metrics(parameters: dict):
    """Render the shared final parameter set used by every result tab."""

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Residence Time (inlet/STP)",
        _metric_value(parameters.get("residence_time_inlet_min"), "min"),
    )
    c2.metric("Concentration", _metric_value(parameters.get("concentration_M"), "M"))
    c3.metric("Liquid Flow Rate", _metric_value(parameters.get("flow_rate_mL_min"), "mL/min"))
    c4.metric("Total Reactor Volume", _metric_value(parameters.get("reactor_volume_mL"), "mL"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Tubing ID", _metric_value(parameters.get("tubing_ID_mm"), "mm"))
    c6.metric("Temperature", _metric_value(parameters.get("temperature_C"), "C"))
    c7.metric("BPR", _metric_value(parameters.get("BPR_bar"), "bar"))
    c8.metric("Wavelength", _metric_value(parameters.get("wavelength_nm"), "nm"))
    st.caption(f"Residence-time basis: {parameters.get('residence_time_basis') or 'not specified'}")


def _render_stage_table(stages: list[dict]):
    if not stages:
        return
    import pandas as pd

    rows = []
    for stage in stages:
        rows.append(
            {
                "Stage": stage.get("stage_number"),
                "Name": stage.get("stage_name"),
                "Reactor ID": stage.get("reactor_equipment_id"),
                "Light ID": stage.get("light_equipment_id"),
                "Volume (mL)": stage.get("reactor_volume_mL"),
                "Material": stage.get("material"),
                "Tubing ID (mm)": stage.get("d_mm") or stage.get("tubing_ID_mm"),
                "Temperature (C)": stage.get("temperature_C"),
                "Wavelength (nm)": stage.get("wavelength_nm"),
                "Q liquid (mL/min)": stage.get("Q_liquid_mL_min"),
                "Q gas inlet/STP (mL/min)": stage.get("Q_gas_sccm"),
                "t inlet/STP (min)": stage.get("residence_time_inlet_min"),
            }
        )
    st.markdown("### Validated Stage Design")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_final_engineering(final_design: dict):
    st.markdown("### Final Engineering Design")
    if len(final_design.get("stages") or []) <= 1:
        _render_parameter_metrics(final_design.get("parameters") or {})
    _render_stage_table(final_design.get("stages") or [])
    manifest = final_design.get("instrument_manifest") or []
    if manifest:
        st.markdown("### Instrument Manifest")
        st.dataframe(manifest, hide_index=True, use_container_width=True)


def _render_reconciliation(final_design: dict):
    """Explain why no executable design exists without promoting stale values."""

    st.error(
        "FlowPilot withheld run parameters because the final proposal, engineering "
        "calculation, topology, and inventory assignment did not close to one design."
    )
    issues = (final_design.get("consistency") or {}).get("issues") or []
    for issue in issues:
        st.markdown(f"- **{issue.get('code', 'FINAL-CHECK')}**: {issue.get('message', '')}")

    diagnostic = final_design.get("diagnostic") or {}
    with st.expander("Intermediate values for diagnosis only", expanded=False):
        st.warning(diagnostic.get("message") or "These values are not run instructions.")
        preliminary = diagnostic.get("preliminary_candidate") or {}
        if preliminary:
            st.json(preliminary)
        stages = diagnostic.get("stage_requirements") or []
        if stages:
            import pandas as pd
            st.dataframe(pd.DataFrame(stages), hide_index=True, use_container_width=True)


def _metric_value(value, unit: str) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:g} {unit}".strip()
    return f"{value} {unit}".strip()


def _render_summary(result: dict, final_design: dict):
    """Render only the canonical post-validation design contract."""
    proposal = final_design.get("parameters") or {}

    if final_design.get("status") != "executable":
        st.markdown("### No Executable Final Design")
        _render_reconciliation(final_design)
    else:
        st.markdown("### Final Design Parameters")
        if len(final_design.get("stages") or []) <= 1:
            _render_parameter_metrics(proposal)
        _render_stage_table(final_design.get("stages") or [])

    intensification = final_design.get("intensification") or {}
    if intensification:
        st.divider()
        st.markdown("### Intensification Status")
        i1, i2, i3 = st.columns(3)
        target = intensification.get("target_factor")
        realized = intensification.get("realized_factor")
        i1.metric("Policy", intensification.get("policy") or "evidence_first")
        i2.metric("Enforced target", f"{target:g}x" if target else "None")
        i3.metric("Realized final IF", f"{realized:g}x" if realized else "Not available")
        if intensification.get("applied_as_hard_constraint"):
            st.info("The target factor is active because explicit intensification mode is enabled.")
        else:
            st.caption(
                "No intensification factor is enforced under the evidence-first "
                "policy. Realized IF is reported only after an executable design closes."
            )

    realization = result.get("design_realization") or {}
    if realization:
        st.divider()
        st.markdown("### Final Design Basis")
        st.caption(
            "Post-council deterministic realization. These decisions produced "
            "the final values shown above."
        )
        for decision in realization.get("decisions") or []:
            kind = str(decision.get("decision") or "engineering decision").replace("_", " ").title()
            if kind == "Liquid Flow Solution":
                st.markdown(
                    f"- **{kind}:** total liquid flow "
                    f"{decision.get('total_reactive_liquid_flow_mL_min', 'N/A')} mL/min "
                    "from stoichiometric ratios and per-pump limits."
                )
            elif kind == "Reactor Selection":
                st.markdown(
                    f"- **{kind}:** `{decision.get('equipment_id', 'unresolved')}`, "
                    f"{decision.get('volume_mL', 'N/A')} mL."
                )
            elif kind == "Pressure Selection":
                st.markdown(
                    f"- **{kind}:** {decision.get('selected_BPR_bar', 'N/A')} bar "
                    "from declared pressure hardware."
                )
            elif kind in {
                "Single Stage Residence Time Closure",
                "Stage Reactor Selection",
            }:
                st.markdown(f"- **{kind}:** numerical and inventory closure verified.")

        safety = final_design.get("safety") or {}
        controls = [
            item.get("description")
            for item in safety.get("controls") or []
            if item.get("required")
        ]
        if controls:
            st.markdown("### Required Safety Controls")
            for control in controls:
                st.warning(control)

    # ── AI narrative in expander ──────────────────────────────────────────────
    explanation = result.get("explanation", "")
    if explanation:
        st.divider()
        with st.expander("Pre-realization model narrative (audit only)", expanded=False):
            st.caption(
                "This narrative records upstream/council reasoning and may contain "
                "superseded candidate values. It is not a run instruction."
            )
            st.markdown(explanation)
            notes = proposal.get("chemistry_notes", "")
            if notes:
                st.info(notes)


def _render_before_after_table(pre: dict, post: dict, delib_log: dict | None):
    """Show a before/after council comparison for key design parameters."""
    st.divider()
    st.markdown("### Candidate vs Final Realized Design")
    st.caption("Initial model candidate versus the deterministic, inventory-validated design.")

    fields = [
        ("residence_time_min",  "τ (Residence Time)",   "min"),
        ("flow_rate_mL_min",    "Q (Flow Rate)",        "mL/min"),
        ("tubing_ID_mm",        "d (Tubing ID)",        "mm"),
        ("reactor_volume_mL",   "V_R (Reactor Volume)", "mL"),
        ("tubing_material",     "Material",             ""),
        ("temperature_C",       "Temperature",          "°C"),
        ("BPR_bar",             "BPR",                  "bar"),
        ("wavelength_nm",       "Wavelength",           "nm"),
        ("concentration_M",     "Concentration",        "M"),
        ("deoxygenation_method","Deoxygenation",        ""),
    ]

    changed_fields = set((delib_log or {}).get("all_changes_applied", {}).keys())

    rows = []
    for field, label, unit in fields:
        v_pre  = pre.get(field)
        v_post = post.get(field)
        if v_pre is None and v_post is None:
            continue

        def _fmt(v, u):
            if v is None:
                return "—"
            try:
                return f"{float(v):.4g} {u}".strip() if u else str(v)
            except (TypeError, ValueError):
                return str(v)

        changed = field in changed_fields or _fmt(v_pre, unit) != _fmt(v_post, unit)
        rows.append({
            "Parameter": label,
            "Initial Candidate": _fmt(v_pre, unit),
            "Final Realized":  _fmt(v_post, unit),
            "Changed": "★" if changed else "",
        })

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Changed": st.column_config.TextColumn("★", width="small"),
            },
        )


def _render_design_result(result):
    """Render output from FLORA-Design (from-goal mode)."""
    topo = result.topology
    conf = topo.topology_confidence
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    st.markdown(f"### Confidence: :{conf_color}[{conf}]")

    feats = result.chem_features
    st.info(
        f"**{feats.reaction_class.replace('_', ' ').title()}** · "
        f"Catalyst: {feats.photocatalyst or '?'} · "
        f"λ = {feats.wavelength_nm or '?'} nm"
    )

    tabs = st.tabs(["Summary", "Process Diagram", "Engineering", "Equipment", "Raw JSON"])

    with tabs[0]:
        st.markdown(result.explanation)
        c1, c2, c3 = st.columns(3)
        c1.metric("Residence time", f"{topo.residence_time_min:.1f} min")
        c2.metric("Flow rate", f"{topo.total_flow_rate_mL_min:.2f} mL/min")
        c3.metric("Reactor volume", f"{topo.reactor_volume_mL:.1f} mL")
        st.divider()
        st.code(topo.pid_description)
        for i, op in enumerate(topo.unit_operations, 1):
            with st.expander(f"{i}. {op.label}"):
                st.json(op.parameters)
                st.caption(op.rationale)

    with tabs[1]:
        from components.process_diagram import render_process_diagram
        render_process_diagram(
            result.svg_path,
            result.png_path,
            topology=topo.model_dump(),
            render_manifest=result.diagram_render_manifest,
        )

    with tabs[2]:
        dc = result.design_candidate
        if dc:
            from components.council_report import render_council_report
            render_council_report(dc)

    with tabs[3]:
        from components.inventory_result import render_inventory_result

        render_inventory_result(result.model_dump())

    with tabs[4]:
        st.json(result.model_dump(exclude_none=True))

    fb_result = {"proposal": {
        "residence_time_min": topo.residence_time_min,
        "flow_rate_mL_min": topo.total_flow_rate_mL_min,
        "confidence": topo.topology_confidence,
    }}
    from components.feedback import render_feedback_widget
    render_feedback_widget(fb_result, context="flora_design_goal")


def _render_conditions(proposal: dict):
    c1, c2, c3 = st.columns(3)
    with c1:
        residence_label = "Residence time (inlet/STP)" if _proposal_has_gas(proposal) else "Residence time"
        st.metric(residence_label, f"{proposal.get('residence_time_min', 0):.1f} min")
        st.metric("Flow rate", f"{proposal.get('flow_rate_mL_min', 0):.2f} mL/min")
        st.metric("Temperature", f"{proposal.get('temperature_C', 25):.0f} °C")
        st.metric("Concentration", f"{proposal.get('concentration_M', 0):.3f} M")
    with c2:
        st.metric("Reactor", proposal.get("reactor_type", "N/A"))
        st.metric("Tubing", proposal.get("tubing_material", "N/A"))
        st.metric("Tubing ID", f"{proposal.get('tubing_ID_mm', 0):.1f} mm")
        st.metric("Volume", f"{proposal.get('reactor_volume_mL', 0):.1f} mL")
    with c3:
        st.metric("BPR", f"{proposal.get('BPR_bar', 0):.0f} bar")
        st.metric("Wavelength", f"{proposal.get('wavelength_nm', 'N/A')} nm")
        st.metric("Deoxygenation", proposal.get("deoxygenation_method", "N/A"))

    reasoning = proposal.get("reasoning_per_field", {})
    if reasoning:
        st.divider()
        st.markdown("**Reasoning per field**")
        for field, reason in reasoning.items():
            st.markdown(f"- **{field}:** {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Council Deliberation — narrative summary of multi-agent conversation
# ─────────────────────────────────────────────────────────────────────────────

def _render_council_deliberation(result: dict):
    """Render the ENGINE council deliberation with full chain-of-thought."""
    delib_log = result.get("deliberation_log")
    proposal = result.get("proposal", {})
    rounds_count = result.get("council_rounds", 0)

    if not delib_log:
        _render_legacy_deliberation(result)
        return

    rounds = delib_log.get("rounds", [])
    sanity_checks = delib_log.get("sanity_checks", [])
    total_rounds = delib_log.get("total_rounds", len(rounds))
    consensus = delib_log.get("consensus_reached", False)
    council_summary = delib_log.get("summary", "")

    # ── Input Design ──────────────────────────────────────────────────────────
    design_calc = ((result.get("engineering_history") or {}).get("before_council") or {}).get("calculations", {})
    pre = result.get("pre_council_proposal", {})
    if design_calc or pre:
        with st.expander("Input Design (fed to the Designer agent)", expanded=True):
            st.caption(
                "These are the conditions the Designer received as the starting "
                "center-point. The council then generated a shortlist around these values."
            )
            src = pre if pre else proposal
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("τ center", f"{src.get('residence_time_min', '?')} min")
            c2.metric("Q center", f"{src.get('flow_rate_mL_min', '?')} mL/min")
            c3.metric("d center", f"{src.get('tubing_ID_mm', '?')} mm")
            c4.metric("V_R center", f"{src.get('reactor_volume_mL', '?')} mL")
            if design_calc:
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Re", f"{design_calc.get('reynolds_number', '?'):.0f}"
                           if isinstance(design_calc.get('reynolds_number'), (int, float)) else "?")
                c6.metric("ΔP", f"{design_calc.get('pressure_drop_bar', '?'):.4f} bar"
                           if isinstance(design_calc.get('pressure_drop_bar'), (int, float)) else "?")
                c7.metric("Da", f"{design_calc.get('damkohler_mass', '?'):.2f}"
                           if isinstance(design_calc.get('damkohler_mass'), (int, float)) else "?")
                c8.metric("τ_lit (analogy)", f"{design_calc.get('tau_analogy_min', '?')} min"
                           if design_calc.get('tau_analogy_min') else "—")

    # ── Designer candidate shortlist ──────────────────────────────────────────
    if council_summary and "### Shortlist" in council_summary:
        shortlist_md = council_summary.split("### Shortlist")[1].split("\n###")[0].strip()
        with st.expander("Designer Candidate Shortlist", expanded=True):
            st.caption(
                "The Designer agent sampled the design space and generated these "
                "feasible candidates. The Expert panel then each advocated for one."
            )
            st.markdown(shortlist_md)
        # Also show Designer strategy if available
        strategy_line = ""
        for line in council_summary.split("\n"):
            if "**Designer strategy**:" in line:
                strategy_line = line.replace("**Designer strategy**:", "").strip()
                break
        if strategy_line:
            st.caption(f"Designer strategy: {strategy_line}")

    st.divider()
    st.markdown(f"### Multi-Agent Deliberation — {total_rounds} Round{'s' if total_rounds != 1 else ''}")

    # ── Header stats ──────────────────────────────────────────────────
    all_deliberations = [d for rnd in rounds for d in rnd]
    n_accept = sum(1 for d in all_deliberations if d.get("status") == "ACCEPT")
    n_warn   = sum(1 for d in all_deliberations if d.get("status") == "WARNING")
    n_revise = sum(1 for d in all_deliberations if d.get("status") == "REVISE")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds", total_rounds)
    c2.metric("Accepted", n_accept)
    c3.metric("Warnings", n_warn)
    c4.metric("Revisions", n_revise)

    _STATUS_ICONS = {"ACCEPT": "✅", "WARNING": "⚠️", "REVISE": "🔄"}
    _AGENT_AVATARS = {
        "Dr. Kinetics": "⏱️",
        "Dr. Fluidics": "🌊",
        "Dr. Safety": "🛡️",
        "Dr. Chemistry": "🧪",
        "Dr. Process": "🏗️",
        "Chief Engineer": "👷",
    }

    # ── Round-by-round rendering ──────────────────────────────────────
    for r_idx, round_delibs in enumerate(rounds):
        round_num = r_idx + 1
        st.divider()
        round_label = {1: "Independent Analysis", 2: "Peer-Informed Revision"}.get(round_num, f"Round {round_num}")
        st.markdown(f"#### Round {round_num} — {round_label}")

        for d in round_delibs:
            agent_name = d.get("agent_display_name", d.get("agent", "?"))
            status = d.get("status", "?")
            had_error = d.get("had_error", False)
            icon = "💥" if had_error else _STATUS_ICONS.get(status, "•")
            avatar = _AGENT_AVATARS.get(agent_name, "🤖")
            error_suffix = " [ERROR — blocks convergence]" if had_error else ""

            cot = d.get("chain_of_thought", "")
            is_green_skip = cot.startswith("Domain ") and "green zone per triage" in cot
            with st.expander(
                f"{avatar} {agent_name} — {icon} {status}{error_suffix}",
                expanded=(had_error or (bool(cot) and not is_green_skip)),
            ):
                # Chain of thought
                if cot:
                    st.markdown(cot)

                # Values Referenced (from DesignCalculator)
                calcs = d.get("values_referenced", d.get("calculations", []))
                if calcs:
                    st.markdown("**Values Referenced:**")
                    for calc in calcs:
                        st.code(calc, language=None)

                # Findings
                findings = d.get("findings", [])
                if findings:
                    st.markdown("**Findings:**")
                    for f in findings:
                        st.markdown(f"- {f}")

                # Proposals (structured FieldProposal objects)
                proposals = d.get("proposals", [])
                if proposals:
                    st.markdown("**Proposals:**")
                    for p in proposals:
                        if isinstance(p, dict) and p.get("field"):
                            st.info(f"💡 `{p['field']}` → **{p['value']}** — {p.get('reason', '')}")
                        elif isinstance(p, dict) and p.get("reason"):
                            st.info(f"💡 {p['reason']}")
                        elif isinstance(p, str):
                            st.info(f"💡 {p}")

                # Concerns (peer critiques for Expert agents, assumption attacks for Skeptic)
                concerns = d.get("concerns", [])
                agent = d.get("agent_display_name", d.get("agent", ""))
                if concerns:
                    label = "**Concerns about other picks:**" if "Expert" in d.get("agent", "") else "**Concerns:**"
                    st.markdown(label)
                    for c in concerns:
                        st.warning(c)

                # References to other agents
                refs = d.get("references_to_agents", [])
                if refs:
                    st.caption("References: " + ", ".join(refs))

                # Rules cited
                rules = d.get("rules_cited", [])
                if rules:
                    st.caption("Handbook rules cited: " + ", ".join(rules[:5]))

                # Tool calls made by this agent
                tool_calls = d.get("tool_calls", [])
                if tool_calls:
                    st.markdown(f"**Tool calls ({len(tool_calls)})**")
                    for tc in tool_calls:
                        tool_name = tc.get("tool", "unknown")
                        tool_input = tc.get("input", {})
                        tool_result = tc.get("result", {})
                        st.markdown(f"**`{tool_name}`**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("Input")
                            st.json(tool_input)
                        with col2:
                            st.caption("Result")
                            st.json(tool_result)

        # Show Skeptic's comparative trade-off summary for this round
        sk_delib = next((d for d in round_delibs if "Skeptic" in d.get("agent_display_name", "")), None)
        if sk_delib:
            ts = next((f for f in sk_delib.get("findings", []) if f.startswith("Trade-off:")), "")
            if ts:
                st.info(f"**Cross-pick trade-off:** {ts[len('Trade-off:'):].strip()}")

        # ── Sanity check for this round ───────────────────────────────
        if r_idx < len(sanity_checks):
            sc = sanity_checks[r_idx]
            st.divider()
            with st.expander(f"👷 Chief Engineer — Sanity Check (Round {round_num})",
                             expanded=bool(sc.get("conflicts_found"))):
                sc_cot = sc.get("chain_of_thought", "")
                if sc_cot:
                    st.markdown(sc_cot)

                conflicts = sc.get("conflicts_found", [])
                if conflicts:
                    st.markdown("**Conflicts found:**")
                    for c in conflicts:
                        st.error(c)

                resolutions = sc.get("resolutions", [])
                if resolutions:
                    st.markdown("**Resolutions:**")
                    for r in resolutions:
                        st.success(r)

                changes = sc.get("final_changes", {})
                if changes:
                    st.markdown("**Applied changes:**")
                    for field, val in changes.items():
                        st.markdown(f"- `{field}` → **{val}**")

    # Trade-off matrix (pre-Chief comparison)
    trade_off_matrix = delib_log.get("trade_off_matrix", "")
    if trade_off_matrix:
        st.divider()
        st.markdown("#### Surviving Picks — Comparison Matrix")
        st.caption("This table was computed before the Chief's decision to make the trade-offs explicit.")
        st.markdown(trade_off_matrix)

    # Trade-off summary
    trade_off_summary = delib_log.get("trade_off_summary", "")
    if trade_off_summary:
        st.info(f"**Skeptic's comparative assessment:** {trade_off_summary}")

    # ── Final outcome ─────────────────────────────────────────────────
    st.divider()
    # Build final design parameters string — always show τ, Q, V_R plus any
    # field that was actually changed during deliberation
    _FIELD_LABELS = {
        "residence_time_min": ("τ", "min"),
        "flow_rate_mL_min": ("Q", "mL/min"),
        "reactor_volume_mL": ("V_R", "mL"),
        "tubing_ID_mm": ("d", "mm"),
        "tubing_material": ("material", ""),
        "temperature_C": ("T", "°C"),
        "concentration_M": ("C", "M"),
        "BPR_bar": ("BPR", "bar"),
        "wavelength_nm": ("λ", "nm"),
        "deoxygenation_method": ("degas", ""),
        "mixer_type": ("mixer", ""),
    }
    always_show = ["residence_time_min", "flow_rate_mL_min", "reactor_volume_mL"]
    changed_fields = list(delib_log.get("all_changes_applied", {}).keys())
    show_fields = list(dict.fromkeys(always_show + changed_fields))  # preserve order, dedupe

    param_parts = []
    for field in show_fields:
        if field not in _FIELD_LABELS:
            continue
        label, unit = _FIELD_LABELS[field]
        val = proposal.get(field, "?")
        suffix = f" {unit}" if unit else ""
        marker = " ★" if field in changed_fields else ""  # star = modified by council
        param_parts.append(f"{label} = {val}{suffix}{marker}")
    param_str = "  ·  ".join(param_parts)

    if consensus:
        msg = (
            f"Consensus reached after {total_rounds} round{'s' if total_rounds != 1 else ''}  "
            f"·  {param_str}"
        )
        if changed_fields:
            msg += f"  ·  (★ = modified by council: {', '.join(changed_fields)})"
        st.success(msg)
    else:
        n_errors = sum(
            1 for rnd in rounds for d in rnd if d.get("had_error")
        )
        warning_msg = f"Max rounds ({total_rounds}) reached"
        if n_errors:
            warning_msg += f" — {n_errors} agent error(s) prevented convergence"
        warning_msg += f"  ·  {param_str}"
        st.warning(warning_msg)


def _render_legacy_deliberation(result: dict):
    """Fallback renderer for results without the new deliberation log."""
    msgs_raw = result.get("council_messages", [])
    rounds = result.get("council_rounds", 0)
    proposal = result.get("proposal", {})

    if not msgs_raw:
        st.info("No council deliberation data available.")
        return

    msgs = []
    for m in msgs_raw:
        if isinstance(m, dict):
            msgs.append(m)
        elif hasattr(m, "model_dump"):
            msgs.append(m.model_dump())

    st.markdown(f"### Council Review — {rounds} Round{'s' if rounds != 1 else ''}")

    _ICONS = {"REJECT": "❌", "WARNING": "⚠️", "ACCEPT": "✅"}
    _NAMES = {
        "DesignCalculator": "Physics Engine",
        "KineticsAgent": "Dr. Kinetics", "KineticsSpecialist": "Dr. Kinetics",
        "FluidicsAgent": "Dr. Fluidics", "FluidicsSpecialist": "Dr. Fluidics",
        "SafetyCriticAgent": "Dr. Safety", "SafetySpecialist": "Dr. Safety",
        "ChemistryValidator": "Dr. Chemistry", "ChemistrySpecialist": "Dr. Chemistry",
        "ProcessArchitectAgent": "Dr. Process", "IntegrationSpecialist": "Dr. Process",
    }
    for m in msgs:
        agent = m.get("agent", "?")
        status = m.get("status", "?")
        concern = m.get("concern", "")
        value = m.get("value", "")
        icon = _ICONS.get(status, "•")
        name = _NAMES.get(agent, agent)
        text = concern or value or "OK"
        st.markdown(f"{icon} **{name}**: {text[:200]}")

    st.divider()
    validated = proposal.get("engine_validated", False)
    if validated:
        st.success(f"Design validated after {rounds} rounds")
    else:
        st.warning("Design did not fully converge.")


# ─────────────────────────────────────────────────────────────────────────────
# Recipe — step-by-step experimental instructions for the bench chemist
# ─────────────────────────────────────────────────────────────────────────────

def _render_canonical_procedure(final_design: dict):
    """Render the frozen procedure compiled from the executable design graph."""
    import re

    procedure = list(final_design.get("operating_procedure") or [])
    safety = dict(final_design.get("safety") or {})
    st.markdown("### Canonical Operating Procedure")
    st.caption(
        "Compiled after final engineering realization from the same canonical "
        "streams, equipment, topology, and safety contract used by every tab."
    )
    if not procedure:
        st.error("No canonical operating procedure is available.")
        return

    hazards = safety.get("hazards") or []
    if hazards:
        st.warning("Declared hazards: " + ", ".join(str(item) for item in hazards))

    labels = {
        "preparation": "Preparation",
        "setup": "System Setup",
        "startup": "Startup",
        "steady_state": "Steady State",
        "collection": "Collection",
        "shutdown": "Shutdown",
        "emergency": "Emergency Response",
        "waste": "Waste Handling",
    }
    for section, label in labels.items():
        steps = [item for item in procedure if item.get("section") == section]
        if not steps:
            continue
        st.markdown(f"#### {label}")
        for item in steps:
            instruction = re.sub(r"\bsccm\b", "mL/min at STP", item.get("instruction", ""), flags=re.IGNORECASE)
            st.markdown(
                f"**{item.get('step_id', 'STEP')}**  \n{instruction}"
            )
            bindings = {
                key.replace("sccm", "inlet_STP_mL_min"): value
                for key, value in (item.get("parameter_bindings") or {}).items()
                if value is not None and "actual" not in key and "in_channel" not in key and "holdup" not in key
            }
            equipment = item.get("equipment_ids") or []
            if equipment:
                st.caption("Equipment: " + ", ".join(f"`{value}`" for value in equipment))
            if bindings:
                st.caption(
                    "Bindings: "
                    + ", ".join(f"{key}={value}" for key, value in bindings.items())
                )

    digest = final_design.get("canonical_sha256")
    if digest:
        st.caption(f"Canonical design SHA-256: `{digest}`")

_GAS_NAMES = {
    "o2": "oxygen (O₂)", "o₂": "oxygen (O₂)", "oxygen": "oxygen (O₂)",
    "h2": "hydrogen (H₂)", "h₂": "hydrogen (H₂)", "hydrogen": "hydrogen (H₂)",
    "co2": "carbon dioxide (CO₂)", "co₂": "carbon dioxide (CO₂)",
    "co": "carbon monoxide (CO)", "n2": "nitrogen (N₂)", "n₂": "nitrogen (N₂)",
    "nitrogen": "nitrogen (N₂)", "ar": "argon (Ar)", "argon": "argon (Ar)",
    "air": "compressed air",
}

def _is_gas_stream(stream: dict) -> bool:
    """Return True if this stream carries a gas (not a liquid solution)."""
    if str(stream.get("phase") or "").lower() == "gas":
        return True
    _GAS_KW = {"o2", "o₂", "oxygen", "h2", "h₂", "hydrogen", "co2", "co₂",
               "syngas", "ethylene", "acetylene", "carbon monoxide", "carbonylation",
               "mfc", "gas", "n2 gas", "argon gas"}
    contents = stream.get("contents", [])
    pump_role = (stream.get("pump_role") or "").lower()
    label = (stream.get("stream_label") or "").lower()
    all_text = " ".join(str(c) for c in contents).lower() + " " + pump_role + " " + label
    return any(kw in all_text for kw in _GAS_KW)


def _proposal_has_gas(proposal: dict) -> bool:
    return any(_is_gas_stream(stream) for stream in proposal.get("streams", []) or [])


def _identify_gas(stream: dict) -> str:
    """Return the human-readable gas name from a stream."""
    contents = stream.get("contents", [])
    pump_role = (stream.get("pump_role") or "").lower()
    all_text = " ".join(str(c) for c in contents).lower() + " " + pump_role
    for kw, name in _GAS_NAMES.items():
        if kw in all_text:
            return name
    return "gas"

def _render_recipe(result: dict, proposal: dict | None = None):
    """Generate and display a step-by-step experimental recipe."""
    proposal = proposal or result.get("proposal", {})
    chem_plan = result.get("chemistry_plan", {})
    streams_raw = proposal.get("streams", [])

    if not proposal:
        return

    # Normalise streams to dicts
    streams = []
    for s in streams_raw:
        if isinstance(s, dict):
            streams.append(s)
        elif hasattr(s, "model_dump"):
            streams.append(s.model_dump())

    # Separate liquid and gas streams
    liquid_streams = [s for s in streams if not _is_gas_stream(s)]
    gas_streams    = [s for s in streams if _is_gas_stream(s)]

    st.markdown("### Experimental Recipe")
    st.markdown("*Step-by-step instructions for the bench chemist.*")

    step_num = [0]

    def step(text):
        step_num[0] += 1
        st.markdown(f"**{step_num[0]}.** {text}")

    # ── Safety note ─────────────────────────────────────────────────────
    safety_flags = proposal.get("safety_flags", [])
    if safety_flags or gas_streams:
        warnings_text = []
        if gas_streams:
            gas_names = [_identify_gas(s) for s in gas_streams]
            warnings_text.append(f"This process uses pressurised gas: **{', '.join(gas_names)}**. "
                                  "Follow your institution's gas handling safety procedures.")
        for flag in safety_flags[:3]:
            warnings_text.append(flag)
        if warnings_text:
            st.warning("⚠️ **Safety notes:** " + "  \n".join(warnings_text))

    # ── Section A: Solution Preparation ─────────────────────────────────
    st.markdown("#### A. Preparation")

    if liquid_streams:
        st.markdown("**Liquid solutions:**")
        for s in liquid_streams:
            label   = s.get("stream_label", "?")
            contents = s.get("contents", [])
            solvent  = s.get("solvent", "")
            conc     = s.get("concentration_M")

            # Clean content names (strip loading in brackets)
            names = [str(c).split("(")[0].strip() for c in contents] if contents else ["reagents"]
            content_str = ", ".join(names)
            conc_str = f" to give a **{conc} M** solution" if conc else ""

            step(
                f"**Stream {label} — Liquid solution:** Weigh out {content_str} "
                f"and dissolve in **{solvent or 'the appropriate solvent'}**{conc_str}. "
                "Transfer to a clean, dry, inert-atmosphere-compatible flask or syringe. "
                "Cap and label."
            )
    elif not gas_streams:
        step("Prepare reagent solutions at the specified concentrations in the appropriate solvents.")

    if gas_streams:
        st.markdown("**Gas feeds:**")
        for s in gas_streams:
            label    = s.get("stream_label", "?")
            gas_name = _identify_gas(s)
            inlet_sccm = s.get("gas_flow_sccm")
            channel_flow = s.get("gas_flow_actual_mL_min")
            equiv = s.get("molar_equiv")
            fr_str = (
                f" at **{inlet_sccm:.4g} sccm at inlet/STP**"
                if inlet_sccm is not None else ""
            )
            if equiv is not None:
                fr_str += f" (**{equiv:.3g} equiv** relative to limiting substrate)"
            if channel_flow is not None:
                fr_str += f"; derived in-channel flow **{channel_flow:.4g} mL/min**"

            step(
                f"**Stream {label} — Gas feed ({gas_name}):** Connect the "
                f"**{gas_name}** cylinder to a **mass flow controller (MFC)**"
                f"{fr_str}. Use gas-rated stainless steel or PTFE fittings "
                "and a check valve to prevent back-flow. "
                "Do **not** use a syringe pump for gas delivery."
            )

    # Deoxygenation
    deoxy = proposal.get("deoxygenation_method")
    if not deoxy and chem_plan.get("deoxygenation_required"):
        deoxy = "N₂ sparging"
    if deoxy and liquid_streams:
        step(
            f"**Deoxygenate all liquid solutions:** Sparge each with **{deoxy}** "
            "for 15 minutes using a stainless steel needle. "
            "Keep capped under inert atmosphere until use."
        )

    # ── Section B: System Setup ─────────────────────────────────────────
    st.markdown("#### B. Flow System Assembly")

    tubing_mat  = proposal.get("tubing_material", "FEP")
    tubing_id   = proposal.get("tubing_ID_mm", 1.0)
    reactor_vol = proposal.get("reactor_volume_mL", 0)
    reactor_type = proposal.get("reactor_type", "coil")

    step(
        f"Assemble the **{reactor_type} reactor**: cut **{tubing_mat}** tubing "
        f"(ID = **{tubing_id} mm**) to give a reactor volume of **{reactor_vol:.1f} mL**. "
        "Coil neatly and secure."
    )

    wl = proposal.get("wavelength_nm")
    if wl:
        step(
            f"Mount the **{wl:.0f} nm LED** light source around the reactor coil. "
            "Ensure uniform, full-length irradiation. Shield from ambient light."
        )

    temp = proposal.get("temperature_C", 25)
    if temp and temp != 25:
        step(
            f"Pre-heat or pre-cool the **temperature-controlled bath** to "
            f"**{temp:.0f} °C** and allow 10 minutes to equilibrate. "
            "Submerge the reactor coil completely."
        )

    bpr = proposal.get("BPR_bar", 0)
    if bpr and bpr > 0:
        step(
            f"Install the **back-pressure regulator (BPR)** set to **{bpr:.0f} bar** "
            "at the reactor outlet. "
            + ("This is required to maintain gas solubility throughout the reactor. " if gas_streams else
               "This prevents solvent boiling at the operating temperature. ")
        )

    mixer = proposal.get("mixer_type", "T-mixer")
    n_liquid = len(liquid_streams)
    n_gas    = len(gas_streams)
    total_inlets = n_liquid + n_gas
    step(
        f"Connect all **{total_inlets} inlet line{'s' if total_inlets != 1 else ''}** "
        f"({n_liquid} liquid pump{'s' if n_liquid != 1 else ''}"
        + (f", {n_gas} gas MFC{'s' if n_gas != 1 else ''}" if n_gas else "")
        + f") to the **{mixer}** at the reactor inlet. "
        "Finger-tighten all fittings, then confirm with wrench. Check for leaks."
    )

    if liquid_streams:
        n_syringes = len(liquid_streams)
        step(
            f"Load **{n_syringes} syringe{'s' if n_syringes != 1 else ''}** with "
            "the prepared liquid solutions. "
            "Remove all air bubbles — invert and tap. Mount on pump(s)."
        )

    # ── Section C: Running the Process ──────────────────────────────────
    st.markdown("#### C. Running the Process")

    flow_rate = proposal.get("flow_rate_mL_min", 0)
    rt        = proposal.get("residence_time_min", 0)

    if liquid_streams:
        for s in liquid_streams:
            fr = s.get("flow_rate_mL_min")
            label = s.get("stream_label", "?")
            if fr:
                step(f"Set **Pump {label}** (liquid) to **{fr:.3f} mL/min**.")
            elif flow_rate and len(liquid_streams) > 0:
                fr_each = flow_rate / len(liquid_streams)
                step(f"Set **Pump {label}** (liquid) to **{fr_each:.3f} mL/min**.")

    if gas_streams:
        for s in gas_streams:
            inlet_sccm = s.get("gas_flow_sccm")
            channel_flow = s.get("gas_flow_actual_mL_min")
            label = s.get("stream_label", "?")
            gas_name = _identify_gas(s)
            if inlet_sccm is not None:
                step(
                    f"Set **MFC {label}** ({gas_name}) to "
                    f"**{inlet_sccm:.4g} sccm at inlet/STP**. "
                    + (
                        f"The pressure-corrected in-channel flow is "
                        f"**{channel_flow:.4g} mL/min** (derived; do not enter this on the MFC)."
                        if channel_flow is not None else ""
                    )
                )
            else:
                step(
                    f"Set **MFC {label}** ({gas_name}) to the target flow rate "
                    "from the inlet/STP stoichiometry calculation."
                )

    step(
        "**Prime the system:** Start all liquid pumps at low flow (0.1 mL/min) "
        "to fill the tubing. Once solution exits at the outlet, increase to target flow."
    )

    if gas_streams:
        step(
            "Open the gas supply: slowly increase MFC flow to the target rate. "
            "Observe the outlet for stable slug/segmented flow formation."
        )

    step(
        f"Allow **{3 * rt:.0f} min** ({3}× residence time of {rt:.0f} min) "
        "for steady state. Discard the initial effluent."
        if rt > 0 else
        "Allow the system to reach steady state before collecting product."
    )

    step(
        "Collect product effluent into a pre-weighed vial. "
        "Record exact collection time, volume, and weight for yield calculation."
    )

    # ── Section D: Shutdown & Workup ────────────────────────────────────
    st.markdown("#### D. Shutdown & Workup")

    if gas_streams:
        step("Close the gas supply valves. Wait 30 seconds for pressure to equalise.")

    step("Stop all liquid pumps.")

    step(
        f"Flush the system with **{reactor_vol * 3:.1f} mL** of pure solvent "
        f"({3}× reactor volume) to recover residual product and clean the lines."
    )

    post_steps = proposal.get("post_reactor_steps", [])
    for ps in post_steps:
        step(f"{ps}")

    quench_reagent = chem_plan.get("quench_reagent", "")
    if chem_plan.get("quench_required") and quench_reagent:
        step(f"Quench the combined product fractions with **{quench_reagent}**.")

    step(
        "Work up the product: standard aqueous extraction, dry over anhydrous MgSO₄ "
        "or Na₂SO₄, filter, and concentrate under reduced pressure. "
        "Purify by column chromatography or recrystallisation as appropriate."
    )
