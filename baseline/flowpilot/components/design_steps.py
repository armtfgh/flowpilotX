"""Streamlit component: renders the 9-step engineering design calculation.

Shows each step as an expandable card with:
  - Status indicator (✓ / ⚠ / ✗ / ↻ / ≈)
  - Summary line
  - LaTeX equations with actual numbers
  - Computed values as metrics
  - Warnings and adjustments
  - Assumptions made
"""

import streamlit as st


_STATUS_ICON = {
    "PASS": "✅",
    "WARNING": "⚠️",
    "FAIL": "❌",
    "ADJUSTED": "🔄",
    "ESTIMATED": "≈",
}

_STATUS_COLOR = {
    "PASS": "green",
    "WARNING": "orange",
    "FAIL": "red",
    "ADJUSTED": "blue",
    "ESTIMATED": "gray",
}


def render_design_steps(calc_dict: dict, key_prefix: str = "ds"):
    """Render the full 9-step engineering design from a dict (serialised DesignCalculations)."""
    steps = calc_dict.get("steps", [])
    if not steps:
        st.info("No design calculations available.")
        return

    # ── Header metrics ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("τ (residence time)", f"{calc_dict.get('residence_time_min', 0):.1f} min")
    c2.metric("Q (flow rate)", f"{calc_dict.get('flow_rate_mL_min', 0):.3f} mL/min")
    stage_total = calc_dict.get("stage_corrected_total_reactor_volume_mL")
    if stage_total and abs(float(stage_total) - float(calc_dict.get("reactor_volume_mL", 0) or 0)) > 1e-6:
        c3.metric(
            "V_R calc zone",
            f"{calc_dict.get('reactor_volume_mL', 0):.2f} mL",
            help="Single-zone deterministic calculator volume. See topology/per-reactor table for staged total volume.",
        )
    else:
        c3.metric("V_R (reactor vol)", f"{calc_dict.get('reactor_volume_mL', 0):.2f} mL")
    c4.metric("Re", f"{calc_dict.get('reynolds_number', 0):.1f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("d (tubing ID)", f"{calc_dict.get('tubing_ID_mm', 0):.2f} mm")
    c6.metric("L (tube length)", f"{calc_dict.get('tubing_length_m', 0):.2f} m")
    c7.metric("ΔP", f"{calc_dict.get('pressure_drop_bar', 0):.4f} bar")
    bpr = calc_dict.get("bpr_pressure_bar", 0)
    c8.metric("BPR", f"{bpr:.1f} bar" if bpr else "Not required")
    if calc_dict.get("bpr_reconciliation_note"):
        st.warning(calc_dict["bpr_reconciliation_note"])

    # ── Intensification anchors (where does IF come from?) ─────────────
    # Surface the three independent IF anchors so users can see how the
    # selected intensification factor was justified: class default vs
    # literature analogy (quality-weighted) vs limitation-driven.
    if_used = calc_dict.get("intensification_factor")
    if_class = calc_dict.get("if_class")
    if_analogy = calc_dict.get("if_analogy")
    n_analogy = calc_dict.get("n_analogy_datapoints")
    if any(v is not None for v in (if_used, if_class, if_analogy)):
        st.markdown("#### Intensification audit")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric(
            "IF implied by candidate",
            f"{float(if_used):.1f}×" if if_used else "—",
            help="Batch time divided by this preliminary candidate's residence time. It is not automatically a design constraint.",
        )
        a2.metric(
            "IF class default",
            f"{float(if_class):.1f}×" if if_class else "—",
            help="Empirical class-level expectation (photoredox 6×, thermal 10×, etc.).",
        )
        a3.metric(
            "IF analogy (weighted)",
            f"{float(if_analogy):.1f}×" if if_analogy else "—",
            help=f"Similarity-score-weighted geometric mean from {n_analogy or 0} literature analogies.",
        )
        # IF_limitation: read from the chemistry plan's mandate if present.
        # The calculator stores it indirectly via tau_reduction_target;
        # since design_steps gets the calc dict only, we'll show "see Chemistry tab" here.
        a4.markdown(
            "**Limitation target**  \n"
            "A screening hypothesis under evidence-first policy; see the Chemistry tab.",
            help="This target is enforced only when the configured translation policy is explicit intensification.",
        )

    if calc_dict.get("is_gas_liquid"):
        st.markdown("#### Gas-Liquid Design")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Gas", calc_dict.get("gas_species") or "gas")
        g2.metric("MFC setpoint", f"{calc_dict.get('gas_flow_sccm', 0):.2f} sccm")
        g3.metric("Gas holdup", f"{calc_dict.get('gas_holdup', 0):.2f}")
        g4.metric("Two-phase ΔP", f"{calc_dict.get('two_phase_pressure_drop_bar', calc_dict.get('pressure_drop_bar', 0)):.3f} bar")
        g5, g6, g7, g8 = st.columns(4)
        g5.metric("Liquid holdup", f"{calc_dict.get('liquid_holdup_volume_mL', 0):.2f} mL")
        g6.metric("Gas actual", f"{calc_dict.get('gas_flow_actual_mL_min', 0):.3f} mL/min")
        if stage_total:
            g7.metric("Topology V_total", f"{stage_total:.2f} mL")
        else:
            g7.metric("Topology V_total", "N/A")
        g8.metric("GLR", f"{calc_dict.get('gas_liquid_ratio', 0):.2f}")

        o2_suff = calc_dict.get("o2_transfer_sufficiency")
        if o2_suff:
            st.caption(f"O2 transfer sufficiency: {o2_suff:.2f}x")
        if calc_dict.get("gas_flow_capped_by_holdup"):
            st.info(
                "Air/O2 MFC setpoint was capped to stay consistent with the gas-holdup "
                "used for two-phase reactor sizing. Liquid residence time is sized as "
                "τ_liq = V_liquid / Q_liquid, with V_total = V_liquid / (1 - ε_g)."
            )

    if calc_dict.get("UA_W_K") or calc_dict.get("heat_transfer_area_m2"):
        st.markdown("#### Heat-Transfer Design")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Wall area", f"{calc_dict.get('heat_transfer_area_m2', 0):.4f} m²")
        h2.metric("UA", f"{calc_dict.get('UA_W_K', 0):.3f} W/K")
        h3.metric("Da_th", f"{calc_dict.get('thermal_damkohler', 0):.4g}")
        h4.metric("Heat score", f"{calc_dict.get('heat_transfer_score', 0):.2f}")

    # Row 3 — flow chemistry specific
    c9, c10, c11, c12 = st.columns(4)
    Pe = calc_dict.get("Pe")
    Pe_ok = calc_dict.get("Pe_adequate", True)
    c9.metric("Péclet (Pe)", f"{Pe:.0f}" if Pe else "N/A",
              delta="plug flow ✓" if Pe_ok else "⚠ axial dispersion",
              delta_color="normal" if Pe_ok else "inverse",
              help="Pe = 192·τ·D_mol/d² — must be ≥100 for plug-flow approximation")
    n_lim = calc_dict.get("n_molar_flow_mmol_min")
    c10.metric("ṅ_limiting", f"{n_lim:.4f} mmol/min" if n_lim else "N/A",
               help="Molar flow of limiting reagent = P_batch/(Y×60)")
    P_flow = calc_dict.get("P_flow_mmol_h")
    c11.metric("P_flow", f"{P_flow:.1f} mmol/h" if P_flow else "N/A",
               help="Forward productivity check: ṅ_lim × Y × 60")
    startup = calc_dict.get("startup_waste_mL")
    c12.metric("Startup Waste", f"{startup} mL" if startup else "N/A",
               help="Volume wasted during startup = 3×τ×Q")

    # Consistency badge
    if calc_dict.get("consistent", True):
        st.success("All consistency checks passed — τ = V_R/Q, L = 4V/(πd²), Re = ρvd/μ ✓")
    else:
        notes = calc_dict.get("consistency_notes", [])
        st.error("Consistency issues:\n" + "\n".join(f"- {n}" for n in notes))

    st.divider()

    # ── Step-by-step cards ──────────────────────────────────────────────
    for step in steps:
        _render_step(step, key_prefix)


def _render_step(step: dict, key_prefix: str):
    """Render a single step as an expandable card."""
    num = step.get("step", 0)
    name = step.get("name", "")
    status = step.get("status", "PASS")
    summary = step.get("summary", "")
    icon = _STATUS_ICON.get(status, "?")
    color = _STATUS_COLOR.get(status, "gray")

    with st.expander(
        f"{icon}  Step {num}: {name} — :{color}[{status}]",
        expanded=(status in ("FAIL", "ADJUSTED", "WARNING")),
    ):
        st.markdown(f"**{summary}**")

        # Equations (LaTeX)
        equations = step.get("equations", [])
        if equations:
            st.markdown("**Equations:**")
            for eq in equations:
                try:
                    st.latex(eq)
                except Exception:
                    st.code(eq)

        # Key values
        values = step.get("values", {})
        if values:
            # Pick the most useful values to display as metrics
            displayable = {
                k: v for k, v in values.items()
                if isinstance(v, (int, float)) and v is not None
            }
            if displayable:
                cols = st.columns(min(len(displayable), 4))
                for i, (k, v) in enumerate(displayable.items()):
                    label = k.replace("_", " ")
                    if isinstance(v, float):
                        if abs(v) < 0.01 and v != 0:
                            display = f"{v:.3e}"
                        elif abs(v) > 1e4:
                            display = f"{v:.0f}"
                        else:
                            display = f"{v:.4g}"
                    else:
                        display = str(v)
                    cols[i % len(cols)].metric(label, display)

            # Non-numeric values
            text_vals = {
                k: v for k, v in values.items()
                if isinstance(v, str) and v
            }
            if text_vals:
                for k, v in text_vals.items():
                    st.markdown(f"**{k.replace('_', ' ')}:** {v}")

        # Warnings
        for w in step.get("warnings", []):
            st.warning(w)

        # Adjustments
        for a in step.get("adjustments", []):
            st.info(f"↻ {a}")

        # Assumptions
        assumptions = step.get("assumptions", [])
        if assumptions:
            st.caption("Assumptions: " + " · ".join(assumptions))
