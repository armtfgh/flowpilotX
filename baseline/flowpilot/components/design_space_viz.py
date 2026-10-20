"""
FLORA — Design Space Visualization Component.

Renders the grid search candidates with:
  1. Summary metrics (N candidates, top score, τ range explored)
  2. Candidates table (sortable dataframe with color coding)
  3. Scatter plot — Productivity vs. L colored by d
  4. Score breakdown bar chart for top-5 feasible candidates
  5. "Council starting point" highlighted box
"""

import streamlit as st
import pandas as pd

# Colors for d values (consistent across plots)
_D_COLORS = {
    "0.5":  "#636EFA",
    "0.75": "#EF553B",
    "1.0":  "#00CC96",
    "1.6":  "#AB63FA",
}

def _d_color(d_mm: float) -> str:
    key = str(round(d_mm, 2)).rstrip("0").rstrip(".")
    # Try exact and common forms
    for candidate_key in [f"{d_mm:.2f}", f"{d_mm:.1f}", f"{d_mm}"]:
        stripped = candidate_key.rstrip("0").rstrip(".")
        if stripped in _D_COLORS:
            return _D_COLORS[stripped]
    # fallback
    return "#888888"


def render_design_space(design_space: list[dict], key_prefix: str = "") -> None:
    """Render the design space grid search results."""
    if not design_space:
        st.info("No design space data available.")
        return

    feasible = [c for c in design_space if c.get("feasible", False)]
    infeasible = [c for c in design_space if not c.get("feasible", False)]

    # ── 1. Summary metrics ────────────────────────────────────────────────
    tau_all = [c["tau_min"] for c in design_space if "tau_min" in c]
    min_tau = min(tau_all) if tau_all else 0.0
    max_tau = max(tau_all) if tau_all else 0.0
    best_score = max((c.get("score", 0.0) for c in feasible), default=0.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates explored", len(design_space))
    col2.metric("Feasible", len(feasible))
    col3.metric("τ range", f"{min_tau:.0f}–{max_tau:.0f} min")
    col4.metric("Best score", f"{best_score:.3f}")

    # ── 2. Council starting point highlight ───────────────────────────────
    council_candidate = next(
        (c for c in design_space if c.get("is_council_candidate", False)), None
    )
    if council_candidate:
        tau = council_candidate.get("tau_min", "?")
        Q = council_candidate.get("Q_mL_min", "?")
        d = council_candidate.get("d_mm", "?")
        L = council_candidate.get("L_m", "?")
        VR = council_candidate.get("V_R_mL", "?")
        sc = council_candidate.get("score", 0.0)
        Re = council_candidate.get("Re", "?")
        dP = council_candidate.get("delta_P_bar", "?")
        st.success(
            f"**Council starting point:** "
            f"τ = {tau} min | Q = {Q} mL/min | d = {d} mm | "
            f"L = {L} m | V_R = {VR} mL | Re = {Re} | ΔP = {dP} bar | "
            f"Score: {sc:.3f}"
        )

    # ── 3. Feasible candidates table ──────────────────────────────────────
    if feasible:
        st.markdown("#### Feasible Candidates")
        rows = []
        for c in feasible:
            viols = c.get("violations", [])
            warns = c.get("warnings", [])
            rows.append({
                "τ (min)": round(c.get("tau_min", 0), 1),
                "d (mm)": c.get("d_mm", 0),
                "Q (mL/min)": round(c.get("Q_mL_min", 0), 4),
                "L (m)": round(c.get("L_m", 0), 2),
                "V_R (mL)": round(c.get("V_R_mL", 0), 3),
                "Re": round(c.get("Re", 0), 1),
                "ΔP (bar)": round(c.get("delta_P_bar", 0), 4),
                "r_mix": round(c.get("r_mix", 0), 4),
                "Conversion": round(c.get("expected_conversion", 0), 3),
                "Productivity (mg/h)": round(c.get("productivity_mg_h", 0), 2),
                "Score": round(c.get("score", 0), 4),
                "τ source": c.get("tau_source", ""),
                "Warnings": "; ".join(warns) if warns else "",
                "Council": "★" if c.get("is_council_candidate") else "",
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            key=f"{key_prefix}_feasible_table",
            column_config={
                "Score": st.column_config.NumberColumn(format="%.4f"),
                "Conversion": st.column_config.NumberColumn(format="%.3f"),
                "ΔP (bar)": st.column_config.NumberColumn(format="%.4f"),
                "Council": st.column_config.TextColumn(help="★ = council starting point"),
            },
            hide_index=True,
        )

    # ── 4. Charts ─────────────────────────────────────────────────────────
    if not feasible:
        st.warning("No feasible candidates found — check constraints.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### Productivity vs. Reactor Length")
        try:
            import plotly.graph_objects as go

            # Group by d_mm for discrete color
            d_groups: dict[float, list] = {}
            for c in feasible:
                d = c.get("d_mm", 0)
                d_groups.setdefault(d, []).append(c)

            fig = go.Figure()
            for d_mm, group in sorted(d_groups.items()):
                color = _d_color(d_mm)
                L_vals = [c.get("L_m", 0) for c in group]
                P_vals = [c.get("productivity_mg_h", 0) for c in group]
                sc_vals = [c.get("score", 0) for c in group]
                # Normalize size to 5-20 range
                sc_max = max(sc_vals) if sc_vals else 1.0
                sizes = [5 + 15 * (s / sc_max if sc_max > 0 else 0) for s in sc_vals]
                hover_texts = [
                    f"τ={c.get('tau_min', '?')} min<br>"
                    f"Q={c.get('Q_mL_min', '?'):.4f} mL/min<br>"
                    f"d={c.get('d_mm', '?')} mm<br>"
                    f"Re={c.get('Re', '?'):.1f}<br>"
                    f"r_mix={c.get('r_mix', '?'):.4f}<br>"
                    f"Score={c.get('score', '?'):.4f}<br>"
                    f"{'★ Council start' if c.get('is_council_candidate') else ''}"
                    for c in group
                ]
                fig.add_trace(go.Scatter(
                    x=L_vals,
                    y=P_vals,
                    mode="markers",
                    name=f"d = {d_mm} mm",
                    marker=dict(color=color, size=sizes, opacity=0.8,
                                line=dict(width=1, color="white")),
                    text=hover_texts,
                    hovertemplate="%{text}<extra></extra>",
                ))

            # Highlight council candidate
            if council_candidate:
                fig.add_trace(go.Scatter(
                    x=[council_candidate.get("L_m", 0)],
                    y=[council_candidate.get("productivity_mg_h", 0)],
                    mode="markers",
                    name="Council start",
                    marker=dict(
                        color="gold", size=18, symbol="star",
                        line=dict(width=2, color="black")
                    ),
                    hovertemplate="Council starting point<extra></extra>",
                ))

            fig.update_layout(
                xaxis_title="Reactor Length L (m)",
                yaxis_title="Productivity (mg/h)",
                legend_title="Tubing ID",
                height=380,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"{key_prefix}_scatter")

        except ImportError:
            # Fallback: simple scatter chart
            scatter_data = pd.DataFrame([{
                "L (m)": c.get("L_m", 0),
                "Productivity (mg/h)": c.get("productivity_mg_h", 0),
            } for c in feasible])
            st.scatter_chart(scatter_data, x="L (m)", y="Productivity (mg/h)")

    with col_right:
        st.markdown("##### Score Breakdown — Top 5 Candidates")
        top5 = feasible[:5]
        try:
            import plotly.graph_objects as go

            labels = [
                f"τ={c.get('tau_min', '?'):.0f}min d={c.get('d_mm', '?'):.2f}mm Q={c.get('Q_mL_min', '?'):.3f}"
                for c in top5
            ]
            breakdown_keys = ["productivity", "L", "mixing", "conversion", "re"]
            breakdown_colors = {
                "productivity": "#636EFA",
                "L": "#EF553B",
                "mixing": "#00CC96",
                "conversion": "#AB63FA",
                "re": "#FFA15A",
            }
            breakdown_labels = {
                "productivity": "Productivity",
                "L": "Length",
                "mixing": "Mixing",
                "conversion": "Conversion",
                "re": "Re (laminar)",
            }

            fig2 = go.Figure()
            for key in breakdown_keys:
                values = []
                for c in top5:
                    sb = c.get("score_breakdown", {})
                    raw = sb.get(key, 0.0)
                    # Multiply by weight to show weighted contribution
                    # Use default weights for display
                    values.append(round(raw, 4))
                fig2.add_trace(go.Bar(
                    name=breakdown_labels.get(key, key),
                    x=labels,
                    y=values,
                    marker_color=breakdown_colors.get(key, "#888"),
                ))

            fig2.update_layout(
                barmode="stack",
                xaxis_title="Candidate",
                yaxis_title="Score component (0–1)",
                legend_title="Component",
                height=380,
                margin=dict(l=40, r=20, t=30, b=80),
                xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
            )
            st.plotly_chart(fig2, use_container_width=True,
                            key=f"{key_prefix}_score_bar")

        except ImportError:
            # Fallback: bar chart
            bar_data = pd.DataFrame([{
                "candidate": f"τ={c.get('tau_min', '?'):.0f}m d={c.get('d_mm', '?')}mm",
                "score": c.get("score", 0),
            } for c in top5])
            st.bar_chart(bar_data.set_index("candidate"))

    # ── 5. Infeasible candidates (collapsed) ──────────────────────────────
    if infeasible:
        with st.expander(f"Infeasible candidates ({len(infeasible)})", expanded=False):
            infeas_rows = []
            for c in infeasible:
                viols = c.get("violations", [])
                infeas_rows.append({
                    "τ (min)": round(c.get("tau_min", 0), 1),
                    "d (mm)": c.get("d_mm", 0),
                    "Q (mL/min)": round(c.get("Q_mL_min", 0), 4),
                    "L (m)": round(c.get("L_m", 0), 2),
                    "V_R (mL)": round(c.get("V_R_mL", 0), 3),
                    "Re": round(c.get("Re", 0), 1),
                    "Violations": "; ".join(viols) if viols else "unknown",
                })
            st.dataframe(
                pd.DataFrame(infeas_rows),
                use_container_width=True,
                hide_index=True,
                key=f"{key_prefix}_infeasible_table",
            )
