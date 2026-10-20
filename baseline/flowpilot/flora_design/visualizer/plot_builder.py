"""FLORA-Design — Plot builder: ΔP vs flow rate, yield vs residence time."""

import logging
import math

from flora_translate.config import SOLVENT_VISCOSITY_cP
from flora_translate.schemas import ProcessTopology

logger = logging.getLogger("flora.design.plots")


def _get_viscosity(solvent: str | None) -> float:
    if solvent:
        for key in [solvent, solvent.lower()]:
            if key in SOLVENT_VISCOSITY_cP:
                return SOLVENT_VISCOSITY_cP[key]
    return 1.0


def _hagen_poiseuille(Q_mL_min: float, ID_mm: float, length_m: float, eta_cP: float) -> float:
    """Pressure drop in bar."""
    Q = Q_mL_min * 1e-6 / 60
    d = ID_mm * 1e-3
    eta = eta_cP * 1e-3
    if d <= 0:
        return 0
    return (128 * eta * length_m * Q) / (math.pi * d**4) * 1e-5


def plot_dp_vs_flowrate(topology: ProcessTopology, solvent: str | None = None):
    """Plot ΔP vs flow rate. Returns a Plotly figure."""
    import plotly.graph_objects as go

    # Extract reactor params
    tubing_ID = 1.0
    tubing_length = 5.0
    for op in topology.unit_operations:
        if op.op_type in ("coil_reactor", "chip_reactor"):
            tubing_ID = op.parameters.get("ID_mm", 1.0)
            tubing_length = op.parameters.get("length_m", 5.0)
            break

    eta = _get_viscosity(solvent)
    flow_rates = [i * 0.1 for i in range(1, 51)]
    dp_values = [_hagen_poiseuille(q, tubing_ID, tubing_length, eta) for q in flow_rates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=flow_rates, y=dp_values,
        mode="lines", name=f"ID={tubing_ID}mm, L={tubing_length:.1f}m",
        line=dict(color="#4A90D9", width=2),
    ))

    # Operating point
    op_q = topology.total_flow_rate_mL_min
    op_dp = _hagen_poiseuille(op_q, tubing_ID, tubing_length, eta)
    fig.add_trace(go.Scatter(
        x=[op_q], y=[op_dp],
        mode="markers", name=f"Operating point ({op_dp:.2f} bar)",
        marker=dict(size=12, color="red", symbol="diamond"),
    ))

    fig.update_layout(
        xaxis_title="Flow rate (mL/min)",
        yaxis_title="ΔP (bar)",
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig


def plot_yield_vs_residence_time(records: list[dict], proposed_tau: float):
    """Scatter plot of yield vs residence time from retrieved records. Returns Plotly figure."""
    import plotly.graph_objects as go

    # Extract data points from records (if they have the needed fields)
    taus, yields, labels = [], [], []
    for r in records:
        if isinstance(r, dict):
            meta = r.get("metadata", r)
            tau = meta.get("residence_time_min")
            yld = meta.get("yield_pct")
            if tau and yld:
                taus.append(tau)
                yields.append(yld)
                labels.append(meta.get("doi", ""))

    fig = go.Figure()
    if taus:
        fig.add_trace(go.Scatter(
            x=taus, y=yields,
            mode="markers", name="Literature data",
            marker=dict(size=10, color="#4A90D9"),
            text=labels, hovertemplate="%{text}<br>τ=%{x}min, yield=%{y}%",
        ))

    # Proposed residence time as vertical line
    fig.add_vline(
        x=proposed_tau, line_dash="dash", line_color="red",
        annotation_text=f"Proposed: {proposed_tau}min",
    )

    fig.update_layout(
        xaxis_title="Residence time (min)",
        yaxis_title="Yield (%)",
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig
