"""FLORA-Optimize — Gaussian Process Bayesian Optimization engine.

Self-contained: no dependency on ChromaDB or LLM APIs.
Uses scikit-learn's GP and Expected Improvement acquisition.
"""

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def suggest_next(
    observations: list[dict],
    param_defs: list[dict],
    objective: str = "yield",
    maximize: bool = True,
    n_candidates: int = 5000,
):
    """Fit GP, compute EI, return next suggestion + plots.

    Args:
        observations: list of dicts, each with param values + "result" key
        param_defs: list of {"name", "min", "max", "units"}
        objective: name of the objective (for labelling)
        maximize: True to maximize, False to minimize
        n_candidates: number of random candidates for EI search

    Returns:
        (suggestion_dict, fig_gp, fig_acq)
    """
    import plotly.graph_objects as go

    param_names = [p["name"] for p in param_defs]
    n_params = len(param_names)

    # Build X and y arrays
    X_raw = np.array([[obs[p] for p in param_names] for obs in observations])
    y_raw = np.array([obs["result"] for obs in observations])
    if not maximize:
        y_raw = -y_raw

    # Normalize X to [0, 1]
    X_mins = np.array([p["min"] for p in param_defs])
    X_maxs = np.array([p["max"] for p in param_defs])
    X_range = X_maxs - X_mins
    X_range[X_range == 0] = 1.0  # avoid division by zero
    X_norm = (X_raw - X_mins) / X_range

    # Fit GP
    kernel = Matern(nu=2.5, length_scale=np.ones(n_params))
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True, alpha=1e-6
    )
    gp.fit(X_norm, y_raw)

    # Generate candidates (Latin hypercube-ish: stratified random)
    X_cand = np.random.uniform(0, 1, (n_candidates, n_params))
    mu, sigma = gp.predict(X_cand, return_std=True)

    # Expected Improvement
    y_best = y_raw.max()
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (mu - y_best) / (sigma + 1e-9)
        EI = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma < 1e-9] = 0.0

    best_idx = np.argmax(EI)
    best_norm = X_cand[best_idx]
    best_real = best_norm * X_range + X_mins

    suggestion = {
        "params": {
            p: float(round(best_real[i], 4)) for i, p in enumerate(param_names)
        },
        "expected_improvement": float(EI[best_idx]),
        "predicted_value": float(mu[best_idx]) if maximize else float(-mu[best_idx]),
    }

    # ── Plot 1: GP surrogate (1D slice along first parameter) ──
    p0 = param_defs[0]
    grid = np.linspace(0, 1, 200).reshape(-1, 1)
    if n_params > 1:
        # Fix other dims at suggested values
        fixed = best_norm.copy()
        X_grid = np.tile(fixed, (200, 1))
        X_grid[:, 0] = grid.ravel()
    else:
        X_grid = grid

    mu_grid, sigma_grid = gp.predict(X_grid, return_std=True)
    x_real = grid.ravel() * (p0["max"] - p0["min"]) + p0["min"]

    if not maximize:
        mu_grid = -mu_grid
        y_plot = -y_raw
    else:
        y_plot = y_raw

    fig_gp = go.Figure()
    fig_gp.add_trace(go.Scatter(
        x=x_real, y=mu_grid,
        mode="lines", name="GP mean",
        line=dict(color="#3b82f6", width=2),
    ))
    fig_gp.add_trace(go.Scatter(
        x=np.concatenate([x_real, x_real[::-1]]),
        y=np.concatenate([mu_grid + 2 * sigma_grid, (mu_grid - 2 * sigma_grid)[::-1]]),
        fill="toself", fillcolor="rgba(59,130,246,0.15)",
        line=dict(width=0), name="95% CI",
    ))
    fig_gp.add_trace(go.Scatter(
        x=X_raw[:, 0], y=y_plot,
        mode="markers", name="Observations",
        marker=dict(size=10, color="#1e293b", symbol="circle"),
    ))
    fig_gp.add_vline(
        x=best_real[0], line_dash="dash", line_color="#ef4444",
        annotation_text=f"Suggested: {best_real[0]:.2f}",
    )
    fig_gp.update_layout(
        xaxis_title=f"{p0['name']} ({p0.get('units', '')})",
        yaxis_title=objective,
        template="plotly_white",
        height=350, margin=dict(l=50, r=20, t=30, b=50),
    )

    # ── Plot 2: Acquisition function (EI) ──
    mu_acq, sigma_acq = gp.predict(X_grid, return_std=True)
    Z_acq = (mu_acq - y_best) / (sigma_acq + 1e-9)
    EI_acq = (mu_acq - y_best) * norm.cdf(Z_acq) + sigma_acq * norm.pdf(Z_acq)
    EI_acq[sigma_acq < 1e-9] = 0.0

    fig_acq = go.Figure()
    fig_acq.add_trace(go.Scatter(
        x=x_real, y=EI_acq,
        mode="lines", name="Expected Improvement",
        line=dict(color="#f59e0b", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.15)",
    ))
    fig_acq.add_vline(
        x=best_real[0], line_dash="dash", line_color="#ef4444",
        annotation_text="Next experiment",
    )
    fig_acq.update_layout(
        xaxis_title=f"{p0['name']} ({p0.get('units', '')})",
        yaxis_title="Expected Improvement",
        template="plotly_white",
        height=350, margin=dict(l=50, r=20, t=30, b=50),
    )

    return suggestion, fig_gp, fig_acq
