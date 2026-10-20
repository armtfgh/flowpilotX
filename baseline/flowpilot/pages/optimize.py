"""FLORA — Condition Optimization (Bayesian Optimization) page."""

import streamlit as st


def render():
    st.title("Condition Optimization")
    st.markdown(
        "Provide your experimental observations and FLORA will suggest "
        "the next conditions to run using Bayesian optimization."
    )

    # ── Session state init ──
    if "bo_observations" not in st.session_state:
        st.session_state.bo_observations = []

    # ── Step 1: Define parameter space ──
    with st.expander("Step 1 — Define parameter space", expanded=True):
        n_params = st.number_input("Number of parameters", 1, 4, value=2)

        param_defs = []
        defaults = [
            ("residence_time_min", 1.0, 30.0, "min"),
            ("temperature_C", 0.0, 100.0, "°C"),
            ("concentration_M", 0.01, 1.0, "M"),
            ("flow_rate_mL_min", 0.05, 5.0, "mL/min"),
        ]
        for i in range(n_params):
            cols = st.columns(4)
            d = defaults[i] if i < len(defaults) else (f"param_{i+1}", 0.0, 10.0, "")
            name = cols[0].text_input("Name", value=d[0], key=f"pname_{i}")
            pmin = cols[1].number_input("Min", value=d[1], key=f"pmin_{i}")
            pmax = cols[2].number_input("Max", value=d[2], key=f"pmax_{i}")
            units = cols[3].text_input("Units", value=d[3], key=f"punits_{i}")
            param_defs.append({"name": name, "min": pmin, "max": pmax, "units": units})

        col1, col2 = st.columns(2)
        with col1:
            objective = st.selectbox(
                "Objective",
                ["yield (%)", "conversion (%)", "selectivity (%)",
                 "space-time yield (g/L/h)", "custom"],
            )
        with col2:
            maximize = st.radio("Direction", ["Maximize", "Minimize"], horizontal=True)

    # ── Step 2: Enter observations ──
    with st.expander("Step 2 — Enter observations", expanded=True):
        with st.form("add_obs"):
            obs_cols = st.columns(n_params + 2)
            obs_vals = {}
            for i, p in enumerate(param_defs):
                obs_vals[p["name"]] = obs_cols[i].number_input(
                    f"{p['name']}", key=f"obs_{i}"
                )
            obs_result = obs_cols[n_params].number_input("Result", key="obs_result")
            obs_note = obs_cols[n_params + 1].text_input("Note", key="obs_note")
            if st.form_submit_button("Add observation"):
                st.session_state.bo_observations.append({
                    **obs_vals, "result": obs_result, "note": obs_note
                })

        # CSV upload
        csv = st.file_uploader("Or upload CSV", type=["csv"], key="bo_csv")
        if csv:
            import pandas as pd
            df = pd.read_csv(csv)
            st.session_state.bo_observations = df.to_dict("records")
            st.success(f"Loaded {len(df)} observations.")

        # Show current observations
        if st.session_state.bo_observations:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(st.session_state.bo_observations),
                use_container_width=True, hide_index=True,
            )
            if st.button("Clear all"):
                st.session_state.bo_observations = []
                st.rerun()

    # ── Step 3: Run BO ──
    n_obs = len(st.session_state.bo_observations)
    if n_obs < 2:
        st.info(f"Add at least 2 observations to enable BO ({n_obs}/2).")
        return

    if st.button("Suggest next experiment", type="primary", use_container_width=True):
        with st.spinner("Fitting Gaussian Process..."):
            try:
                from flora_optimize.bo_engine import suggest_next

                suggestion, fig_gp, fig_acq = suggest_next(
                    observations=st.session_state.bo_observations,
                    param_defs=param_defs,
                    objective=objective.split(" ")[0],
                    maximize=(maximize == "Maximize"),
                )

                st.subheader("Suggested next experiment")
                sug_cols = st.columns(len(param_defs) + 1)
                for i, p in enumerate(param_defs):
                    sug_cols[i].metric(
                        f"{p['name']} ({p['units']})",
                        f"{suggestion['params'][p['name']]:.3g}",
                    )
                sug_cols[-1].metric("Expected improvement", f"{suggestion['expected_improvement']:.3f}")

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("GP Surrogate Model")
                    st.plotly_chart(fig_gp, use_container_width=True)
                with c2:
                    st.subheader("Acquisition Function (EI)")
                    st.plotly_chart(fig_acq, use_container_width=True)

                import json
                st.download_button(
                    "Download suggestion (JSON)",
                    json.dumps(suggestion, indent=2),
                    "bo_suggestion.json", "application/json",
                )

            except Exception as e:
                from components.error_card import render_error
                render_error(e, "Bayesian Optimization")
