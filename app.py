import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


st.set_page_config(page_title="Uplift Simulator", layout="wide")


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_models(workdir: Path):
    t_bundle = None
    s_bundle = None

    t_path = workdir / "t_learner.pkl"
    s_path = workdir / "s_learner.pkl"

    if t_path.exists():
        t_bundle = _load_pickle(t_path)
    if s_path.exists():
        s_bundle = _load_pickle(s_path)

    return t_bundle, s_bundle


def ensure_columns(df: pd.DataFrame, required_cols: list[str]):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input data: {missing}")


def cast_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def predict_uplift(model_choice: str, data: pd.DataFrame, t_bundle, s_bundle):
    if model_choice == "T-learner":
        if t_bundle is None:
            raise ValueError("t_learner.pkl not found.")
        feature_cols = t_bundle["feature_cols"]
        categorical_cols = t_bundle.get("categorical_cols", [])
        ensure_columns(data, feature_cols)
        x = cast_categoricals(data[feature_cols], categorical_cols)

        treated_model = t_bundle["treated_model"]
        control_model = t_bundle["control_model"]

        mu1 = treated_model.predict_proba(x)[:, 1]
        mu0 = control_model.predict_proba(x)[:, 1]
        return mu1, mu0

    if model_choice == "S-learner":
        if s_bundle is None:
            raise ValueError("s_learner.pkl not found.")
        feature_cols = s_bundle["feature_cols"]
        treatment_col = s_bundle.get("treatment_col", "treatment")
        categorical_cols = s_bundle.get("categorical_cols", [])

        ensure_columns(data, [c for c in feature_cols if c != treatment_col])

        x_base = data.copy()
        x_treated = x_base.copy()
        x_control = x_base.copy()
        x_treated[treatment_col] = 1
        x_control[treatment_col] = 0

        ensure_columns(x_treated, feature_cols)
        ensure_columns(x_control, feature_cols)

        x_treated = cast_categoricals(x_treated[feature_cols], categorical_cols)
        x_control = cast_categoricals(x_control[feature_cols], categorical_cols)

        model = s_bundle["model"]
        mu1 = model.predict_proba(x_treated)[:, 1]
        mu0 = model.predict_proba(x_control)[:, 1]
        return mu1, mu0

    raise ValueError("Unknown model type.")


def build_target_mask(strategy: str, uplift: np.ndarray, target_frac: float, rng: np.random.Generator):
    n = uplift.shape[0]
    target_n = int(round(n * target_frac))
    target_n = max(0, min(target_n, n))

    if strategy == "Treat none":
        return np.zeros(n, dtype=bool)
    if strategy == "Treat all":
        return np.ones(n, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    if target_n == 0:
        return mask

    if strategy == "Random split":
        idx = rng.choice(n, size=target_n, replace=False)
        mask[idx] = True
        return mask

    # Optimized uplift targeting.
    idx = np.argsort(-uplift)[:target_n]
    mask[idx] = True
    return mask


def evaluate_strategy(name: str, mask: np.ndarray, mu1: np.ndarray, mu0: np.ndarray, price: float, unit_cost: float, contact_cost: float):
    p = np.where(mask, mu1, mu0)
    expected_conversions = float(np.sum(p))
    users = len(p)
    targeted_users = int(mask.sum())
    conversion_rate = expected_conversions / users if users > 0 else 0.0
    treated_conversion_rate = float(np.mean(mu1[mask])) if targeted_users > 0 else 0.0
    revenue = expected_conversions * price
    variable_cost = expected_conversions * unit_cost
    marketing_cost = targeted_users * contact_cost
    profit = revenue - variable_cost - marketing_cost

    return {
        "Scenario": name,
        "Users": users,
        "Targeted users": targeted_users,
        "% of people treated": targeted_users / users if users > 0 else 0.0,
        "Expected conversions": expected_conversions,
        "Expected conversion rate": conversion_rate,
        "Expected treated conversion rate": treated_conversion_rate,
        "Expected revenue": revenue,
        "Expected variable cost": variable_cost,
        "Expected marketing cost": marketing_cost,
        "Expected profit": profit,
    }


def find_best_target_pct(
    uplift: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
    price: float,
    unit_cost: float,
    contact_cost: float,
    objective: str,
):
    objective_to_metric = {
        "Revenue": "Expected revenue",
        "Profit": "Expected profit",
    }
    metric_col = objective_to_metric[objective]

    best_pct = 0
    best_value = -np.inf

    for pct in range(0, 101):
        frac = pct / 100.0
        mask = build_target_mask("Optimized uplift", uplift, frac, np.random.default_rng())
        metrics = evaluate_strategy("Optimized uplift", mask, mu1, mu0, price, unit_cost, contact_cost)
        value = float(metrics[metric_col])
        if value > best_value:
            best_value = value
            best_pct = pct

    return best_pct, best_value, metric_col


def plot_sensitivity_with_split_marker(curve_df: pd.DataFrame, value_cols: list[str], target_frac: float, y_axis_title: str):
    plot_df = curve_df.reset_index().melt(
        id_vars="% of people treated",
        value_vars=value_cols,
        var_name="Series",
        value_name="value",
    )
    marker_df = pd.DataFrame(
        {
            "% of people treated": [float(target_frac)],
            "Marker": ["Current split"],
        }
    )

    lines = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("% of people treated:Q", axis=alt.Axis(format="%")),
            y=alt.Y("value:Q", title=y_axis_title),
            color=alt.Color("Series:N", title="Series"),
            tooltip=[
                "Series:N",
                alt.Tooltip("% of people treated:Q", format=".0%"),
                alt.Tooltip("value:Q", format=",.4f"),
            ],
        )
    )
    split_rule = (
        alt.Chart(marker_df)
        .mark_rule(strokeDash=[4, 4], size=2)
        .encode(
            x=alt.X("% of people treated:Q"),
            color=alt.Color("Marker:N", legend=alt.Legend(title="Marker")),
            tooltip=["Marker:N", alt.Tooltip("% of people treated:Q", format=".0%")],
        )
    )
    st.altair_chart((lines + split_rule).interactive(), use_container_width=True)


def main():
    st.title("Uplift Modeling Scenario Simulator")
    st.caption("Compare expected conversion rate, revenue, and profit under different treatment strategies.")

    workdir = Path(__file__).resolve().parent
    t_bundle, s_bundle = load_models(workdir)

    if t_bundle is None and s_bundle is None:
        st.error("No model files found. Add s_learner.pkl and/or t_learner.pkl in this folder.")
        st.stop()

    # Apply pending optimized split before creating the slider widget.
    if "pending_target_pct" in st.session_state:
        st.session_state.target_pct = int(st.session_state.pending_target_pct)
        del st.session_state["pending_target_pct"]

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV for simulation", type=["csv"])
        default_data_path = workdir / "test.csv"

        if uploaded is not None:
            sim_df = pd.read_csv(uploaded)
            st.success("Using uploaded data.")
        elif default_data_path.exists():
            sim_df = load_data(default_data_path)
            st.info("Using local test.csv.")
        else:
            st.error("No data file found. Upload a CSV.")
            st.stop()

        available_models = []
        if t_bundle is not None:
            available_models.append("T-learner")
        if s_bundle is not None:
            available_models.append("S-learner")

        model_choice = st.selectbox("Model", available_models)

        st.header("Business Assumptions")
        price = st.number_input("Price per unit", min_value=0.0, value=25.0, step=0.5)
        unit_cost = st.number_input("Variable cost per unit", min_value=0.0, value=8.0, step=0.5)
        contact_cost = st.number_input("Marketing cost per treated user", min_value=0.0, value=0.01, step=0.05)

        st.header("Scenario Controls")
        if "target_pct" not in st.session_state:
            st.session_state.target_pct = 30

        target_pct = st.slider("% of people treated", min_value=0, max_value=100, step=1, key="target_pct")
        auto_objective = st.selectbox(
            "Optimize split for",
            ["Profit", "Revenue"],
        )
        auto_optimize_clicked = st.button("Find optimal split")

    try:
        mu1, mu0 = predict_uplift(model_choice, sim_df, t_bundle, s_bundle)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    uplift = mu1 - mu0
    rng = np.random.default_rng()
    target_frac = target_pct / 100.0

    if auto_optimize_clicked:
        best_pct, best_value, metric_col = find_best_target_pct(
            uplift=uplift,
            mu1=mu1,
            mu0=mu0,
            price=price,
            unit_cost=unit_cost,
            contact_cost=contact_cost,
            objective=auto_objective,
        )
        st.session_state.pending_target_pct = int(best_pct)
        st.success(f"Optimal split found: {best_pct}% (best {auto_objective.lower()}: ${best_value:,.2f}).")
        st.rerun()

    scenario_defs = [
        ("Treat none", "Treat none"),
        ("Treat all", "Treat all"),
        ("Random split", "Random split"),
        ("Optimized uplift", "Optimized uplift"),
    ]

    rows = []
    for scenario_name, strategy in scenario_defs:
        mask = build_target_mask(strategy, uplift, target_frac, rng)
        rows.append(evaluate_strategy(scenario_name, mask, mu1, mu0, price, unit_cost, contact_cost))

    summary = pd.DataFrame(rows)

    baseline_conversions = float(summary.loc[summary["Scenario"] == "Treat none", "Expected conversions"].iloc[0])
    baseline_revenue = float(summary.loc[summary["Scenario"] == "Treat none", "Expected revenue"].iloc[0])
    baseline_profit = float(summary.loc[summary["Scenario"] == "Treat none", "Expected profit"].iloc[0])

    summary["Incremental conversions vs Treat none"] = summary["Expected conversions"] - baseline_conversions
    summary["Incremental revenue vs Treat none"] = summary["Expected revenue"] - baseline_revenue
    summary["Incremental profit vs Treat none"] = summary["Expected profit"] - baseline_profit

    st.subheader("Scenario Comparison")
    summary_display = summary.copy()
    summary_display["% of people treated"] = summary_display["% of people treated"].map(lambda v: f"{v:.1%}")
    summary_display["Expected conversions"] = summary_display["Expected conversions"].map(lambda v: f"{v:.2f}")
    summary_display["Expected conversion rate"] = summary_display["Expected conversion rate"].map(lambda v: f"{v:.3%}")
    summary_display["Expected treated conversion rate"] = summary_display["Expected treated conversion rate"].map(lambda v: f"{v:.3%}")
    summary_display["Expected revenue"] = summary_display["Expected revenue"].map(lambda v: f"${v:,.2f}")
    summary_display["Expected variable cost"] = summary_display["Expected variable cost"].map(lambda v: f"${v:,.2f}")
    summary_display["Expected marketing cost"] = summary_display["Expected marketing cost"].map(lambda v: f"${v:,.2f}")
    summary_display["Expected profit"] = summary_display["Expected profit"].map(lambda v: f"${v:,.2f}")
    summary_display["Incremental conversions vs Treat none"] = summary_display["Incremental conversions vs Treat none"].map(lambda v: f"{v:.2f}")
    summary_display["Incremental revenue vs Treat none"] = summary_display["Incremental revenue vs Treat none"].map(lambda v: f"${v:,.2f}")
    summary_display["Incremental profit vs Treat none"] = summary_display["Incremental profit vs Treat none"].map(lambda v: f"${v:,.2f}")
    st.table(summary_display)

    optimized_row = summary[summary["Scenario"] == "Optimized uplift"].iloc[0]
    random_row = summary[summary["Scenario"] == "Random split"].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimized conversion rate", f"{optimized_row['Expected conversion rate']:.2%}", f"{optimized_row['Expected conversion rate'] - random_row['Expected conversion rate']:.2%} vs random")
    c2.metric("Optimized expected revenue", f"${optimized_row['Expected revenue']:,.2f}", f"${optimized_row['Expected revenue'] - random_row['Expected revenue']:,.2f} vs random")
    c3.metric("Optimized expected profit", f"${optimized_row['Expected profit']:,.2f}", f"${optimized_row['Expected profit'] - random_row['Expected profit']:,.2f} vs random")

    st.subheader("Profit by % of People Treated")
    rates = np.linspace(0, 1, 21)
    curve_rows = []
    for r in rates:
        random_mask = build_target_mask("Random split", uplift, float(r), rng)
        optimized_mask = build_target_mask("Optimized uplift", uplift, float(r), rng)

        random_metrics = evaluate_strategy("Random split", random_mask, mu1, mu0, price, unit_cost, contact_cost)
        optimized_metrics = evaluate_strategy("Optimized uplift", optimized_mask, mu1, mu0, price, unit_cost, contact_cost)

        curve_rows.append(
            {
                "% of people treated": r,
                "Random profit": random_metrics["Expected profit"],
                "Optimized profit": optimized_metrics["Expected profit"],
                "Random revenue": random_metrics["Expected revenue"],
                "Optimized revenue": optimized_metrics["Expected revenue"],
                "Random conversions": random_metrics["Expected conversions"],
                "Optimized conversions": optimized_metrics["Expected conversions"],
            }
        )

    curve_df = pd.DataFrame(curve_rows).set_index("% of people treated")
    plot_sensitivity_with_split_marker(
        curve_df=curve_df,
        value_cols=["Random profit", "Optimized profit"],
        target_frac=target_frac,
        y_axis_title="Expected profit",
    )

    st.subheader("Revenue by % of People Treated")
    plot_sensitivity_with_split_marker(
        curve_df=curve_df,
        value_cols=["Random revenue", "Optimized revenue"],
        target_frac=target_frac,
        y_axis_title="Expected revenue",
    )

    st.subheader("Conversion by % of People Treated")
    plot_sensitivity_with_split_marker(
        curve_df=curve_df,
        value_cols=["Random conversions", "Optimized conversions"],
        target_frac=target_frac,
        y_axis_title="Expected conversions",
    )

    results = sim_df.copy()
    results["p_conversion_treated"] = mu1
    results["p_conversion_control"] = mu0
    results["uplift"] = uplift
    results["recommended_treatment"] = (np.argsort(np.argsort(-uplift)) < int(round(len(uplift) * target_frac))).astype(int)

    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored results (CSV)",
        data=csv_data,
        file_name="uplift_scored_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
