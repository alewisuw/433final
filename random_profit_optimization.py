from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BusinessCondition:
    price: float
    unit_cost: float
    contact_cost: float


def load_bundle(bundle_path: Path) -> dict:
    with bundle_path.open("rb") as f:
        return pickle.load(f)


def predict_mu1_mu0_t_learner(data: pd.DataFrame, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = bundle["feature_cols"]
    x = data[feature_cols].copy()

    for col in bundle.get("categorical_cols", []):
        if col in x.columns:
            x[col] = x[col].astype("category")

    treated_model = bundle["treated_model"]
    control_model = bundle["control_model"]
    mu1 = treated_model.predict_proba(x)[:, 1]
    mu0 = control_model.predict_proba(x)[:, 1]
    return mu1, mu0


def evaluate(mask: np.ndarray, mu1: np.ndarray, mu0: np.ndarray, c: BusinessCondition) -> dict[str, float]:
    p = np.where(mask, mu1, mu0)
    expected_conversions = float(np.sum(p))
    expected_revenue = expected_conversions * c.price
    expected_profit = expected_revenue - (expected_conversions * c.unit_cost) - (int(mask.sum()) * c.contact_cost)
    return {
        "conversions": expected_conversions,
        "revenue": expected_revenue,
        "profit": expected_profit,
    }


def metrics_from_conversions_and_k(conversions: float, k: int, c: BusinessCondition) -> dict[str, float]:
    revenue = conversions * c.price
    profit = revenue - (conversions * c.unit_cost) - (k * c.contact_cost)
    return {
        "conversions": float(conversions),
        "revenue": float(revenue),
        "profit": float(profit),
    }


def pick_top_uplift_mask(uplift: np.ndarray, target_frac: float) -> np.ndarray:
    n = uplift.size
    k = int(round(n * float(np.clip(target_frac, 0.0, 1.0))))
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    idx = np.argsort(-uplift)[:k]
    mask[idx] = True
    return mask


def pick_random_mask(n: int, target_frac: float, rng: np.random.Generator) -> np.ndarray:
    k = int(round(n * float(np.clip(target_frac, 0.0, 1.0))))
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    idx = rng.choice(n, size=k, replace=False)
    mask[idx] = True
    return mask


def optimize_for_profit(
    conversions_by_k: np.ndarray,
    condition: BusinessCondition,
) -> tuple[int, dict[str, float]]:
    n = conversions_by_k.size - 1
    k_grid = np.rint((np.arange(0, 101) / 100.0) * n).astype(int)

    candidate_conversions = conversions_by_k[k_grid]
    candidate_revenue = candidate_conversions * condition.price
    candidate_profit = candidate_revenue - (candidate_conversions * condition.unit_cost) - (k_grid * condition.contact_cost)

    best_idx = int(np.argmax(candidate_profit))
    best_k = int(k_grid[best_idx])
    best_metrics = {
        "conversions": float(candidate_conversions[best_idx]),
        "revenue": float(candidate_revenue[best_idx]),
        "profit": float(candidate_profit[best_idx]),
    }
    return best_k, best_metrics


def optimize_for_visits(visits_by_k: np.ndarray) -> tuple[int, dict[str, float]]:
    n = visits_by_k.size - 1
    k_grid = np.rint((np.arange(0, 101) / 100.0) * n).astype(int)
    candidate_visits = visits_by_k[k_grid]

    best_idx = int(np.argmax(candidate_visits))
    best_k = int(k_grid[best_idx])
    best_metrics = {
        "visits": float(candidate_visits[best_idx]),
    }
    return best_k, best_metrics


def sample_condition(rng: np.random.Generator) -> BusinessCondition:
    # Keep margins positive and realistic for simulation.
    price = float(rng.uniform(60.0, 240.0))
    unit_cost = float(rng.uniform(10.0, min(price * 0.75, 140.0)))
    contact_cost = float(rng.uniform(0.5, 12.0))
    return BusinessCondition(price=price, unit_cost=unit_cost, contact_cost=contact_cost)


def run_simulation(data_path: Path, model_path: Path, n_conditions: int = 50, seed: int = 42) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    bundle = load_bundle(model_path)
    mu1, mu0 = predict_mu1_mu0_t_learner(data, bundle)
    uplift = mu1 - mu0
    n = uplift.size

    # Precompute conversions for targeting top-k uplift users once; reused across all conditions.
    order = np.argsort(-uplift)
    delta_sorted = uplift[order]
    total_mu0 = float(np.sum(mu0))
    cumsum_delta = np.concatenate(([0.0], np.cumsum(delta_sorted, dtype=float)))
    conversions_by_k = total_mu0 + cumsum_delta

    sum_mu1 = float(np.sum(mu1))
    sum_mu0 = float(np.sum(mu0))
    rng = np.random.default_rng(seed)

    rows: list[dict[str, float]] = []
    for i in range(n_conditions):
        condition = sample_condition(rng)
        best_k, optimized = optimize_for_profit(conversions_by_k, condition)

        # Expected random baseline at the same treatment volume.
        treat_frac = best_k / n if n > 0 else 0.0
        baseline_conversions = (treat_frac * sum_mu1) + ((1.0 - treat_frac) * sum_mu0)
        baseline = metrics_from_conversions_and_k(baseline_conversions, best_k, condition)

        rows.append(
            {
                "run": i + 1,
                "best_treatment_frac": treat_frac,
                "price": condition.price,
                "unit_cost": condition.unit_cost,
                "contact_cost": condition.contact_cost,
                "optimized_profit": optimized["profit"],
                "baseline_profit": baseline["profit"],
                "profit_increase": optimized["profit"] - baseline["profit"],
                "optimized_revenue": optimized["revenue"],
                "baseline_revenue": baseline["revenue"],
                "revenue_increase": optimized["revenue"] - baseline["revenue"],
                "optimized_conversions": optimized["conversions"],
                "baseline_conversions": baseline["conversions"],
                "conversions_increase": optimized["conversions"] - baseline["conversions"],
            }
        )

    return pd.DataFrame(rows)


def run_visit_simulation(data_path: Path, model_path: Path, n_conditions: int = 50, seed: int = 42) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    bundle = load_bundle(model_path)
    mu1, mu0 = predict_mu1_mu0_t_learner(data, bundle)
    uplift = mu1 - mu0
    n = uplift.size

    order = np.argsort(-uplift)
    delta_sorted = uplift[order]
    total_mu0 = float(np.sum(mu0))
    cumsum_delta = np.concatenate(([0.0], np.cumsum(delta_sorted, dtype=float)))
    visits_by_k = total_mu0 + cumsum_delta

    sum_mu1 = float(np.sum(mu1))
    sum_mu0 = float(np.sum(mu0))
    rng = np.random.default_rng(seed)

    rows: list[dict[str, float]] = []
    for i in range(n_conditions):
        _ = rng.random()
        best_k, optimized = optimize_for_visits(visits_by_k)

        # Expected random baseline at the same treatment volume.
        treat_frac = best_k / n if n > 0 else 0.0
        baseline_visits = (treat_frac * sum_mu1) + ((1.0 - treat_frac) * sum_mu0)

        rows.append(
            {
                "run": i + 1,
                "best_treatment_frac": treat_frac,
                "optimized_visits": optimized["visits"],
                "baseline_visits": baseline_visits,
                "visits_increase": optimized["visits"] - baseline_visits,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    results = run_simulation(
        data_path=root / "test.csv",
        model_path=root / "models" / "t_learner.pkl",
        n_conditions=50,
        seed=42,
    )
    visit_results = run_visit_simulation(
        data_path=root / "test.csv",
        model_path=root / "models" / "t_learner-visit.pkl",
        n_conditions=50,
        seed=42,
    )

    def pct_lift(num: pd.Series, den: pd.Series) -> pd.Series:
        den_safe = den.replace(0, np.nan)
        return ((num - den_safe) / den_safe) * 100.0

    results["profit_increase_pct"] = pct_lift(results["optimized_profit"], results["baseline_profit"])
    results["revenue_increase_pct"] = pct_lift(results["optimized_revenue"], results["baseline_revenue"])
    results["conversions_increase_pct"] = pct_lift(results["optimized_conversions"], results["baseline_conversions"])

    avg_optimized_profit = float(results["optimized_profit"].mean())
    avg_baseline_profit = float(results["baseline_profit"].mean())
    avg_optimized_revenue = float(results["optimized_revenue"].mean())
    avg_baseline_revenue = float(results["baseline_revenue"].mean())
    avg_optimized_conversions = float(results["optimized_conversions"].mean())
    avg_baseline_conversions = float(results["baseline_conversions"].mean())

    avg_profit_increase_pct = float(results["profit_increase_pct"].mean(skipna=True))
    avg_revenue_increase_pct = float(results["revenue_increase_pct"].mean(skipna=True))
    avg_conversions_increase_pct = float(results["conversions_increase_pct"].mean(skipna=True))

    print("Average strategy metrics over 50 random conditions:")
    print(f"- Profit (optimized):      {avg_optimized_profit:,.4f}")
    print(f"- Profit (random baseline): {avg_baseline_profit:,.4f}")
    print(f"- Revenue (optimized):     {avg_optimized_revenue:,.4f}")
    print(f"- Revenue (random baseline): {avg_baseline_revenue:,.4f}")
    print(f"- Conversions (optimized): {avg_optimized_conversions:,.4f}")
    print(f"- Conversions (random baseline): {avg_baseline_conversions:,.4f}")

    print("Average percent increase over 50 random conditions (optimized vs random baseline):")
    print(f"- Profit:      {avg_profit_increase_pct:,.4f}%")
    print(f"- Revenue:     {avg_revenue_increase_pct:,.4f}%")
    print(f"- Conversions: {avg_conversions_increase_pct:,.4f}%")

    visit_results["visits_increase_pct"] = pct_lift(visit_results["optimized_visits"], visit_results["baseline_visits"])
    avg_optimized_visits = float(visit_results["optimized_visits"].mean())
    avg_baseline_visits = float(visit_results["baseline_visits"].mean())
    avg_visits_increase_pct = float(visit_results["visits_increase_pct"].mean(skipna=True))

    print("\nAverage visit strategy metrics over 50 random conditions:")
    print(f"- Visits (optimized):        {avg_optimized_visits:,.4f}")
    print(f"- Visits (random baseline):  {avg_baseline_visits:,.4f}")
    print("Average visit percent increase over 50 random conditions (optimized vs random baseline):")
    print(f"- Visits: {avg_visits_increase_pct:,.4f}%")

    out_path = root / "profit_optimization_50_runs.csv"
    results.to_csv(out_path, index=False)
    print(f"\nDetailed per-run results saved to: {out_path}")

    visit_out_path = root / "visit_optimization_50_runs.csv"
    visit_results.to_csv(visit_out_path, index=False)
    print(f"Detailed visit per-run results saved to: {visit_out_path}")


if __name__ == "__main__":
    main()
