"""Decision-focused demand forecasting benchmark for a newsvendor decision.

The benchmark intentionally evaluates forecasts with both statistical accuracy
and downstream inventory cost. For asymmetric overage/underage costs, the
newsvendor-optimal order is the corresponding conditional demand quantile, so a
quantile forecaster trained with pinball loss is a natural decision-aware model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


@dataclass(frozen=True)
class NewsvendorCosts:
    underage: float = 5.0
    overage: float = 1.0

    def __post_init__(self) -> None:
        if self.underage <= 0 or self.overage <= 0:
            raise ValueError("underage and overage costs must be positive")

    @property
    def critical_fractile(self) -> float:
        return self.underage / (self.underage + self.overage)


def pinball_loss(actual: np.ndarray, forecast: np.ndarray, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    error = actual - forecast
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def newsvendor_cost(
    actual: np.ndarray,
    order: np.ndarray,
    costs: NewsvendorCosts,
) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    order = np.asarray(order, dtype=float)
    if actual.shape != order.shape:
        raise ValueError("actual and order must have the same shape")
    under = np.maximum(actual - order, 0.0)
    over = np.maximum(order - actual, 0.0)
    return costs.underage * under + costs.overage * over


def make_promotional_demand(
    n_points: int = 900,
    seed: int = 19,
) -> pd.DataFrame:
    """Create daily intermittent demand with trend, weekly seasonality and promotions."""
    if n_points < 120:
        raise ValueError("n_points must be at least 120")
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)
    weekday = np.arange(n_points) % 7
    promo = (rng.random(n_points) < 0.12).astype(int)

    latent = (
        7.5
        + 0.006 * t
        + 2.1 * np.sin(2.0 * np.pi * weekday / 7.0)
        + 7.0 * promo
    )
    latent = np.maximum(latent, 0.2)
    demand = rng.poisson(latent).astype(float)

    zero_probability = np.where(promo == 1, 0.04, 0.18)
    demand[rng.random(n_points) < zero_probability] = 0.0

    index = pd.date_range("2024-01-01", periods=n_points, freq="D")
    return pd.DataFrame({"demand": demand, "promo": promo}, index=index)


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = {"demand", "promo"}
    if not required.issubset(frame.columns):
        raise ValueError("frame must contain demand and promo columns")

    work = frame.copy()
    day = np.arange(len(work), dtype=float)
    work["trend"] = day / max(len(work) - 1, 1)
    work["dow_sin"] = np.sin(2.0 * np.pi * work.index.dayofweek / 7.0)
    work["dow_cos"] = np.cos(2.0 * np.pi * work.index.dayofweek / 7.0)
    work["lag_1"] = work["demand"].shift(1)
    work["lag_7"] = work["demand"].shift(7)
    work["rolling_mean_7"] = work["demand"].shift(1).rolling(7).mean()
    work["rolling_zero_rate_28"] = (
        work["demand"].eq(0).astype(float).shift(1).rolling(28, min_periods=7).mean()
    )
    work = work.dropna()

    feature_columns = [
        "promo",
        "trend",
        "dow_sin",
        "dow_cos",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_zero_rate_28",
    ]
    return work[feature_columns], work["demand"]


def _metrics(
    name: str,
    actual: np.ndarray,
    order: np.ndarray,
    costs: NewsvendorCosts,
) -> dict[str, float | str]:
    actual = np.asarray(actual, dtype=float)
    order = np.maximum(np.asarray(order, dtype=float), 0.0)
    costs_per_period = newsvendor_cost(actual, order, costs)
    fulfilled = np.minimum(actual, order)
    total_demand = float(actual.sum())
    return {
        "model": name,
        "rmse": float(np.sqrt(np.mean((actual - order) ** 2))),
        "pinball": pinball_loss(actual, order, costs.critical_fractile),
        "mean_newsvendor_cost": float(np.mean(costs_per_period)),
        "total_newsvendor_cost": float(np.sum(costs_per_period)),
        "fill_rate": float(fulfilled.sum() / total_demand) if total_demand > 0 else 1.0,
        "stockout_rate": float(np.mean(order + 1e-12 < actual)),
        "mean_order": float(np.mean(order)),
    }


def run_benchmark(
    n_points: int = 900,
    train_fraction: float = 0.70,
    seed: int = 19,
    costs: NewsvendorCosts | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit point and quantile forecasters and evaluate downstream decisions."""
    if not 0.5 <= train_fraction < 0.95:
        raise ValueError("train_fraction must be in [0.5, 0.95)")
    costs = costs or NewsvendorCosts()
    frame = make_promotional_demand(n_points=n_points, seed=seed)
    X, y = make_features(frame)
    split = int(len(X) * train_fraction)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    common_params = dict(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=8,
        random_state=seed,
    )
    mean_model = GradientBoostingRegressor(loss="squared_error", **common_params)
    quantile_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=costs.critical_fractile,
        **common_params,
    )
    mean_model.fit(X_train, y_train)
    quantile_model.fit(X_train, y_train)

    actual = y_test.to_numpy(dtype=float)
    mean_forecast = np.maximum(mean_model.predict(X_test), 0.0)
    quantile_forecast = np.maximum(quantile_model.predict(X_test), 0.0)
    seasonal_naive = np.maximum(X_test["lag_7"].to_numpy(dtype=float), 0.0)

    metrics = pd.DataFrame(
        [
            _metrics("seasonal_naive", actual, seasonal_naive, costs),
            _metrics("gb_point_mean", actual, mean_forecast, costs),
            _metrics(
                f"gb_quantile_q{costs.critical_fractile:.3f}",
                actual,
                quantile_forecast,
                costs,
            ),
        ]
    ).sort_values("mean_newsvendor_cost", ignore_index=True)

    predictions = pd.DataFrame(
        {
            "actual": actual,
            "seasonal_naive": seasonal_naive,
            "gb_point_mean": mean_forecast,
            "gb_quantile": quantile_forecast,
        },
        index=y_test.index,
    )
    return metrics, predictions


def save_artifacts(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: str | Path = "artifacts",
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "newsvendor_forecast_benchmark.csv", index=False)
    predictions.to_csv(output / "newsvendor_forecast_predictions.csv")

    sample = predictions.iloc[:120]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample.index, sample["actual"], label="actual", linewidth=1.8)
    ax.plot(sample.index, sample["gb_point_mean"], label="point mean", alpha=0.9)
    ax.plot(sample.index, sample["gb_quantile"], label="decision quantile", alpha=0.9)
    ax.set_title("Forecast accuracy vs newsvendor ordering decision")
    ax.set_ylabel("daily units")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "newsvendor_forecast_benchmark.png", dpi=150)
    plt.close(fig)


def main() -> None:
    costs = NewsvendorCosts(underage=5.0, overage=1.0)
    metrics, predictions = run_benchmark(costs=costs)
    save_artifacts(metrics, predictions)
    print(f"critical fractile: {costs.critical_fractile:.3f}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
