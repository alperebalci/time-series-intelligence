import numpy as np
import pytest

from src.newsvendor_benchmark import (
    NewsvendorCosts,
    newsvendor_cost,
    pinball_loss,
    run_benchmark,
)


def test_critical_fractile_matches_newsvendor_ratio():
    costs = NewsvendorCosts(underage=5.0, overage=1.0)
    assert costs.critical_fractile == pytest.approx(5.0 / 6.0)


def test_newsvendor_cost_penalizes_underage_and_overage_asymmetrically():
    actual = np.array([10.0, 10.0])
    order = np.array([8.0, 13.0])
    costs = NewsvendorCosts(underage=5.0, overage=1.0)
    result = newsvendor_cost(actual, order, costs)
    assert result.tolist() == pytest.approx([10.0, 3.0])


def test_pinball_loss_is_zero_for_exact_forecast():
    values = np.array([0.0, 2.0, 7.0])
    assert pinball_loss(values, values, 0.8) == pytest.approx(0.0)


def test_decision_focused_benchmark_is_reproducible_and_complete():
    metrics1, predictions1 = run_benchmark(n_points=360, seed=17)
    metrics2, predictions2 = run_benchmark(n_points=360, seed=17)
    assert metrics1.equals(metrics2)
    assert predictions1.equals(predictions2)
    assert set(metrics1["model"]) == {
        "seasonal_naive",
        "gb_point_mean",
        "gb_quantile_q0.833",
    }
    assert np.isfinite(metrics1["mean_newsvendor_cost"]).all()
    assert (metrics1["fill_rate"].between(0.0, 1.0)).all()
