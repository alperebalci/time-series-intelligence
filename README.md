# Time Series Intelligence

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is intentionally maintained as a standalone primary repository in the consolidated Jors Academy portfolio. It is not used as a container for other projects.
<!-- portfolio-umbrella:end -->

A reproducible project demonstrating **forecasting**, **anomaly detection**, **multi-model benchmarking**, and **production-like rolling backtesting** with Salesforce's [Merlion](https://github.com/salesforce/Merlion) time-series library.

The repository contains four workflows:

1. a small synthetic end-to-end demo,
2. a multi-model benchmark using real US macroeconomic data packaged with `statsmodels`,
3. a production-like evaluator workflow with rolling retraining and sliding/expanding training windows, and
4. a full **model × retraining-policy matrix** for walk-forward comparison, and\n5. a **decision-focused newsvendor benchmark** that compares point forecasts with critical-fractile quantile forecasts using downstream ordering cost.

## What this repository demonstrates

- Merlion `TimeSeries` conversion from pandas data
- Forecasting with multiple Merlion models
- Anomaly detection with multiple Merlion detectors
- Forecast evaluation with sMAPE and RMSE
- Anomaly evaluation with precision, recall, and F1
- Merlion `ForecastEvaluator` for historical deployment simulation
- Merlion `TSADEvaluator` for live-style anomaly detection simulation
- Train-once, expanding-window, and sliding-window retraining policies
- Full model × policy comparison matrices
- Model runtime measurement
- Reproducible synthetic data generation
- A real-data benchmark with no external data download
- CSV benchmark tables and PNG comparison plots\n- Newsvendor critical-fractile quantile regression with pinball loss\n- Decision-cost evaluation alongside RMSE, fill rate, and stockout rate

## Models benchmarked

### Forecasting

- `DefaultForecaster`
- `ARIMA`
- `ETS`
- `Prophet`

The forecasting benchmark uses quarterly US real GDP from `statsmodels.datasets.macrodata`.

### Anomaly detection

- `DefaultDetector`
- `IsolationForest`
- `StatThreshold`

Because the macroeconomic dataset does not contain anomaly labels, anomaly detection is evaluated on **real GDP growth with controlled point anomalies injected only into the held-out test period**. This keeps the underlying signal real while providing known ground truth for precision, recall, and F1.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── src
    ├── __init__.py
    ├── data.py
    ├── run_demo.py
    ├── benchmark.py
    ├── production_backtest.py
    └── model_policy_matrix.py
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This project pins Merlion to `salesforce-merlion==2.0.2`.

## Run the synthetic demo

```bash
python -m src.run_demo
```

It creates:

- `artifacts/forecast.png`
- `artifacts/anomaly_detection.png`
- `artifacts/metrics.csv`

## Run the real-data benchmark

```bash
python -m src.benchmark
```

It creates:

- `artifacts/forecast_benchmark.csv` — forecasting model ranking by sMAPE
- `artifacts/anomaly_benchmark.csv` — detector ranking by F1
- `artifacts/forecast_benchmark.png` — actual GDP and model forecasts
- `artifacts/anomaly_benchmark.png` — true and detected anomalies

The benchmark is fault-tolerant: if an optional model fails because of a platform-specific dependency, the remaining models still run and the failure is recorded in the `status` column.

## Run the decision-focused newsvendor benchmark

```bash
python -m src.newsvendor_benchmark
```

This workflow generates intermittent daily demand with promotion effects, fits a point-mean gradient-boosting model and a quantile gradient-boosting model, and evaluates both as ordering policies. With underage cost `c_u` and overage cost `c_o`, the benchmark uses the newsvendor critical fractile

```text
tau = c_u / (c_u + c_o)
```

and trains the decision-aware model directly at quantile `tau` using pinball loss.

The key comparison is intentionally not only forecast error. The output reports RMSE, pinball loss, mean and total newsvendor cost, fill rate, stockout rate, and mean order quantity. This makes it possible for a model with slightly worse RMSE to be preferred when it produces materially better operational decisions under asymmetric costs.

It creates:

- `artifacts/newsvendor_forecast_benchmark.csv`
- `artifacts/newsvendor_forecast_predictions.csv`
- `artifacts/newsvendor_forecast_benchmark.png`

Pretrained foundation models such as Chronos, TimesFM, or Moirai are deliberately not hard dependencies here because model-weight downloads would make the core benchmark network-dependent. Their forecasts can be evaluated with the same `newsvendor_cost` and `pinball_loss` functions, which keeps model selection separate from the downstream decision test.

## Run the production-like backtest

```bash
python -m src.production_backtest
```

This workflow uses Merlion's evaluator layer rather than fitting a model once and forecasting the entire test set in one batch.

Three deployment policies are compared:

| Policy | Retraining | Training history |
|---|---|---|
| `train_once` | never | initial training set only |
| `expanding_annual` | approximately yearly | all history available up to each retraining point |
| `sliding_10y_annual` | approximately yearly | most recent 10 years only |

For forecasting, the workflow uses `ForecastEvaluator` with ARIMA and a roughly one-quarter prediction horizon. For anomaly detection, it uses `TSADEvaluator` with Isolation Forest.

It creates:

- `artifacts/production_forecast_backtest.csv`
- `artifacts/production_anomaly_backtest.csv`
- `artifacts/production_forecast_backtest.png`
- `artifacts/production_anomaly_backtest.png`

## Run the model × retraining-policy matrix

```bash
python -m src.model_policy_matrix
```

This is the most comprehensive workflow in the repository. It evaluates each model under each deployment policy using Merlion's walk-forward evaluator framework.

### Forecast matrix

The following 9 combinations are evaluated:

| Model | `train_once` | `expanding_annual` | `sliding_10y_annual` |
|---|---:|---:|---:|
| ARIMA | ✓ | ✓ | ✓ |
| ETS | ✓ | ✓ | ✓ |
| Prophet | ✓ | ✓ | ✓ |

Each combination reports:

- sMAPE,
- RMSE,
- runtime,
- training-window policy,
- retraining frequency,
- execution status.

### Anomaly-detection matrix

The following 9 combinations are evaluated:

| Model | `train_once` | `expanding_annual` | `sliding_10y_annual` |
|---|---:|---:|---:|
| DefaultDetector | ✓ | ✓ | ✓ |
| IsolationForest | ✓ | ✓ | ✓ |
| StatThreshold | ✓ | ✓ | ✓ |

Each combination reports:

- precision,
- recall,
- F1,
- runtime,
- training-window policy,
- retraining frequency,
- execution status.

The matrix runner is also fault-tolerant. A failure in one model/policy pair is written to the `status` column and does not stop the remaining combinations.

It creates:

- `artifacts/forecast_model_policy_matrix.csv`
- `artifacts/anomaly_model_policy_matrix.csv`
- `artifacts/forecast_model_policy_matrix.png`
- `artifacts/anomaly_model_policy_matrix.png`

The PNG files are heatmaps: forecast combinations are compared by sMAPE, while anomaly combinations are compared by F1.

## Benchmark methodology

### Forecasting

The quarterly real-GDP series is split chronologically: 80% training and 20% testing. Static benchmarking fits each model on the same training set. Walk-forward workflows instead simulate historical deployment and periodically retrain according to the configured policy.

The primary forecast metric is **sMAPE**; lower is better. RMSE and runtime are also recorded in the evaluator-based workflows.

### Anomaly detection

Real GDP is converted to quarterly percentage growth. The first 80% is used as clean training history. Five deterministic high-magnitude point anomalies are injected into the held-out 20% only.

Each detector is evaluated using:

- precision,
- recall,
- F1 score,
- runtime.

Higher F1 is better.

## Why rolling retraining matters

A static holdout benchmark answers: "How well does this model work if trained once?" Production systems usually need a different answer: "How well does this model work as observations arrive and the model is periodically refreshed?"

Merlion's evaluator framework can:

- train an initial model,
- walk forward through unseen observations,
- generate predictions at a configured cadence,
- periodically reset and retrain the model,
- use either the full expanding history or a bounded sliding window,
- evaluate the resulting prediction sequence against ground truth.

`production_backtest.py` isolates the effect of retraining policy for one model per task. `model_policy_matrix.py` goes further by measuring the interaction between **model choice and retraining strategy**.

## Original synthetic workflow

The synthetic signal combines linear trend, daily and weekly seasonality, Gaussian noise, and deterministic injected anomalies. The first 70% is used for training and the remaining 30% for evaluation.

`DefaultForecaster` produces the forecast and `DefaultDetector` produces anomaly labels. This workflow is intentionally small and useful for learning the Merlion API before moving to the benchmark.

## Why Merlion?

Merlion provides a unified interface for time-series forecasting, anomaly detection, change-point detection, ensembles, AutoML-style model selection, evaluation pipelines, visualization, and distributed execution.

The upstream `salesforce/Merlion` repository is archived and read-only, so this repository is intended for learning, reproducible experiments, and comparative analysis rather than representing an actively maintained upstream stack.

## References

- Merlion source: https://github.com/salesforce/Merlion
- Merlion documentation: https://opensource.salesforce.com/Merlion/
- Merlion technical report: https://arxiv.org/abs/2109.09265
- statsmodels macrodata dataset: https://www.statsmodels.org/stable/datasets/generated/macrodata.html\n- Makridakis, Spiliotis, Assimakopoulos et al., *The M5 accuracy competition: Results, findings and conclusions*, International Journal of Forecasting, 2022.
