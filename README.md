# arcx_v1_gold

Gold-market version of the Arcx v1 anomaly detector: an **XGBoost model that flags candles where the trend is about to change**, using volume and price features.

> Research code only. Not financial advice. Data and trained models are not included.

## Contents

| File | Description |
| --- | --- |
| `anomaly_detector.py` | Full pipeline: load OHLCV CSV, build volume/price features over multiple windows, label future trend changes, drop leakage-prone features, train XGBoost, print metrics and plot anomalies. |
| `anomaly_detector_test.py` | Applies a trained model and feature list to new data. |
| `Anomaly_power.py` / `Anomaly_power_test.py` | Variants that evaluate the predictive power of the anomaly signal. |

## Requirements

Python 3.9+, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `joblib`.

## Usage

Input CSVs need a `timestamp` column and OHLCV columns (`Volume` included). Training writes `xgb_model.pkl` and `xgb_feature_list.pkl`, which the test scripts load.
