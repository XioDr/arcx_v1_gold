import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def preprocess_features(df, window_sizes=[1, 3, 5, 15, 30, 45, 60, 90]):
    df = df.copy()
    for w in window_sizes:
        df[f'volume_mean_{w}'] = df['Volume'].rolling(window=w, min_periods=1).mean()
        df[f'volume_std_{w}'] = df['Volume'].rolling(window=w, min_periods=1).std()
        df[f'volume_change_{w}'] = df['Volume'].diff(w)
        df[f'volume_ratio_{w}'] = df['Volume'] / (df[f'volume_mean_{w}'] + 1e-6)

        df[f'price_mean_{w}'] = df['Close'].rolling(window=w, min_periods=1).mean()
        df[f'price_std_{w}'] = df['Close'].rolling(window=w, min_periods=1).std()
        df[f'price_change_{w}'] = df['Close'].diff(w)
        df[f'price_ratio_{w}'] = df['Close'] / (df[f'price_mean_{w}'] + 1e-6)
    return df

def label_change_in_trend(df, future_windows=[15, 30, 60, 90]):
    df = df.copy()
    for w in future_windows:
        df[f'future_close_{w}'] = df['Close'].shift(-w)
        df[f'price_change_{w}'] = ((df[f'future_close_{w}'] - df['Close']) / df['Close']) * 100
        df[f'price_direction_{w}'] = np.sign(df[f'price_change_{w}'])

    for w in future_windows:
        df[f'trend_change_{w}'] = df[f'price_direction_{w}'] != df[f'price_direction_{w}'].shift(1)

    trend_cols = [f'trend_change_{w}' for w in future_windows]
    df['true_anomaly'] = df[trend_cols].any(axis=1).astype(int).replace({0: 1, 1: -1})
    return df

if __name__ == "__main__":
    filepath = r"D:\project\split_output_data\BTCUSDT_ohlc_data_1m(5year)_test_4pct.csv"
    df = load_data(filepath)
    df = preprocess_features(df)
    df = label_change_in_trend(df)

    future_windows = [15, 30, 60, 90]
    price_change_cols = [f'price_change_{w}' for w in future_windows]
    df['anomaly_strength'] = df[price_change_cols].abs().max(axis=1)

    clf = joblib.load('xgb_model.pkl')
    clf_features = joblib.load('xgb_feature_list.pkl')
    meta_model = joblib.load('xgb_meta_model.pkl')
    meta_features = joblib.load('xgb_meta_feature_list.pkl')

    df_meta = df.dropna(subset=clf_features + ['anomaly_strength']).copy()
    X_base = df_meta[clf_features]
    df_meta['anomaly_prob'] = clf.predict_proba(X_base)[:, 1]

    X_meta = df_meta[meta_features]
    y_true = df_meta['anomaly_strength']
    y_pred = meta_model.predict(X_meta)

    print("=== Meta Model Evaluation on Test Data ===")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.4f}")

    df_meta['predicted_anomaly_strength'] = y_pred

    plt.figure(figsize=(12, 6))
    plt.plot(df_meta['timestamp'], y_true, label='Actual', marker='o')
    plt.plot(df_meta['timestamp'], y_pred, label='Predicted', marker='x')
    plt.title("Anomaly Strength on Test Data")
    plt.xlabel("Timestamp")
    plt.ylabel("Anomaly Strength (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    df_meta[['timestamp', 'anomaly_strength', 'predicted_anomaly_strength']].to_csv(
        'anomaly_strength_predictions.csv', index=False
    )
    print("✅ Results saved to 'anomaly_strength_predictions.csv'")
