import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# 1. Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# 2. Feature engineering
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

# 3. Labeling with anomaly strength
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

    print("✅ True anomaly labels created.")
    return df

# 4. Remove leakage-prone features
def remove_leakage_features(all_features):
    leakage_keywords = ['price_direction_', 'trend_change_', 'future_close_', 'price_change_']
    return [f for f in all_features if not any(kw in f for kw in leakage_keywords)]

# === MAIN ===
if __name__ == "__main__":
    filepath = r"D:\project\split_output_data\BTCUSDT_ohlc_data_1m(5year)_train_80pct.csv"
    df = load_data(filepath)

    df = preprocess_features(df)
    df = label_change_in_trend(df)

    all_features = [col for col in df.columns if col.startswith('volume_') or col.startswith('price_')]
    valid_features = [col for col in all_features if df[col].isnull().mean() < 0.1]
    clean_features = remove_leakage_features(valid_features)
    print(f"✅ {len(clean_features)} clean features selected.")

    # Calculate anomaly strength
    future_windows = [15, 30, 60, 90]
    price_change_cols = [f'price_change_{w}' for w in future_windows]
    df['anomaly_strength'] = df[price_change_cols].abs().max(axis=1)

    # Load classifier
    clf = joblib.load('xgb_model.pkl')
    clf_features = joblib.load('xgb_feature_list.pkl')

    df_meta = df.dropna(subset=clf_features + ['anomaly_strength']).copy()
    X_base = df_meta[clf_features]
    df_meta['anomaly_prob'] = clf.predict_proba(X_base)[:, 1]

    # Train meta model
    meta_features = clf_features + ['anomaly_prob']
    X = df_meta[meta_features]
    y = df_meta['anomaly_strength']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    meta_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    meta_model.fit(X_train, y_train)

    # Evaluate
    y_pred = meta_model.predict(X_test)
    print("=== Meta Model Performance ===")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    # Plot
    test_timestamps = df_meta.loc[X_test.index, 'timestamp']
    plt.figure(figsize=(12, 6))
    plt.plot(test_timestamps, y_test, label='Actual', marker='o')
    plt.plot(test_timestamps, y_pred, label='Predicted', marker='x')
    plt.title("Anomaly Strength (Actual vs Predicted)")
    plt.xlabel("Timestamp")
    plt.ylabel("Anomaly Strength (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save
    joblib.dump(meta_model, 'xgb_meta_model.pkl')
    joblib.dump(meta_features, 'xgb_meta_feature_list.pkl')
    print("✅ Meta model and feature list saved.")
