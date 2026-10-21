import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# 2. Feature engineering (volume + price)
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

# 3. Labeling: anomaly if trend changes in future
def label_change_in_trend(df, future_windows=[15, 30, 60, 90]):
    df = df.copy()
    df['true_anomaly'] = 1  # normal = 1, anomaly = -1

    for w in future_windows:
        df[f'future_close_{w}'] = df['Close'].shift(-w)
        df[f'price_change_{w}'] = df[f'future_close_{w}'] - df['Close']
        df[f'price_direction_{w}'] = np.sign(df[f'price_change_{w}'])

    for w in future_windows:
        col = f'price_direction_{w}'
        df[f'trend_change_{w}'] = df[col] != df[col].shift(1)

    trend_cols = [f'trend_change_{w}' for w in future_windows]
    df['true_anomaly'] = df[trend_cols].any(axis=1).astype(int).replace({0: 1, 1: -1})

    print("✅ True anomaly labels created for test data.")
    return df

# 4. Remove leakage-prone features
def remove_leakage_features(all_features):
    leakage_keywords = ['price_direction_', 'trend_change_', 'future_close_']
    clean_features = [f for f in all_features if not any(kw in f for kw in leakage_keywords)]
    return clean_features

# 5. Plot anomalies
def plot_anomalies(df, pred_col='xgb_pred'):
    plt.figure(figsize=(15, 5))
    plt.plot(df['timestamp'], df['Volume'], label='Volume')
    anomalies = df[df[pred_col] == -1]
    plt.scatter(anomalies['timestamp'], anomalies['Volume'], color='red', label='Anomaly', s=30)
    plt.title("Detected Anomalies on Test Data (XGBoost)")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    print("\n--- TESTING PHASE ---")

    # تنظیم مسیر فایل‌ها
    model_path = 'xgb_model.pkl'
    feature_list_path = 'xgb_feature_list.pkl'
    test_filepath = r"D:\project\arcx_v1_gold\split_output_data(gold)\gold_backtest_15pct.csv"

    # 1. Load model
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded from '{model_path}'")
    except FileNotFoundError:
        print(f"❌ Model file not found at '{model_path}'")
        exit()

    # 2. Load feature list
    try:
        features = joblib.load(feature_list_path)
        print(f"✅ Feature list loaded from '{feature_list_path}'")
    except FileNotFoundError:
        print(f"❌ Feature list file not found at '{feature_list_path}'")
        exit()

    # 3. Load and preprocess test data
    df_test = load_data(test_filepath)
    print(f"✅ Loaded {len(df_test)} rows of test data")

    df_test = preprocess_features(df_test)
    print("✅ Features created for test data")

    df_test = label_change_in_trend(df_test)

    # 4. Drop rows with missing values
    df_test_clean = df_test.dropna(subset=features + ['true_anomaly'])
    X_test = df_test_clean[features]
    y_test = df_test_clean['true_anomaly'].replace({-1: 1, 1: 0})

    # 5. Predict
    y_pred = model.predict(X_test)

    print("\n=== Evaluation Metrics (Test Data) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Plot and save results
    df_test_clean['xgb_pred'] = y_pred
    df_test_clean['xgb_pred'] = df_test_clean['xgb_pred'].replace({0: 1, 1: -1})

    print("\n--- Anomalies on Test Data ---")
    plot_anomalies(df_test_clean)

    output_path = "processed_test_with_xgboost.csv"
    df_test_clean.to_csv(output_path, index=False)
    print(f"✅ Processed test data saved to '{output_path}'")
