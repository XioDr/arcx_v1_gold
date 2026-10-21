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
        # Volume-based features
        df[f'volume_mean_{w}'] = df['Volume'].rolling(window=w, min_periods=1).mean()
        df[f'volume_std_{w}'] = df['Volume'].rolling(window=w, min_periods=1).std()
        df[f'volume_change_{w}'] = df['Volume'].diff(w)
        df[f'volume_ratio_{w}'] = df['Volume'] / (df[f'volume_mean_{w}'] + 1e-6)

        # Price-based features
        df[f'price_mean_{w}'] = df['Close'].rolling(window=w, min_periods=1).mean()
        df[f'price_std_{w}'] = df['Close'].rolling(window=w, min_periods=1).std()
        df[f'price_change_{w}'] = df['Close'].diff(w)
        df[f'price_ratio_{w}'] = df['Close'] / (df[f'price_mean_{w}'] + 1e-6)
    return df

# 3. Labeling: anomaly if trend changes in future
def label_change_in_trend(df, future_windows=[15, 30, 60, 90]):
    df = df.copy()
    df['true_anomaly'] = 1  # default: normal (1), anomaly (-1)

    for w in future_windows:
        df[f'future_close_{w}'] = df['Close'].shift(-w)
        df[f'price_change_{w}'] = df[f'future_close_{w}'] - df['Close']
        df[f'price_direction_{w}'] = np.sign(df[f'price_change_{w}'])

    for w in future_windows:
        col = f'price_direction_{w}'
        df[f'trend_change_{w}'] = df[col] != df[col].shift(1)

    trend_cols = [f'trend_change_{w}' for w in future_windows]
    df['true_anomaly'] = df[trend_cols].any(axis=1).astype(int).replace({0: 1, 1: -1})

    print("✅ True anomaly labels created.")
    return df

# 4. Remove leakage-prone features
def remove_leakage_features(all_features):
    leakage_keywords = ['price_direction_', 'trend_change_', 'future_close_']
    clean_features = [f for f in all_features if not any(kw in f for kw in leakage_keywords)]
    return clean_features

# 5. Train model without Optuna
def train_model(df, features, label_col='true_anomaly',
                model_path='xgb_model.pkl',
                feature_list_path='xgb_feature_list.pkl'):
    model_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'gamma': 0,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0
    }

    df_clean = df.dropna(subset=features + [label_col])
    X = df_clean[features]
    y = df_clean[label_col].replace({-1: 1, 1: 0})

    model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        **model_params
    )

    model.fit(X, y)

    y_pred = model.predict(X)
    print("\n=== Evaluation Metrics (Training Data) ===")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    joblib.dump(model, model_path)
    joblib.dump(features, feature_list_path)
    print(f"✅ Model saved to '{model_path}'")
    print(f"✅ Feature list saved to '{feature_list_path}'")

    return model, features

# 6. Plot anomalies
def plot_anomalies(df, pred_col='xgb_pred'):
    plt.figure(figsize=(15, 5))
    plt.plot(df['timestamp'], df['Volume'], label='Volume')
    anomalies = df[df[pred_col] == -1]
    plt.scatter(anomalies['timestamp'], anomalies['Volume'], color='red', label='Anomaly', s=30)
    plt.title("Detected Anomalies (XGBoost)")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    print("--- TRAINING PHASE ---")

    train_filepath = r"D:\project\arcx_v1_gold\split_output_data(gold)\gold_train_80pct.csv"

    # Load and preprocess data
    df_train = load_data(train_filepath)
    print(f"✅ Loaded {len(df_train)} rows of training data")

    df_train = preprocess_features(df_train)
    print("✅ Features created for training data")

    df_train = label_change_in_trend(df_train)

    all_train_features = [col for col in df_train.columns if col.startswith('volume_') or col.startswith('price_')]
    valid_train_features = [col for col in all_train_features if df_train[col].isnull().mean() < 0.1]
    clean_train_features = remove_leakage_features(valid_train_features)
    print(f"✅ {len(clean_train_features)} clean features selected for training (no leakage)")

    # Train model
    model, final_features = train_model(df_train, clean_train_features)

    # Predict
    df_train_clean = df_train.dropna(subset=final_features)
    df_train_clean['xgb_pred'] = model.predict(df_train_clean[final_features])
    df_train_clean['xgb_pred'] = df_train_clean['xgb_pred'].replace({0: 1, 1: -1})

    df_train = df_train.merge(df_train_clean[['timestamp', 'xgb_pred']], on='timestamp', how='left')

    # Plot
    print("\n--- Anomalies on Training Data ---")
    plot_anomalies(df_train)
