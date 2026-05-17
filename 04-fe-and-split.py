# ==================================================
# FEATURE ENGINEERING
# Scaling with Min-Max + Winsorization then Split the data
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def main():
    print("=== STARTING FEATURE ENGINEERING & DYNAMIC SPLIT (ROBUST MODE) ===")

    # ==========================================
    # 1. FOLDER CONFIGURATION
    # ==========================================
    DATASET_DIR = "dataset"
    SPLIT_DIR = f"{DATASET_DIR}/splits"
    SCALER_DIR = f"{DATASET_DIR}/scalers"

    STAT_EVAL_DIR = "evaluations/statistical/fe-and-split"
    GRAPH_EVAL_DIR = "evaluations/graphical/fe-and-split"

    for directory in [SPLIT_DIR, SCALER_DIR, STAT_EVAL_DIR, GRAPH_EVAL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # ==========================================
    # 2. LOAD MASTER DATASET
    # ==========================================
    data_path = f'{DATASET_DIR}/master-features-dataset.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # ==========================================
    # 3. FEATURE ENGINEERING (t-1 lags)
    # ==========================================
    print("[*] Performing feature lagging...")
    df['Target_Price'] = df['Price']

    features_to_lag = [
        'Price', 'Rolling_LOWESS_Trend', 'Reconstructed_Signal',
        'USD_Index', 'SP500', 'US10Y', 'VIX'
    ]

    for col in features_to_lag:
        df[f'Lag1_{col}'] = df[col].shift(1)

    feature_cols = [f'Lag1_{c}' for c in features_to_lag]

    # Targeted dropna: only check columns used in ML
    subset_to_check = ['Target_Price'] + feature_cols
    df = df.dropna(subset=subset_to_check).reset_index(drop=True)

    # ==========================================
    # 4. CHRONOLOGICAL SPLIT
    # ==========================================
    print("[*] Performing chronological splitting...")
    total_rows = len(df)
    train_size = int(total_rows * 0.8)

    train_df = df.iloc[:train_size].copy()
    temp_df = df.iloc[train_size:].copy()

    test_size = int(len(temp_df) * 0.5)
    test_df = temp_df.iloc[:test_size].copy()
    unseen_df = temp_df.iloc[test_size:].copy()

    # Declare date boundaries for plotting
    test_start_date = test_df['Date'].min()
    unseen_start_date = unseen_df['Date'].min()

    # Winsorization (prevent extreme outliers)
    print("[*] Applying winsorization (clipping outliers)...")
    for col in feature_cols:
        lower_bound = train_df[col].quantile(0.01)
        upper_bound = train_df[col].quantile(0.99)

        train_df[col] = np.clip(train_df[col], lower_bound, upper_bound)
        test_df[col] = np.clip(test_df[col], lower_bound, upper_bound)
        unseen_df[col] = np.clip(unseen_df[col], lower_bound, upper_bound)

    # Min-Max scaling
    print("[*] Performing Min-Max scaling...")
    scaler_X = MinMaxScaler(clip=True)
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(train_df[feature_cols])
    y_train_scaled = scaler_y.fit_transform(train_df[['Target_Price']])

    X_test_scaled = scaler_X.transform(test_df[feature_cols])
    y_test_scaled = scaler_y.transform(test_df[['Target_Price']])

    X_unseen_scaled = scaler_X.transform(unseen_df[feature_cols])
    y_unseen_scaled = scaler_y.transform(unseen_df[['Target_Price']])

    # ==========================================
    # 5. SAVE SPLITS AND SCALERS
    # ==========================================
    pd.DataFrame(X_train_scaled, index=train_df['Date'], columns=feature_cols).to_csv(
        f"{SPLIT_DIR}/X_train.csv")
    pd.DataFrame(y_train_scaled, index=train_df['Date'], columns=['Target_Price']).to_csv(
        f"{SPLIT_DIR}/y_train.csv")

    pd.DataFrame(X_test_scaled, index=test_df['Date'], columns=feature_cols).to_csv(
        f"{SPLIT_DIR}/X_test.csv")
    pd.DataFrame(y_test_scaled, index=test_df['Date'], columns=['Target_Price']).to_csv(
        f"{SPLIT_DIR}/y_test.csv")

    pd.DataFrame(X_unseen_scaled, index=unseen_df['Date'], columns=feature_cols).to_csv(
        f"{SPLIT_DIR}/X_unseen.csv")
    pd.DataFrame(y_unseen_scaled, index=unseen_df['Date'], columns=['Target_Price']).to_csv(
        f"{SPLIT_DIR}/y_unseen.csv")

    joblib.dump(scaler_X, f"{SCALER_DIR}/scaler_X.pkl")
    joblib.dump(scaler_y, f"{SCALER_DIR}/scaler_y.pkl")

    # ==========================================
    # 6. GRAPHICAL EVALUATION
    # ==========================================
    print("[*] Creating split evaluation plot...")
    plt.figure(figsize=(15, 6))
    plt.plot(train_df['Date'], train_df['Target_Price'],
             label=f'Train Data ({len(train_df)} days)', color='darkblue')
    plt.plot(test_df['Date'], test_df['Target_Price'],
             label=f'Test Data ({len(test_df)} days)', color='darkorange')
    plt.plot(unseen_df['Date'], unseen_df['Target_Price'],
             label=f'Unseen Data ({len(unseen_df)} days)', color='green')

    plt.axvline(x=test_start_date, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=unseen_start_date, color='black', linestyle='--', alpha=0.5)

    plt.title('Dynamic Train / Test / Unseen Split Over Time', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('WTI Crude Oil Price (USD)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    graph_path = f"{GRAPH_EVAL_DIR}/01_train_test_unseen_split.png"
    plt.savefig(graph_path, dpi=300)
    plt.close()

    # ==========================================
    # 7. STATISTICAL EVALUATION (Text report)
    # ==========================================
    print("[*] Creating statistical report...")
    report_text = (
        "FEATURE ENGINEERING & DYNAMIC SPLIT REPORT\n"
        "===========================================\n"
        "1. FEATURES USED (t-1 lags):\n"
    )
    for f in feature_cols:
        report_text += f"   - {f}\n"

    report_text += (
        "\n2. DATASET SPLIT (Chronological):\n"
        f"   [TRAIN]  : {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')} | Count: {len(train_df)} rows\n"
        f"   [TEST]   : {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')} | Count: {len(test_df)} rows\n"
        f"   [UNSEEN] : {unseen_df['Date'].min().strftime('%Y-%m-%d')} to {unseen_df['Date'].max().strftime('%Y-%m-%d')} | Count: {len(unseen_df)} rows\n"
        "\n3. SCALING & PREPROCESSING METHOD:\n"
        "   - Outlier Handling: Winsorization (clipping at 1st and 99th percentiles from Train Data)\n"
        "   - Scaling Algorithm: MinMaxScaler (range 0 to 1) with clip=True\n"
        "   - Fit Strategy: Fitted ONLY on TRAIN data to prevent data leakage.\n"
        "===========================================\n"
    )

    stat_path = f"{STAT_EVAL_DIR}/feature_split_report.txt"
    with open(stat_path, 'w') as f:
        f.write(report_text)

    print("\n[+] Report and graphs successfully saved in evaluations/ directory")
    print("[✓] DATASET READY FOR TRAINING! (Scaler: MinMaxScaler + Winsorization)")


if __name__ == "__main__":
    main()