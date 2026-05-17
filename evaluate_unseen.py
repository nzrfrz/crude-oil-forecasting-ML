"""
evaluate_unseen.py
Compute forecasting metrics on the unseen future dataset (2024-2026)
using the pre-trained models and scalers.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# ==========================================
# 1. CONFIGURATION
# ==========================================
SPLIT_DIR = "dataset/splits"
SCALER_DIR = "dataset/scalers"
MODEL_DIR = "models"

# Output directory for unseen evaluation results
OUTPUT_DIR = "evaluations/statistical/unseen-eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD UNSEEN DATA AND SCALER
# ==========================================
X_unseen = pd.read_csv(f"{SPLIT_DIR}/X_unseen.csv", index_col=0)
y_unseen = pd.read_csv(f"{SPLIT_DIR}/y_unseen.csv", index_col=0)

# Inverse transform target scaler to get actual USD prices
scaler_y = joblib.load(f"{SCALER_DIR}/scaler_y.pkl")
y_actual = scaler_y.inverse_transform(y_unseen).flatten()

# ==========================================
# 3. LOAD MODELS AND GENERATE PREDICTIONS
# ==========================================
model_names = ["XGBoost", "Random_Forest", "SVR", "Stacking_Ensemble"]
predictions = {}

for name in model_names:
    model_path = f"{MODEL_DIR}/{name}_final.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        y_pred_scaled = model.predict(X_unseen)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        predictions[name] = y_pred
        print(f"Loaded {name} model from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Skipping {name}.")

# ==========================================
# 4. COMPUTE METRICS FOR EACH MODEL
# ==========================================
def directional_accuracy(actual, pred):
    """Calculate percentage of correctly predicted price direction."""
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(pred[1:] - actual[:-1])
    return np.mean(actual_dir == pred_dir) * 100

results = []

for name, y_pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
    da = directional_accuracy(y_actual, y_pred)
    
    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "DA (%)": da
    })

df_results = pd.DataFrame(results)
print("\n=== UNSEEN DATASET PERFORMANCE (2024-01-02 to 2026-04-30) ===")
print(df_results.to_string(index=False))

# Save to CSV
output_csv = f"{OUTPUT_DIR}/unseen_metrics.csv"
df_results.to_csv(output_csv, index=False)
print(f"\n✅ Metrics saved to {output_csv}")

# ==========================================
# 5. (Optional) Generate daily error table for first N days
# ==========================================
# Create a DataFrame with actual and predictions for inspection
dates = pd.to_datetime(X_unseen.index)
comparison_df = pd.DataFrame({"Date": dates, "Actual": y_actual})
for name, y_pred in predictions.items():
    comparison_df[f"{name}_Pred"] = y_pred
    comparison_df[f"{name}_AbsErr_%"] = np.abs((y_pred - y_actual) / y_actual) * 100

# Save full comparison to CSV (optional)
comparison_df.to_csv(f"{OUTPUT_DIR}/unseen_full_comparison.csv", index=False)
print(f"Full comparison saved to {OUTPUT_DIR}/unseen_full_comparison.csv")

# Print first 10 rows as example
print("\nFirst 10 rows of unseen predictions (Actual vs. Models):")
print(comparison_df.head(10).to_string())