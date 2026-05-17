import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ==========================================
# 1. OUTPUT FOLDER PREPARATION
# ==========================================
STATISTICAL_EVAL_PATH = "./evaluations/statistical/model-train"
GRAPHICAL_EVAL_PATH = "./evaluations/graphical/model-train"
SAVED_MODEL_PATH = "./models"

for directory in [STATISTICAL_EVAL_PATH, GRAPHICAL_EVAL_PATH, SAVED_MODEL_PATH]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==========================================
# 2. LOAD DATA & SCALER
# ==========================================
print("\n[+] Loading data splits and scaler...")
X_train = pd.read_csv("dataset/splits/X_train.csv", index_col=0)
X_test = pd.read_csv("dataset/splits/X_test.csv", index_col=0)
y_train = pd.read_csv("dataset/splits/y_train.csv", index_col=0)
y_test = pd.read_csv("dataset/splits/y_test.csv", index_col=0)

# Load scaler_y to convert predictions back to original price (USD)
scaler_y = joblib.load("dataset/scalers/scaler_y.pkl")
feature_names = X_train.columns.tolist()

# ==========================================
# 2.5. AUTO-TUNING XGBoost, SVR & RANDOM FOREST (HYPERPARAMETER OPTIMIZATION)
# ==========================================
print("\n[+] Starting hyperparameter tuning for SVR, RF & XGBoost...")
tscv = TimeSeriesSplit(n_splits=5)

# --- A. TUNING SVR ---
print("    -> Tuning SVR...")
param_grid_svr = {
    'C': [0.1, 1, 10, 50],
    'gamma': ['scale', 'auto', 0.1, 1],
    'epsilon': [0.001, 0.01, 0.05, 0.1],
    'kernel': ['rbf']
}
svr_tuner = GridSearchCV(SVR(), param_grid_svr, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
svr_tuner.fit(X_train, y_train.values.ravel())
best_svr_params = svr_tuner.best_params_
print(f"       [✓] Best SVR parameters: {best_svr_params}")

# --- B. TUNING RANDOM FOREST ---
print("    -> Tuning Random Forest (please wait)...")
param_grid_rf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_tuner = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
rf_tuner.fit(X_train, y_train.values.ravel())
best_rf_params = rf_tuner.best_params_
print(f"       [✓] Best Random Forest parameters: {best_rf_params}")

# --- C. TUNING XGBOOST ---
print("    -> Tuning XGBoost (please wait)...")
param_grid_xgb = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0]
}
xgb_tuner = GridSearchCV(XGBRegressor(random_state=42), param_grid_xgb, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
xgb_tuner.fit(X_train, y_train.values.ravel())
best_xgb_params = xgb_tuner.best_params_
print(f"       [✓] Best XGBoost parameters: {best_xgb_params}")

# ==========================================
# 3. MODEL INITIALIZATION & STACKING
# ==========================================
print("\n[+] Initializing base models and stacking ensemble...")
base_models = {
    "XGBoost": XGBRegressor(**best_xgb_params, random_state=42),
    "Random_Forest": RandomForestRegressor(**best_rf_params, random_state=42),
    "SVR": SVR(**best_svr_params)
}

# Create stacking ensemble (meta-learner: Ridge Regression)
estimators = [(name, model) for name, model in base_models.items()]
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5,
    n_jobs=-1
)

# Combine all models for training
all_models = base_models.copy()
all_models["Stacking_Ensemble"] = stacking_model

# ==========================================
# 4. TRAINING & PREDICTION
# ==========================================
predictions = {}
y_test_inv = scaler_y.inverse_transform(y_test).flatten()

for name, model in all_models.items():
    print(f"[*] Training {name}...")
    model.fit(X_train, y_train.values.ravel())

    # Save physical model (.pkl) to final-models folder
    model_path = f"{SAVED_MODEL_PATH}/{name}_final.pkl"
    joblib.dump(model, model_path)
    print(f"    -> Model saved at: {model_path}")

    # Prediction
    y_pred_scaled = model.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    predictions[name] = y_pred_inv

# ==========================================
# 5. DIEBOLD-MARIANO TEST FUNCTION
# ==========================================
def dm_test(actual, pred1, pred2):
    """Calculate Diebold-Mariano statistic using Absolute Error metric"""
    e1 = np.abs(actual - pred1)
    e2 = np.abs(actual - pred2)
    d = e1 - e2  # Loss differential

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    # DM Statistic
    stat = mean_d / np.sqrt(var_d / len(d))
    # P-Value (Two-tailed)
    pval = 2 * (1 - stats.t.cdf(np.abs(stat), df=len(d)-1))
    return stat, pval

def directional_accuracy(actual, pred):
    """Calculate percentage of correctly predicted price direction (Up/Down)"""
    # Actual direction (today - yesterday)
    actual_dir = np.sign(np.diff(actual))
    # Predicted direction (predicted today - actual yesterday)
    pred_dir = np.sign(pred[1:] - actual[:-1])
    return np.mean(actual_dir == pred_dir) * 100

# ==========================================
# 6. STATISTICAL EVALUATION (METRICS & DM TEST)
# ==========================================
print("\n[+] Calculating metrics & DA...")
metrics_data = []

for name, y_pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
    mae = mean_absolute_error(y_test_inv, y_pred)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred) * 100
    da = directional_accuracy(y_test_inv, y_pred)

    metrics_data.append({
        "Model": name, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "DA (%)": da
    })

df_metrics = pd.DataFrame(metrics_data)

# Save metrics table to CSV
df_metrics.to_csv(f"{STATISTICAL_EVAL_PATH}/FINAL_METRICS.csv", index=False)
print(df_metrics.to_string(index=False))

# ==========================================
# 7. GRAPHICAL EVALUATION (ACTUAL VS PREDICTED)
# ==========================================
print("\n[+] Creating comparison graphs and heatmap...")

# --- GRAPH 1: BAR CHART OF METRICS (RMSE, MAE, MAPE, DA) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Evaluation Metrics Comparison',
             fontsize=16, fontweight='bold')

sns.barplot(data=df_metrics, x='Model', y='RMSE',
            ax=axes[0, 0], hue='Model', palette='Blues_r', legend=False)
axes[0, 0].set_title('RMSE (Lower is Better)')

sns.barplot(data=df_metrics, x='Model', y='MAE',
            ax=axes[0, 1], hue='Model', palette='Greens_r', legend=False)
axes[0, 1].set_title('MAE (Lower is Better)')

sns.barplot(data=df_metrics, x='Model', y='MAPE (%)',
            ax=axes[1, 0], hue='Model', palette='Oranges_r', legend=False)
axes[1, 0].set_title('MAPE % (Lower is Better)')

sns.barplot(data=df_metrics, x='Model', y='DA (%)',
            ax=axes[1, 1], hue='Model', palette='Purples', legend=False)
axes[1, 1].set_title('Directional Accuracy % (Higher is Better)')
# Show legend ONLY for the 50% red dashed line
axes[1, 1].axhline(50, color='red', linestyle='--', label='50% Random Guess')
axes[1, 1].legend(loc='lower right')

for ax in axes.flat:
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
metrics_plot_path = f"{GRAPHICAL_EVAL_PATH}/Metrics_BarCharts.png"
plt.savefig(metrics_plot_path, dpi=300)

# --- GRAPH 2: DIEBOLD-MARIANO P-VALUE HEATMAP ---
model_names = list(predictions.keys())
n_models = len(model_names)
dm_pvalues = np.zeros((n_models, n_models))

# Compute DM test combinations
for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if i == j:
            dm_pvalues[i, j] = 1.0  # P-value with itself
        else:
            _, pval = dm_test(y_test_inv, predictions[m1], predictions[m2])
            dm_pvalues[i, j] = pval

plt.figure(figsize=(8, 6))
# Create heatmap, annotate values below 0.05 as significant
sns.heatmap(dm_pvalues, annot=True, cmap='coolwarm', fmt=".4f",
            xticklabels=model_names, yticklabels=model_names,
            vmin=0, vmax=0.1)
plt.title('Diebold-Mariano Test (P-Values)\n< 0.05 indicates significant difference', fontsize=14)
plt.tight_layout()
dm_heatmap_path = f"{GRAPHICAL_EVAL_PATH}/DM_Test_Heatmap.png"
plt.savefig(dm_heatmap_path, dpi=300)

# ==========================================
# SAVE DIEBOLD-MARIANO RESULTS TO CSV
# ==========================================
print("\n[+] Saving Diebold-Mariano test results...")
# Create DataFrame from p-value matrix
df_dm = pd.DataFrame(dm_pvalues, index=model_names, columns=model_names)

# Save to CSV
dm_csv_path = f"{STATISTICAL_EVAL_PATH}/Diebold_Mariano_PValues.csv"
df_dm.to_csv(dm_csv_path)
print(f"    -> DM Test P-Value table saved at: {dm_csv_path}")

# Create text report summarizing DM test significance (p < 0.05)
dm_report_path = f"{STATISTICAL_EVAL_PATH}/Diebold_Mariano_Report.txt"
with open(dm_report_path, "w") as f:
    f.write("DIEBOLD-MARIANO TEST SIGNIFICANCE REPORT\n")
    f.write("=========================================\n")
    f.write("* Note: If p-value < 0.05, the two models' performances are significantly different.\n\n")

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i < j:  # Avoid duplicate comparisons (A vs B and B vs A)
                pval = dm_pvalues[i, j]
                significance = "SIGNIFICANT (Different)" if pval < 0.05 else "NOT SIGNIFICANT (Similar)"
                f.write(f"- {m1} vs {m2} : p-value = {pval:.4f} => {significance}\n")

print(f"    -> DM Test conclusion report saved at: {dm_report_path}")

# --- GRAPH 3: ACTUAL VS PREDICTED (TIME SERIES) ---
print("\n[+] Creating evaluation plot...")
plt.figure(figsize=(16, 8))
plt.plot(pd.to_datetime(X_test.index), y_test_inv,
         color='black', linewidth=2, label='Actual WTI Price')

# Colors and line styles for each model
styles = {
    "Stacking_Ensemble": {"color": "purple", "linestyle": "-", "linewidth": 2},
    "SVR": {"color": "green", "linestyle": "--", "linewidth": 1.5},
    "XGBoost": {"color": "red", "linestyle": "--", "linewidth": 1.5},
    "Random_Forest": {"color": "blue", "linestyle": "-.", "linewidth": 1}
}

for name, y_pred in predictions.items():
    plt.plot(pd.to_datetime(X_test.index), y_pred,
             color=styles[name]["color"],
             linestyle=styles[name]["linestyle"],
             linewidth=styles[name]["linewidth"],
             alpha=0.8,
             label=f"{name} Predicted")

plt.title('Final Evaluation: WTI Crude Oil Price Prediction (Actual vs Models)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

graph_path = f"{GRAPHICAL_EVAL_PATH}/Actual_vs_Predicted.png"
plt.savefig(graph_path, dpi=300)
print(f"[+] Graph saved at: {graph_path}")

# ==========================================
# 8. VISUALIZATION 3: FEATURE IMPORTANCE & PERMUTATION
# ==========================================
print("Calculating feature importance V2...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# XGBoost Native Importance
xgb_imp = base_models['XGBoost'].feature_importances_
axes[0].barh(feature_names, xgb_imp, color='red')
axes[0].set_title('XGBoost Feature Importance')
axes[0].invert_yaxis()

# Random Forest Native Importance
rf_imp = base_models['Random_Forest'].feature_importances_
axes[1].barh(feature_names, rf_imp, color='blue')
axes[1].set_title('Random Forest Feature Importance')
axes[1].invert_yaxis()

# SVR Permutation Importance
svr_result = permutation_importance(
    base_models['SVR'], X_test, y_test, n_repeats=10, random_state=42)
svr_imp = svr_result.importances_mean
axes[2].barh(feature_names, svr_imp, color='green')
axes[2].set_title('SVR Permutation Importance')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(f"{GRAPHICAL_EVAL_PATH}/feature_importance.png", dpi=300)
plt.close()
print("\n=== ALL PROCESSES COMPLETED SUCCESSFULLY! ===")

print(f"[+] All graphs saved in {GRAPHICAL_EVAL_PATH} directory")
print("=== FINAL EVALUATION PROCESS COMPLETED ===")