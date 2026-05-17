"""
rolling-lowess.py
Rolling LOWESS decomposition for WTI crude oil prices.
Computes trend and residual using a rolling window (past data only).
Generates statistical report and graphical evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis
from tqdm import tqdm

print("=== STARTING ROLLING LOWESS DECOMPOSITION ===")

# ==========================================
# 1. FOLDER CONFIGURATION
# ==========================================
DATASET_DIR = "dataset"
STAT_EVAL_DIR = "evaluation/statistical/rolling-lowess"
GRAPH_EVAL_DIR = "evaluations/graphical/rolling-lowess"

# Create folders if missing
for directory in [DATASET_DIR, STAT_EVAL_DIR, GRAPH_EVAL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==========================================
# 2. LOAD DYNAMIC DATASET
# ==========================================
data_path = f'{DATASET_DIR}/WTI-Macro-Master-UpToDate.csv'
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ==========================================
# 3. ROLLING LOWESS PROCESS
# ==========================================
window_size = 252          # ~1 trading year
lowess_frac = 0.2

prices = df['Price'].values
trends = np.full(len(prices), np.nan)
residuals = np.full(len(prices), np.nan)

print(f"\nProcessing {len(prices)} data rows (Window: {window_size} days)...")
start_time = time.time()

for i in tqdm(range(window_size, len(prices)), desc="Computing Rolling LOWESS"):
    y_window = prices[i - window_size : i + 1]
    x_window = np.arange(len(y_window))

    # Compute LOWESS using only past data
    smoothed = lowess(y_window, x_window, frac=lowess_frac, return_sorted=False)

    current_trend = smoothed[-1]
    current_residual = y_window[-1] - current_trend

    trends[i] = current_trend
    residuals[i] = current_residual

end_time = time.time()
execution_time = end_time - start_time
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = execution_time % 60
time_str = f"{hours}h {minutes}m {seconds:.2f}s"

# Merge results and drop warm-up NaN rows
df['Rolling_LOWESS_Trend'] = trends
df['Rolling_LOWESS_Residual'] = residuals
df_clean = df.dropna(subset=['Rolling_LOWESS_Trend']).reset_index(drop=True)

# ==========================================
# 4. SAVE DATASET FOR CEEMDAN
# ==========================================
dataset_output = f'{DATASET_DIR}/rolling-lowess-dataset.csv'
df_clean.to_csv(dataset_output, index=False)
print(f"\n[+] Dataset saved at: {dataset_output}")

# ==========================================
# 5. STATISTICAL EVALUATION (for thesis)
# ==========================================
print("\nComputing statistical evaluation...")

# Prepare summary statistics for Price, Trend, and Residual
price_stats = df_clean['Price'].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
trend_stats = df_clean['Rolling_LOWESS_Trend'].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
res_stats = df_clean['Rolling_LOWESS_Residual'].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()

# Add skewness
price_skew = skew(df_clean['Price'].dropna())
trend_skew = skew(df_clean['Rolling_LOWESS_Trend'].dropna())
res_skew = skew(df_clean['Rolling_LOWESS_Residual'].dropna())

# Augmented Dickey-Fuller test on residual
res_data = df_clean['Rolling_LOWESS_Residual'].values
adf_result = adfuller(res_data)
adf_stat = adf_result[0]
p_value = adf_result[1]
is_stationary = "Yes (Stationary)" if p_value < 0.05 else "No (Non-Stationary)"

# Descriptive stats for residual (extra)
mean_res = np.mean(res_data)
std_res = np.std(res_data)
skew_res = res_skew
kurt_res = kurtosis(res_data)

# ==========================================
# 5a. Generate requested report format
# ==========================================
stat_report = f"""
ROLLING LOWESS DECOMPOSITION REPORT
===================================
Input trading days       : {len(df)}
Window size (days)       : {window_size}
LOWESS frac              : {lowess_frac}
Output rows (after warmup): {len(df_clean)}
Date range              : {df_clean['Date'].min()} to {df_clean['Date'].max()}
Execution time          : {time_str}
===================================

SUMMARY STATISTICS:
                   count        mean         std         min         25%         50%         75%         max    skewness
WTI                {price_stats['count']:.0f}   {price_stats['mean']:.6f}   {price_stats['std']:.6f}   {price_stats['min']:.6f}   {price_stats['25%']:.6f}   {price_stats['50%']:.6f}   {price_stats['75%']:.6f}   {price_stats['max']:.6f}   {price_skew:.6f}
LOWESS_Trend       {trend_stats['count']:.0f}   {trend_stats['mean']:.6f}   {trend_stats['std']:.6f}   {trend_stats['min']:.6f}   {trend_stats['25%']:.6f}   {trend_stats['50%']:.6f}   {trend_stats['75%']:.6f}   {trend_stats['max']:.6f}   {trend_skew:.6f}
Rolling_LOWESS_Residual {res_stats['count']:.0f}   {res_stats['mean']:.6f}   {res_stats['std']:.6f}   {res_stats['min']:.6f}   {res_stats['25%']:.6f}   {res_stats['50%']:.6f}   {res_stats['75%']:.6f}   {res_stats['max']:.6f}   {res_skew:.6f}

STATIONARITY TEST (Residuals - ADF):
- ADF Statistic      : {adf_stat:.6f}
- P-Value            : {p_value:.6e}
- Conclusion         : {is_stationary} (important for CEEMDAN)

DESCRIPTIVE STATISTICS (Residuals):
- Mean               : {mean_res:.6f} (near zero indicates good detrending)
- Std Deviation      : {std_res:.6f}
- Skewness           : {skew_res:.6f}
- Kurtosis           : {kurt_res:.6f}
===================================
"""

stat_output = f'{STAT_EVAL_DIR}/residual_statistics_report.txt'
with open(stat_output, "w") as text_file:
    text_file.write(stat_report)
print(f"[+] Statistical report saved at: {stat_output}")

# ==========================================
# 6. GRAPHICAL EVALUATION (for paper/thesis)
# ==========================================
print("Generating graphical evaluation...")
plt.style.use('default')

# Graph 1: Actual price vs Trend
plt.figure(figsize=(12, 6))
plt.plot(df_clean['Date'], df_clean['Price'], label='Actual WTI Price', color='lightgray', alpha=0.8)
plt.plot(df_clean['Date'], df_clean['Rolling_LOWESS_Trend'], label='Rolling LOWESS Trend', color='red', linewidth=1.5)
plt.title('WTI Crude Oil Price vs Rolling Macro Trend', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{GRAPH_EVAL_DIR}/01_actual_vs_trend.png', dpi=300)
plt.close()

# Graph 2: Residual time series
plt.figure(figsize=(12, 4))
plt.plot(df_clean['Date'], df_clean['Rolling_LOWESS_Residual'], color='teal', linewidth=1)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Extracted Residuals (Input for CEEMDAN)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{GRAPH_EVAL_DIR}/02_residual_timeseries.png', dpi=300)
plt.close()

# Graph 3: Residual distribution (histogram)
plt.figure(figsize=(8, 6))
plt.hist(df_clean['Rolling_LOWESS_Residual'], bins=50, color='teal', alpha=0.7, edgecolor='black')
plt.axvline(mean_res, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_res:.2f}')
plt.title('Distribution of LOWESS Residuals', fontsize=14)
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{GRAPH_EVAL_DIR}/03_residual_distribution.png', dpi=300)
plt.close()

print(f"[+] Graphs saved in: {GRAPH_EVAL_DIR}/")
print("\n✅ ROLLING LOWESS PIPELINE COMPLETED!")