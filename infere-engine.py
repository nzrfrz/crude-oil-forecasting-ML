import os
import sys
import subprocess
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess
from PyEMD import CEEMDAN
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==========================================
# 1. INCREMENTAL DATASET UPDATE ENGINE
# ==========================================
def update_dataset_incrementally():
    print("\n[*] Initializing incremental update engine...")

    master_dyn_path = "dataset/WTI-Master-Dynamic.csv"
    lowess_path = "dataset/rolling-lowess-dataset.csv"
    master_feat_path = "dataset/master-features-dataset.csv"

    if not (os.path.exists(master_dyn_path) and os.path.exists(lowess_path) and os.path.exists(master_feat_path)):
        print("[!] Error: Historical dataset files not found.")
        return

    # 1. Load historical data
    master_dyn = pd.read_csv(master_dyn_path)
    lowess_df = pd.read_csv(lowess_path)
    master_feat = pd.read_csv(master_feat_path)

    # ---------------------------------------------------------
    # AUTO-TRIMMING: Remove corrupted tail rows (empty data)
    # ---------------------------------------------------------
    print("[*] Checking dataset column integrity...")
    null_counts = master_feat.isnull().sum(axis=1)

    # If a row has many NaNs, it is a corrupted leftover row
    if null_counts.max() > 5:
        good_rows = master_feat[null_counts <= 5]
        last_valid_date_str = good_rows['Date'].max()
        last_valid_date = pd.to_datetime(last_valid_date_str)

        print(f"    -> Anomaly detected. Rolling back to safe date: {last_valid_date.strftime('%Y-%m-%d')}")

        master_dyn['Date'] = pd.to_datetime(master_dyn['Date'])
        master_dyn = master_dyn[master_dyn['Date'] <= last_valid_date]

        lowess_df['Date'] = pd.to_datetime(lowess_df['Date'])
        lowess_df = lowess_df[lowess_df['Date'] <= last_valid_date]

        master_feat['Date'] = pd.to_datetime(master_feat['Date'])
        master_feat = master_feat[master_feat['Date'] <= last_valid_date]
    else:
        last_valid_date = pd.to_datetime(master_feat['Date'].max())
        master_dyn['Date'] = pd.to_datetime(master_dyn['Date'])
        lowess_df['Date'] = pd.to_datetime(lowess_df['Date'])
        master_feat['Date'] = pd.to_datetime(master_feat['Date'])

    today_str = datetime.today().strftime('%Y-%m-%d')
    print(f"    -> Data ready to update from {last_valid_date.strftime('%Y-%m-%d')} to {today_str}.")

    # 2. Download new data
    start_dl = (last_valid_date - timedelta(days=10)).strftime('%Y-%m-%d')
    wti_ticker = "CL=F"
    macro_tickers = {"DX-Y.NYB": "USD_Index",
                     "^GSPC": "SP500", "^VIX": "VIX", "^TNX": "US10Y"}

    raw_wti = yf.download(wti_ticker, start=start_dl,
                          end=today_str, progress=False).reset_index()
    if isinstance(raw_wti.columns, pd.MultiIndex):
        raw_wti.columns = raw_wti.columns.droplevel(1)

    # Always rename 'Close' from YFinance to 'Price'
    if 'Close' in raw_wti.columns:
        raw_wti.rename(columns={'Close': 'Price'}, inplace=True)
    if 'Adj Close' in raw_wti.columns:
        raw_wti.drop(columns=['Adj Close'], inplace=True)

    raw_macro = yf.download(list(macro_tickers.keys()), start=start_dl,
                            end=today_str, progress=False)['Close'].reset_index()
    raw_macro.rename(columns=macro_tickers, inplace=True)

    raw_wti['Date'] = pd.to_datetime(raw_wti['Date']).dt.tz_localize(None)
    raw_macro['Date'] = pd.to_datetime(raw_macro['Date']).dt.tz_localize(None)

    new_master = pd.merge(raw_wti, raw_macro, on='Date', how='left').ffill().bfill()
    new_days = new_master[new_master['Date'] > last_valid_date]

    if len(new_days) == 0:
        print("[✓] Great! Your dataset is already up to date. No update needed.")
        return

    print(f"[*] Adding {len(new_days)} new trading days sequentially...")

    window_size = 252
    ceemdan_op = CEEMDAN(trials=50, parallel=False)

    # 3. Incremental processing
    for _, row in new_days.iterrows():
        # A. Master Dynamic
        new_dyn_row = {col: row.get(col, np.nan) for col in master_dyn.columns}
        master_dyn = pd.concat([master_dyn, pd.DataFrame([new_dyn_row])], ignore_index=True)

        # B. LOWESS (using 'Price' column)
        window_raw = master_dyn['Price'].iloc[-window_size:].values
        smoothed = lowess(window_raw, np.arange(len(window_raw)), frac=0.1, return_sorted=False)
        residual = window_raw[-1] - smoothed[-1]

        new_lowess_row = new_dyn_row.copy()
        new_lowess_row['Rolling_LOWESS_Trend'] = smoothed[-1]
        new_lowess_row['Rolling_LOWESS_Residual'] = residual

        aligned_lowess_row = {col: new_lowess_row.get(col, np.nan) for col in lowess_df.columns}
        lowess_df = pd.concat([lowess_df, pd.DataFrame([aligned_lowess_row])], ignore_index=True)

        # C. CEEMDAN & Master Features
        window_resid = lowess_df['Rolling_LOWESS_Residual'].iloc[-window_size:].values
        imfs = ceemdan_op.ceemdan(window_resid)
        recon_signal = np.sum(imfs[3:]) if len(imfs) > 3 else 0

        new_feat_row = new_lowess_row.copy()
        new_feat_row['Reconstructed_Signal'] = recon_signal

        # Absolute alignment to prevent "Ghost Columns"
        aligned_feat_row = {}
        for col in master_feat.columns:
            if col in new_feat_row:
                aligned_feat_row[col] = new_feat_row[col]
            else:
                aligned_feat_row[col] = 0.0

        master_feat = pd.concat([master_feat, pd.DataFrame([aligned_feat_row])], ignore_index=True)

    # 4. Save updated CSV files
    print("[*] Saving updated data...")
    master_dyn.to_csv(master_dyn_path, index=False)
    lowess_df.to_csv(lowess_path, index=False)
    master_feat.to_csv(master_feat_path, index=False)

    # 5. Trigger feature engineering script
    print("[*] Re-running Feature Engineering (fe-and-split-v2.py)...")
    try:
        subprocess.run([sys.executable, "fe-and-split-v2.py"], check=True)
        print("[✓] UPDATE COMPLETE! CSV structure cleaned and database ready for use.")
    except Exception as e:
        print(f"[!] Failed to run fe-and-split-v2.py. Error: {e}")

# ==========================================
# 2. FORECASTING FEATURE
# ==========================================
def parse_date(date_str):
    """Parse English date format (e.g., '2 january 2024' or '7 march 2026') to datetime."""
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    try:
        parts = date_str.strip().lower().split()
        if len(parts) != 3:
            return None
        day, month_str, year = int(parts[0]), parts[1], int(parts[2])
        if month_str not in month_map:
            return None
        return pd.to_datetime(f"{year}-{month_map[month_str]:02d}-{day:02d}")
    except Exception:
        return None


def forecast_price():
    print("\n--- MENU: FORECAST PRICE ---")
    # Define file paths
    scaler_path = "dataset/scalers/scaler_y.pkl"
    x_test_path = "dataset/splits/X_test.csv"
    x_unseen_path = "dataset/splits/X_unseen.csv"
    master_dyn_path = "dataset/WTI-Macro-Master-UpToDate.csv"

    # Peek at Unseen data dates
    X_unseen_temp = pd.read_csv(x_unseen_path, index_col=0)
    X_unseen_temp.index = pd.to_datetime(X_unseen_temp.index)
    if not X_unseen_temp.empty:
        min_date = X_unseen_temp.index.min().strftime('%d %B %Y')
        max_date = X_unseen_temp.index.max().strftime('%d %B %Y')
        print(f"Info: Unseen data available from {min_date} to {max_date}")
    else:
        print("Info: Unseen data is empty.")

    print("Input format: Day Month Year (Example: 26 march 2026)\n")
    start_str = input("Enter Start Date : ")
    end_str = input("Enter End Date   : ")
    print("-" * 53)

    start_date = parse_date(start_str)
    end_date = parse_date(end_str)

    if start_date is None or end_date is None or start_date > end_date:
        print("[!] Invalid date input.")
        return

    business_days = pd.bdate_range(start=start_date, end=end_date)

    if len(business_days) > 7:
        print("[!] Forecast range too long (max 7 business days).")
        return
    if len(business_days) == 0:
        print("[!] Date range contains only holidays (Saturday/Sunday).")
        return

    print("[*] Loading models and dataset...")
    try:
        # Verify file existence
        for path in [scaler_path, x_test_path, x_unseen_path]:
            if not os.path.exists(path):
                print(f"[!] Missing file: {path}")
                return

        scaler_y = joblib.load(scaler_path)
        X_test = pd.read_csv(x_test_path, index_col=0)
        X_unseen = pd.read_csv(x_unseen_path, index_col=0)

        # Load Master Dynamic to get actual prices
        master_dyn = pd.read_csv(master_dyn_path)
        master_dyn['Date'] = pd.to_datetime(master_dyn['Date'])
        master_dyn.set_index('Date', inplace=True)

        # Combine historical and future data
        X_combined = pd.concat([X_test, X_unseen]).sort_index()
        X_combined.index = pd.to_datetime(X_combined.index)
        X_combined = X_combined[~X_combined.index.duplicated(keep='first')]

        # Load final models
        models = {
            "SVR": joblib.load("./models/SVR_final.pkl"),
            "XGBoost": joblib.load("./models/XGBoost_final.pkl"),
            "Random_Forest": joblib.load("./models/Random_Forest_final.pkl"),
            "Stacking_Ensemble": joblib.load("./models/Stacking_Ensemble_final.pkl")
        }

    except Exception as e:
        print(f"[!] Failed to load models/dataset. Error: {e}")
        return

    available_dates = business_days.intersection(X_combined.index)

    if len(available_dates) == 0:
        print(f"[!] Sorry, the requested dates are not available in the processed dataset.")
        latest_date = X_combined.index.max().date() if not X_combined.empty else 'Empty'
        print(f"    Latest available feature data is: {latest_date}")
        print("    Solution: Select Menu '1' (Update Dataset) first to process the latest daily features.")
        return

    X_target = X_combined.loc[available_dates]

    # Get actual prices if available (future dates will be NaN)
    actual_vals = [master_dyn.loc[date, 'Price'] if date in master_dyn.index else np.nan for date in available_dates]

    results_df = pd.DataFrame({
        "Date": available_dates.strftime('%Y-%m-%d'),
        "Day": available_dates.day_name(),
        "Actual": [f"${val:.2f}" if pd.notna(val) else "N/A" for val in actual_vals]
    })

    for name, model in models.items():
        pred_scaled = model.predict(X_target)
        # Reshape for inverse_transform if needed
        if len(pred_scaled.shape) == 1:
            pred_scaled = pred_scaled.reshape(-1, 1)

        pred_usd = scaler_y.inverse_transform(pred_scaled).flatten()
        results_df[name] = [f"${x:.2f}" for x in pred_usd]

        # Calculate error percentage
        error_col = []
        for pred, actual in zip(pred_usd, actual_vals):
            if pd.notna(actual) and actual != 0:
                err = abs(pred - actual) / actual * 100
                error_col.append(f"{err:.2f}%")
            else:
                error_col.append("N/A")
        results_df[f"{name}_Err"] = error_col

    # Create output folder and save to CSV
    output_dir = "inference-output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"forecast_minmaxwinsor_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    results_df.to_csv(filepath, index=False)

    # Print to terminal
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print("\n" + "=" * 120)
    print(" " * 35 + "WTI CRUDE OIL FORECASTING & EVALUATION RESULTS (USD)")
    print("=" * 120)
    print(results_df.to_string(index=False))
    print("=" * 120)
    print(f"[✓] Success! Evaluation table saved to: {filepath}\n")

    # ==========================================
    # ENTERPRISE SUBPLOT VISUALIZATION (OPTION 3)
    # ==========================================
    # Only create if at least one actual price row exists
    if any(pd.notna(val) for val in actual_vals):
        print("[*] Creating enterprise visualization plot...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        dates_plot = pd.to_datetime(results_df['Date'])

        # --- TOP GRAPH (Line Chart - Price) ---
        ax1.plot(dates_plot, actual_vals, color='black', linewidth=3, marker='o', label='Actual Price')

        styles = {
            "Stacking_Ensemble": {"color": "purple", "linestyle": "-", "linewidth": 2, "marker": "s"},
            "SVR": {"color": "green", "linestyle": "--", "linewidth": 1.5, "marker": "^"},
            "XGBoost": {"color": "red", "linestyle": "--", "linewidth": 1.5, "marker": "x"},
            "Random_Forest": {"color": "blue", "linestyle": "-.", "linewidth": 1.5, "marker": "d"}
        }

        for name in models.keys():
            pred_vals = results_df[name].str.replace('$', '', regex=False).astype(float)
            ax1.plot(dates_plot, pred_vals,
                     color=styles[name]["color"],
                     linestyle=styles[name]["linestyle"],
                     linewidth=styles[name]["linewidth"],
                     marker=styles[name]["marker"],
                     alpha=0.8,
                     label=f"{name} Predicted")

        ax1.set_title(f'WTI Crude Oil Price Forecast ({dates_plot.min().strftime("%d %B %Y")} to {dates_plot.max().strftime("%d %B %Y")})',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- BOTTOM GRAPH (Bar Chart - Error/MAPE) ---
        error_data = {}
        for name in models.keys():
            # Remove '%' symbol and convert 'N/A' to NaN for plotting
            error_data[name] = results_df[f"{name}_Err"].replace("N/A", np.nan).str.replace('%', '', regex=False).astype(float)

        err_df = pd.DataFrame(error_data)
        err_df.index = dates_plot.dt.strftime('%m-%d')

        colors_list = [styles[name]["color"] for name in models.keys()]
        err_df.plot(kind='bar', ax=ax2, color=colors_list, alpha=0.8)

        ax2.set_title('Absolute Percentage Error (MAPE) per Day', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Error (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='upper right', ncol=len(models))
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax2.tick_params(axis='x', rotation=0)

        plt.tight_layout()

        # Save plot to inference-output folder
        plot_filename = f"forecast_plot_minmaxwinsor_{timestamp}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close()

        print(f"[+] Enterprise subplot graph saved at: {plot_filepath}\n")
    else:
        print("\n[!] Graph not created because this is a pure future forecast (no actual prices available).\n")

# ==========================================
# MAIN MENU LOOP
# ==========================================
def main():
    while True:
        print("\n=====================================================")
        print("    WTI CRUDE OIL AI INFERENCE ENGINE v4.1           ")
        print("=====================================================")
        print("1. Update Dataset (Pull data up to yesterday & update features)")
        print("2. Forecast Price (Future / historical prediction)")
        print("0. Exit")
        print("=====================================================")

        choice = input("Select menu (0/1/2): ")

        if choice == '1':
            update_dataset_incrementally()
        elif choice == '2':
            forecast_price()
        elif choice == '0':
            print("Exiting program. Happy forecasting!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()