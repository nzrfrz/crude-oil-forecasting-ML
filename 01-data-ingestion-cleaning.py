"""
get_dataset.py
Download WTI crude oil futures (CL=F) and macroeconomic indicators:
- US Dollar Index (DX-Y.NYB)
- S&P 500 (^GSPC)
- VIX (^VIX)
- US 10Y Treasury yield (^TNX)

Merges on trading days, forward/backward fills missing values
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

print("=== STARTING DYNAMIC DATA FETCHING PIPELINE ===")

# ==========================================
# 1. CONFIGURATION
# ==========================================
WTI_TICKER = "CL=F"          # Crude Oil WTI Futures
MACRO_TICKERS = {
    "DX-Y.NYB": "USD_Index",
    "^GSPC":    "SP500",
    "^VIX":     "VIX",
    "^TNX":     "US10Y"
}

START_DATE = "2001-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')   # today's date automatically

# ==========================================
# 2. DOWNLOAD & CLEAN WTI (main target)
# ==========================================
print(f"\n1. Downloading WTI ({WTI_TICKER}) from {START_DATE} to {END_DATE}...")
wti_raw = yf.download(WTI_TICKER, start=START_DATE, end=END_DATE, progress=False)
wti_raw = wti_raw.reset_index()

# Flatten MultiIndex columns if using recent yfinance version
if isinstance(wti_raw.columns, pd.MultiIndex):
    wti_raw.columns = wti_raw.columns.get_level_values(0)

# Keep only relevant columns and rename 'Close' -> 'Price'
wti_clean = wti_raw[['Date', 'Close', 'Open', 'High', 'Low']].copy()
wti_clean.rename(columns={'Close': 'Price'}, inplace=True)

# Remove timezone information for safe merging
wti_clean['Date'] = pd.to_datetime(wti_clean['Date']).dt.tz_localize(None)

# Extra safeguard: remove weekends (Sat=5, Sun=6)
wti_clean['Weekday'] = wti_clean['Date'].dt.dayofweek
wti_clean = wti_clean[~wti_clean['Weekday'].isin([5, 6])].drop(columns=['Weekday'])

print(f"   -> WTI clean (no weekends): {len(wti_clean)} rows")

# ==========================================
# 3. DOWNLOAD MACROECONOMIC DATA
# ==========================================
ticker_list = list(MACRO_TICKERS.keys())
print(f"\n2. Downloading macro data: {ticker_list} ...")
macro_close = yf.download(ticker_list, start=START_DATE, end=END_DATE, progress=False)['Close']
macro_df = macro_close.reset_index()
macro_df.rename(columns=MACRO_TICKERS, inplace=True)
macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.tz_localize(None)

# ==========================================
# 4. MERGE & HANDLE MISSING VALUES
# ==========================================
print("\n3. Merging WTI and macro data...")
# Left join using WTI's trading calendar as backbone
master = pd.merge(wti_clean, macro_df, on='Date', how='left')

# Fill missing values caused by different holiday calendars
master = master.ffill()   # forward fill (previous day's value)
master = master.bfill()   # backward fill for any remaining leading NaNs

# ==========================================
# 5. SAVE TO DISK
# ==========================================
OUTPUT_DIR = './dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = f'{OUTPUT_DIR}/WTI-Macro-Master-UpToDate.csv'
master.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ DONE! Up To Date dataset saved at: {OUTPUT_PATH}")
print("\nLast 5 rows (most recent data):")
print(master.tail())