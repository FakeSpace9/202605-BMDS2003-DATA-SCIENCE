"""
Data Preprocessing — Gold Price Dataset
CRISP-DM: Data Preparation phase
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from utils import load_raw_dataset
# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
output_dir = project_root / "data" / "processed"


# 2. Load the data
df = load_raw_dataset()

print("Initial shape:", df.shape)
print(df.head())
print(df.dtypes)

# ------------------------------------------------------------------
# 2. Convert 'Date' to datetime and sort chronologically
#    (raw file is sorted newest -> oldest, we want oldest -> newest
#     for time-series modelling)
# ------------------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.sort_values('Date').reset_index(drop=True)

# ------------------------------------------------------------------
# 3. Check for missing values
# ------------------------------------------------------------------
print("\nMissing values per column before drop:")
print(df.isnull().sum())

print("\nZero values per column before drop:")
print((df == 0).sum())

# Only Price/Open/High/Low being 0 is a genuine data error (gold price
# can't legitimately be 0). Volume==0 or Chg%==0 can be real (a quiet
# trading day), so we don't nuke those to NaN.
price_cols = ['Price', 'Open', 'High', 'Low']
df[price_cols] = df[price_cols].replace(0, np.nan)
before = len(df)
df = df.dropna(subset=price_cols).reset_index(drop=True)
print(f"\nDropped {before - len(df)} rows with missing/zero price fields "
      f"(these are true data errors, safe to remove -- unlike statistical "
      f"outliers, they don't carry information).")

# ------------------------------------------------------------------
# 4. Check and remove duplicate rows / duplicate dates
# ------------------------------------------------------------------
print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates(subset='Date', keep='first').reset_index(drop=True)

# ------------------------------------------------------------------
# 5. Ensure correct data types
# ------------------------------------------------------------------
numeric_cols = ['Price', 'Open', 'High', 'Low', 'Volume', 'Chg%']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_cols).reset_index(drop=True)

# ------------------------------------------------------------------
# 6. Outlier detection (IQR method) — flag, don't blindly drop,
#    since genuine price spikes are meaningful in financial data
# ------------------------------------------------------------------
def flag_outliers_iqr(series, factor=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (series < lower) | (series > upper)

df['Is_Outlier_Price'] = flag_outliers_iqr(df['Price'])
df['Is_Outlier_Volume'] = flag_outliers_iqr(df['Volume'])
df['Is_Outlier_ChgPct'] = flag_outliers_iqr(df['Chg%'])
df['Is_LowVolume'] = df['Volume'] < 1000
 
print(f"\nFlagged (not dropped) {df['Is_Outlier_Price'].sum()} Price outliers, "
      f"{df['Is_Outlier_Volume'].sum()} Volume outliers, "
      f"{df['Is_Outlier_ChgPct'].sum()} Chg% outliers, "
      f"{df['Is_LowVolume'].sum()} low-volume rows.")
print("Rows by year among flagged Price outliers:")
print(df.loc[df['Is_Outlier_Price'], 'Date'].dt.year.value_counts().sort_index())

# ------------------------------------------------------------------
# 7. Feature engineering (common for gold-price / time-series models)
# ------------------------------------------------------------------
# Lags
df["Price_Lag1"] = df["Price"].shift(1)
df["Price_Lag2"] = df["Price"].shift(2)
df["Volume_Lag1"] = df["Volume"].shift(1)
 
df["MA_7"] = df["Price"].shift(1).rolling(7).mean()
df["MA_30"] = df["Price"].shift(1).rolling(30).mean()
df["Volatility_7"] = df["Price"].shift(1).rolling(7).std()
df["Volatility_30"] = df["Price"].shift(1).rolling(30).std()
 
df["daily_return_lag1"] = df["Price_Lag1"] / df["Price_Lag2"] - 1
df["daily_return_lag2"] = df["Price_Lag2"] / df["Price"].shift(3) - 1
 
df["Price_Range"] = df["High"] - df["Low"]
df["Price_Change"] = df["Price"] - df["Open"]  # today's move, NOT usable
                                                # as a same-day predictor
df["Price_Range_Lag1"] = df["Price_Range"].shift(1)
df["Price_Change_Lag1"] = df["Price_Change"].shift(1)
 
# --- Additional features required by train_linear_regression.py (Chg% target) ---
# Chg% momentum lags
df["Chg%_Lag1"] = df["Chg%"].shift(1)
df["Chg%_Lag2"] = df["Chg%"].shift(2)
df["Chg%_Lag3"] = df["Chg%"].shift(3)
 
# Volume momentum: yesterday's volume relative to its own recent (10-day) average.
# >1 means recent volume is running above its local average.
df["Volume_Momentum"] = df["Volume"].shift(1) / df["Volume"].shift(1).rolling(10).mean()
 
# MA ratio: short vs long moving average (both already computed on lagged price,
# so no leakage). >1 means short-term average is above long-term average.
df["MA_Ratio"] = df["MA_7"] / df["MA_30"]
 
# Bollinger %B: where yesterday's price sat within its 20-day Bollinger Band,
# computed entirely on lagged price so no same-day leakage.
# 0 = at lower band, 1 = at upper band, <0 or >1 = outside the bands (breakout).
_bb_mid = df["Price"].shift(1).rolling(20).mean()
_bb_std = df["Price"].shift(1).rolling(20).std()
_bb_upper = _bb_mid + 2 * _bb_std
_bb_lower = _bb_mid - 2 * _bb_std
df["BB_Status"] = (df["Price"].shift(1) - _bb_lower) / (_bb_upper - _bb_lower)
 
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Day"] = df["Date"].dt.day
 
# Drop only the leading rows where lag/rolling windows aren't full yet
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"\nDropped {before - len(df)} leading rows with incomplete lag/rolling windows.")
print(f"Final cleaned dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

# ------------------------------------------------------------------
# 8. Feature scaling (Min-Max scaling numeric features)
#    Fit the scaler only on the training split to avoid leakage.
# ------------------------------------------------------------------
# Choose a compact set of numeric features that downstream models and
# the Streamlit app will use (keep consistent across training and app):
scaler_feature_cols = ['Open', 'High', 'Low', 'Volume']

# ------------------------------------------------------------------
# 9. Train-test split
# ------------------------------------------------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    shuffle=False,  
    random_state=42
)
train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)

print("\nSingle chronological split (for baseline comparison only):")
print("Train:", train_df.shape, train_df['Date'].min().date(), "->", train_df['Date'].max().date())
print("Test :", test_df.shape, test_df['Date'].min().date(), "->", test_df['Date'].max().date())

# Fit scaler on training partition only and apply to both sets
scaler = StandardScaler()
train_df[scaler_feature_cols] = train_df[scaler_feature_cols].apply(pd.to_numeric, errors='coerce')
test_df[scaler_feature_cols] = test_df[scaler_feature_cols].apply(pd.to_numeric, errors='coerce')
scaler.fit(train_df[scaler_feature_cols])
 
train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()
train_df_scaled[scaler_feature_cols] = scaler.transform(train_df[scaler_feature_cols])
test_df_scaled[scaler_feature_cols] = scaler.transform(test_df[scaler_feature_cols])

# ------------------------------------------------------------------
# 10. Save cleaned dataset and fitted scaler
# ------------------------------------------------------------------
# Save the cleaned (unscaled) full dataset for reproducibility
output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_dir / "Gold_Price_cleaned.csv", index=False)
# Save the scaled training and test splits used for modelling
train_df_scaled.to_csv(output_dir / "Gold_Price_train.csv", index=False)
test_df_scaled.to_csv(output_dir / "Gold_Price_test.csv", index=False)
# Persist the fitted scaler for use in the Streamlit app
joblib.dump(scaler, project_root / "prototype" / "scaler.pkl")

print("\nPreprocessing complete. Files saved:")
print(" - Gold_Price_cleaned.csv (full cleaned + engineered features)")
print(" - Gold_Price_train.csv (scaled training split)")
print(" - Gold_Price_test.csv (scaled test split)")
print(" - scaler.pkl (fitted StandardScaler)")