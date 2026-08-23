import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
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

price_cols = ['Price', 'Open', 'High', 'Low']
df[price_cols] = df[price_cols].replace(0, np.nan)
before = len(df)
df = df.dropna(subset=price_cols).reset_index(drop=True)
print(f"\nDropped {before - len(df)} rows with missing/zero price fields.")

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
# 6. Outlier detection (IQR method)
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

# ------------------------------------------------------------------
# 7. Feature engineering
# ------------------------------------------------------------------
# Lags
df["Price_Lag1"] = df["Price"].shift(1)
df["Price_Lag2"] = df["Price"].shift(2)
df["Price_Lag30"] = df["Price"].shift(30)
df["Price_Lag60"] = df["Price"].shift(60)
df["Price_Lag90"] = df["Price"].shift(90)
df["Volume_Lag1"] = df["Volume"].shift(1)

df["MA_7"] = df["Price"].shift(1).rolling(7).mean()
df["MA_30"] = df["Price"].shift(1).rolling(30).mean()
df["MA_60"] = df["Price"].shift(1).rolling(60).mean()
df["Volatility_7"] = df["Price"].shift(1).rolling(7).std()
df["Volatility_30"] = df["Price"].shift(1).rolling(30).std()

df["daily_return_lag1"] = df["Price_Lag1"] / df["Price_Lag2"] - 1
df["daily_return_lag2"] = df["Price_Lag2"] / df["Price"].shift(3) - 1

df["Price_Range"] = df["High"] - df["Low"]
df["Price_Change"] = df["Price"] - df["Open"] 
df["Price_Range_Lag1"] = df["Price_Range"].shift(1)
df["Price_Change_Lag1"] = df["Price_Change"].shift(1)

df["Chg%_Lag1"] = df["Chg%"].shift(1)
df["Chg%_Lag2"] = df["Chg%"].shift(2)
df["Chg%_Lag3"] = df["Chg%"].shift(3)

df["Volume_Momentum"] = df["Volume"].shift(1) / df["Volume"].shift(1).rolling(10).mean()
df["MA_Ratio"] = df["MA_7"] / df["MA_30"]

_bb_mid = df["Price"].shift(1).rolling(20).mean()
_bb_std = df["Price"].shift(1).rolling(20).std()
_bb_upper = _bb_mid + 2 * _bb_std
_bb_lower = _bb_mid - 2 * _bb_std

df["BB_Status"] = (df["Price"].shift(1) - _bb_lower) / (_bb_upper - _bb_lower)

# Momentum & Trend Indicators
delta = df["Price"].shift(1).diff()
gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
rs = gain / loss
df["RSI_14"] = 100 - (100 / (1 + rs))

ema_12 = df["Price"].shift(1).ewm(span=12, adjust=False).mean()
ema_26 = df["Price"].shift(1).ewm(span=26, adjust=False).mean()
df["MACD"] = ema_12 - ema_26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

df["Price_to_MA7"] = (df["Price"].shift(1) - df["MA_7"]) / df["MA_7"]
df["Price_to_MA30"] = (df["Price"].shift(1) - df["MA_30"]) / df["MA_30"]
df["Price_to_MA60"] = (df["Price"].shift(1) - df["MA_60"]) / df["MA_60"]

# Volume-Price Interactions
direction = np.sign(df["Price"].shift(1).diff()).fillna(0)
df["OBV"] = (df["Volume"].shift(1) * direction).cumsum()
df["Volume_Weighted_Chg_Lag1"] = df["Chg%_Lag1"] * df["Volume_Lag1"]

# Advanced Volatility & Ranges
df["BB_Width"] = (_bb_upper - _bb_lower) / _bb_mid

high_lag1 = df["High"].shift(1)
low_lag1 = df["Low"].shift(1)
prev_close = df["Price"].shift(2)

tr1 = high_lag1 - low_lag1
tr2 = (high_lag1 - prev_close).abs()
tr3 = (low_lag1 - prev_close).abs()
true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["ATR_14"] = true_range.rolling(14).mean()

# Extended Horizon Lags
df["Price_Lag7"] = df["Price"].shift(7)
df["Chg%_Lag7"] = df["Chg%"].shift(7)
df["Chg%_Lag30"] = df["Chg%"].shift(30)

# Time Encoding
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Day"] = df["Date"].dt.day

df["Return"] = df["Price"] / df["Price_Lag1"] - 1
df["Return_future"] = df["Return"].shift(-1)
df["Return_Lag1"] = df["Return"].shift(1)
df["Return_Lag2"] = df["Return"].shift(2)

df["LogReturn"] = np.log(df["Price"] / df["Price_Lag1"])
df["LogReturn_future"] = df["LogReturn"].shift(-1)
df["LogReturn_Lag1"] = df["LogReturn"].shift(1)
df["LogReturn_Lag2"] = df["LogReturn"].shift(2)

# Drop leading rows with incomplete rolling windows
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"\nDropped {before - len(df)} leading rows with incomplete lag/rolling windows.")
print(f"Final cleaned dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

# ------------------------------------------------------------------
# 8. Feature scaling 
# ------------------------------------------------------------------
scaler_feature_cols = ['Open', 'High', 'Low', 'Volume']
scaler = StandardScaler()

df[scaler_feature_cols] = df[scaler_feature_cols].apply(pd.to_numeric, errors='coerce')

# Fitting the scaler on the entire dataset and scaling the columns
df[scaler_feature_cols] = scaler.fit_transform(df[scaler_feature_cols])

# ------------------------------------------------------------------
# 9. Save cleaned dataset and fitted scaler
# ------------------------------------------------------------------
output_dir.mkdir(parents=True, exist_ok=True)

# Save the final cleaned and scaled dataset
df.to_csv(output_dir / "Gold_Price_cleaned.csv", index=False)

# Persist the fitted scaler 
prototype_dir = project_root / "prototype"
prototype_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, prototype_dir / "scaler.pkl")

print("\nPreprocessing complete. Files saved:")
print(" - Gold_Price_cleaned.csv (full cleaned, engineered, and scaled dataset)")
print(" - scaler.pkl (fitted StandardScaler)")