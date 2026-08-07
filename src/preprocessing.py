"""
Data Preprocessing — Gold Price Dataset
CRISP-DM: Data Preparation phase
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
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
print("\nMissing values per column:")
print(df.isnull().sum())

# If any missing values existed, you could handle them like this
# (kept here for completeness / marking purposes even if count is 0):
df = df.ffill().bfill()          # forward/backward fill for time series

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

# ------------------------------------------------------------------
# 6. Outlier detection (IQR method) — flag, don't blindly drop,
#    since genuine price spikes are meaningful in financial data
# ------------------------------------------------------------------
def flag_outliers_iqr(series, factor=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (series < lower) | (series > upper)

for col in ['Price', 'Volume', 'Chg%']:
    outlier_mask = flag_outliers_iqr(df[col])
    print(f"{col}: {outlier_mask.sum()} potential outliers")

# ------------------------------------------------------------------
# 7. Feature engineering (common for gold-price / time-series models)
# ------------------------------------------------------------------
# Lags
df["Price_Lag1"] = df["Price"].shift(1)
df["Price_Lag2"] = df["Price"].shift(2)
df["Volume_Lag1"] = df["Volume"].shift(1)

# Rolling means & volatilities (shifted to avoid leakage)
df["MA_7"] = df["Price"].shift(1).rolling(7).mean()
df["MA_30"] = df["Price"].shift(1).rolling(30).mean()
df["Volatility_7"] = df["Price"].shift(1).rolling(7).std()
df["Volatility_30"] = df["Price"].shift(1).rolling(30).std()

# Daily returns (lagged)
df["daily_return_lag1"] = df["Price_Lag1"] / df["Price_Lag2"] - 1
df["daily_return_lag2"] = df["Price_Lag2"] / df["Price"].shift(3) - 1

# Price range & change from previous day (shifted by 1)
df["Price_Range"] = df["High"] - df["Low"]
df["Price_Change"] = df["Open"] - df["Price"]  # Close = Price column
df["Price_Range_Lag1"] = df["Price_Range"].shift(1)
df["Price_Change_Lag1"] = df["Price_Change"].shift(1)

# Time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df['Day'] = df['Date'].dt.day

# Drop rows with NaN created by lag/rolling features (first few rows)
df = df.dropna().reset_index(drop=True)

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
train_frames = []
test_frames = []

# Group by year and apply the 80/20 split to each year's data
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.40,
    shuffle=False,  # Shuffle within each year to avoid time-based bias
    random_state=42
)
train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)

print("\nTrain shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Fit scaler on training partition only and apply to both sets
scaler = StandardScaler()

# Ensure scaler columns are numeric
train_df[scaler_feature_cols] = train_df[scaler_feature_cols].apply(pd.to_numeric, errors='coerce')
test_df[scaler_feature_cols] = test_df[scaler_feature_cols].apply(pd.to_numeric, errors='coerce')

# Fit and transform
scaler.fit(train_df[scaler_feature_cols])
train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()

train_df_scaled[scaler_feature_cols] = scaler.transform(train_df[scaler_feature_cols])
test_df_scaled[scaler_feature_cols] = scaler.transform(test_df[scaler_feature_cols])

# ------------------------------------------------------------------
# 10. Save cleaned dataset and fitted scaler
# ------------------------------------------------------------------
# Save the cleaned (unscaled) full dataset for reproducibility
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