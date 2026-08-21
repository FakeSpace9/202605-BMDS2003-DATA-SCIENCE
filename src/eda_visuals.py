import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_raw_dataset

# ============================================================
# 1. PROJECT PATHS
# ============================================================

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# Folder where all report graphs will be saved
PLOT_PATH = project_root / "report_assets" / "plots"
PLOT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = load_raw_dataset()

# Clean column names
df.columns = [c.strip() for c in df.columns]

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 3. IDENTIFY IMPORTANT COLUMNS
# ============================================================

# Identify date column
date_col = next(
    (c for c in df.columns if c.lower() in ["date", "datetime", "timestamp"]),
    None
)

# Identify price column
price_col = next(
    (c for c in df.columns if c.lower() in ["price", "gold_price", "gold price", "close"]),
    None
)

# Convert date
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

# Convert price
if price_col:
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")


if date_col is None:
    print("WARNING: Date column was not found.")

if price_col is None:
    print("WARNING: Price column was not found.")


# ============================================================
# 4. PLOT SETTINGS
# ============================================================

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True

created = []


# ============================================================
# 5. SAVE PLOT FUNCTION
# ============================================================

def save_plot(filename, title, xlabel=None, ylabel=None):
    """Save the current matplotlib figure into report_assets/plots/."""
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    
    output_path = PLOT_PATH / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    created.append(output_path)
    print(f"Saved: {output_path}")


# ============================================================
# 6. NUMERIC COLUMNS & 7. CALCULATE RETURNS
# ============================================================

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if price_col:
    df["Price_Change"] = df[price_col].diff()
    df["Daily_Return"] = df[price_col].pct_change() * 100
    df["Volatility_7"] = df["Daily_Return"].rolling(7).std()
    df["Volatility_30"] = df["Daily_Return"].rolling(30).std()


# ============================================================
# GRAPH 1: GOLD PRICE OVER TIME
# ============================================================
if date_col and price_col:
    plt.figure()
    plt.plot(df[date_col], df[price_col], linewidth=1.2)
    save_plot("01_gold_price_time_series.png", "Gold Price Over Time", "Date", "Gold Price")


# ============================================================
# GRAPH 2: GOLD PRICE DISTRIBUTION
# ============================================================
if price_col:
    plt.figure()
    plt.hist(df[price_col].dropna(), bins=30, color='skyblue', edgecolor='black')
    save_plot("02_gold_price_distribution.png", "Distribution of Gold Prices", "Gold Price", "Frequency")


# ============================================================
# GRAPH 3 [NEW]: YEARLY PRICE RANGE (MIN, AVG, MAX)
# ============================================================
if price_col and date_col:
    plt.figure()
    df['Year'] = df[date_col].dt.year
    yearly_stats = df.groupby('Year')[price_col].agg(['min', 'mean', 'max'])
    
    plt.fill_between(yearly_stats.index, yearly_stats['min'], yearly_stats['max'], color='lightblue', alpha=0.5, label='Min-Max Range')
    plt.plot(yearly_stats.index, yearly_stats['mean'], color='darkblue', marker='o', linewidth=2, label='Average Price')
    
    plt.legend()
    save_plot("03_yearly_price_range.png", "Yearly Gold Price Range (Min, Avg, Max)", "Year", "Gold Price")


# ============================================================
# GRAPH 4 [NEW]: AVERAGE PRICE CHANGE BY MONTH (SEASONALITY)
# ============================================================
if price_col and date_col:
    plt.figure()
    df['Month_Num'] = df[date_col].dt.month
    monthly_seasonality = df.groupby('Month_Num')["Price_Change"].mean()
    
    # Color green for positive months, red for negative months
    colors = ['green' if val > 0 else 'red' for val in monthly_seasonality]
    
    monthly_seasonality.plot(kind='bar', color=colors, edgecolor='black')
    plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.axhline(0, color='black', linewidth=1)
    
    save_plot("04_monthly_seasonality.png", "Average Price Change by Month (Seasonality)", "Month", "Average Price Change")


# ============================================================
# GRAPH 5: DAILY RETURN DISTRIBUTION
# ============================================================
if price_col:
    plt.figure()
    plt.hist(df["Daily_Return"].dropna(), bins=40, color='lightgreen', edgecolor='black')
    save_plot("05_daily_return_distribution.png", "Distribution of Daily Gold Returns", "Daily Return (%)", "Frequency")


# ============================================================
# GRAPH 6 [NEW]: RISK DISTRIBUTION (HISTOGRAM OF VOLATILITY)
# ============================================================
if price_col:
    plt.figure()
    plt.hist(df["Volatility_7"].dropna(), bins=30, color='purple', edgecolor='black', alpha=0.7)
    save_plot("06_volatility_distribution.png", "How Often is the Market Highly Volatile?", "7-Day Volatility", "Number of Days")


# ============================================================
# GRAPH 7 [NEW]: AVERAGE VOLATILITY BY YEAR (RISK BY YEAR)
# ============================================================
if price_col and date_col:
    plt.figure()
    yearly_risk = df.groupby('Year')["Volatility_30"].mean()
    
    yearly_risk.plot(kind='bar', color='coral', edgecolor='black')
    
    save_plot("07_yearly_risk_levels.png", "Average Market Risk (Volatility) by Year", "Year", "Average 30-Day Volatility")


# ============================================================
# GRAPH 8 [NEW]: NEXT-DAY MOMENTUM (UP DAYS VS DOWN DAYS)
# ============================================================
if price_col:
    plt.figure()
    
    # Determine if yesterday was an Up day or Down day
    valid_dir = df[['Price_Change', 'Daily_Return']].copy().dropna()
    valid_dir['Yesterday_Direction'] = np.where(valid_dir['Price_Change'].shift(1) > 0, 'Up Yesterday', 'Down Yesterday')
    valid_dir = valid_dir.dropna() # Drop the first NaN row
    
    momentum = valid_dir.groupby('Yesterday_Direction')['Daily_Return'].mean()
    
    momentum.plot(kind='bar', color=['red', 'green'], edgecolor='black')
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(rotation=0)
    
    save_plot("08_next_day_momentum.png", "Average Today's Return based on Yesterday's Trend", "Yesterday's Market Direction", "Average Daily Return Today (%)")


# ============================================================
# GRAPH 9: PRICE VS 30-DAY MOVING AVERAGE
# ============================================================
if price_col:
    moving_average_30 = df[price_col].rolling(30).mean()
    plt.figure()
    plt.plot(df[price_col], label="Gold Price", linewidth=1)
    plt.plot(moving_average_30, label="30-Day Moving Average", linewidth=1.5, color='orange')
    plt.legend()
    save_plot("09_price_vs_ma30.png", "Gold Price and 30-Day Moving Average", "Observation", "Gold Price")


# ============================================================
# GRAPH 10: MONTHLY AVERAGE GOLD PRICE
# ============================================================
if date_col and price_col:
    monthly_price = df.set_index(date_col)[price_col].resample("ME").mean()
    plt.figure()
    plt.plot(monthly_price.index, monthly_price.values, linewidth=1.2, color='teal')
    save_plot("10_monthly_average_price.png", "Monthly Average Gold Price", "Month", "Average Gold Price")


# ============================================================
# GRAPH 11: MONTHLY GOLD PRICE VOLATILITY
# ============================================================
if date_col and price_col:
    monthly_volatility = df.set_index(date_col)["Daily_Return"].resample("ME").std() * 100
    plt.figure()
    plt.bar(monthly_volatility.index, monthly_volatility.values, width=20, color='crimson')
    save_plot("11_monthly_volatility.png", "Monthly Gold Price Volatility", "Month", "Volatility (%)")


# ============================================================
# GRAPH 12: YEARLY AVERAGE GOLD PRICE
# ============================================================
if date_col and price_col:
    yearly_price = df.set_index(date_col)[price_col].resample("YE").mean()
    plt.figure()
    plt.bar(yearly_price.index.year.astype(str), yearly_price.values, color='gold', edgecolor='black')
    save_plot("12_yearly_average_price.png", "Yearly Average Gold Price", "Year", "Average Gold Price")


# ============================================================
# GRAPH 13 [NEW]: RETURNS BY DAY OF THE WEEK
# ============================================================
if price_col and date_col:
    plt.figure()
    
    # Get Day of Week (0 = Monday, 4 = Friday)
    df['Day_of_Week'] = df[date_col].dt.dayofweek
    
    # Group by day and calculate mean return
    dow_returns = df.groupby('Day_of_Week')['Daily_Return'].mean()
    
    # Plot
    colors = ['green' if val > 0 else 'red' for val in dow_returns]
    dow_returns.plot(kind='bar', color=colors, edgecolor='black')
    plt.xticks(range(5), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], rotation=0)
    plt.axhline(0, color='black', linewidth=1)
    
    save_plot("13_day_of_week_returns.png", "Average Daily Return by Day of the Week", "Day of the Week", "Average Return (%)")


# ============================================================
# GRAPH 14 [NEW]: RETURN BY VOLUME TIERS
# ============================================================
volume_col = next((c for c in numeric_cols if "volume" in c.lower()), None)

if volume_col and price_col:
    plt.figure()
    
    valid_vol = df[[volume_col, "Daily_Return"]].dropna().copy()
    
    # Divide Volume into 3 buckets: Low, Medium, High
    valid_vol['Volume_Tier'] = pd.qcut(valid_vol[volume_col], q=3, labels=['Low Volume', 'Medium Volume', 'High Volume'])
    
    vol_impact = valid_vol.groupby('Volume_Tier')['Daily_Return'].mean()
    
    colors = ['green' if val > 0 else 'red' for val in vol_impact]
    vol_impact.plot(kind='bar', color=colors, edgecolor='black')
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(rotation=0)
    
    save_plot("14_return_by_volume.png", "Average Price Return by Trading Volume Levels", "Trading Volume Tier", "Average Return (%)")


# ============================================================
# GRAPH 15: CORRELATION MATRIX
# ============================================================
if len(numeric_cols) >= 2:
    correlation_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[correlation_cols].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest", aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()

    output_path = PLOT_PATH / "15_correlation_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    created.append(output_path)
    print(f"Saved: {output_path}")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("GRAPH GENERATION COMPLETE")
print("=" * 60)

print(f"Dataset shape : {df.shape}")
print(f"Date column   : {date_col}")
print(f"Price column  : {price_col}")
print(f"Graphs created: {len(created)}")
print(f"Output folder : {PLOT_PATH}")

print()
print("Generated files:")

for i, path in enumerate(created, start=1):
    print(f"{i:02d}. {path.name}")

print("=" * 60)