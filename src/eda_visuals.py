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
    (
        c for c in df.columns
        if c.lower() in ["date", "datetime", "timestamp"]
    ),
    None
)

# Identify price column
price_col = next(
    (
        c for c in df.columns
        if c.lower() in [
            "price",
            "gold_price",
            "gold price",
            "close"
        ]
    ),
    None
)

# Convert date
if date_col:
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df = df.sort_values(
        date_col
    ).reset_index(drop=True)


# Convert price
if price_col:
    df[price_col] = pd.to_numeric(
        df[price_col],
        errors="coerce"
    )


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

def save_plot(
    filename,
    title,
    xlabel=None,
    ylabel=None
):
    """
    Save the current matplotlib figure
    into report_assets/plots/.
    """

    plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    plt.tight_layout()

    output_path = PLOT_PATH / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    created.append(output_path)

    print(f"Saved: {output_path}")


# ============================================================
# 6. NUMERIC COLUMNS
# ============================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()


# ============================================================
# 7. CALCULATE RETURNS
# ============================================================

if price_col:

    # Daily price change
    df["Price_Change"] = (
        df[price_col].diff()
    )

    # Daily percentage return
    df["Daily_Return"] = (
        df[price_col]
        .pct_change()
        * 100
    )

    # 7-day rolling volatility
    df["Volatility_7"] = (
        df["Daily_Return"]
        .rolling(7)
        .std()
    )

    # 30-day rolling volatility
    df["Volatility_30"] = (
        df["Daily_Return"]
        .rolling(30)
        .std()
    )


# ============================================================
# GRAPH 1
# GOLD PRICE OVER TIME
# ============================================================

if date_col and price_col:

    plt.figure()

    plt.plot(
        df[date_col],
        df[price_col],
        linewidth=1.2
    )

    save_plot(
        "01_gold_price_time_series.png",
        "Gold Price Over Time",
        "Date",
        "Gold Price"
    )


# ============================================================
# GRAPH 2
# GOLD PRICE DISTRIBUTION
# ============================================================

if price_col:

    plt.figure()

    plt.hist(
        df[price_col].dropna(),
        bins=30
    )

    save_plot(
        "02_gold_price_distribution.png",
        "Distribution of Gold Prices",
        "Gold Price",
        "Frequency"
    )


# ============================================================
# GRAPH 3
# GOLD PRICE BOXPLOT
# ============================================================

if price_col:

    plt.figure()

    plt.boxplot(
        df[price_col].dropna(),
        vert=True
    )

    save_plot(
        "03_gold_price_boxplot.png",
        "Gold Price Boxplot",
        ylabel="Gold Price"
    )


# ============================================================
# GRAPH 4
# DAILY PRICE CHANGE
# ============================================================

if price_col:

    plt.figure()

    plt.plot(
        df["Price_Change"],
        linewidth=0.8
    )

    save_plot(
        "04_daily_price_change.png",
        "Daily Gold Price Change",
        "Observation",
        "Price Change"
    )


# ============================================================
# GRAPH 5
# DAILY RETURN DISTRIBUTION
# ============================================================

if price_col:

    plt.figure()

    plt.hist(
        df["Daily_Return"].dropna(),
        bins=40
    )

    save_plot(
        "05_daily_return_distribution.png",
        "Distribution of Daily Gold Returns",
        "Daily Return (%)",
        "Frequency"
    )


# ============================================================
# GRAPH 6
# 7-DAY ROLLING VOLATILITY
# ============================================================

if price_col:

    plt.figure()

    plt.plot(
        df["Volatility_7"],
        linewidth=1
    )

    save_plot(
        "06_rolling_7day_volatility.png",
        "7-Day Rolling Gold Price Volatility",
        "Observation",
        "Volatility (%)"
    )


# ============================================================
# GRAPH 7
# 30-DAY ROLLING VOLATILITY
# ============================================================

if price_col:

    plt.figure()

    plt.plot(
        df["Volatility_30"],
        linewidth=1
    )

    save_plot(
        "07_rolling_30day_volatility.png",
        "30-Day Rolling Gold Price Volatility",
        "Observation",
        "Volatility (%)"
    )


# ============================================================
# GRAPH 8
# PRICE VS PREVIOUS-DAY PRICE
# ============================================================

if price_col:

    lag1 = df[price_col].shift(1)

    valid = pd.DataFrame({
        "Previous_Price": lag1,
        "Current_Price": df[price_col]
    }).dropna()

    plt.figure()

    plt.scatter(
        valid["Previous_Price"],
        valid["Current_Price"],
        s=12,
        alpha=0.6
    )

    save_plot(
        "08_price_vs_lag1.png",
        "Gold Price vs Previous-Day Price",
        "Previous-Day Price",
        "Current Price"
    )


# ============================================================
# GRAPH 9
# PRICE VS 30-DAY MOVING AVERAGE
# ============================================================

if price_col:

    moving_average_30 = (
        df[price_col]
        .rolling(30)
        .mean()
    )

    plt.figure()

    plt.plot(
        df[price_col],
        label="Gold Price",
        linewidth=1
    )

    plt.plot(
        moving_average_30,
        label="30-Day Moving Average",
        linewidth=1.5
    )

    plt.legend()

    save_plot(
        "09_price_vs_ma30.png",
        "Gold Price and 30-Day Moving Average",
        "Observation",
        "Gold Price"
    )


# ============================================================
# GRAPH 10
# MONTHLY AVERAGE GOLD PRICE
# ============================================================

if date_col and price_col:

    monthly_price = (
        df.set_index(date_col)[price_col]
        .resample("ME")
        .mean()
    )

    plt.figure()

    plt.plot(
        monthly_price.index,
        monthly_price.values,
        linewidth=1.2
    )

    save_plot(
        "10_monthly_average_price.png",
        "Monthly Average Gold Price",
        "Month",
        "Average Gold Price"
    )


# ============================================================
# GRAPH 11
# MONTHLY GOLD PRICE VOLATILITY
# ============================================================

if date_col and price_col:

    monthly_volatility = (
        df.set_index(date_col)["Daily_Return"]
        .resample("ME")
        .std()
        * 100
    )

    plt.figure()

    plt.bar(
        monthly_volatility.index,
        monthly_volatility.values,
        width=20
    )

    save_plot(
        "11_monthly_volatility.png",
        "Monthly Gold Price Volatility",
        "Month",
        "Volatility (%)"
    )


# ============================================================
# GRAPH 12
# YEARLY AVERAGE GOLD PRICE
# ============================================================

if date_col and price_col:

    yearly_price = (
        df.set_index(date_col)[price_col]
        .resample("YE")
        .mean()
    )

    plt.figure()

    plt.bar(
        yearly_price.index.year.astype(str),
        yearly_price.values
    )

    save_plot(
        "12_yearly_average_price.png",
        "Yearly Average Gold Price",
        "Year",
        "Average Gold Price"
    )


# ============================================================
# GRAPH 13
# GOLD PRICE AUTOCORRELATION
# ============================================================

if price_col:

    max_lag = min(
        30,
        len(df) // 4
    )

    autocorrelation = [
        df[price_col].autocorr(lag=i)
        for i in range(1, max_lag + 1)
    ]

    plt.figure()

    plt.bar(
        range(1, max_lag + 1),
        autocorrelation
    )

    save_plot(
        "13_price_autocorrelation.png",
        "Gold Price Autocorrelation by Lag",
        "Lag",
        "Autocorrelation"
    )


# ============================================================
# GRAPH 14
# DAILY RETURN VS TRADING VOLUME
# ============================================================

volume_col = next(
    (
        c for c in numeric_cols
        if "volume" in c.lower()
    ),
    None
)


if volume_col and price_col:

    valid = pd.DataFrame({
        "Volume": df[volume_col],
        "Daily_Return": df["Daily_Return"]
    }).dropna()

    plt.figure()

    plt.scatter(
        valid["Volume"],
        valid["Daily_Return"],
        s=12,
        alpha=0.6
    )

    save_plot(
        "14_return_vs_volume.png",
        "Daily Return vs Trading Volume",
        "Trading Volume",
        "Daily Return (%)"
    )


# ============================================================
# GRAPH 15
# CORRELATION MATRIX
# ============================================================

if len(numeric_cols) >= 2:

    # Recalculate numeric columns after adding
    # derived variables
    correlation_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    corr = df[correlation_cols].corr()

    plt.figure(figsize=(10, 8))

    plt.imshow(
        corr,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title(
        "Correlation Matrix of Numeric Features"
    )

    plt.tight_layout()

    output_path = (
        PLOT_PATH /
        "15_correlation_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

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