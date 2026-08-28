import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_raw_dataset



# 1. PROJECT PATHS
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# Folder where all report graphs will be saved
PLOT_PATH = project_root / "report_assets" / "plots"
PLOT_PATH.mkdir(parents=True, exist_ok=True)



# 2. LOAD DATASET
df = load_raw_dataset()

# Clean column names
df.columns = [c.strip() for c in df.columns]

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 3. IDENTIFY IMPORTANT COLUMNS
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


# 4. CONVERT DATE AND PRICE
if date_col:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    # Sort chronologically
    # IMPORTANT for calculating returns correctly
    df = (
        df.sort_values(date_col)
        .reset_index(drop=True)
    )


if price_col:

    df[price_col] = pd.to_numeric(
        df[price_col],
        errors="coerce"
    )


if date_col is None:
    print("WARNING: Date column was not found.")

if price_col is None:
    print("WARNING: Price column was not found.")


# 5. PLOT SETTINGS
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True

created = []


# 6. SAVE PLOT FUNCTION
def save_plot(
    filename,
    title,
    xlabel=None,
    ylabel=None
):
    """
    Save the current matplotlib figure into
    report_assets/plots/
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


# 7. NUMERIC COLUMNS
numeric_cols = (
    df
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)



# 8. CALCULATE PRICE MOVEMENT FEATURES
if price_col:

    # Daily price change
    df["Price_Change"] = (
        df[price_col].diff()
    )

    # Daily return (%)
    df["Daily_Return"] = (
        df[price_col].pct_change() * 100
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

    # Absolute daily price movement
    df["Absolute_Price_Change"] = (
        df["Price_Change"].abs()
    )


# GRAPH 1
# GOLD PRICE OVER TIME
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


# GRAPH 2
# GOLD PRICE DISTRIBUTION
if price_col:

    plt.figure()

    plt.hist(
        df[price_col].dropna(),
        bins=30,
        color="skyblue",
        edgecolor="black"
    )

    save_plot(
        "02_gold_price_distribution.png",
        "Distribution of Gold Prices",
        "Gold Price",
        "Frequency"
    )


# GRAPH 3
# YEARLY PRICE RANGE
if price_col and date_col:

    plt.figure()

    df["Year"] = (
        df[date_col]
        .dt.year
    )

    yearly_stats = (
        df.groupby("Year")[price_col]
        .agg(["min", "mean", "max"])
    )

    plt.fill_between(
        yearly_stats.index,
        yearly_stats["min"],
        yearly_stats["max"],
        color="lightblue",
        alpha=0.5,
        label="Min-Max Range"
    )

    plt.plot(
        yearly_stats.index,
        yearly_stats["mean"],
        color="darkblue",
        marker="o",
        linewidth=2,
        label="Average Price"
    )

    plt.legend()

    save_plot(
        "03_yearly_price_range.png",
        "Yearly Gold Price Range",
        "Year",
        "Gold Price"
    )


# GRAPH 4
# AVERAGE PRICE CHANGE BY QUARTER

if price_col and date_col:

    plt.figure()

    df["Quarter"] = (
        df[date_col]
        .dt.quarter
    )

    quarterly_change = (
        df.groupby("Quarter")["Price_Change"]
        .mean()
    )

    quarter_names = [
        "Q1",
        "Q2",
        "Q3",
        "Q4"
    ]

    plt.bar(
        quarter_names,
        quarterly_change.values,
        edgecolor="black"
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1
    )

    save_plot(
        "04_quarterly_price_change.png",
        "Average Gold Price Change by Quarter",
        "Quarter",
        "Average Price Change"
    )


# GRAPH 5
# DAILY RETURN DISTRIBUTION
if price_col:

    plt.figure()

    plt.hist(
        df["Daily_Return"].dropna(),
        bins=40,
        color="lightgreen",
        edgecolor="black"
    )

    save_plot(
        "05_daily_return_distribution.png",
        "Distribution of Daily Gold Returns",
        "Daily Return (%)",
        "Frequency"
    )


# GRAPH 6
# VOLATILITY DISTRIBUTION
if price_col:

    plt.figure()

    plt.hist(
        df["Volatility_7"].dropna(),
        bins=30,
        color="purple",
        edgecolor="black",
        alpha=0.7
    )

    save_plot(
        "06_volatility_distribution.png",
        "Distribution of 7-Day Gold Price Volatility",
        "7-Day Volatility",
        "Number of Days"
    )


# GRAPH 7
# AVERAGE VOLATILITY BY YEAR
if price_col and date_col:

    plt.figure()

    yearly_risk = (
        df.groupby("Year")["Volatility_30"]
        .mean()
    )

    yearly_risk.plot(
        kind="bar",
        color="coral",
        edgecolor="black"
    )

    save_plot(
        "07_yearly_risk_levels.png",
        "Average Gold Price Volatility by Year",
        "Year",
        "Average 30-Day Volatility"
    )


# GRAPH 8
# TOP 10 LARGEST DAILY PRICE INCREASES
# Question:
# What were the largest daily increases in gold price?
if price_col and date_col:

    largest_increases = (
        df[
            [date_col, "Price_Change"]
        ]
        .dropna()
        .nlargest(
            10,
            "Price_Change"
        )
        .sort_values(
            "Price_Change"
        )
    )

    plt.figure()

    plt.barh(
        largest_increases[
            date_col
        ].dt.strftime("%Y-%m-%d"),
        largest_increases[
            "Price_Change"
        ],
        edgecolor="black"
    )

    save_plot(
        "08_top_10_price_increases.png",
        "Top 10 Largest Daily Gold Price Increases",
        "Date",
        "Price Increase"
    )


# GRAPH 9
# TOP 10 LARGEST DAILY PRICE DECREASES
# Question:
# What were the largest daily decreases in gold price?
if price_col and date_col:

    largest_decreases = (
        df[
            [date_col, "Price_Change"]
        ]
        .dropna()
        .nsmallest(
            10,
            "Price_Change"
        )
        .sort_values(
            "Price_Change",
            ascending=False
        )
    )

    plt.figure()

    plt.barh(
        largest_decreases[
            date_col
        ].dt.strftime("%Y-%m-%d"),
        largest_decreases[
            "Price_Change"
        ],
        edgecolor="black"
    )

    save_plot(
        "09_top_10_price_decreases.png",
        "Top 10 Largest Daily Gold Price Decreases",
        "Date",
        "Price Decrease"
    )

# GRAPH 10
# YEARLY GOLD PRICE CHANGE
# Question:
# How much did the gold price change from the beginning
# to the end of each year?
if date_col and price_col:

    yearly_open = (
        df
        .set_index(date_col)[price_col]
        .resample("YE")
        .first()
    )

    yearly_close = (
        df
        .set_index(date_col)[price_col]
        .resample("YE")
        .last()
    )

    yearly_change = (
        yearly_close - yearly_open
    )

    plt.figure()

    plt.bar(
        yearly_change.index.year.astype(str),
        yearly_change.values,
        edgecolor="black"
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1
    )

    save_plot(
        "10_yearly_price_change.png",
        "Yearly Gold Price Change",
        "Year",
        "Price Change"
    )


# GRAPH 11
# MONTHLY POSITIVE VS NEGATIVE DAYS
# Question:
# Do gold prices increase or decrease more often
# during different months?
if price_col and date_col:

    monthly_data = (
        df[
            [
                date_col,
                "Daily_Return"
            ]
        ]
        .dropna()
        .copy()
    )

    monthly_data["Month"] = (
        monthly_data[date_col]
        .dt.month
    )

    monthly_data["Direction"] = np.where(
        monthly_data["Daily_Return"] > 0,
        "Positive",
        "Negative"
    )

    monthly_direction = (
        monthly_data
        .groupby(
            ["Month", "Direction"]
        )
        .size()
        .unstack(fill_value=0)
    )

    # Make sure both columns exist
    if "Positive" not in monthly_direction.columns:
        monthly_direction["Positive"] = 0

    if "Negative" not in monthly_direction.columns:
        monthly_direction["Negative"] = 0

    monthly_direction = (
        monthly_direction[
            ["Positive", "Negative"]
        ]
    )

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec"
    ]

    # Make sure all 12 months appear
    monthly_direction = (
        monthly_direction
        .reindex(
            range(1, 13),
            fill_value=0
        )
    )

    monthly_direction.index = month_names

    plt.figure()

    monthly_direction.plot(
        kind="bar",
        stacked=True,
        ax=plt.gca(),
        edgecolor="black"
    )

    plt.legend(
        title="Daily Movement"
    )

    plt.xticks(
        rotation=45
    )

    save_plot(
        "11_monthly_positive_negative_days.png",
        "Positive and Negative Gold Price Days by Month",
        "Month",
        "Number of Days"
    )


# GRAPH 12
# YEARLY AVERAGE GOLD PRICE
if date_col and price_col:

    yearly_price = (
        df
        .set_index(date_col)[price_col]
        .resample("YE")
        .mean()
    )

    plt.figure()

    plt.bar(
        yearly_price.index.year.astype(str),
        yearly_price.values,
        color="gold",
        edgecolor="black"
    )

    save_plot(
        "12_yearly_average_price.png",
        "Yearly Average Gold Price",
        "Year",
        "Average Gold Price"
    )


# GRAPH 13
# YEARLY GOLD PRICE GROWTH
#
# Question:
# Which years experienced the largest price increases?
if date_col and price_col:

    yearly_avg = (
        df
        .set_index(date_col)[price_col]
        .resample("YE")
        .mean()
    )

    yearly_growth = (
        yearly_avg
        .pct_change()
        * 100
    ).dropna()

    plt.figure()

    plt.bar(
        yearly_growth.index.year.astype(str),
        yearly_growth.values,
        edgecolor="black"
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1
    )

    save_plot(
        "13_yearly_price_growth.png",
        "Yearly Gold Price Growth",
        "Year",
        "Price Growth (%)"
    )


# GRAPH 14
# AVERAGE MONTHLY GOLD PRICE
# Question:
# Which months have higher or lower average gold prices?
if date_col and price_col:

    monthly_average = (
        df
        .assign(
            Month=df[date_col].dt.month
        )
        .groupby("Month")[price_col]
        .mean()
    )

    # Make sure all months are included
    monthly_average = (
        monthly_average
        .reindex(range(1, 13))
    )

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec"
    ]

    plt.figure()

    plt.bar(
        month_names,
        monthly_average.values,
        edgecolor="black"
    )

    save_plot(
        "14_average_monthly_price.png",
        "Average Gold Price by Month",
        "Month",
        "Average Gold Price"
    )


# GRAPH 15
# CORRELATION MATRIX
if len(numeric_cols) >= 2:

    correlation_cols = (
        df
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    corr = (
        df[correlation_cols]
        .corr()
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        corr,
        interpolation="nearest",
        aspect="auto",
        cmap="coolwarm"
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

    print(
        f"Saved: {output_path}"
    )



# 16. FINAL SUMMARY
print()
print("=" * 60)
print("GRAPH GENERATION COMPLETE")
print("=" * 60)

print(
    f"Dataset shape : {df.shape}"
)

print(
    f"Date column   : {date_col}"
)

print(
    f"Price column  : {price_col}"
)

print(
    f"Graphs created: {len(created)}"
)

print(
    f"Output folder : {PLOT_PATH}"
)

print()
print("Generated files:")

for i, path in enumerate(
    created,
    start=1
):

    print(
        f"{i:02d}. {path.name}"
    )

print("=" * 60)