import warnings
from datetime import datetime
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ==========================================
# 0. PATHS & CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent          # .../prototype
PROJECT_ROOT = BASE_DIR.parent                        # repo root
METRICS_DIR = BASE_DIR / "summary_metrics"
PLOTS_DIR = PROJECT_ROOT / "report_assets" / "plots"
EDA_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "Gold_Price_cleaned.csv"  # clickable graph

st.set_page_config(layout="wide", page_title="Gold Price Predictor", page_icon="\U0001F947")

ALGO_LR = "Linear Regression (Walk-Forward)"
ALGO_KNN = "KNN Regression (Walk-Forward)"
ALGO_RF = "Random Forest (Walk-Forward)"
ALGO_GB = "Gradient Boosting (Walk-Forward)"

MODEL_FILES = {
    ALGO_LR: "linear_regression_walkforward_price.pkl",
    ALGO_KNN: "knn_walkforward_price.pkl",
    ALGO_RF: "random_forest_price.pkl",
    ALGO_GB: "gradient_boosting_price.pkl"
}

METRIC_FILES = {
    ALGO_LR: "walkforward_price_summary_metrics.json",
    ALGO_KNN: "walkforward_price_knn_summary_metrics.json",
    ALGO_RF: "random_forest_summary_metrics.json",
    ALGO_GB: "gradient_boosting_summary_metrics.json"
    
}

FEATURES = {
    ALGO_LR: ["Volume", "Month", "Day", "Volatility_7", "MA_7"],
    ALGO_KNN: ["Volume_Momentum", "Volatility_7", "Volatility_30", "RSI_14",
               "daily_return_lag1", "daily_return_lag2"],
    ALGO_RF: ["Volume", "Month", "Day", "Volatility_7", "Return_Lag1"],
    ALGO_GB: ["Volume", "Volatility_7", "Return_Lag1", "Momentum_7"]
}

# 5 most relevant plots for a forecasting dashboard, out of the 15 saved
# under report_assets/plots -- picked to explain the trend, the moving-
# average / volatility / seasonality features the models are built on,
# and how those engineered features relate to each other.
PLOT_SELECTION = [
    ("01_gold_price_time_series.png", "Gold Price Over Time",
     "Long-run closing-price trend that motivates the forecasting problem."),
    ("09_price_vs_ma30.png", "Price vs 30-Day Moving Average",
     "Price tracking its moving average -- the same signal MA_7 captures for the Linear Regression model."),
    ("06_volatility_distribution.png", "Volatility Distribution",
     "Spread of rolling volatility -- the Volatility_7 / Volatility_30 features used by all three models."),
    ("04_monthly_seasonality.png", "Monthly Seasonality",
     "Average price behaviour by calendar month -- the seasonal signal behind the Month feature."),
    ("15_correlation_matrix.png", "Feature Correlation Matrix",
     "Correlation between engineered features -- shows why each model was built on a different feature set."),
]

METRIC_COLUMN_ORDER = ["test_MAE", "test_RMSE", "test_MAPE", "test_LogMAE",
                        "test_LogRMSE", "train_R2", "test_R2", "R2_gap"]


# ==========================================
# 1. CACHED LOADERS
# ==========================================
@st.cache_resource
def load_models():
    loaded = {}
    for algo, fname in MODEL_FILES.items():
        path = BASE_DIR / "model"/fname
        loaded[algo] = joblib.load(path) if path.exists() else None
    return loaded


@st.cache_data
def load_metrics():
    loaded = {}
    for algo, fname in METRIC_FILES.items():
        path = METRICS_DIR / fname
        if path.exists():
            with open(path) as f:
                loaded[algo] = json.load(f)
        else:
            loaded[algo] = None
    return loaded

@st.cache_data
def load_eda_data():
    """Processed dataset that backs the interactive Market Insights charts."""
    if not EDA_DATA_FILE.exists():
        return None
    df = pd.read_csv(EDA_DATA_FILE, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "Gold_Price.csv"

@st.cache_data
def load_raw_eda_data():
    """Raw dataset with the SAME derived columns, in the SAME order, as
    src/eda_visuals.py (the script that generated report_assets/plots).
    Used so the Market Insights charts match those report images exactly,
    instead of the differently-defined engineered features in the
    processed/cleaned CSV used for model training."""
    if not RAW_DATA_FILE.exists():
        return None
    df = pd.read_csv(RAW_DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df["Price_Change"] = df["Price"].diff()
    df["Daily_Return"] = df["Price"].pct_change() * 100
    df["Volatility_7"] = df["Daily_Return"].rolling(7).std()
    df["Volatility_30"] = df["Daily_Return"].rolling(30).std()
    df["Absolute_Price_Change"] = df["Price_Change"].abs()
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter
    df["Month"] = df["Date"].dt.month
    return df

def _selected_points(event):
    """Pull the list of clicked/selected points out of a st.plotly_chart on_select event, regardless of whether anything was clicked yet."""
    if not event:
        return []
    sel = event.get("selection") if isinstance(event, dict) else None
    return sel.get("points", []) if sel else []

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "select2d", "lasso2d", "autoScale2d",
        "hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines",
    ],
}

models = load_models()
metrics = load_metrics()
eda_df = load_eda_data()
raw_eda_df = load_raw_eda_data()


# ==========================================
# 2. SHARED FEATURE-ENGINEERING HELPERS
#    (raw numbers in -> engineered features out, computed in the backend
#    so the user never has to compute MA_7 / RSI_14 / etc. by hand)
# ==========================================
def calculate_rsi(prices, window=14):
    """Relative Strength Index of the most recent point in `prices`."""
    s = pd.Series(prices, dtype=float)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def parse_number_list(raw_text, min_count, label):
    """Parse a comma-separated list of numbers, with a friendly error
    message on anything invalid or incomplete."""
    if not raw_text or not raw_text.strip():
        return None, f"{label}: this field is empty. Please enter comma-separated numbers."
    try:
        values = [float(x.strip()) for x in raw_text.split(",") if x.strip() != ""]
    except ValueError:
        return None, f"{label}: contains a value that is not a number."
    if len(values) < min_count:
        return None, f"{label}: needs at least {min_count} values, you entered {len(values)}."
    return values, None


def next_business_day(d):
    nd = d + pd.Timedelta(days=1)
    while nd.weekday() >= 5:  # skip Sat/Sun
        nd += pd.Timedelta(days=1)
    return nd


def run_recursive_forecast(algo, model, seed_prices, seed_volumes, anchor_date, n_days):
    """Recursively forecast `n_days` of future closing prices for one
    algorithm, starting the day after `anchor_date`, seeded ONLY from the
    prices/volumes the user typed into that algorithm's own section.

    Future Trading Volume can't be known in advance, so it is approximated
    as the average of the seed volumes at each step (held roughly constant).
    """
    price_hist = list(seed_prices)[-30:]
    vol_hist = list(seed_volumes)[-10:] if seed_volumes else [0.0]
    current_date = anchor_date

    dates, preds = [], []
    for _ in range(n_days):
        current_date = next_business_day(current_date)
        vol_forecast = float(np.mean(vol_hist))
        month, day = current_date.month, current_date.day
        prices_7 = price_hist[-7:]
        vol7 = float(np.std(prices_7, ddof=1)) if len(prices_7) > 1 else 0.0

        if algo == ALGO_LR:
            ma7 = float(np.mean(prices_7))
            X = np.array([[vol_forecast, month, day, vol7, ma7]])
            pred_price = float(model.predict(X)[0])

        elif algo == ALGO_RF:
            price_lag1 = price_hist[-1]
            return_lag1 = np.log(price_hist[-1] / price_hist[-2])
            X = np.array([[vol_forecast, month, day, vol7, return_lag1]])
            change = float(model.predict(X)[0])
            pred_price = price_lag1 * np.exp(change)

        elif algo == ALGO_KNN:
            prices_30 = price_hist[-30:]
            vol30 = float(np.std(prices_30, ddof=1))
            rsi = calculate_rsi(prices_30, window=14)
            ret1 = (prices_30[-1] - prices_30[-2]) / prices_30[-2]
            ret2 = (prices_30[-2] - prices_30[-3]) / prices_30[-3]
            vol_mom = vol_hist[-1] / np.mean(vol_hist)
            price_lag1 = prices_30[-1]
            X = np.array([[vol_mom, vol7, vol30, rsi, ret1, ret2]])
            diff = float(model.predict(X)[0])
            pred_price = price_lag1 + diff

        elif algo == ALGO_GB:
            prices_7 = price_hist[-7:]
            vol7 = float(np.std(prices_7, ddof=1)) if len(prices_7) > 1 else 0.0

            ret_lag1 = np.log(price_hist[-1] / price_hist[-2])
            mom7 = (price_hist[-1] / price_hist[-8]) - 1 if len(price_hist) >= 8 else 0.0

            X = np.array([[vol_forecast, vol7, ret_lag1, mom7]])

            pred_log_ret = float(model.predict(X)[0])
            pred_price = price_hist[-1] * np.exp(pred_log_ret)

        else:
            continue

        dates.append(current_date)
        preds.append(pred_price)
        price_hist = (price_hist + [pred_price])[-30:]
        vol_hist = (vol_hist + [vol_forecast])[-10:]

    return dates, preds


def compute_and_store_forecast(algo, seed_prices, seed_volumes, n_days):
    """Runs a multi-day forecast for one algorithm, seeded only from that
    algorithm's own entered inputs, and stores it in session_state so it
    can be drawn -- together with any other algorithm's forecast -- in the
    shared "Your Multi-Day Forecasts" section further down the page."""
    model = models[algo]
    if model is None:
        st.error(f"Model file not found: {MODEL_FILES[algo]}")
        return

    anchor_date = pd.Timestamp(datetime.now().date())
    dates, preds = run_recursive_forecast(algo, model, seed_prices, seed_volumes, anchor_date, n_days)
    forecast_series = pd.Series(preds, index=pd.DatetimeIndex(dates), name=algo)

    if "forecast_results" not in st.session_state:
        st.session_state.forecast_results = {}
    st.session_state.forecast_results[algo] = forecast_series
    st.success(f"{int(n_days)}-day forecast added \u2014 see \u201cYour Multi-Day Forecasts\u201d below.")


# ==========================================
# 3. PAGE HEADER
# ==========================================
st.title("\U0001F947 Gold Price Prediction & Analytics Dashboard")
st.caption("BMDS2003 Data Science Group Project -- CRISP-DM prototype")

tab_insights, tab_compare, tab_predict = st.tabs(
    ["\U0001F4C8 Market Insights", "\u2696\uFE0F Model Comparison",
     "\U0001F52E Predict & Forecast"]
)

# ==========================================
# TAB 1: MARKET INSIGHTS (5 EDA plots)
# ==========================================
with tab_insights:
    st.subheader("Key Exploratory Data Analysis Plots")
    st.write("5 interactive charts built live from the same processed data the models below are trained on. Click a point, bar or cell on any chart to see its exact values.")
    if eda_df is None:
        st.error(f"Processed dataset not found: {EDA_DATA_FILE}")
    else:
        col1, col2 = st.columns(2, gap="large")

        # 1. Gold Price Over Time
        with col1:
            st.markdown("**Gold Price Over Time**")
            fig1 = go.Figure(go.Scatter(
                x=eda_df["Date"], y=eda_df["Price"], mode="lines",
                line=dict(color="#1f77b4", width=1.5),
                customdata=np.stack([eda_df["MA_7"], eda_df["MA_30"], eda_df["Volatility_7"]], axis=-1),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Price: %{y:,.0f}<br>"
                    "MA_7: %{customdata[0]:,.0f}<br>MA_30: %{customdata[1]:,.0f}<extra></extra>"
                ),
            ))
            fig1.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis_title="Date", yaxis_title="Gold Price",
                                title=dict(text="Gold Price Over Time", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            event1 = st.plotly_chart(fig1, use_container_width=True, on_select="rerun",
                                      key="plot_price_time", config=PLOTLY_CONFIG)
            st.caption("The fundamental long-term macro trend of gold prices that motivates the forecasting problem.")
            pts = _selected_points(event1)
            if pts:
                p = pts[0]
                ma7, ma30, vol7 = p["customdata"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Date:", pd.to_datetime(p["x"]).strftime("%Y-%m-%d"))
                c2.metric("Price:", f"{p['y']:,.0f}")
                c3.metric("MA_7:", f"{ma7:,.0f}")
                c4.metric("MA_30:", f"{ma30:,.0f}")

        # 2. Positive and Negative Days by Month
        with col2:
            st.markdown("**Positive and Negative Days by Month**")
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            md = eda_df[["Date", "Chg%"]].dropna().copy()
            md["Month"] = md["Date"].dt.month
            md["Direction"] = np.where(md["Chg%"] > 0, "Positive", "Negative")
            monthly_direction = (
                md.groupby(["Month", "Direction"]).size().unstack(fill_value=0)
                .reindex(columns=["Positive", "Negative"], fill_value=0)
                .reindex(range(1, 13), fill_value=0)
            )
            monthly_direction.index = month_names
            fig2 = go.Figure()
            fig2.add_bar(x=month_names, y=monthly_direction["Positive"], name="Positive",
                          marker_color="#1f77b4",
                          hovertemplate="Month: %{x}<br>Positive days: %{y}<extra></extra>")
            fig2.add_bar(x=month_names, y=monthly_direction["Negative"], name="Negative",
                          marker_color="#ff7f0e",
                          hovertemplate="Month: %{x}<br>Negative days: %{y}<extra></extra>")
            fig2.update_layout(barmode="stack", height=380, margin=dict(l=10, r=10, t=20, b=10),
                                 xaxis_title="Month", yaxis_title="Number of Days",
                                 legend_title="Daily Movement",
                                 title=dict(text="Positive and Negative Gold Price Days by Month", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            event2 = st.plotly_chart(fig2, use_container_width=True, on_select="rerun",
                                       key="plot_monthly_direction",config=PLOTLY_CONFIG)
            st.caption("Explores whether prices historically increase or decrease more often in specific months, justifying the 'Month' feature.")
            pts = _selected_points(event2)
            if pts:
                p = pts[0]
                curve_name = "Positive" if p.get("curve_number", 0) == 0 else "Negative"
                st.write(f"**{p['x']}** \u2192 **{int(p['y'])} {curve_name.lower()} days**")

        st.markdown("<br>", unsafe_allow_html=True)
        col3, col4 = st.columns(2, gap="large")
        # 3. 7-Day Volatility Distribution 
        with col3:
            st.markdown("**7-Day Volatility Distribution**")
            vol = raw_eda_df["Volatility_7"].dropna()
            counts, bin_edges = np.histogram(vol, bins=30)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            fig3 = go.Figure(go.Bar(
                x=bin_centers, y=counts, marker_color="purple", marker_line_color="black",
                marker_line_width=0.5, opacity=0.7, width=(bin_edges[1] - bin_edges[0]),
                customdata=np.stack([bin_edges[:-1], bin_edges[1:]], axis=-1),
                hovertemplate="Range: %{customdata[0]:.2f}\u2013%{customdata[1]:.2f}<br>Count: %{y}<extra></extra>",
            ))
            fig3.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis_title="7-Day Volatility", yaxis_title="Number of Days",
                                title=dict(text="Distribution of 7-Day Gold Price Volatility", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            event3 = st.plotly_chart(fig3, use_container_width=True, on_select="rerun",
                                      key="plot_volatility_dist",config=PLOTLY_CONFIG)
            st.caption("Shows the spread of short-term rolling volatility\u2014a key feature (Volatility_7) used across all four algorithms.")
            pts = _selected_points(event3)
            if pts:
                p = pts[0]
                lo, hi = p["customdata"]
                st.write(f"**Selected bin:** {lo:.2f} \u2013 {hi:.2f} \u2192 **{int(p['y'])} days** fall in this range.")

        # 4. Daily Return Distribution
        with col4:
            st.markdown("**Daily Return Distribution**")
            ret = raw_eda_df["Daily_Return"].dropna()
            counts, bin_edges = np.histogram(ret, bins=40)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            fig4 = go.Figure(go.Bar(
                x=bin_centers, y=counts, marker_color="lightgreen", marker_line_color="black",
                marker_line_width=0.5, width=(bin_edges[1] - bin_edges[0]),
                customdata=np.stack([bin_edges[:-1], bin_edges[1:]], axis=-1),
                hovertemplate="Range: %{customdata[0]:.2f}\u2013%{customdata[1]:.2f}%<br>Count: %{y}<extra></extra>",
            ))
            fig4.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis_title="Daily Return (%)", yaxis_title="Frequency",
                                title=dict(text="Distribution of Daily Gold Returns", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            event4 = st.plotly_chart(fig4, use_container_width=True, on_select="rerun",
                                      key="plot_return_dist",config=PLOTLY_CONFIG)
            st.caption("Visualizes the spread of daily returns, which models like Gradient Boosting and Random Forest rely on as lagged input features.")
            pts = _selected_points(event4)
            if pts:
                p = pts[0]
                lo, hi = p["customdata"]
                st.write(f"**Selected bin:** {lo:.2f}% \u2013 {hi:.2f}% \u2192 **{int(p['y'])} days** fall in this range.")

        st.markdown("<br>", unsafe_allow_html=True)
        col5, col6 = st.columns(2, gap="large")
        # 5. Feature Correlation Matrix 
        with col5:
            st.markdown("**Feature Correlation Matrix**")
            corr_cols = ["Price", "Open", "High", "Low", "Volume", "Chg%",
                        "Price_Change", "Daily_Return", "Volatility_7", "Volatility_30",
                        "Absolute_Price_Change", "Year", "Quarter"]
            corr_cols = [c for c in corr_cols if c in raw_eda_df.columns]
            corr = raw_eda_df[corr_cols].corr()
            fig5 = go.Figure(go.Heatmap(
                z=corr.values, x=list(corr.columns), y=list(corr.columns),
                colorscale="RdBu_r", zmin=float(corr.values.min()), zmax=1,
                colorbar=dict(title="Correlation"),
                hovertemplate="%{x} vs %{y}<br>Correlation %{z:.2f}<extra></extra>",
            ))
            fig5.update_layout(height=650, margin=dict(l=5, r=5, t=40, b=5), 
                               title=dict(text="Correlation Matrix of Numeric Features", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            fig5.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
            fig5.update_xaxes(constrain="domain")
            event5 = st.plotly_chart(fig5, use_container_width=True, on_select="rerun", key="plot_corr_matrix",config=PLOTLY_CONFIG)
            st.caption("Displays the mathematical relationships between numeric variables, guiding the overall feature selection strategy.")
            pts = _selected_points(event5)
            if pts:
                p = pts[0]
                st.write(f"**{p['x']}** vs **{p['y']}** \u2192 correlation of **{p['z']:.2f}**")

# ==========================================
# TAB 2: MODEL COMPARISON (4 algorithms)
# ==========================================
with tab_compare:
    st.subheader("Walk-Forward Validation Metrics -- All 4 Algorithms")

    rows = {}
    for algo in MODEL_FILES:
        m = metrics.get(algo)
        if m and "mean" in m:
            rows[algo] = m["mean"]
        elif m:
            rows[algo] = m

    if not rows:
        st.error("No metrics files found under prototype/metrics/.")
    else:
        comp_df = pd.DataFrame(rows).T
        cols_present = [c for c in METRIC_COLUMN_ORDER if c in comp_df.columns]
        comp_df = comp_df[cols_present]

        st.dataframe(comp_df.style.format("{:,.4f}"), use_container_width=True)

        c1, c2 = st.columns(2)
        if "test_R2" in comp_df:
            best_r2 = comp_df["test_R2"].idxmax()
            c1.metric("Highest Test R\u00b2", f"{comp_df.loc[best_r2, 'test_R2']:.4f}")
            c1.caption(f"Model: {best_r2}")
        if "test_MAE" in comp_df:
            best_mae = comp_df["test_MAE"].idxmin()
            c2.metric("Lowest Test MAE", f"{comp_df.loc[best_mae, 'test_MAE']:,.2f}")
            c2.caption(f"Model: {best_mae}")

        st.markdown("**Test R\u00b2 by algorithm**")
        if "test_R2" in comp_df:
            st.bar_chart(comp_df[["test_R2"]])
        st.markdown("**Test MAE / RMSE by algorithm**")
        err_cols = [c for c in ["test_MAE", "test_RMSE"] if c in comp_df.columns]
        if err_cols:
            st.bar_chart(comp_df[err_cols])

# ==========================================
# TAB 3: PREDICT & FORECAST -- static per-algorithm manual input
# ==========================================
with tab_predict:
    st.subheader("Predict From Manual Input")
    st.write(
        "Every algorithm has its own section below -- fill in the raw numbers it needs "
        "and the app calculates the engineered features automatically. Fill in exactly "
        "one section for a single prediction, or fill in two or three to compare them. "
        "Each section also lets you forecast several days ahead using that same data."
    )

    if "pred_results" not in st.session_state:
        st.session_state.pred_results = {}  # algo_name -> predicted price

    # ---------------- Linear Regression ----------------
    with st.expander(f"\U0001F4D0 {ALGO_LR}", expanded=True):
        st.caption(f"Features used: {', '.join(FEATURES[ALGO_LR])}")
        c1, c2, c3 = st.columns(3)
        with c1:
            lr_volume = st.number_input("Yesterday's Trading Volume", min_value=0.0,
                             value=51877.0, key="lr_volume")
        with c2:
            lr_month = st.number_input("Month (1-12) [blank = today]", min_value=1, max_value=12,
                                        value=None, key="lr_month")
        with c3:
            lr_day = st.number_input("Day (1-31) [blank = today]", min_value=1, max_value=31,
                                      value=None, key="lr_day")
        lr_prices_raw = st.text_area(
            "Last 7 closing prices (comma-separated, oldest \u2192 newest)",
            value="136104, 137789, 132595, 133974, 135454, 135771, 135793", key="lr_prices")

        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        with bcol1:
            lr_do_predict = st.button("Predict Next Day", key="lr_btn")
        with bcol2:
            lr_ndays = st.number_input("Days ahead", min_value=1, max_value=60, value=10, key="lr_ndays")
        with bcol3:
            lr_do_forecast = st.button("Forecast Ahead", key="lr_forecast_btn")

        if lr_do_predict or lr_do_forecast:
            prices, err = parse_number_list(lr_prices_raw, 7, "Last 7 closing prices")
            if err:
                st.error(err)
            elif len(prices) != 7:
                st.error(f"Please enter exactly 7 prices. You entered {len(prices)}.")
            elif models[ALGO_LR] is None:
                st.error(f"Model file not found: {MODEL_FILES[ALGO_LR]}")
            else:
                if lr_do_predict:
                    final_month = int(lr_month) if lr_month is not None else datetime.now().month
                    final_day = int(lr_day) if lr_day is not None else datetime.now().day
                    ma7 = float(np.mean(prices))
                    vol7 = float(np.std(prices, ddof=1))
                    X = np.array([[lr_volume, final_month, final_day, vol7, ma7]])
                    pred = float(models[ALGO_LR].predict(X)[0])
                    st.session_state.pred_results[ALGO_LR] = pred

                    st.success(f"### Predicted Next Closing Price: ${pred:,.2f}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MA_7", f"{ma7:,.2f}")
                    m2.metric("Volatility_7", f"{vol7:,.2f}")
                    m3.metric("Month / Day used", f"{final_month} / {final_day}")

                if lr_do_forecast:
                    compute_and_store_forecast(ALGO_LR, prices, [lr_volume], int(lr_ndays))

    # ---------------- KNN Regression ----------------
    with st.expander(f"\U0001F4CA {ALGO_KNN}", expanded=True):
        st.caption(f"Features used: {', '.join(FEATURES[ALGO_KNN])}")
        knn_prices_raw = st.text_area(
            "Last 30 closing prices (comma-separated, oldest \u2192 newest)",
            value=", ".join(str(128000.0 + (i * 250)) for i in range(30)), key="knn_prices")
        knn_vol_raw = st.text_area(
            "Last 10 trading volumes (comma-separated, oldest \u2192 newest)",
            value=", ".join(str(45000.0 + (i * 500)) for i in range(10)), key="knn_vols")

        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        with bcol1:
            knn_do_predict = st.button("Predict Next Day", key="knn_btn")
        with bcol2:
            knn_ndays = st.number_input("Days ahead", min_value=1, max_value=60, value=10, key="knn_ndays")
        with bcol3:
            knn_do_forecast = st.button("Forecast Ahead", key="knn_forecast_btn")

        if knn_do_predict or knn_do_forecast:
            prices, err1 = parse_number_list(knn_prices_raw, 30, "Last 30 closing prices")
            vols, err2 = parse_number_list(knn_vol_raw, 10, "Last 10 trading volumes")
            errs = [e for e in (err1, err2) if e]
            if errs:
                for e in errs:
                    st.error(e)
            elif models[ALGO_KNN] is None:
                st.error(f"Model file not found: {MODEL_FILES[ALGO_KNN]}")
            else:
                prices_30 = prices[-30:]
                vols_10 = vols[-10:]

                if knn_do_predict:
                    prices_7 = prices_30[-7:]
                    vol30 = float(np.std(prices_30, ddof=1))
                    vol7 = float(np.std(prices_7, ddof=1))
                    rsi = calculate_rsi(prices_30, window=14)
                    ret1 = (prices_30[-1] - prices_30[-2]) / prices_30[-2]
                    ret2 = (prices_30[-2] - prices_30[-3]) / prices_30[-3]
                    vol_mom = vols_10[-1] / np.mean(vols_10)
                    price_lag1 = prices_30[-1]

                    X = np.array([[vol_mom, vol7, vol30, rsi, ret1, ret2]])
                    diff = float(models[ALGO_KNN].predict(X)[0])
                    pred = price_lag1 + diff
                    st.session_state.pred_results[ALGO_KNN] = pred

                    st.success(f"### Predicted Next Closing Price: ${pred:,.2f}")
                    st.caption(
                        f"Model predicts the day-over-day price change (${diff:,.2f}), "
                        f"added to the last known close of ${price_lag1:,.2f}."
                    )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Volatility_7 / _30", f"{vol7:,.1f} / {vol30:,.1f}")
                    m2.metric("RSI_14", f"{rsi:.2f}")
                    m3.metric("Volume Momentum", f"{vol_mom:.3f}")

                if knn_do_forecast:
                    compute_and_store_forecast(ALGO_KNN, prices_30, vols_10, int(knn_ndays))

    # ---------------- Random Forest ----------------
    with st.expander(f"\U0001F332 {ALGO_RF}", expanded=True):
        st.caption(f"Features used: {', '.join(FEATURES[ALGO_RF])}")
        c1, c2, c3 = st.columns(3)
        with c1:
            rf_volume = st.number_input("Yesterday's Trading Volume", min_value=0.0,
                             value=51877.0, key="rf_volume")
        with c2:
            rf_month = st.number_input("Month (1-12) [blank = today]", min_value=1, max_value=12,
                                        value=None, key="rf_month")
        with c3:
            rf_day = st.number_input("Day (1-31) [blank = today]", min_value=1, max_value=31,
                                      value=None, key="rf_day")
        rf_prices_raw = st.text_area(
            "Last 7 closing prices (comma-separated, oldest \u2192 newest; last value = yesterday's close)",
            value="136104, 137789, 132595, 133974, 135454, 135771, 135793", key="rf_prices")

        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        with bcol1:
            rf_do_predict = st.button("Predict Next Day", key="rf_btn")
        with bcol2:
            rf_ndays = st.number_input("Days ahead", min_value=1, max_value=60, value=10, key="rf_ndays")
        with bcol3:
            rf_do_forecast = st.button("Forecast Ahead", key="rf_forecast_btn")

        if rf_do_predict or rf_do_forecast:
            prices, err = parse_number_list(rf_prices_raw, 7, "Last 7 closing prices")
            if err:
                st.error(err)
            elif len(prices) != 7:
                st.error(f"Please enter exactly 7 prices. You entered {len(prices)}.")
            elif models[ALGO_RF] is None:
                st.error(f"Model file not found: {MODEL_FILES[ALGO_RF]}")
            else:
                if rf_do_predict:
                    final_month = int(rf_month) if rf_month is not None else datetime.now().month
                    final_day = int(rf_day) if rf_day is not None else datetime.now().day
                    vol7 = float(np.std(prices, ddof=1))
                    price_lag1 = prices[-1]
                    return_lag1 = np.log(prices[-1] / prices[-2])
                    X = np.array([[rf_volume, final_month, final_day, vol7, return_lag1]])
                    change = float(models[ALGO_RF].predict(X)[0])
                    pred = price_lag1 * np.exp(change)
                    st.session_state.pred_results[ALGO_RF] = pred

                    st.success(f"### Predicted Next Closing Price: ${pred:,.2f}")
                    st.caption(
                        f"Model predicts a log return of {change:+.6f}, "
                        f"which is converted to a predicted closing price from "
                        f"the last known close of ${price_lag1:,.2f}."
                    )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Volatility_7", f"{vol7:,.2f}")
                    m2.metric("Return_Lag1", f"{return_lag1:+.6f}")
                    m3.metric("Month / Day used", f"{final_month} / {final_day}")

                if rf_do_forecast:
                    compute_and_store_forecast(ALGO_RF, prices, [rf_volume], int(rf_ndays))

    # ---------------- Gradient Boosting ----------------
    with st.expander(f"🚀 {ALGO_GB}", expanded=True):
        st.caption(f"Features used: {', '.join(FEATURES[ALGO_GB])}")
        
        c1 = st.columns(1)[0]
        with c1:
            gb_volume = st.number_input("Yesterday's Trading Volume", min_value=0.0, value=51877.0, key="gb_volume")
            
        gb_prices_raw = st.text_area(
            "Last 8 closing prices (comma-separated, oldest → newest; at least 8 required for 7-day Momentum)",
            value="134000, 136104, 137789, 132595, 133974, 135454, 135771, 135793", 
            key="gb_prices"
        )

        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        with bcol1:
            gb_do_predict = st.button("Predict Next Day", key="gb_btn")
        with bcol2:
            gb_ndays = st.number_input("Days ahead", min_value=1, max_value=60, value=10, key="gb_ndays")
        with bcol3:
            gb_do_forecast = st.button("Forecast Ahead", key="gb_forecast_btn")

        if gb_do_predict or gb_do_forecast:
            prices, err = parse_number_list(gb_prices_raw, 8, "Last 8 closing prices")
            if err:
                st.error(err)
            elif models[ALGO_GB] is None:
                st.error(f"Model file not found: {MODEL_FILES[ALGO_GB]}")
            else:
                if gb_do_predict:
                    vol7 = float(np.std(prices[-7:], ddof=1))
                    ret_lag1 = float(np.log(prices[-1] / prices[-2]))
                    mom7 = float((prices[-1] / prices[-8]) - 1)
                    
                    X = np.array([[gb_volume, vol7, ret_lag1, mom7]])
                    pred_log_ret = float(models[ALGO_GB].predict(X)[0])
                    
                    # Reconstruct price from log return
                    pred = prices[-1] * np.exp(pred_log_ret)
                    st.session_state.pred_results[ALGO_GB] = pred

                    st.success(f"### Predicted Next Closing Price: ${pred:,.2f}")
                    st.caption(
                        f"Model predicts log return ({pred_log_ret:+.6f}), "
                        f"reconstructed from last close (${prices[-1]:,.2f})."
                    )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Volatility_7", f"{vol7:,.2f}")
                    m2.metric("Return_Lag1 (Log)", f"{ret_lag1:.4f}")
                    m3.metric("Momentum_7", f"{mom7:.2%}")

                if gb_do_forecast:
                    compute_and_store_forecast(ALGO_GB, prices, [gb_volume], int(gb_ndays))

    # ---------------- Comparison of filled-in next-day results ----------------
    st.markdown("---")
    st.subheader("Your Next-Day Predictions")
    results = st.session_state.pred_results

    if len(results) == 0:
        st.info("Click \u201cPredict Next Day\u201d on at least one algorithm above to see a result here.")
    else:
        if len(results) == 1:
            algo, pred = next(iter(results.items()))
            st.write(f"**{algo}** \u2192 **${pred:,.2f}**")
        else:
            res_df = pd.DataFrame(list(results.items()),
                                   columns=["Algorithm", "Predicted Price"]).set_index("Algorithm")
            st.dataframe(res_df.style.format("${:,.2f}"), use_container_width=True)
            st.bar_chart(res_df)
            spread = res_df["Predicted Price"].max() - res_df["Predicted Price"].min()
            st.caption(f"Spread across filled-in algorithms: ${spread:,.2f}")

        if st.button("Clear predictions"):
            st.session_state.pred_results = {}
            st.rerun()

    # ---------------- Combined multi-day forecast chart ----------------
    st.markdown("---")
    st.subheader("Your Multi-Day Forecasts")
    forecasts = st.session_state.get("forecast_results", {})

    if len(forecasts) == 0:
        st.info("Click \u201cForecast Ahead\u201d on at least one algorithm above to see its forecast here. "
                 "Forecast multiple algorithms to draw their lines together on the same chart.")
    else:
        forecast_df = pd.concat(forecasts.values(), axis=1)
        forecast_df.index.name = "Date"
        forecast_df = forecast_df.sort_index()

        st.dataframe(forecast_df.style.format("${:,.2f}"), use_container_width=True)
        st.line_chart(forecast_df)
        st.caption(
            "Each line is a recursive day-by-day forecast seeded only from that algorithm's "
            "own entered data -- each day's output feeds the next day's inputs. Future Trading "
            "Volume is held near the average of what you entered, since it can't be known in advance."
        )

        if st.button("Clear forecasts"):
            st.session_state.forecast_results = {}
            st.rerun()