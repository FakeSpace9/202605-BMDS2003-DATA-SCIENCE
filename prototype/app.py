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

st.set_page_config(layout="wide", page_title="Gold Price Predictor", page_icon="🥇")

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
    if not EDA_DATA_FILE.exists():
        return None
    df = pd.read_csv(EDA_DATA_FILE, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "Gold_Price.csv"

@st.cache_data
def load_raw_eda_data():
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
    if not event:
        return []
    sel = event.get("selection") if isinstance(event, dict) else None
    return sel.get("points", []) if sel else []

PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": [
        "hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines",
    ],
}

models = load_models()
metrics = load_metrics()
eda_df = load_eda_data()
raw_eda_df = load_raw_eda_data()

# ==========================================
# 2. SHARED FEATURE-ENGINEERING HELPERS
# ==========================================
def calculate_rsi(prices, window=14):
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
    while nd.weekday() >= 5:
        nd += pd.Timedelta(days=1)
    return nd

def run_recursive_forecast(algo, model, seed_prices, seed_volumes, anchor_date, n_days):
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
            X = pd.DataFrame([[vol_forecast, month, day, vol7, ma7]], columns=FEATURES[ALGO_LR])
            pred_price = float(model.predict(X)[0])
        elif algo == ALGO_RF:
            price_lag1 = price_hist[-1]
            return_lag1 = np.log(price_hist[-1] / price_hist[-2])
            X = pd.DataFrame([[vol_forecast, month, day, vol7, return_lag1]], columns=FEATURES[ALGO_RF])
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
            X = pd.DataFrame([[vol_mom, vol7, vol30, rsi, ret1, ret2]], columns=FEATURES[ALGO_KNN])
            diff = float(model.predict(X)[0])
            pred_price = price_lag1 + diff
        elif algo == ALGO_GB:
            prices_7 = price_hist[-7:]
            vol7 = float(np.std(prices_7, ddof=1)) if len(prices_7) > 1 else 0.0
            ret_lag1 = np.log(price_hist[-1] / price_hist[-2])
            mom7 = (price_hist[-1] / price_hist[-8]) - 1 if len(price_hist) >= 8 else 0.0
            X = pd.DataFrame([[vol_forecast, vol7, ret_lag1, mom7]], columns=FEATURES[ALGO_GB])
            pred_log_ret = float(model.predict(X)[0])
            pred_price = price_hist[-1] * np.exp(pred_log_ret)
        else:
            continue

        dates.append(current_date)
        preds.append(pred_price)
        price_hist = (price_hist + [pred_price])[-30:]
        vol_hist = (vol_hist + [vol_forecast])[-10:]
    return dates, preds

def compute_and_store_forecast(algo, seed_prices, seed_volumes, anchor_date, n_days):
    model = models[algo]
    if model is None:
        st.error(f"Model file not found: {MODEL_FILES[algo]}")
        return
    dates, preds = run_recursive_forecast(algo, model, seed_prices, seed_volumes, anchor_date, n_days)
    forecast_series = pd.Series(preds, index=pd.DatetimeIndex(dates), name=algo)
    if "forecast_results" not in st.session_state:
        st.session_state.forecast_results = {}
    st.session_state.forecast_results[algo] = forecast_series
    st.success(f"{int(n_days)}-day forecast added successfully!")

# ==========================================
# 3. PAGE HEADER
# ==========================================
st.title("🥇 Gold Price Prediction & Analytics Dashboard")
st.caption("BMDS2003 Data Science Group Project -- CRISP-DM prototype")

tab_insights, tab_compare, tab_predict = st.tabs(
    ["📈 Market Insights", "⚖️ Model Comparison", "🔮 Predict & Forecast"]
)

# ==========================================
# TAB 1: MARKET INSIGHTS
# ==========================================
with tab_insights:
    st.subheader("Key Exploratory Data Analysis Plots")
    st.write("5 interactive charts built live from the same processed data the models below are trained on. **Scroll your mouse wheel to zoom in/out.** Click and drag to pan. Click the 'Home' icon in the top right to reset the view.")
    if eda_df is None:
        st.error(f"Processed dataset not found: {EDA_DATA_FILE}")
    else:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**Gold Price Over Time**")
            fig1 = go.Figure(go.Scatter(
                x=eda_df["Date"], y=eda_df["Price"], mode="lines",
                line=dict(color="#1f77b4", width=2, shape="spline"),
                customdata=np.stack([eda_df["MA_7"], eda_df["MA_30"], eda_df["Volatility_7"]], axis=-1),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Price: %{y:,.0f}<br>"
                    "MA_7: %{customdata[0]:,.0f}<br>MA_30: %{customdata[1]:,.0f}<extra></extra>"
                ),
            ))
            fig1.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis_title="Date", yaxis_title="Gold Price",
                                title=dict(text="Gold Price Over Time", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            event1 = st.plotly_chart(fig1, use_container_width=True, on_select="rerun", key="plot_price_time", config=PLOTLY_CONFIG)
            st.caption("The fundamental long-term macro trend of gold prices that motivates the forecasting problem.")
            
        with col2:
            st.markdown("**Positive and Negative Days by Month**")
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
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
            fig2.add_bar(x=month_names, y=monthly_direction["Positive"], name="Positive", marker_color="#1f77b4", hovertemplate="Month: %{x}<br>Positive days: %{y}<extra></extra>")
            fig2.add_bar(x=month_names, y=monthly_direction["Negative"], name="Negative", marker_color="#ff7f0e", hovertemplate="Month: %{x}<br>Negative days: %{y}<extra></extra>")
            fig2.update_layout(barmode="stack", height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Month", yaxis_title="Number of Days", legend_title="Daily Movement", title=dict(text="Positive and Negative Gold Price Days by Month", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            event2 = st.plotly_chart(fig2, use_container_width=True, on_select="rerun", key="plot_monthly_direction",config=PLOTLY_CONFIG)
            st.caption("Explores whether prices historically increase or decrease more often in specific months.")

        st.markdown("<br>", unsafe_allow_html=True)
        col3, col4 = st.columns(2, gap="large")
        
        with col3:
            st.markdown("**7-Day Volatility Distribution**")
            vol = raw_eda_df["Volatility_7"].dropna()
            counts, bin_edges = np.histogram(vol, bins=30)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            fig3 = go.Figure(go.Scatter(
                x=bin_centers, y=counts, mode="lines", fill="tozeroy",
                line=dict(color="purple", width=2, shape="spline"),
                opacity=0.7, customdata=np.stack([bin_edges[:-1], bin_edges[1:]], axis=-1),
                hovertemplate="Range: %{customdata[0]:.2f}–%{customdata[1]:.2f}<br>Count: %{y}<extra></extra>",
            ))
            fig3.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="7-Day Volatility", yaxis_title="Number of Days", title=dict(text="Distribution of 7-Day Gold Price Volatility", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            event3 = st.plotly_chart(fig3, use_container_width=True, on_select="rerun", key="plot_volatility_dist",config=PLOTLY_CONFIG)
            st.caption("Shows the spread of short-term rolling volatility—a key feature (Volatility_7).")

        with col4:
            st.markdown("**Daily Return Distribution**")
            ret = raw_eda_df["Daily_Return"].dropna()
            counts, bin_edges = np.histogram(ret, bins=40)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            fig4 = go.Figure(go.Scatter(
                x=bin_centers, y=counts, mode="lines", fill="tozeroy",
                line=dict(color="#2ca02c", width=2, shape="spline"),
                opacity=0.7, customdata=np.stack([bin_edges[:-1], bin_edges[1:]], axis=-1),
                hovertemplate="Range: %{customdata[0]:.2f}–%{customdata[1]:.2f}%<br>Count: %{y}<extra></extra>",
            ))
            fig4.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Daily Return (%)", yaxis_title="Frequency", title=dict(text="Distribution of Daily Gold Returns", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            event4 = st.plotly_chart(fig4, use_container_width=True, on_select="rerun", key="plot_return_dist",config=PLOTLY_CONFIG)
            st.caption("Visualizes the spread of daily returns.")

        st.markdown("<br>", unsafe_allow_html=True)
        col5, col6 = st.columns(2, gap="large")
        
        with col5:
            st.markdown("**Feature Correlation Matrix**")
            corr_cols = ["Price", "Open", "High", "Low", "Volume", "Chg%", "Price_Change", "Daily_Return", "Volatility_7", "Volatility_30", "Absolute_Price_Change", "Year", "Quarter"]
            corr_cols = [c for c in corr_cols if c in raw_eda_df.columns]
            corr = raw_eda_df[corr_cols].corr()
            fig5 = go.Figure(go.Heatmap(
                z=corr.values, x=list(corr.columns), y=list(corr.columns),
                colorscale="RdBu_r", zmin=float(corr.values.min()), zmax=1,
                colorbar=dict(title="Correlation"), hovertemplate="%{x} vs %{y}<br>Correlation %{z:.2f}<extra></extra>",
            ))
            fig5.update_layout(height=650, margin=dict(l=5, r=5, t=40, b=5), title=dict(text="Correlation Matrix of Numeric Features", x=0.5, xanchor="center", y=0.98, yanchor="top"))
            fig5.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
            fig5.update_xaxes(constrain="domain")
            event5 = st.plotly_chart(fig5, use_container_width=True, on_select="rerun", key="plot_corr_matrix",config=PLOTLY_CONFIG)
            st.caption("Displays the mathematical relationships between numeric variables.")

# ==========================================
# TAB 2: MODEL COMPARISON
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
            c1.metric("Highest Test R²", f"{comp_df.loc[best_r2, 'test_R2']:.4f}")
            c1.caption(f"Model: {best_r2}")
        if "test_MAE" in comp_df:
            best_mae = comp_df["test_MAE"].idxmin()
            c2.metric("Lowest Test MAE", f"{comp_df.loc[best_mae, 'test_MAE']:,.2f}")
            c2.caption(f"Model: {best_mae}")

        st.markdown("**Test R² by algorithm**")
        if "test_R2" in comp_df:
            st.bar_chart(comp_df[["test_R2"]])
        st.markdown("**Test MAE / RMSE by algorithm**")
        err_cols = [c for c in ["test_MAE", "test_RMSE"] if c in comp_df.columns]
        if err_cols:
            st.bar_chart(comp_df[err_cols])

# ==========================================
# TAB 3: PREDICT & FORECAST (Restructured)
# ==========================================
with tab_predict:
    # Initialize session state for storing next-day predictions
    if "pred_results" not in st.session_state:
        st.session_state.pred_results = {}

    st.write("First, choose the algorithm you want to use. The data input form below will automatically disable the fields that the chosen model doesn't need.")

    # 1. Selection
    st.markdown("### 1. Select Algorithm")
    selected_algo = st.selectbox("Choose the model to use for prediction:", list(MODEL_FILES.keys()), label_visibility="collapsed")

    # Flag booleans for UI
    en_lr_rf = selected_algo in [ALGO_LR, ALGO_RF]
    en_gb = selected_algo == ALGO_GB
    en_knn = selected_algo == ALGO_KNN
    en_vol = selected_algo in [ALGO_LR, ALGO_RF, ALGO_GB]

    # 2. Input Frame
    st.markdown("### 2. Enter Data")
    st.caption(f"Features mathematically used by **{selected_algo}**: {', '.join(FEATURES[selected_algo])}")
    
    with st.container(border=True):
        st.markdown("#### Timeline & Base Inputs")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            help_vol = "Used by LR, RF, GB." if en_vol else "This algorithm does not use single-day volume."
            vol_val = st.number_input("Yesterday's Trading Volume", value=51877.0, disabled=not en_vol, help=help_vol)
        with c2:
            year_val = st.number_input("Year [blank=today]", min_value=2000, max_value=2100, value=None, help="Sets the starting year for the forecast timeline.")
        with c3:
            month_val = st.number_input("Month (1-12) [blank=today]", min_value=1, max_value=12, value=None, help="Sets the starting month. Also used as a mathematical feature by LR and RF.")
        with c4:
            day_val = st.number_input("Day (1-31) [blank=today]", min_value=1, max_value=31, value=None, help="Sets the starting day. Also used as a mathematical feature by LR and RF.")

        st.markdown("#### Historical Prices & Volumes")
        # Render the text areas, graying out those not associated with the selected algorithm
        prices_7_val = st.text_area(
            "Last 7 closing prices (Linear Regression & Random Forest)", 
            value="136104, 137789, 132595, 133974, 135454, 135771, 135793", 
            disabled=not en_lr_rf, help="Required by LR and RF to compute Volatility_7, Moving Averages, and Returns."
        )
        
        prices_8_val = st.text_area(
            "Last 8 closing prices (Gradient Boosting)", 
            value="134000, 136104, 137789, 132595, 133974, 135454, 135771, 135793", 
            disabled=not en_gb, help="Required by GB to compute 7-day Momentum and Volatility."
        )
        
        prices_30_val = st.text_area(
            "Last 30 closing prices (KNN)", 
            value=", ".join(str(128000.0 + (i * 250)) for i in range(30)), 
            disabled=not en_knn, help="Required by KNN to calculate RSI_14, Volatility_30, and multiple lags."
        )
        
        vols_10_val = st.text_area(
            "Last 10 trading volumes (KNN)", 
            value=", ".join(str(45000.0 + (i * 500)) for i in range(10)), 
            disabled=not en_knn, help="Required by KNN to calculate Volume Momentum."
        )

        st.markdown("#### Actions")
        bc1, bc2, bc3 = st.columns([1.5, 1, 1.5])
        with bc1:
            btn_predict = st.button("Predict Next Day", type="primary", use_container_width=True)
        with bc2:
            forecast_days = st.number_input("Days ahead to forecast", min_value=1, max_value=60, value=10, label_visibility="collapsed")
        with bc3:
            btn_forecast = st.button("Forecast Ahead", use_container_width=True)

    # 3. Process Button Clicks (Calculate and show detailed metrics for the *current run*)
    if btn_predict or btn_forecast:
        # Safely parse the exact date from user input
        now = datetime.now()
        start_year = int(year_val) if year_val is not None else now.year
        start_month = int(month_val) if month_val is not None else now.month
        start_day = int(day_val) if day_val is not None else now.day
        
        try:
            anchor_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
        except ValueError:
            st.error("Invalid date provided (e.g. February 31st). Defaulting to today's date.")
            anchor_date = pd.Timestamp(now.date())
            start_month, start_day = now.month, now.day

        # Extract correct data inputs based on selection
        prices, vols = None, None
        errs = []

        if selected_algo in [ALGO_LR, ALGO_RF]:
            prices, err = parse_number_list(prices_7_val, 7, "Last 7 closing prices")
            if err: errs.append(err)
            vols = [vol_val]
        elif selected_algo == ALGO_GB:
            prices, err = parse_number_list(prices_8_val, 8, "Last 8 closing prices")
            if err: errs.append(err)
            vols = [vol_val]
        elif selected_algo == ALGO_KNN:
            prices, err1 = parse_number_list(prices_30_val, 30, "Last 30 closing prices")
            vols_list, err2 = parse_number_list(vols_10_val, 10, "Last 10 trading volumes")
            if err1: errs.append(err1)
            if err2: errs.append(err2)
            vols = vols_list

        # Execute if no validation errors
        if errs:
            for e in errs: st.error(e)
        elif models[selected_algo] is None:
            st.error(f"Model file not found: {MODEL_FILES[selected_algo]}")
        else:
            # SINGLE DAY PREDICTION (Always calculate and show if either button is clicked)
            target_date = next_business_day(anchor_date).strftime('%A, %b %d, %Y')
            st.markdown(f"### 3. Detailed Results: **{selected_algo}**")
            st.markdown(f"**Target Date:** {target_date}")
            
            if selected_algo == ALGO_LR:
                ma7 = float(np.mean(prices))
                vol7 = float(np.std(prices, ddof=1))
                X = pd.DataFrame([[vol_val, start_month, start_day, vol7, ma7]], columns=FEATURES[ALGO_LR])
                pred = float(models[ALGO_LR].predict(X)[0])
                st.session_state.pred_results[ALGO_LR] = pred # Save to state for comparison
                
                st.success(f"### Predicted Price: **${pred:,.2f}**")
                m1, m2, m3 = st.columns(3)
                m1.metric("MA_7", f"{ma7:,.2f}")
                m2.metric("Volatility_7", f"{vol7:,.2f}")
                m3.metric("Month / Day used", f"{start_month} / {start_day}")

            elif selected_algo == ALGO_RF:
                vol7 = float(np.std(prices, ddof=1))
                price_lag1 = prices[-1]
                return_lag1 = np.log(prices[-1] / prices[-2])
                X = pd.DataFrame([[vol_val, start_month, start_day, vol7, return_lag1]], columns=FEATURES[ALGO_RF])
                change = float(models[ALGO_RF].predict(X)[0])
                pred = price_lag1 * np.exp(change)
                st.session_state.pred_results[ALGO_RF] = pred
                
                st.success(f"### Predicted Price: **${pred:,.2f}**")
                st.caption(f"Model predicts a log return of {change:+.6f}, derived from the last known close of ${price_lag1:,.2f}.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Volatility_7", f"{vol7:,.2f}")
                m2.metric("Return_Lag1", f"{return_lag1:+.6f}")
                m3.metric("Month / Day used", f"{start_month} / {start_day}")

            elif selected_algo == ALGO_KNN:
                prices_30 = prices[-30:]
                vols_10 = vols[-10:]
                vol30 = float(np.std(prices_30, ddof=1))
                vol7 = float(np.std(prices_30[-7:], ddof=1))
                rsi = calculate_rsi(prices_30, window=14)
                ret1 = (prices_30[-1] - prices_30[-2]) / prices_30[-2]
                ret2 = (prices_30[-2] - prices_30[-3]) / prices_30[-3]
                vol_mom = vols_10[-1] / np.mean(vols_10)
                price_lag1 = prices_30[-1]
                X = pd.DataFrame([[vol_mom, vol7, vol30, rsi, ret1, ret2]], columns=FEATURES[ALGO_KNN])
                diff = float(models[ALGO_KNN].predict(X)[0])
                pred = price_lag1 + diff
                st.session_state.pred_results[ALGO_KNN] = pred
                
                st.success(f"### Predicted Price: **${pred:,.2f}**")
                st.caption(f"Model predicts the day-over-day price change (${diff:,.2f}), added to the last known close of ${price_lag1:,.2f}.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Volatility_7 / _30", f"{vol7:,.1f} / {vol30:,.1f}")
                m2.metric("RSI_14", f"{rsi:.2f}")
                m3.metric("Volume Momentum", f"{vol_mom:.3f}")

            elif selected_algo == ALGO_GB:
                vol7 = float(np.std(prices[-7:], ddof=1))
                ret_lag1 = float(np.log(prices[-1] / prices[-2]))
                mom7 = float((prices[-1] / prices[-8]) - 1)
                X = pd.DataFrame([[vol_val, vol7, ret_lag1, mom7]], columns=FEATURES[ALGO_GB])
                pred_log_ret = float(models[ALGO_GB].predict(X)[0])
                pred = prices[-1] * np.exp(pred_log_ret)
                st.session_state.pred_results[ALGO_GB] = pred
                
                st.success(f"### Predicted Price: **${pred:,.2f}**")
                st.caption(f"Model predicts log return ({pred_log_ret:+.6f}), reconstructed from last close (${prices[-1]:,.2f}).")
                m1, m2, m3 = st.columns(3)
                m1.metric("Volatility_7", f"{vol7:,.2f}")
                m2.metric("Return_Lag1 (Log)", f"{ret_lag1:.4f}")
                m3.metric("Momentum_7", f"{mom7:.2%}")

            # FORECAST AHEAD
            if btn_forecast:
                compute_and_store_forecast(selected_algo, prices, vols, anchor_date, int(forecast_days))

    # 4. Global Display Check (This runs every time to persist the comparison table & graphs)
    
    # 4A. Next-Day Predictions Comparison Table
    if st.session_state.pred_results:
        st.markdown("---")
        if len(st.session_state.pred_results) == 1:
            st.info("Run predictions for additional algorithms above to unlock the comparison table.")
        else:
            st.markdown("#### Next-Day Predictions Comparison")
            res_df = pd.DataFrame(list(st.session_state.pred_results.items()), columns=["Algorithm", "Predicted Price"]).set_index("Algorithm")
            st.dataframe(res_df.style.format("${:,.2f}"), use_container_width=True)
            st.bar_chart(res_df)
            spread = res_df["Predicted Price"].max() - res_df["Predicted Price"].min()
            st.caption(f"Spread across filled-in algorithms: ${spread:,.2f}")

            if st.button("Clear Single-Day Predictions", key="clear_single"):
                st.session_state.pred_results = {}
                st.rerun()

    # 4B. Combined Multi-Day Forecast Chart
    forecasts = st.session_state.get("forecast_results", {})
    if forecasts:
        st.markdown("---")
        st.markdown("#### Combined Multi-Day Forecast Chart")
        
        forecast_df = pd.concat(forecasts.values(), axis=1)
        forecast_df.index.name = "Date"
        forecast_df = forecast_df.sort_index()

        st.dataframe(forecast_df.style.format("${:,.2f}"), use_container_width=True)
        st.line_chart(forecast_df)
        st.caption(
            "Each line is a recursive day-by-day forecast seeded only from that algorithm's "
            "own entered data. Future Trading Volume is held near the average of what you entered."
        )

        if st.button("Clear Multi-Day Forecasts", key="clear_forecast"):
            st.session_state.forecast_results = {}
            st.rerun()

    # Empty State Hint
    if not st.session_state.pred_results and not forecasts and not btn_predict and not btn_forecast:
        st.info("Click **Predict Next Day** or **Forecast Ahead** to see results generated here.")