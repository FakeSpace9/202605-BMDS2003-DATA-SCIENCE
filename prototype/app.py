"""
Gold Price Prediction App -- Streamlit prototype.

Sections:
    1. Market Insights        -- 5 selected EDA plots from report_assets/plots
    2. Model Comparison        -- walk-forward metrics for all 3 trained algorithms
    3. Predict & Forecast      -- static, per-algorithm manual input forms
                                  (no dynamic single dropdown). Each algorithm
                                  section lets you:
                                    a) Predict just the next day, and/or
                                    b) Forecast N days ahead
                                  Both use ONLY the numbers you typed into that
                                  section -- nothing is pulled from a hidden
                                  historical CSV.
"""

import warnings
from datetime import datetime
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ==========================================
# 0. PATHS & CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent          # .../prototype
PROJECT_ROOT = BASE_DIR.parent                        # repo root
METRICS_DIR = BASE_DIR / "metrics"
PLOTS_DIR = PROJECT_ROOT / "report_assets" / "plots"

st.set_page_config(layout="wide", page_title="Gold Price Predictor", page_icon="\U0001F947")

ALGO_LR = "Linear Regression (Walk-Forward)"
ALGO_KNN = "KNN Regression (Walk-Forward)"
ALGO_RF = "Random Forest (Walk-Forward)"

MODEL_FILES = {
    ALGO_LR: "linear_regression_walkforward_price.pkl",
    ALGO_KNN: "knn_walkforward_price.pkl",
    ALGO_RF: "random_forest_price.pkl",
}

METRIC_FILES = {
    ALGO_LR: "walkforward_price_summary_metrics.json",
    ALGO_KNN: "walkforward_price_knn_summary_metrics.json",
    ALGO_RF: "random_forest_summary_metrics.json",
}

FEATURES = {
    ALGO_LR: ["Volume", "Month", "Day", "Volatility_7", "MA_7"],
    ALGO_KNN: ["Volume_Momentum", "Volatility_7", "Volatility_30", "RSI_14",
               "daily_return_lag1", "daily_return_lag2"],
    ALGO_RF: ["Volume", "Month", "Day", "Volatility_7"],
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
        path = BASE_DIR / fname
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


models = load_models()
metrics = load_metrics()


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
            X = np.array([[vol_forecast, month, day, vol7]])
            change = float(model.predict(X)[0])
            pred_price = price_lag1 + change

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
    st.write("5 plots selected from the EDA report that explain the signals the models below are built on.")
    cols = st.columns(2)
    for i, (fname, title, caption) in enumerate(PLOT_SELECTION):
        path = PLOTS_DIR / fname
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            if path.exists():
                st.image(str(path), use_container_width=True, caption=caption)
            else:
                st.warning(f"Plot not found: {path}")

# ==========================================
# TAB 2: MODEL COMPARISON (3 algorithms)
# ==========================================
with tab_compare:
    st.subheader("Walk-Forward Validation Metrics -- All 3 Algorithms")

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
                                         value=150000.0, key="lr_volume")
        with c2:
            lr_month = st.number_input("Month (1-12) [blank = today]", min_value=1, max_value=12,
                                        value=None, key="lr_month")
        with c3:
            lr_day = st.number_input("Day (1-31) [blank = today]", min_value=1, max_value=31,
                                      value=None, key="lr_day")
        lr_prices_raw = st.text_area(
            "Last 7 closing prices (comma-separated, oldest \u2192 newest)",
            value="2380.5, 2390.0, 2385.2, 2395.1, 2400.0, 2398.5, 2405.0", key="lr_prices")

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
            value=", ".join(str(2300.0 + i) for i in range(30)), key="knn_prices")
        knn_vol_raw = st.text_area(
            "Last 10 trading volumes (comma-separated, oldest \u2192 newest)",
            value=", ".join(str(150000.0 + i * 1000) for i in range(10)), key="knn_vols")

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
                                         value=150000.0, key="rf_volume")
        with c2:
            rf_month = st.number_input("Month (1-12) [blank = today]", min_value=1, max_value=12,
                                        value=None, key="rf_month")
        with c3:
            rf_day = st.number_input("Day (1-31) [blank = today]", min_value=1, max_value=31,
                                      value=None, key="rf_day")
        rf_prices_raw = st.text_area(
            "Last 7 closing prices (comma-separated, oldest \u2192 newest; last value = yesterday's close)",
            value="2380.5, 2390.0, 2385.2, 2395.1, 2400.0, 2398.5, 2405.0", key="rf_prices")

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
                    X = np.array([[rf_volume, final_month, final_day, vol7]])
                    change = float(models[ALGO_RF].predict(X)[0])
                    pred = price_lag1 + change
                    st.session_state.pred_results[ALGO_RF] = pred

                    st.success(f"### Predicted Next Closing Price: ${pred:,.2f}")
                    st.caption(
                        f"Model predicts a small daily change ({change:+.4f}), "
                        f"added to the last known close of ${price_lag1:,.2f}."
                    )
                    m1, m2 = st.columns(2)
                    m1.metric("Volatility_7", f"{vol7:,.2f}")
                    m2.metric("Month / Day used", f"{final_month} / {final_day}")

                if rf_do_forecast:
                    compute_and_store_forecast(ALGO_RF, prices, [rf_volume], int(rf_ndays))

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