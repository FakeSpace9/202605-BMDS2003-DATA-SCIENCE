import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Set page layout to wide for a better dashboard look
st.set_page_config(layout="wide", page_title="Gold Price Predictor")

# --- PATH DEFINITIONS ---
current_dir = Path(__file__).resolve().parent

SCALER_FILENAME = "scaler.pkl"
DATA_FILENAME = "Gold_Price.csv"

# name shown in the UI -> pkl filename
MODEL_FILES = {
    "Linear Regression (Baseline)": "linear_regression_price_r0.5.pkl",
}

FEATURE_COLUMNS = ["Open", "High", "Low", "Volume"]


# --- ASSET LOADING ---
@st.cache_resource
def load_ml_assets():
    """Load the scaler and every model, failing loudly (but cleanly) if any file is missing."""
    missing = [
        f for f in [SCALER_FILENAME, *MODEL_FILES.values()]
        if not (current_dir / f).exists()
    ]
    if missing:
        st.error(
            "Missing model file(s): " + ", ".join(missing) +
            f". Make sure they sit next to app.py at `{current_dir}`."
        )
        st.stop()

    scaler = joblib.load(current_dir / SCALER_FILENAME)
    models = {name: joblib.load(current_dir / fname) for name, fname in MODEL_FILES.items()}
    return scaler, models


@st.cache_data
def load_data():
    """Load and sort the historical gold price data."""
    data_path = current_dir / DATA_FILENAME
    if not data_path.exists():
        st.error(f"Data file not found: `{DATA_FILENAME}` (expected at `{current_dir}`).")
        st.stop()
    data = pd.read_csv(data_path)
    data["Date"] = pd.to_datetime(data["Date"])
    return data.sort_values("Date")


scaler, models = load_ml_assets()
df = load_data()

# --- APP UI ---
st.title("🥇 Daily Gold Price Prediction & Analytics Dashboard")
st.write(
    "This application predicts the daily closing price of gold based on "
    "intra-day trading metrics and provides interactive historical market analysis."
)

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("1. Select Machine Learning Model")
selected_model_name = st.sidebar.selectbox("Active Prediction Model:", list(models.keys()))
active_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.header("2. Input Market Features")


def user_input_features() -> pd.DataFrame:
    open_price = st.sidebar.number_input("Open Price", min_value=20000.0, max_value=150000.0, value=136143.0)
    high_price = st.sidebar.number_input("High Price", min_value=20000.0, max_value=150000.0, value=137037.0)
    low_price = st.sidebar.number_input("Low Price", min_value=20000.0, max_value=150000.0, value=135525.0)
    volume = st.sidebar.number_input("Trading Volume", min_value=0.0, max_value=150000.0, value=51877.0)

    data = {
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Volume": volume,
    }
    # Enforce the exact column order the scaler/model were fit on
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


input_df = user_input_features()

# --- MAIN PAGE LAYOUT ---
col1, col2 = st.columns([1, 2])

# LEFT COLUMN: Prediction Engine
with col1:
    st.subheader(f"🔮 Price Prediction ({selected_model_name})")
    st.write("Current Input Parameters:")
    st.dataframe(input_df, use_container_width=True)

    # PREDICTION LOGIC
    try:
        scaled_input = scaler.transform(input_df)
        main_prediction = active_model.predict(scaled_input)[0]
        st.success(f"### Estimated Close Price: ₹{main_prediction:,.2f} (INR)")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        scaled_input = None

    st.markdown("---")

    # --- MODEL COMPARISON FEATURE (TABLE ONLY) ---
    st.subheader("⚖️ Model Comparison")
    if len(models) < 2:
        st.caption("Only one model is currently registered — add more to `MODEL_FILES` to compare.")
    else:
        compare_toggle = st.checkbox("Compare all models")
        if compare_toggle and scaled_input is not None:
            comparison_data = {name: mod.predict(scaled_input)[0] for name, mod in models.items()}
            comp_df = pd.DataFrame(
                list(comparison_data.items()), columns=["Model", "Predicted Price (INR)"]
            ).set_index("Model")
            st.dataframe(comp_df.style.format("₹{:,.2f}"), use_container_width=True)

# RIGHT COLUMN: Interactive Historical Chart
with col2:
    st.subheader("📈 Interactive Historical Trend")
    st.write("Filter the historical chart by typing exact dates or using the drag bar.")

    # --- ADVANCED INTERACTIVE DATE FILTERING ---
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    default_start = max(min_date, pd.to_datetime("2014-01-01").date())

    if "start_input" not in st.session_state:
        st.session_state.start_input = default_start
        st.session_state.end_input = max_date
        st.session_state.slider_dates = (default_start, max_date)

    def sync_from_inputs():
        start = st.session_state.start_input
        end = st.session_state.end_input

        if start is None or end is None:
            return

        if start > end:
            st.toast(
                "⚠️ Error: Start Date cannot be after End Date. Dates automatically adjusted.",
                icon="❌",
            )
            st.session_state.end_input = start
            st.session_state.slider_dates = (start, start)
        else:
            st.session_state.slider_dates = (start, end)

    def sync_from_slider():
        st.session_state.start_input = st.session_state.slider_dates[0]
        st.session_state.end_input = st.session_state.slider_dates[1]

    col2_a, col2_b = st.columns(2)
    with col2_a:
        st.date_input(
            "Type Start Date:",
            key="start_input",
            min_value=min_date,
            max_value=max_date,
            on_change=sync_from_inputs,
        )
    with col2_b:
        st.date_input(
            "Type End Date:",
            key="end_input",
            min_value=min_date,
            max_value=max_date,
            on_change=sync_from_inputs,
        )

    st.slider(
        "Or drag to select date range:",
        min_value=min_date,
        max_value=max_date,
        key="slider_dates",
        format="YYYY-MM-DD",
        on_change=sync_from_slider,
    )

    final_start = st.session_state.slider_dates[0]
    final_end = st.session_state.slider_dates[1]
    filtered_df = df[(df["Date"].dt.date >= final_start) & (df["Date"].dt.date <= final_end)]

    chart_data = filtered_df.set_index("Date")[["Price"]]
    st.line_chart(chart_data)