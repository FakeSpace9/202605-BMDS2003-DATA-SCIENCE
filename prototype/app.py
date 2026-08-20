import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ==========================================
# 1. ALGORITHM & MODEL CONFIGURATION
# ==========================================
ALGO_NAME_1 = "Linear Regression (Walk-Forward)"
ALGO_NAME_2 = "KNN Regression (Walk-Forward)"

PKL_PATH_1 = "linear_regression_walkforward_price.pkl"
PKL_PATH_2 = "knn_walkforward_price.pkl"

model_paths = {
    ALGO_NAME_1: PKL_PATH_1,
    ALGO_NAME_2: PKL_PATH_2
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_saved_object(path):
    if not path or not os.path.exists(path):
        return None
    return joblib.load(path)

def calculate_rsi(prices, window=14):
    """Calculates the Relative Strength Index (RSI) using Pandas."""
    df = pd.DataFrame(prices, columns=['price'])
    delta = df['price'].diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1]

# ==========================================
# 3. STREAMLIT APP UI
# ==========================================
def main():
    st.title("Gold Price Prediction App")
    st.write("Enter recent market history to automatically calculate technical indicators and predict the next closing price.")

    st.sidebar.header("Model Selection")
    selected_algo = st.sidebar.selectbox("Choose an Algorithm:", list(model_paths.keys()))

    st.subheader("Input Recent Data")
    
    input_data = None
    calculated_metrics = {}

    # ---------------------------------------------------------
    # UI FOR LINEAR REGRESSION
    # Features: ["Volume", "Month", "Day", "Volatility_7", "MA_7"]
    # ---------------------------------------------------------
    if selected_algo == ALGO_NAME_1:
        col1, col2, col3 = st.columns(3)
        with col1:
            in_volume = st.number_input("Yesterday's Volume", value=150000.0)
        with col2:
            # value=None allows the box to be completely blank
            in_month = st.number_input("Month (1-12) [Blank = Today]", min_value=1, max_value=12, value=None)
        with col3:
            in_day = st.number_input("Day (1-31) [Blank = Today]", min_value=1, max_value=31, value=None)
            
        default_7_days = "2380.5, 2390.0, 2385.2, 2395.1, 2400.0, 2398.5, 2405.0"
        price_input = st.text_area("Enter the last 7 closing prices (comma-separated, oldest to newest):", value=default_7_days)
        
        if st.button("Predict Price"):
            try:
                prices = [float(x.strip()) for x in price_input.split(',')]
                
                if len(prices) != 7:
                    st.error(f"Please enter exactly 7 prices. You entered {len(prices)}.")
                    return
                
                # 1. Background Calculations
                calc_ma7 = np.mean(prices)
                calc_vol7 = np.std(prices, ddof=1) 
                
                # 2. Date Fallback Logic (If blank, use current date)
                final_month = in_month if in_month is not None else datetime.now().month
                final_day = in_day if in_day is not None else datetime.now().day
                
                calculated_metrics = {
                    "Month Used": final_month,
                    "Day Used": final_day,
                    "MA_7": calc_ma7,
                    "Volatility_7": calc_vol7
                }
                
                # 3. Pack strictly ordered array
                input_data = np.array([[in_volume, final_month, final_day, calc_vol7, calc_ma7]])
                
            except ValueError:
                st.error("Invalid input. Please ensure prices are numbers separated by commas.")
                return

    # ---------------------------------------------------------
    # UI FOR KNN REGRESSION
    # Features: ["Volume_Momentum", "Volatility_7", "Volatility_30", "RSI_14", "daily_return_lag1", "daily_return_lag2"]
    # ---------------------------------------------------------
    elif selected_algo == ALGO_NAME_2:
        
        # 1. 30 Days of Prices (for Volatility and RSI)
        default_30_days = ", ".join([str(2300.0 + i) for i in range(30)])
        price_input = st.text_area("Enter the last 30 closing prices (comma-separated, oldest to newest):", value=default_30_days)
        
        # 2. 10 Days of Volume (for Volume_Momentum)
        default_10_vols = ", ".join([str(150000.0 + (i * 1000)) for i in range(10)])
        vol_input = st.text_area("Enter the last 10 volume figures (comma-separated, oldest to newest):", value=default_10_vols)

        if st.button("Predict Price"):
            try:
                prices = [float(x.strip()) for x in price_input.split(',')]
                volumes = [float(x.strip()) for x in vol_input.split(',')]
                
                if len(prices) < 30:
                    st.error(f"Please enter at least 30 prices. You entered {len(prices)}.")
                    return
                if len(volumes) < 10:
                    st.error(f"Please enter at least 10 volume figures. You entered {len(volumes)}.")
                    return
                
                # Take exactly the correct windows
                prices_30 = prices[-30:]
                prices_7 = prices[-7:]
                vols_10 = volumes[-10:]
                
                # --- Background Calculations ---
                
                # Volatility and RSI
                calc_vol30 = np.std(prices_30, ddof=1)
                calc_vol7 = np.std(prices_7, ddof=1)
                calc_rsi = calculate_rsi(prices_30, window=14)
                
                # Daily Returns
                ret_lag1 = (prices_30[-1] - prices_30[-2]) / prices_30[-2]
                ret_lag2 = (prices_30[-2] - prices_30[-3]) / prices_30[-3]
                
                # Volume Momentum (Yesterday's Vol / 10-day Avg Vol)
                calc_vol_mom = vols_10[-1] / np.mean(vols_10)
                
                calculated_metrics = {
                    "Vol Momentum": calc_vol_mom,
                    "Volatility_30": calc_vol30,
                    "Volatility_7": calc_vol7,
                    "RSI_14": calc_rsi,
                    "Return Lag 1 (%)": ret_lag1 * 100,
                    "Return Lag 2 (%)": ret_lag2 * 100
                }
                
                input_data = np.array([[calc_vol_mom, calc_vol7, calc_vol30, calc_rsi, ret_lag1, ret_lag2]])
                
            except ValueError:
                st.error("Invalid input. Please ensure prices and volumes are numbers separated by commas.")
                return

    # ==========================================
    # 4. EXECUTE PREDICTION & SHOW RESULTS
    # ==========================================
    if input_data is not None:
        selected_pkl_path = model_paths[selected_algo]
        model = load_saved_object(selected_pkl_path)
        
        if model:
            st.markdown("---")
            st.write("### Calculated Background Features")
            
            # Dynamically format columns based on how many metrics we calculated
            cols = st.columns(len(calculated_metrics))
            for i, (metric_name, metric_value) in enumerate(calculated_metrics.items()):
                # Format differently if it's Month or Day (no decimals) vs other metrics
                if "Month" in metric_name or "Day" in metric_name:
                    cols[i].metric(label=metric_name, value=f"{int(metric_value)}")
                else:
                    cols[i].metric(label=metric_name, value=f"{metric_value:.4f}")
            
            try:
                prediction = model.predict(input_data)
                st.markdown("---")
                st.success(f"### Predicted Next Closing Price: ${prediction[0]:.2f}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.error(f"Model file not found at: {selected_pkl_path}")

if __name__ == "__main__":
    main()