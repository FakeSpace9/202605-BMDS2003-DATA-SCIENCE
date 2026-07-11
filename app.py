import streamlit as st
import pandas as pd
import joblib

# Set page layout to wide for a better dashboard look
st.set_page_config(layout="wide", page_title="Gold Price Predictor")

# 1. Load the trained models and scaler safely
@st.cache_resource
def load_ml_assets():
    scaler = joblib.load('scaler.pkl')
    models = {
        "Linear Regression (Baseline)": joblib.load('model_lr.pkl'),
        "Decision Tree": joblib.load('model_dt.pkl'),
        "Random Forest": joblib.load('model_rf.pkl'),
        "Gradient Boosting": joblib.load('model_gb.pkl')
    }
    return scaler, models

scaler, models = load_ml_assets()

# Cache the data loading so the app runs faster
@st.cache_data
def load_data():
    df = pd.read_csv('Gold Price_MYR.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

df = load_data()

# App title and description
st.title("🥇 Daily Gold Price Prediction & Analytics Dashboard (MYR)")
st.write("This application predicts the daily closing price of gold based on intra-day trading metrics and provides interactive historical market analysis.")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("1. Select Machine Learning Model")
# Let the user choose the model
selected_model_name = st.sidebar.selectbox("Active Prediction Model:", list(models.keys()))
active_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.header("2. Input Market Features")

def user_input_features():
    # UPDATED: Scaled down for MYR
    Open_price = st.sidebar.number_input('Open Price (MYR)', min_value=1000.0, max_value=15000.0, value=7215.0)
    High_price = st.sidebar.number_input('High Price (MYR)', min_value=1000.0, max_value=15000.0, value=7262.0)
    Low_price = st.sidebar.number_input('Low Price (MYR)', min_value=1000.0, max_value=15000.0, value=7182.0)
    Volume = st.sidebar.number_input('Trading Volume', min_value=0.0, max_value=150000.0, value=51877.0)

    data = {
        'Open': Open_price,
        'High': High_price,
        'Low': Low_price,
        'Volume': Volume
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- MAIN PAGE LAYOUT ---
col1, col2 = st.columns([1, 2])

# LEFT COLUMN: Prediction Engine
with col1:
    st.subheader(f'🔮 Price Prediction ({selected_model_name})')
    st.write("Current Input Parameters:")
    
    # Format the dataframe display
    st.dataframe(input_df, use_container_width=True)
    
    # PREDICTION LOGIC
    scaled_input = scaler.transform(input_df)
    main_prediction = active_model.predict(scaled_input)[0]
    
    # UPDATED: Formatted for RM
    st.success(f"### Estimated Close Price: RM {main_prediction:,.2f}")
        
    st.markdown("---")
    
    # --- MODEL COMPARISON FEATURE (TABLE ONLY) ---
    st.subheader("⚖️ Model Comparison")
    compare_toggle = st.toggle("Compare all 4 algorithms")
    
    if compare_toggle:
        # Calculate predictions for all models
        comparison_data = {}
        for name, mod in models.items():
            comparison_data[name] = mod.predict(scaled_input)[0]
            
        # Create a dataframe for the table
        comp_df = pd.DataFrame(list(comparison_data.items()), columns=['Model', 'Predicted Price (MYR)'])
        comp_df.set_index('Model', inplace=True)
        
        # Display ONLY the beautifully formatted raw numbers
        st.dataframe(comp_df.style.format("RM {:,.2f}"), use_container_width=True)

# RIGHT COLUMN: Interactive Historical Chart
with col2:
    st.subheader('📈 Interactive Historical Trend (MYR)')
    st.write("Filter the historical chart by typing exact dates or using the drag bar.")
    
    # --- ADVANCED INTERACTIVE DATE FILTERING ---
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    if 'start_input' not in st.session_state:
        st.session_state.start_input = pd.to_datetime('2014-01-01').date()
        st.session_state.end_input = max_date
        st.session_state.slider_dates = (pd.to_datetime('2014-01-01').date(), max_date)

    def sync_from_inputs():
        start = st.session_state.start_input
        end = st.session_state.end_input
        
        if start is None or end is None:
            return
            
        if start > end:
            st.toast("⚠️ Error: Start Date cannot be after End Date. Dates automatically adjusted.", icon="❌")
            st.session_state.end_input = start
            st.session_state.slider_dates = (start, start)
        else:
            st.session_state.slider_dates = (start, end)

    def sync_from_slider():
        st.session_state.start_input = st.session_state.slider_dates[0]
        st.session_state.end_input = st.session_state.slider_dates[1]
        
    col2_a, col2_b = st.columns(2)
    with col2_a:
        st.date_input("Type Start Date:", key="start_input", min_value=min_date, max_value=max_date, on_change=sync_from_inputs)
    with col2_b:
        st.date_input("Type End Date:", key="end_input", min_value=min_date, max_value=max_date, on_change=sync_from_inputs)

    st.slider(
        "Or drag to select date range:",
        min_value=min_date,
        max_value=max_date,
        key="slider_dates",
        format="YYYY-MM-DD",
        on_change=sync_from_slider
    )
    
    final_start = st.session_state.slider_dates[0]
    final_end = st.session_state.slider_dates[1]
    filtered_df = df[(df['Date'].dt.date >= final_start) & (df['Date'].dt.date <= final_end)]
    
    chart_data = filtered_df.set_index('Date')[['Price']]
    st.line_chart(chart_data)