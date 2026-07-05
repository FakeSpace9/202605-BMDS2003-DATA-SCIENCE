import streamlit as st
import pandas as pd
import joblib

# Set page layout to wide for a better dashboard look
st.set_page_config(layout="wide", page_title="Gold Price Predictor")

# Load the trained model and scaler
model = joblib.load('gold_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Cache the data loading so the app runs faster
@st.cache_data
def load_data():
    df = pd.read_csv('Gold Price.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

df = load_data()

# App title and description
st.title("🥇 Daily Gold Price Prediction & Analytics Dashboard")
st.write("This application predicts the daily closing price of gold based on intra-day trading metrics and provides interactive historical market analysis.")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Input Market Features")

def user_input_features():
    Open_price = st.sidebar.number_input('Open Price', min_value=20000.0, max_value=150000.0, value=136143.0)
    High_price = st.sidebar.number_input('High Price', min_value=20000.0, max_value=150000.0, value=137037.0)
    Low_price = st.sidebar.number_input('Low Price', min_value=20000.0, max_value=150000.0, value=135525.0)
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
    st.subheader('🔮 Price Prediction')
    st.write("Current Input Parameters:")
    
    # Format the dataframe display
    st.dataframe(input_df, use_container_width=True)
    
    if st.button('Predict Closing Price', type="primary"):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        
        # Display Prediction with Currency formatting
        st.success(f"### Estimated Close Price: ${prediction[0]:,.2f}")
        
    st.markdown("---")
    st.subheader('📊 Market Extremes (All-Time)')
    
    # Display Metrics with Currency formatting
    st.metric(label="Historical Max Price", value=f"${df['Price'].max():,.0f}")
    st.metric(label="Historical Min Price", value=f"${df['Price'].min():,.0f}")
# RIGHT COLUMN: Interactive Historical Chart
with col2:
    st.subheader('📈 Interactive Historical Trend')
    st.write("Filter the historical chart by typing exact dates or using the drag bar.")
    
    # --- ADVANCED INTERACTIVE DATE FILTERING ---
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    # Initialize session state so the inputs and slider stay perfectly synced
    if 'start_input' not in st.session_state:
        st.session_state.start_input = pd.to_datetime('2020-01-01').date()
        st.session_state.end_input = max_date
        st.session_state.slider_dates = (pd.to_datetime('2020-01-01').date(), max_date)

   # Callbacks to sync the fields and the drag bar together
    def sync_from_inputs():
        start = st.session_state.start_input
        end = st.session_state.end_input
        
        # --- SAFETY CHECK ---
        # If the user is currently deleting/typing and the field is empty, do nothing.
        if start is None or end is None:
            return
            
        # --- VALIDATION LOGIC ---
        if start > end:
            # Show a warning pop-up
            st.toast("⚠️ Error: Start Date cannot be after End Date. Dates automatically adjusted.", icon="❌")
            # Automatically fix the error by setting End Date equal to Start Date
            st.session_state.end_input = start
            st.session_state.slider_dates = (start, start)
        else:
            st.session_state.slider_dates = (start, end)

    def sync_from_slider():
        st.session_state.start_input = st.session_state.slider_dates[0]
        st.session_state.end_input = st.session_state.slider_dates[1]
        
    # 1. Type-in Date Fields
    col2_a, col2_b = st.columns(2)
    with col2_a:
        st.date_input("Type Start Date:", key="start_input", min_value=min_date, max_value=max_date, on_change=sync_from_inputs)
    with col2_b:
        st.date_input("Type End Date:", key="end_input", min_value=min_date, max_value=max_date, on_change=sync_from_inputs)

    # 2. Drag Bar (Slider)
    st.slider(
        "Or drag to select date range:",
        min_value=min_date,
        max_value=max_date,
        key="slider_dates",
        format="YYYY-MM-DD",
        on_change=sync_from_slider
    )
    
    # Filter the dataframe based on the synced dates
    final_start = st.session_state.slider_dates[0]
    final_end = st.session_state.slider_dates[1]
    filtered_df = df[(df['Date'].dt.date >= final_start) & (df['Date'].dt.date <= final_end)]
    
    # Plot the filtered data
    chart_data = filtered_df.set_index('Date')[['Price']]
    st.line_chart(chart_data)