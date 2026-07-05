import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('gold_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title and description
st.title("🥇 Daily Gold Price Prediction App")
st.write("""
This application predicts the daily closing price of gold based on intra-day trading metrics.
Enter the market data below to get a prediction.
""")

# Sidebar for user inputs
st.sidebar.header("Input Market Features")

def user_input_features():
    Open_price = st.sidebar.number_input('Open Price', min_value=20000, max_value=150000, value=38000)
    High_price = st.sidebar.number_input('High Price', min_value=20000, max_value=150000, value=39000)
    Low_price = st.sidebar.number_input('Low Price', min_value=20000, max_value=150000, value=37500)
    Volume = st.sidebar.number_input('Trading Volume', min_value=0, max_value=150000, value=11500)

    data = {
        'Open': Open_price,
        'High': High_price,
        'Low': Low_price,
        'Volume': Volume
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display the user input
st.subheader('User Input Parameters')
st.write(input_df)

# Predict the price when the button is clicked
if st.button('Predict Closing Price'):
    # Scale the input data using the saved scaler
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)

    st.subheader('Predicted Gold Closing Price')
    st.success(f"Estimated Price: {prediction[0]:.2f}")