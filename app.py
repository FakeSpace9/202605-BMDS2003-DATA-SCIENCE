import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load all 4 trained models
@st.cache_resource
def load_models():
    models = {
        "Linear Regression (Baseline)": joblib.load('baseline_linear_regression.pkl'),
        "Decision Tree": joblib.load('decision_tree_regressor.pkl'),
        "Random Forest": joblib.load('random_forest_regressor.pkl'),
        "Gradient Boosting": joblib.load('gradient_boosting_regressor.pkl')
    }
    return models

models = load_models()

st.set_page_config(layout="wide")
st.title("Seoul Bike Sharing Demand Prediction")
st.write("Compare predictions across four different machine learning models.")

# --- Sidebar Inputs ---
st.sidebar.header("Input Environmental Factors")

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp = st.sidebar.number_input("Temperature (°C)", -20.0, 40.0, 15.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 10.0, 1.5)
visibility = st.sidebar.number_input("Visibility (10m)", 0, 2000, 1000)
solar_rad = st.sidebar.number_input("Solar Radiation (MJ/m2)", 0.0, 4.0, 1.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 40.0, 0.0)
snowfall = st.sidebar.number_input("Snowfall (cm)", 0.0, 10.0, 0.0)

seasons = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
season_map = {"Spring": 1, "Summer": 2, "Autumn": 0, "Winter": 3} 

holiday = st.sidebar.selectbox("Holiday", ["No Holiday", "Holiday"])
holiday_map = {"No Holiday": 1, "Holiday": 0} 

func_day = st.sidebar.selectbox("Functioning Day", ["Yes", "No"])
func_map = {"Yes": 1, "No": 0}

month = st.sidebar.slider("Month", 1, 12, 6)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

# --- Prediction Logic ---
if st.button("Predict Bike Demand", type="primary"):
    
    input_data = pd.DataFrame({
        'Hour': [hour],
        'Temperature(°C)': [temp],
        'Humidity(%)': [humidity],
        'Wind speed (m/s)': [wind_speed],
        'Visibility (10m)': [visibility],
        'Dew point temperature(°C)': [temp - ((100 - humidity)/5)],
        'Solar Radiation (MJ/m2)': [solar_rad],
        'Rainfall(mm)': [rainfall],
        'Snowfall (cm)': [snowfall],
        'Seasons': [season_map[seasons]],
        'Holiday': [holiday_map[holiday]],
        'Functioning Day': [func_map[func_day]],
        'Month': [month],
        'DayOfWeek': [day_of_week]
    })
    
    st.markdown("### Model Predictions")
    
    cols = st.columns(4)
    
    for idx, (model_name, model) in enumerate(models.items()):
        # The model now predicts the LOG of the bike count
        log_pred = model.predict(input_data)[0]
        
        # We must REVERSE the log transformation to get actual bikes
        raw_pred = np.expm1(log_pred)
        final_pred = max(0, int(np.round(raw_pred)))
        
        with cols[idx]:
            st.info(f"**{model_name}**")
            st.metric(label="Predicted Bikes", value=final_pred)