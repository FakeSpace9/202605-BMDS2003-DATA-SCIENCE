import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("Loading and preprocessing data...")
df = pd.read_csv('Gold Price.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Convert Currency to MYR (Assuming 1 INR = 0.057 MYR)
exchange_rate = 0.057
cols_to_convert = ['Price', 'Open', 'High', 'Low']
for col in cols_to_convert:
    df[col] = df[col] * exchange_rate

# Handle Outliers (IQR Method)
for col in cols_to_convert:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# ==========================================
# PREVENTING FEATURE LEAKAGE
# ==========================================
# Shift the target to predict TOMORROW'S price.
df['Target_Next_Day_Price'] = df['Price'].shift(-1)
model_df = df.dropna().drop(columns=['Date']) 

# ==========================================
# 2. DATA SPLITTING & SCALING
# ==========================================
print("Splitting and scaling data...")
X = model_df.drop(columns=['Target_Next_Day_Price'])
y = model_df['Target_Next_Day_Price']

# shuffle=True allows Tree models to interpolate correctly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Scale ONLY on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING, LOG EVALUATION & EXPORTING
# ==========================================
print("Training and saving models...")
models = {
    "Linear_Regression": LinearRegression(),
    "Decision_Tree": DecisionTreeRegressor(random_state=42),
    "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient_Boosting": GradientBoostingRegressor(random_state=42)
}

joblib.dump(scaler, 'scaler.pkl')

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    predictions = model.predict(X_test_scaled)
    
    # Prevent negative numbers before taking the log (safety measure for Linear Regression)
    safe_predictions = np.clip(predictions, a_min=1e-10, a_max=None)
    
    # Convert to Log Values
    log_y_test = np.log(y_test)
    log_predictions = np.log(safe_predictions)
    
    # Evaluate using Log Values
    log_rmse = np.sqrt(mean_squared_error(log_y_test, log_predictions))
    log_mae = mean_absolute_error(log_y_test, log_predictions)
    r2 = r2_score(y_test, predictions) # R2 usually stays on the standard scale to measure explained variance
    
    # Output to 4 decimal places since log values are smaller
    print(f"{name.replace('_', ' ')} -> Log RMSE: {log_rmse:.4f} | Log MAE: {log_mae:.4f} | R2: {r2:.4f}")
    
    # Save EVERY model
    joblib.dump(model, f'{name}.pkl')

print("\nSuccess: 'scaler.pkl' and all 4 model '.pkl' files saved to directory.")

# ==========================================
# 4. GENERATING VISUALIZATIONS
# ==========================================
print("Generating and saving visualisations...")
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Price'], color='blue')
plt.title('Daily Gold Price Trend (MYR)')
plt.xlabel('Time (Index)')
plt.ylabel('Price (MYR)')
plt.savefig('price_trend.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df[['Price', 'Open', 'High', 'Low']])
plt.title('Boxplot for Price Features')
plt.ylabel('MYR')
plt.savefig('outliers_boxplot.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(model_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

rf_model = models["Random_Forest"]
importance = rf_model.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=importance, y=X.columns)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.savefig('feature_importance.png')
plt.close()

print("All tasks completed.")