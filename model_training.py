import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the dataset
print("Loading data...")
df = pd.read_csv('Gold Price.csv')

# 2. Data Preparation
df['Date'] = pd.to_datetime(df['Date'])
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Price']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardise the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
print("\nTraining models...")
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# 4. Evaluation
print("\n--- Model Performance ---")
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R2': r2}
    print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.4f}")

# 5. Save the best model and the scaler for deployment
print("\nSaving the best model (Linear Regression) and Scaler for deployment...")
best_model = models['Linear Regression']
joblib.dump(best_model, 'gold_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved successfully as 'gold_price_model.pkl' and 'scaler.pkl'")