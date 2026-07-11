import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("Loading data...")
df = pd.read_csv('Gold Price.csv')

# Data Preparation
df['Date'] = pd.to_datetime(df['Date'])
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Price']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardise the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. MODEL TRAINING
# ==========================================
print("\nTraining models...")
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# ==========================================
# 3. EVALUATION
# ==========================================
print("\n--- Model Performance ---")
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R2': r2}
    print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.4f}")

# ==========================================
# 4. SAVE MODELS & SCALER
# ==========================================
print("\nSaving all 4 models and the Scaler for deployment...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(models['Linear Regression'], 'model_lr.pkl')
joblib.dump(models['Decision Tree'], 'model_dt.pkl')
joblib.dump(models['Random Forest'], 'model_rf.pkl')
joblib.dump(models['Gradient Boosting'], 'model_gb.pkl')
print("✅ Success! All models and the scaler have been saved as .pkl files.")

# ==========================================
# 5. ADVANCED EVALUATION VISUALS
# ==========================================
print("\nGenerating Advanced Evaluation Visuals...")

# Set professional visual style 
sns.set_theme(style="whitegrid")

# --- VISUAL A: Residual Analysis Plot (Linear Regression) ---
print("Generating Residual Plot...")
# Predict specifically with the Linear Regression model
lr_model = models['Linear Regression']
y_pred_lr = lr_model.predict(X_test_scaled)
residuals = y_test - y_pred_lr

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='teal', bins=40)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Error Distribution (Linear Regression)', fontsize=14, fontweight='bold')
plt.xlabel('Prediction Error (Actual Price - Predicted Price)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# --- VISUAL B: Feature Importance Bar Chart (Random Forest) ---
print("Generating Feature Importance Chart...")
# Extract importance metrics from the Random Forest model
feature_importance = pd.DataFrame({
    'Feature': ['Open', 'High', 'Low', 'Volume'],
    'Importance': models['Random Forest'].feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Random Forest: Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Relative Importance (0 to 1)', fontsize=12)
plt.ylabel('')
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Success! Advanced evaluation visuals (Residuals & Feature Importance) have been saved.")