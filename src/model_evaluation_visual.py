import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Load data
print("Loading data...")
df = pd.read_csv('Gold Price.csv')

# 2. Prepare Data
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the baseline Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate RMSE for the error band
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 4. Plot Actual vs Predicted
print("Generating Advanced Scatter Plot...")
plt.figure(figsize=(9, 7))
plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue', label='Predicted Points')

# Draw the perfect prediction line
min_val = y_test.min()
max_val = y_test.max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Line')

# Draw the Shaded Error Band (+/- 1 RMSE)
plt.fill_between(
    [min_val, max_val], 
    [min_val - rmse, max_val - rmse], 
    [min_val + rmse, max_val + rmse], 
    color='darkgreen', alpha=0.15, label=f'RMSE Error Band (±{rmse:.0f})'
)

plt.xlabel('Actual Gold Price', fontsize=12)
plt.ylabel('Predicted Gold Price', fontsize=12)
plt.title('Linear Regression: Actual vs Predicted Prices (With Error Band)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 5. Save the image silently
plt.savefig('actual_vs_predicted_advanced.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Success! The advanced scatter plot has been saved as 'actual_vs_predicted_advanced.png'.")