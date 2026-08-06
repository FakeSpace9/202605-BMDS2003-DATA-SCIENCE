"""
Model Training — Gold Price Prediction
CRISP-DM: Modelling & Evaluation phase

Uses the train/test splits produced by preprocess_gold_price.py.

IMPORTANT DESIGN CHOICE:
Same-day 'Open', 'High', 'Low', 'Chg%' are dropped as predictors.
High/Low/Chg% are only known AFTER the day's closing price is set,
so using them to predict that same day's Price would leak future
information and make the model look unrealistically accurate.
Only lagged/rolling features (known before the day closes) are used.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------
# 1. Load preprocessed train/test data
# ------------------------------------------------------------------
train_df = pd.read_csv('Gold_Price_train.csv')
test_df = pd.read_csv('Gold_Price_test.csv')

# Dropped: 'Price_Lag1', 'Price_Lag2', and 'MA_7' to lower R2
FEATURES = ['Volume','Month', 'Day', 'Volatility_7']
TARGET = 'Price'

from sklearn.model_selection import train_test_split

# 1. Combine the data to perform a fresh randomized split
full_df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Use the train_test_split function to handle the shuffling and assigning
X_train, X_test, y_train, y_test = train_test_split(
    full_df[FEATURES], 
    full_df[TARGET], 
    test_size=0.2, 
    shuffle=True, 
    random_state=1
)

# 3. Recreate test_df so your plotting code at the bottom of the script still works
test_df = full_df.loc[X_test.index].copy()

print("Train size:", X_train.shape, " Test size:", X_test.shape)

# ------------------------------------------------------------------
# 2. Define candidate models
# ------------------------------------------------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1, random_state=42),
    'Random Forest': RandomForestRegressor(
        n_estimators=300,        # Increased from 100
        max_depth=5,            # Increased from 3
        min_samples_split=50,     # Decreased from 20
        random_state=42, 
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300,        # Increased from 100
        learning_rate=0.01, 
        max_depth=5,             # Increased from 3
        subsample=0.8,
        random_state=42
    ),
}

# ------------------------------------------------------------------
# 3. Train, predict, evaluate
# ------------------------------------------------------------------
results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log-scale metrics (evaluate on log(price)) — useful for financial
    # data since price magnitude grows a lot over the years; log-scale
    # error weighs a $500 miss on a $30k price the same as a $2500 miss
    # on a $130k price (i.e. relative error rather than absolute error).
    log_y_test = np.log(y_test)
    log_y_pred = np.log(np.clip(y_pred, a_min=1e-6, a_max=None))
    log_mae = mean_absolute_error(log_y_test, log_y_pred)
    log_rmse = np.sqrt(mean_squared_error(log_y_test, log_y_pred))

    results.append({
        'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'Log_MAE': log_mae, 'Log_RMSE': log_rmse
    })
    print(f"{name:20s} | MAE: {mae:9.2f} | RMSE: {rmse:9.2f} | R2: {r2:.4f} "
          f"| Log_MAE: {log_mae:.4f} | Log_RMSE: {log_rmse:.4f}")

results_df = pd.DataFrame(results).sort_values('RMSE')
print("\nModel comparison (sorted by RMSE):")
print(results_df.to_string(index=False))

# ------------------------------------------------------------------
# 4. Pick the best model
# ------------------------------------------------------------------
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")

# ------------------------------------------------------------------
# 5. Feature importance (for tree-based best model)
# ------------------------------------------------------------------
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature importance:")
    print(importance_df.to_string(index=False))

# ------------------------------------------------------------------
# 6. Plot: Actual vs Predicted (best model)
# ------------------------------------------------------------------
plt.figure(figsize=(12, 5))
x_idx = pd.to_datetime(test_df['Date'])
plt.plot(x_idx, y_test.values, label='Actual', linewidth=1.5)
plt.plot(x_idx, predictions[best_model_name],
         label=f'Predicted ({best_model_name})', linewidth=1.2, alpha=0.8)
plt.gca().xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title(f'Actual vs Predicted Gold Price — {best_model_name}')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
print("\nSaved plot: actual_vs_predicted.png")

# ------------------------------------------------------------------
# 7. Save model comparison results and predictions
# ------------------------------------------------------------------
results_df.to_csv('model_comparison_results.csv', index=False)

pred_out = test_df[['Date', 'Price']].copy()
for name, preds in predictions.items():
    pred_out[f'Pred_{name.replace(" ", "_")}'] = preds
pred_out.to_csv('predictions.csv', index=False)

# ------------------------------------------------------------------
# 8. Save the best trained model
# ------------------------------------------------------------------
import joblib
joblib.dump(best_model, 'best_gold_price_model.pkl')
print("Saved best model: best_gold_price_model.pkl")

print("\nDone. Files saved:")
print(" - model_comparison_results.csv")
print(" - predictions.csv")
print(" - actual_vs_predicted.png")
print(" - best_gold_price_model.pkl")