"""
train_linear_regression.py
Gold Price Prediction — CRISP-DM: Modelling & Evaluation phase

Trains a Linear Regression baseline on the preprocessed train/test
splits produced by preprocess_gold_price.py.

Run order:
    1. python preprocess_gold_price.py   (creates Gold_Price_train.csv / Gold_Price_test.csv)
    2. python train_linear_regression.py (this script)

IMPORTANT DESIGN CHOICE:
    Same-day 'Open', 'High', 'Low', 'Chg%' are dropped as predictors.
    High/Low/Chg% are only known AFTER the day's closing price is set,
    so using them to predict that same day's Price would leak future
    information and make the model look unrealistically accurate.
    Only lagged/rolling features (known before the day closes) are used.
"""

from pathlib import Path
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import (
    load_splits,
    print_metrics,
    save_model,
    save_metrics,
)

TARGET = "Price"
FEATURES = [
    "Volume",
    "Day",
    "Month",
    "Volatility_30",
]

def main():
    print("Loading preprocessed train/test splits...")
    train_df, test_df = load_splits()

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print(f"Train size: {X_train.shape}  Test size: {X_test.shape}")
    print(f"Features  : {FEATURES}")

    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    metrics = print_metrics("Linear Regression", y_train, y_train_pred, y_test, y_pred)

    coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": model.coef_})
    print("\nModel coefficients:")
    print(coef_df.to_string(index=False))
    print(f"Intercept: {model.intercept_:,.4f}")

    save_metrics("Linear Regression", metrics)


    print("\n--- Phase 2: Retraining Final Model on 100% Data ---")
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    final_model = LinearRegression()
    final_model.fit(X_full, y_full)

    # Predict on the full dataset to get our final in-sample metrics
    print("Evaluating Phase 2 model on full dataset...")
    y_full_pred = final_model.predict(X_full)
    
    # We pass the full dataset as both Train and Test to satisfy the function requirements
    print_metrics("Final Linear Regression (100% Data)", y_full, y_full_pred, y_full, y_full_pred)

    # Display the final model's new coefficients
    coef_df_final = pd.DataFrame({"Feature": FEATURES, "Coefficient": final_model.coef_})
    print("\nFinal Model coefficients (100% data):")
    print(coef_df_final.to_string(index=False))
    print(f"Intercept: {final_model.intercept_:,.4f}")
    
    save_model(model,"linear_regression 80 20.pkl")
    save_model(final_model,"linear_regression 100 0.pkl")


if __name__ == "__main__":
    main()