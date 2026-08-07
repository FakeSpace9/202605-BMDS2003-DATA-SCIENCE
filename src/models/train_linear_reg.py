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
from xml.parsers.expat import model
import pandas as pd
from sklearn.linear_model import LinearRegression
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import (
    load_splits,
    print_metrics,
    save_model,
    save_metrics,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

TARGET = "Price"
FEATURES = [
    
    # Lagged volume
    "Volume_Lag1",
    
    # Rolling volatilities
    "Volatility_7",
    "Volatility_30",
    
    # Time features
    "Month",
    "DayOfWeek",
    "Year"
]

def main():
    print("Loading preprocessed train/test splits...")
    train_df, test_df = load_splits()

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print(f"Train size: {X_train.shape}  Test size: {X_test.shape}")
    print(f"Features  : {FEATURES}")

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

    save_model(model,"linear_regression.pkl")


if __name__ == "__main__":
    main()