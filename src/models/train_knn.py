"""
Gold Price Prediction — KNN Regression
80/20 chronological train-test split

Uses the same preprocessed train/test files and features as the
Linear Regression baseline.

IMPORTANT:
Same-day Open, High, Low and Chg% are not used because they are
only known after the day's closing price. Only lagged/rolling
features prepared during preprocessing should be used.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import (
    load_splits,
    print_metrics,
    save_model,
    save_metrics,
)

TARGET = "Price"

# IMPORTANT: You must include lagged price features to predict the actual price.
# Without knowing yesterday's price, KNN cannot guess today's price based on Volume/Month alone.
# Ensure "Price_lag_1" or "MA_7" (Moving Average) exist in your preprocessed data.
FEATURES = [
    "Volatility_7",
    "Volatility_30",
    "RSI_14",
    "Volume_Momentum",
    "Volume_Weighted_Chg_Lag1",
    "daily_return_lag1",
    "daily_return_lag2",
]

# KNN hyperparameters 
N_NEIGHBORS = 5      # Lowered from 10 to make predictions stick closer to recent prices
WEIGHTS = "uniform" # 'distance' gives more weight to the most recent/similar days
METRIC = "euclidean" # Standard Euclidean distance works well after scaling


def main():
    print("Loading preprocessed train/test splits...")
    train_df, test_df = load_splits()

    # Validate if new features exist, otherwise fall back gracefully
    missing_features = [f for f in FEATURES if f not in train_df.columns]
    if missing_features:
        print(f"WARNING: Missing features {missing_features}. Model performance will suffer.")
        valid_features = [f for f in FEATURES if f in train_df.columns]
    else:
        valid_features = FEATURES

    X_train = train_df[valid_features]
    y_train = train_df[TARGET]

    X_test = test_df[valid_features]
    y_test = test_df[TARGET]

    print(f"Train size: {X_train.shape}  Test size: {X_test.shape}")
    print(f"Features  : {valid_features}")

    print("\nTraining KNN Regression model...")

    # Scaling is critical for KNN because it calculates Euclidean distances
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=N_NEIGHBORS,
            weights=WEIGHTS,
            metric=METRIC
        ))
    ])

    model.fit(X_train, y_train)

    print("Evaluating model...")

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    metrics = print_metrics(
        "KNN Regression",
        y_train,
        y_train_pred,
        y_test,
        y_pred
    )

    print("\nKNN Parameters:")
    print(f"Number of neighbors (k): {N_NEIGHBORS}")
    print(f"Weights                : {WEIGHTS}")
    print(f"Distance metric        : {METRIC}")

    # Save the 80/20 model
    save_metrics("KNN Regression", metrics)
    save_model(model, "knn_regression_80_20.pkl")

    print("\nModel saved as: knn_regression_80_20.pkl")


if __name__ == "__main__":
    main()