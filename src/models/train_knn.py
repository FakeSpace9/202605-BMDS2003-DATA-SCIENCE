
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

# Same features used by the Linear Regression model
FEATURES = [
    "Volume",
    "Day",
    "Month",
    "Volatility_30",
]

# KNN hyperparameters
N_NEIGHBORS = 2
WEIGHTS = "distance"
METRIC = "euclidean"


def main():
    print("Loading preprocessed train/test splits...")
    train_df, test_df = load_splits()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(f"Train size: {X_train.shape}  Test size: {X_test.shape}")
    print(f"Features  : {FEATURES}")

    print("\nTraining KNN Regression model...")

    # Scaling is important for KNN because it uses distance
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





