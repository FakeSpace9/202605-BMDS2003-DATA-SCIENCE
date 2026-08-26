"""
train_knn_walkforward.py
Gold Price Prediction — CRISP-DM: Modelling & Evaluation phase

Improved KNN: 
Uses "differencing" to predict the daily price change rather than the 
absolute price. This completely fixes the KNN extrapolation problem and 
prevents massive negative R2 scores during walk-forward validation.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import print_metrics2, save_model, save_metrics2, load_cleaned_dataset

current_dir = Path(__file__).resolve().parent.parent
project_root = current_dir.parent
MODEL_OUTPUT_PATH = project_root / "prototype"

TARGET = "Price"

# KNN requires stationary features (percentages/ratios) to calculate accurate 
# distances across different decades. MA_7 is removed because it drifts upwards.
FEATURES = [
    "Volume_Momentum",
    "Volatility_7",
    "Volatility_30",
    "RSI_14",
    "daily_return_lag1",
    "daily_return_lag2"
]

# KNN Hyperparameters
N_NEIGHBORS = 15      # Increased for smoother, more generalized daily difference predictions
WEIGHTS = "uniform"
METRIC = "euclidean"

MIN_TRAIN_YEARS = 4  
MIN_TEST_ROWS = 30   


def build_fold_groups(df: pd.DataFrame, fold_years: list, min_test_rows: int) -> list:
    groups = []
    for year in fold_years:
        n_rows = (df["Year"] == year).sum()
        if n_rows < min_test_rows and groups:
            groups[-1].append(year)
        else:
            groups.append([year])
    return groups


def main():
    print("Loading cleaned dataset...")
    df = load_cleaned_dataset()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Validate if features exist, otherwise fallback
    valid_features = [f for f in FEATURES if f in df.columns]

    years = sorted(df["Year"].unique())
    fold_years = years[MIN_TRAIN_YEARS:]
    fold_groups = build_fold_groups(df, fold_years, MIN_TEST_ROWS)

    fold_rows = []
    last_model = None

    for group in fold_groups:
        test_label = f"{group[0]}" if len(group) == 1 else f"{group[0]}-{group[-1]}"
        train_fold = df.loc[df["Year"] < group[0]]
        test_fold = df.loc[df["Year"].isin(group)]

        if len(train_fold) < 100:
            continue

        X_train, y_train_raw = train_fold[valid_features], train_fold[TARGET]
        X_test, y_test_raw = test_fold[valid_features], test_fold[TARGET]
        y_train_diff = y_train_raw - train_fold["Price_Lag1"]
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=N_NEIGHBORS,
                weights=WEIGHTS,
                metric=METRIC
            ))
        ])

        # Fit model on the price DIFFERENCE
        model.fit(X_train, y_train_diff)
        last_model = model

        # Predict the differences
        y_train_pred_diff = model.predict(X_train)
        y_test_pred_diff = model.predict(X_test)

        # Reconstruct the absolute predicted price: Price_Lag1 + Predicted Difference
        y_train_pred = train_fold["Price_Lag1"] + y_train_pred_diff
        y_pred = test_fold["Price_Lag1"] + y_test_pred_diff

        print(f"\n########## Fold: train < {group[0]}, test = {test_label} "
              f"(n_train={len(train_fold)}, n_test={len(test_fold)}) ##########")

        metrics = print_metrics2(
            f"KNN_{test_label}", 
            y_train_raw, y_train_pred, 
            y_test_raw, y_pred,
            price_lag1_train=train_fold["Price_Lag1"],
            price_lag1_test=test_fold["Price_Lag1"],
        )

        row = {"test_year": test_label, "n_train": len(train_fold), "n_test": len(test_fold)}
        for split in ("train", "test"):
            for k, v in metrics[split].items():
                row[f"{split}_{k}"] = v
        fold_rows.append(row)

    results_df = pd.DataFrame(fold_rows)

    # Train-test R2 gap calculation
    results_df["R2_gap"] = results_df["train_R2"] - results_df["test_R2"]

    print("\n=== Fold-by-fold test metrics ===")
    print(results_df[["test_year", "n_train", "n_test", "test_MAE", "test_RMSE",
                       "test_MAPE", "test_LogMAE", "test_LogRMSE", "train_R2", "test_R2", "R2_gap"]]
          .round(4).to_string(index=False))

    print("\n=== Summary across folds (test set) ===")
    summary = results_df[["test_MAE", "test_RMSE", "test_MAPE",
                           "test_LogMAE", "test_LogRMSE", "train_R2", "test_R2", "R2_gap"]].agg(["mean", "std", "median"])
    print(summary.round(4))


    # Save outputs
    results_path = MODEL_OUTPUT_PATH / "fold_metrics"/"walkforward_price_knn_fold_metrics.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved fold metrics to {results_path}")

    save_metrics2(
        "KNN_walkforward_Price",
        {
            "mean": summary.loc["mean"].to_dict(),
            "std": summary.loc["std"].to_dict(),
        },
        filename="walkforward_price_knn_summary_metrics.json",
    )

    # Save the final fold's trained model pipeline
    save_model(last_model, "knn_walkforward_price.pkl")

if __name__ == "__main__":
    main()