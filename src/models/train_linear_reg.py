"""
train_linear_regression_walkforward.py
Gold Price Prediction — CRISP-DM: Modelling & Evaluation phase

Same model/target/features as train_linear_regression.py (predicting
Chg%), but replaces the single static 75/25 split with expanding-window
walk-forward validation: retrain once per year, test only on the next
year, so you see how the model performs across regimes (incl. 2024-2025)
instead of one blended number.

Run order:
    1. python preprocess_gold_price.py        (creates Gold_Price_cleaned.csv)
    2. python train_linear_regression_walkforward.py   (this script)

No plots are saved -- only per-fold metrics (CSV) and an aggregate
summary (JSON), plus the final fold's fitted model.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import print_metrics2, save_model, save_metrics2, load_cleaned_dataset

current_dir = Path(__file__).resolve().parent.parent
project_root = current_dir.parent
MODEL_OUTPUT_PATH = project_root / "prototype"

TARGET = "Chg%"

FEATURES = [
    "Price_Lag1",
    "Volume_Lag1",
    # Momentum & Lags
    "Chg%_Lag1",
    "Chg%_Lag2",
    "Chg%_Lag3",
    # Technical Indicators
    "Volume_Momentum",
    "MA_Ratio",
    "BB_Status",
    # Context & Volatility
    "Price_Range_Lag1",
    "Volatility_7",
    "Volatility_30",
]

MIN_TRAIN_YEARS = 4  # need enough history before the first test fold


def main():
    print("Loading cleaned dataset...")
    df = load_cleaned_dataset()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    years = sorted(df["Year"].unique())
    fold_years = years[MIN_TRAIN_YEARS:]

    fold_rows = []
    coef_rows = []
    last_model = None

    for test_year in fold_years:
        train_fold = df.loc[df["Year"] < test_year]
        test_fold = df.loc[df["Year"] == test_year]

        if len(test_fold) == 0 or len(train_fold) < 100:
            continue

        X_train, y_train = train_fold[FEATURES], train_fold[TARGET]
        X_test, y_test = test_fold[FEATURES], test_fold[TARGET]

        model = LinearRegression()
        model.fit(X_train, y_train)
        last_model = model

        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        print(f"\n########## Fold: train < {test_year}, test = {test_year} "
              f"(n_train={len(train_fold)}, n_test={len(test_fold)}) ##########")

        metrics = print_metrics2(
            f"LinReg_{test_year}", y_train, y_train_pred, y_test, y_pred,
            price_lag1_train=train_fold["Price_Lag1"],
            price_lag1_test=test_fold["Price_Lag1"],
        )

        row = {"test_year": test_year, "n_train": len(train_fold), "n_test": len(test_fold)}
        for split in ("train", "test"):
            for k, v in metrics[split].items():
                row[f"{split}_{k}"] = v
        fold_rows.append(row)

        coef_row = {"test_year": test_year, "Intercept": model.intercept_}
        coef_row.update(dict(zip(FEATURES, model.coef_)))
        coef_rows.append(coef_row)

    results_df = pd.DataFrame(fold_rows)
    coef_df = pd.DataFrame(coef_rows)

    print("\n=== Fold-by-fold test metrics ===")
    print(results_df[["test_year", "n_train", "n_test", "test_MAE", "test_RMSE",
                       "test_MAPE", "test_LogMAE", "test_LogRMSE", "test_R2"]]
          .round(4).to_string(index=False))

    print("\n=== Coefficients by fold ===")
    print(coef_df.round(6).to_string(index=False))

    print("\n=== Summary across folds (test set) ===")
    summary = results_df[["test_MAE", "test_RMSE", "test_MAPE",
                           "test_LogMAE", "test_LogRMSE", "test_R2"]].agg(["mean", "std"])
    print(summary.round(4))

    # Save (no plots) -- fold results, coefficients, and aggregate summary
    results_path = MODEL_OUTPUT_PATH / "walkforward_chgpct_fold_metrics.csv"
    coef_path = MODEL_OUTPUT_PATH / "walkforward_chgpct_coefficients.csv"
    results_df.to_csv(results_path, index=False)
    coef_df.to_csv(coef_path, index=False)
    print(f"\nSaved fold metrics to {results_path}")
    print(f"Saved coefficients to {coef_path}")

    save_metrics2(
        "LinearRegression_walkforward_ChgPct",
        {
            "mean": summary.loc["mean"].to_dict(),
            "std": summary.loc["std"].to_dict(),
        },
        filename="walkforward_chgpct_summary_metrics.json",
    )

    # Save the last-fold model (trained on the most data) as the deployable one
    save_model(last_model, "linear_regression_walkforward_chgpct.pkl")


if __name__ == "__main__":
    main()