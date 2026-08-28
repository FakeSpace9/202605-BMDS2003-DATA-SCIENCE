from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from utils import (
    load_cleaned_dataset,
    print_metrics2,
    save_metrics2,
    save_model,
)

PRICE_COL = "Price"
LAG1_COL = "Price_Lag1"
RETURN_COL = "Price_Change"

FEATURE_COLS = [
    "Volume",
    "Volatility_7",
    "Return_Lag1",
    "Momentum_7",
]

GB_PARAMS = dict(
    n_estimators=50,
    learning_rate=0.03,
    max_depth=2,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
)

WARMUP_YEARS = 4
MIN_FOLD_SIZE = 30
MIN_TRAIN_SIZE = 100

OUTPUT_DIR = PROJECT_ROOT / "prototype"

def prepare_dataset() -> pd.DataFrame:
    df = load_cleaned_dataset()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Price_Lag1" not in df.columns:
        df["Price_Lag1"] = df["Price"].shift(1)

    df = df[(df[PRICE_COL] > 0) & (df[LAG1_COL] > 0)].copy()

    df[RETURN_COL] = np.log(df[PRICE_COL] / df[LAG1_COL])
    df["Return_Lag1"] = df[RETURN_COL].shift(1)
    df["Momentum_7"] = (df[PRICE_COL] / df[PRICE_COL].shift(7) - 1)

    needed = [
        "Date",
        "Year",
        PRICE_COL,
        LAG1_COL,
    ] + FEATURE_COLS

    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=needed).reset_index(drop=True)
    return df

def make_test_windows(df: pd.DataFrame) -> list[list[int]]:
    all_years = sorted(df["Year"].unique())
    candidate_years = all_years[WARMUP_YEARS:]

    windows = []

    for year in candidate_years:
        row_count = (
            df["Year"] == year
        ).sum()

        if windows and row_count < MIN_FOLD_SIZE:
            windows[-1].append(year)
        else:
            windows.append([year])

    return windows

def window_label(window: list[int]) -> str:
    return (
        str(window[0])
        if len(window) == 1
        else f"{window[0]}-{window[-1]}"
    )

def run_single_fold(
    df: pd.DataFrame,
    window: list[int]
) -> dict | None:

    label = window_label(window)
    cutoff_year = window[0]

    train_df = df.loc[
        df["Year"] < cutoff_year
    ]

    test_df = df.loc[
        df["Year"].isin(window)
    ]

    if len(train_df) < MIN_TRAIN_SIZE:
        print(
            f"Skipping fold {label}: "
            "training data too small."
        )
        return None

    model = GradientBoostingRegressor(
        **GB_PARAMS
    )

    model.fit(
        train_df[FEATURE_COLS],
        train_df[RETURN_COL]
    )

    train_pred_change = model.predict(
        train_df[FEATURE_COLS]
    )

    test_pred_change = model.predict(
        test_df[FEATURE_COLS]
    )

    # Convert predicted log return back to price
    train_pred_price = (
        train_df[LAG1_COL]
        * np.exp(train_pred_change)
    )

    test_pred_price = (
        test_df[LAG1_COL]
        * np.exp(test_pred_change)
    )

    print(
        f"\n======== Fold: train < {cutoff_year}, "
        f"test = {label} "
        f"(n_train={len(train_df)}, "
        f"n_test={len(test_df)}) ========"
    )

    metrics = print_metrics2(
        f"GradBoost_{label}",
        train_df[PRICE_COL],
        train_pred_price,
        test_df[PRICE_COL],
        test_pred_price,
        price_lag1_train=train_df[LAG1_COL],
        price_lag1_test=test_df[LAG1_COL],
    )

    fold_summary = {
        "test_year": label,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    for split in ("train", "test"):
        for metric_name, value in metrics[split].items():
            fold_summary[
                f"{split}_{metric_name}"
            ] = value

    feature_importance = {
        "test_year": label
    }

    feature_importance.update(
        zip(
            FEATURE_COLS,
            model.feature_importances_
        )
    )

    return {
        "fold_summary": fold_summary,
        "feature_importance": feature_importance,
        "model": model,
    }


def build_results_table(
    fold_results: list[dict]
) -> pd.DataFrame:

    results_df = pd.DataFrame(
        [r["fold_summary"] for r in fold_results]
    )

    results_df["R2_gap"] = (
        results_df["train_R2"]
        - results_df["test_R2"]
    )

    return results_df


def report_results(
    results_df: pd.DataFrame,
    importance_df: pd.DataFrame
) -> None:

    print("\n---- Fold-by-fold test metrics ----")

    cols = [
        "test_year",
        "n_train",
        "n_test",
        "test_MAE",
        "test_RMSE",
        "test_MAPE",
        "test_LogMAE",
        "test_LogRMSE",
        "train_R2",
        "test_R2",
        "R2_gap",
    ]

    print(
        results_df[cols]
        .round(4)
        .to_string(index=False)
    )

    print("\n---- Feature Importance by Fold ----")

    print(
        importance_df
        .round(6)
        .to_string(index=False)
    )

    avg_importance = (importance_df[FEATURE_COLS].mean().sort_values(ascending=False))

    print("\n---- Average Feature Importance ----")
    print(avg_importance.round(6).to_string())

    summary = results_df[
        [
            "test_MAE",
            "test_RMSE",
            "test_MAPE",
            "test_LogMAE",
            "test_LogRMSE",
            "train_R2",
            "test_R2",
            "R2_gap",
        ]
    ].agg(["mean"])

    print("\n---- Summary across folds (test set) ----")
    print(summary.round(4))

def persist_outputs(results_df: pd.DataFrame, importance_df: pd.DataFrame, final_model) -> None:

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    fold_metrics_path = (OUTPUT_DIR /"fold_metrics"/"gradient_boosting_fold_metrics.csv")
    results_df.to_csv(fold_metrics_path, index=False)

    print(
        f"\nSaved fold metrics to "
        f"{fold_metrics_path}"
    )

    importance_path = (OUTPUT_DIR /"feature_coefficient"/ "gradient_boosting_feature_coefficient.csv")

    importance_df.to_csv(importance_path, index=False)

    print(
        f"Saved feature importance to "
        f"{importance_path}"
    )

    summary_mean = results_df[
        [
            "test_MAE",
            "test_RMSE",
            "test_MAPE",
            "test_LogMAE",
            "test_LogRMSE",
            "train_R2",
            "test_R2",
            "R2_gap",
        ]
    ].mean().to_dict()

    save_metrics2(
        "GradientBoosting_Price",
        {"mean": summary_mean},
        filename="gradient_boosting_summary_metrics.json",
    )

    save_model(final_model, "gradient_boosting_price.pkl")

    print("\nFinal Gradient Boosting model saved "
        "(trained on Log_Return, "
        "predictions reconstructed to Price scale)."
    )


def main() -> None:

    print("Loading cleaned dataset...")
    df = prepare_dataset()
    years = sorted(df["Year"].unique())

    print("\nAvailable years:")
    print(years)
    windows = make_test_windows(df)

    print("\n---- Walk-forward folds ----")

    for window in windows:
        label = window_label(window)
        cutoff_year = window[0]

        n_train = (df["Year"] < cutoff_year).sum()
        n_test = (df["Year"].isin(window)).sum()

        print(f"Test = {label} | n_train = {n_train} | n_test = {n_test}")

    fold_results = []

    for window in windows:
        result = run_single_fold(df, window)

        if result is not None:
            fold_results.append(result)

    if not fold_results:
        raise ValueError("No valid folds were generated.")

    results_df = build_results_table(fold_results)

    importance_df = pd.DataFrame([r["feature_importance"] for r in fold_results])

    final_model = fold_results[-1]["model"]

    report_results(results_df,importance_df)
    persist_outputs(results_df,importance_df,final_model)

if __name__ == "__main__":
    main()