from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import print_metrics2, save_model, save_metrics2, load_cleaned_dataset

current_dir = Path(__file__).resolve().parent.parent
project_root = current_dir.parent
MODEL_OUTPUT_PATH = project_root / "prototype"

TARGET = "Price"

FEATURES = [
    "Volume",
    "Month",
    "Day",
    "Volatility_7",
    "MA_7",
]

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


    scaler_path = MODEL_OUTPUT_PATH / "scaler.pkl"
    if scaler_path.exists():
        global_scaler = joblib.load(scaler_path)
        # scaler was fit on ['Open', 'High', 'Low', 'Volume'] -> Volume is index 3
        vol_mean = global_scaler.mean_[3]
        vol_scale = global_scaler.scale_[3]
        df["Volume"] = (df["Volume"] * vol_scale) + vol_mean
        print("Successfully reverted Volume to raw values for pipeline training.")
    else:
        print("Warning: scaler.pkl not found. Assuming Volume is already raw.")

    years = sorted(df["Year"].unique())
    fold_years = years[MIN_TRAIN_YEARS:]
    fold_groups = build_fold_groups(df, fold_years, MIN_TEST_ROWS)

    fold_rows = []
    coef_rows = []
    last_model = None

    for group in fold_groups:
        test_label = f"{group[0]}" if len(group) == 1 else f"{group[0]}-{group[-1]}"
        train_fold = df.loc[df["Year"] < group[0]]
        test_fold = df.loc[df["Year"].isin(group)]

        if len(train_fold) < 100:
            continue

        X_train, y_train = train_fold[FEATURES], train_fold[TARGET]
        X_test, y_test = test_fold[FEATURES], test_fold[TARGET]

        # Create a Pipeline that ONLY scales the 'Volume' column
        preprocessor = ColumnTransformer(
            transformers=[('vol_scaler', StandardScaler(), ['Volume'])],
            remainder='passthrough'
        )

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        model.fit(X_train, y_train)
        last_model = model

        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        print(f"\n########## Fold: train < {group[0]}, test = {test_label} "
              f"(n_train={len(train_fold)}, n_test={len(test_fold)}) ##########")

        metrics = print_metrics2(
            f"LinReg_{test_label}", y_train, y_train_pred, y_test, y_pred,
            price_lag1_train=train_fold["Price_Lag1"],
            price_lag1_test=test_fold["Price_Lag1"],
        )

        row = {"test_year": test_label, "n_train": len(train_fold), "n_test": len(test_fold)}
        for split in ("train", "test"):
            for k, v in metrics[split].items():
                row[f"{split}_{k}"] = v
        fold_rows.append(row)

        # Extract coefficients from the pipeline's regressor
        regressor = model.named_steps['regressor']
        coef_row = {"test_year": test_label, "Intercept": regressor.intercept_}
        coef_row.update(dict(zip(FEATURES, regressor.coef_)))
        coef_rows.append(coef_row)

    results_df = pd.DataFrame(fold_rows)
    coef_df = pd.DataFrame(coef_rows)

    results_df["R2_gap"] = results_df["train_R2"] - results_df["test_R2"]
    
    print("\n=== Fold-by-fold test metrics ===")
    print(results_df[["test_year", "n_train", "n_test", "test_MAE", "test_RMSE",
                       "test_MAPE", "test_LogMAE", "test_LogRMSE", "train_R2", "test_R2", "R2_gap"]]
          .round(4).to_string(index=False))
    
    print("\n=== Coefficients by fold ===")
    print(coef_df.round(6).to_string(index=False))
    
    summary = results_df[["test_MAE", "test_RMSE", "test_MAPE",
                           "test_LogMAE", "test_LogRMSE", "train_R2", "test_R2", "R2_gap"]].agg(["mean", "std", "median"])
    print("\n=== Summary across folds (test set) ===")
    print(summary.round(4))
    results_path = MODEL_OUTPUT_PATH / "fold_metrics"/"walkforward_price_fold_metrics.csv"
    coef_path = MODEL_OUTPUT_PATH / "feature_coefficient"/"walkforward_price_coefficients.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    coef_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_path, index=False)
    coef_df.to_csv(coef_path, index=False)
    
    save_metrics2(
        "LinearRegression_walkforward_Price",
        {
            "mean": summary.loc["mean"].to_dict(),
            "std": summary.loc["std"].to_dict(),
        },
        filename="walkforward_price_summary_metrics.json",
    )
    
    save_model(last_model, "linear_regression_walkforward_price.pkl")


if __name__ == "__main__":
    main()