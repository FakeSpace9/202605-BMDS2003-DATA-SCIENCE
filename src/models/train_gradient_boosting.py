from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Connect to group project utilities
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import print_metrics2, save_model, save_metrics2, load_cleaned_dataset

current_dir = Path(__file__).resolve().parent.parent
project_root = current_dir.parent
MODEL_OUTPUT_PATH = project_root / "prototype"

TARGET = "Price"
ANCHOR = "Price_Lag30"
ANCHOR_NOISY = "Price_Lag30_noisy"
RANDOM_SEED = 42
ANCHOR_NOISE_FRAC = 0.18

FEATURES = [
    ANCHOR_NOISY,
    "Volatility_7",
    "Volatility_30",
    "RSI_14",
    "Volume_Momentum",
    "Volume_Weighted_Chg_Lag1",
    "daily_return_lag1",
    "daily_return_lag2",
]

TEST_SIZE = 0.20


def add_noisy_anchor(df: pd.DataFrame, noise_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(df))
    df = df.copy()
    df[ANCHOR_NOISY] = df[ANCHOR] * (1.0 + noise_frac * noise)
    return df


def main():
    print("Loading cleaned dataset...")
    df = load_cleaned_dataset()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = add_noisy_anchor(df, ANCHOR_NOISE_FRAC, RANDOM_SEED)

    split = int(len(df) * (1 - TEST_SIZE))
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    print(f"Train: {train_df['Date'].min().date()} -> {train_df['Date'].max().date()} (n={len(train_df)})")
    print(f"Test : {test_df['Date'].min().date()} -> {test_df['Date'].max().date()} (n={len(test_df)})")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # Model training using Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    # Output formatted metrics
    metrics = print_metrics2(
        "GradientBoosting_r0.5", y_train, y_train_pred, y_test, y_pred,
        price_lag1_train=train_df["Price_Lag1"],
        price_lag1_test=test_df["Price_Lag1"],
    )

    print(f"\nTrain R2: {metrics['train']['R2']:.4f}")
    print(f"Test R2 : {metrics['test']['R2']:.4f}")

    # Feature Importance replacing Linear Regression Coefficients
    imp_df = pd.DataFrame([{**dict(zip(FEATURES, model.feature_importances_))}])
    print("\n=== Feature Importances ===")
    print(imp_df.round(6).to_string(index=False))

    # Save output artifacts to project folder
    metrics_path = MODEL_OUTPUT_PATH / "price_fold_metrics_gb_r0.5.csv"
    imp_path = MODEL_OUTPUT_PATH / "price_feature_importance_gb_r0.5.csv"
    
    pd.DataFrame([{"split": "train", **metrics["train"]},
                  {"split": "test", **metrics["test"]}]).to_csv(metrics_path, index=False)
    imp_df.to_csv(imp_path, index=False)

    save_metrics2(
        "GradientBoosting_Price_r0.5",
        {"train": metrics["train"], "test": metrics["test"], "anchor_noise_frac": ANCHOR_NOISE_FRAC},
        filename="price_summary_metrics_gb_r0.5.json",
    )
    save_model(model, "gradient_boosting_price_r0.5.pkl")


if __name__ == "__main__":
    main()