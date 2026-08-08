"""
train_linear_regression_r2_0.5.py
Gold Price Prediction — CRISP-DM: Modelling & Evaluation phase

Produces a deliberately moderate-skill model (train R2 ~ 0.69, test R2
~ 0.48 on your actual data) instead of the near-perfect fit the full
feature set gives (train R2 ~0.98, test R2 ~0.97).

WHY A STATIC SPLIT INSTEAD OF WALK-FORWARD:
    I tried this on the expanding-window walk-forward setup first. It
    doesn't work: gold trended from ~$1,300 to ~$3,000+ across this
    dataset, so once a fold's model is weakened enough to stop fitting
    almost perfectly, it can't track the *current* price regime either,
    and test R2 on that fold's narrow one-year price range goes
    catastrophically negative (-14 to -100 in testing) rather than
    settling around a moderate value. Averaging those blown-up folds
    with the folds that still fit well produces a "0.5" that's a
    statistical illusion, not a real moderate-skill model. A single
    75/25 chronological split doesn't have this regime-shift-per-fold
    problem, so a moderate R2 is actually achievable and stable there.

WHY R2 WAS NEAR-PERFECT IN THE FIRST PLACE:
    Price_Lag30 gets a coefficient near 1.0 -- the model is mostly just
    echoing "the price ~30 days ago" back out. MACD, BB_Width and
    ATR_14 are computed in absolute price units, so they leak the same
    kind of price-level information.

WHAT THIS SCRIPT DOES:
    1. Drops MACD / BB_Width / ATR_14 (absolute price-scale features).
    2. Degrades Price_Lag30 with multiplicative Gaussian noise
       (ANCHOR_NOISE_FRAC = 0.18, tuned empirically against your data
       to balance train and test R2 near 0.5 -- see the grid search
       printed at the bottom if you want to retune it).
    3. Fits on a single chronological 75/25 split (matching the
       original train_linear_regression.py split style), not
       walk-forward.

NOTE FOR YOUR WRITE-UP: this is a controlled way to produce a
moderate-skill comparison model. It is not something you'd do to a
model you intend to actually deploy -- say so explicitly if you
write this up, so it doesn't read as if 0.5 is this feature set's
natural, undoctored performance.

Run order:
    1. python preprocessing.py                       (creates Gold_Price_cleaned.csv)
    2. python train_linear_regression_r2_0.5.py       (this script)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import print_metrics2, save_model, save_metrics2, load_cleaned_dataset

current_dir = Path(__file__).resolve().parent.parent
project_root = current_dir.parent
MODEL_OUTPUT_PATH = project_root / "prototype"

TARGET = "Price"
ANCHOR = "Price_Lag30"
ANCHOR_NOISY = "Price_Lag30_noisy"
RANDOM_SEED = 42

# Tuned empirically on this dataset (static 75/25 split) to balance
# train/test R2 near 0.5. Larger = weaker model. Re-run the grid search
# at the bottom of this file if you change the feature set or split.
ANCHOR_NOISE_FRAC = 0.18

# MACD / BB_Width / ATR_14 dropped -- absolute price-scale, same
# "encodes the level" problem the undegraded anchor had.
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

TEST_SIZE = 0.25


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
    print(f"Train: {train_df['Date'].min().date()} -> {train_df['Date'].max().date()} "
          f"(n={len(train_df)})")
    print(f"Test : {test_df['Date'].min().date()} -> {test_df['Date'].max().date()} "
          f"(n={len(test_df)})")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    metrics = print_metrics2(
        "LinReg_r0.5", y_train, y_train_pred, y_test, y_pred,
        price_lag1_train=train_df["Price_Lag1"],
        price_lag1_test=test_df["Price_Lag1"],
    )

    print(f"\nTrain R2: {metrics['train']['R2']:.4f}")
    print(f"Test R2 : {metrics['test']['R2']:.4f}")

    coef_df = pd.DataFrame([{
        "Intercept": model.intercept_,
        **dict(zip(FEATURES, model.coef_)),
    }])
    print("\n=== Coefficients ===")
    print(coef_df.round(6).to_string(index=False))

    metrics_path = MODEL_OUTPUT_PATH / "price_fold_metrics_r0.5.csv"
    coef_path = MODEL_OUTPUT_PATH / "price_coefficients_r0.5.csv"
    pd.DataFrame([{"split": "train", **metrics["train"]},
                  {"split": "test", **metrics["test"]}]).to_csv(metrics_path, index=False)
    coef_df.to_csv(coef_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved coefficients to {coef_path}")

    save_metrics2(
        "LinearRegression_Price_r0.5",
        {"train": metrics["train"], "test": metrics["test"], "anchor_noise_frac": ANCHOR_NOISE_FRAC},
        filename="price_summary_metrics_r0.5.json",
    )
    save_model(model, "linear_regression_price_r0.5.pkl")


def grid_search_anchor_noise(noise_grid=None):
    """Optional: retune ANCHOR_NOISE_FRAC if you change features/split.
    Prints train/test R2 for each candidate noise level so you can pick
    a new value by hand. Not called automatically -- run manually:
        python -c "from train_linear_regression_r2_0.5 import grid_search_anchor_noise as g; g()"
    """
    noise_grid = noise_grid or np.arange(0.10, 0.30, 0.01)
    df = load_cleaned_dataset()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    split = int(len(df) * (1 - TEST_SIZE))

    print(f"{'noise':>6} {'train_R2':>10} {'test_R2':>10} {'dist_from_0.5':>15}")
    for nf in noise_grid:
        d = add_noisy_anchor(df, nf, RANDOM_SEED)
        train_df, test_df = d.iloc[:split], d.iloc[split:]
        m = LinearRegression().fit(train_df[FEATURES], train_df[TARGET])
        tr_r2 = r2_score(train_df[TARGET], m.predict(train_df[FEATURES]))
        te_r2 = r2_score(test_df[TARGET], m.predict(test_df[FEATURES]))
        dist = abs(tr_r2 - 0.5) + abs(te_r2 - 0.5)
        print(f"{nf:>6.3f} {tr_r2:>10.4f} {te_r2:>10.4f} {dist:>15.4f}")


if __name__ == "__main__":
    main()