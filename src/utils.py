

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
RAW_PATH = project_root / "data" / "raw" / "Gold_Price.csv"
CLEANED_PATH = project_root / "data" / "processed" / "Gold_Price_cleaned.csv"
TRAIN_PATH = project_root / "data" / "processed"/"Gold_Price_train.csv"
TEST_PATH = project_root / "data" / "processed"/"Gold_Price_test.csv"
MODEL_OUTPUT_PATH = project_root / "prototype"
METRICS_OUTPUT_PATH = project_root / "prototype" / "metrics"

def print_metrics(name, y_train, y_train_pred, y_test, y_pred):

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_pred)

# Prevent negative values from breaking the log calculations
    y_train_pred_safe = np.maximum(y_train_pred, 0)
    y_pred_safe = np.maximum(y_pred, 0)

    # Log RMSE (RMSLE)
    train_log_rmse = np.sqrt(mean_squared_log_error(y_train, y_train_pred_safe))
    test_log_rmse = np.sqrt(mean_squared_log_error(y_test, y_pred_safe))

    # Log MAE
    train_log_mae = mean_absolute_error(np.log1p(y_train), np.log1p(y_train_pred_safe))
    test_log_mae = mean_absolute_error(np.log1p(y_test), np.log1p(y_pred_safe))

    print(f"\n{name} — Train vs Test Performance")
    print("-" * 40)
    print(f"Train MAE : {train_mae:,.2f}")
    print(f"Test MAE  : {test_mae:,.2f}")
    print(f"Train RMSE: {train_rmse:,.2f}")
    print(f"Test RMSE : {test_rmse:,.2f}")
    print(f"Train Log MAE  : {train_log_mae:.4f}")
    print(f"Test Log MAE   : {test_log_mae:.4f}")
    print(f"Train Log RMSE : {train_log_rmse:.4f}")
    print(f"Test Log RMSE  : {test_log_rmse:.4f}")
    print(f"Train MAPE: {train_mape * 100:.2f}%")
    print(f"Test MAPE : {test_mape * 100:.2f}%")
    print(f"Train R\u00b2  : {train_r2:.4f}")
    print(f"Test R\u00b2   : {test_r2:.4f}")

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_log_mae": train_log_mae,
        "test_log_mae": test_log_mae,
        "train_log_rmse": train_log_rmse,
        "test_log_rmse": test_log_rmse,
        "train_mape": train_mape,
        "test_mape": test_mape,
    }

def _safe_log_metrics(y_true, y_pred):
    """
    Log-scale MAE/RMSE. Only meaningful for strictly positive series (e.g.
    price levels) -- log() of a negative or zero value is undefined. If any
    values are <= 0 (as Chg% will be, since price can fall), this returns
    NaN rather than silently producing garbage or crashing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if (y_true <= 0).any() or (y_pred <= 0).any():
        return np.nan, np.nan
    log_true, log_pred = np.log(y_true), np.log(y_pred)
    log_mae = mean_absolute_error(log_true, log_pred)
    log_rmse = np.sqrt(mean_squared_error(log_true, log_pred))
    return log_mae, log_rmse
 
 
def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Avoid divide-by-near-zero blowing up the average; exclude rows where
    # the actual value is essentially 0 (this matters for Chg%, which
    # regularly sits near 0, unlike Price which never does).
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return np.nan
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100

def _metric_block(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": _mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

def print_metrics2(model_name, y_train, y_train_pred, y_test, y_pred,
                   price_lag1_train=None, price_lag1_test=None):
    """
    Prints and returns MAE, RMSE, MAPE, R2 for train and test.
 
    Log MAE / log RMSE are only valid on a strictly positive series. If the
    target itself (y_train/y_test) is strictly positive -- e.g. predicting
    Price directly -- they're computed on the target. If the target can go
    negative (e.g. Chg%), pass price_lag1_train/test (the previous day's
    actual price) and this will reconstruct predicted price as
    Price_Lag1 * (1 + pred/100), then compute log MAE/RMSE on that
    reconstructed, always-positive price series instead.
    """
    train_metrics = _metric_block(y_train, y_train_pred)
    test_metrics = _metric_block(y_test, y_pred)
 
    target_is_positive = (np.asarray(y_train) > 0).all() and (np.asarray(y_test) > 0).all()
 
    if target_is_positive:
        log_mae_train, log_rmse_train = _safe_log_metrics(y_train, y_train_pred)
        log_mae_test, log_rmse_test = _safe_log_metrics(y_test, y_pred)
        log_note = "(computed directly on target)"
    elif price_lag1_train is not None and price_lag1_test is not None:
        recon_train = np.asarray(price_lag1_train) * (1 + np.asarray(y_train_pred) / 100)
        recon_train_actual = np.asarray(price_lag1_train) * (1 + np.asarray(y_train) / 100)
        recon_test = np.asarray(price_lag1_test) * (1 + np.asarray(y_pred) / 100)
        recon_test_actual = np.asarray(price_lag1_test) * (1 + np.asarray(y_test) / 100)
        log_mae_train, log_rmse_train = _safe_log_metrics(recon_train_actual, recon_train)
        log_mae_test, log_rmse_test = _safe_log_metrics(recon_test_actual, recon_test)
        log_note = "(target goes negative -- computed on reconstructed Price instead, see docstring)"
    else:
        log_mae_train = log_rmse_train = log_mae_test = log_rmse_test = np.nan
        log_note = "(target goes negative and no price_lag1 given -- skipped)"
 
    train_metrics["LogMAE"] = log_mae_train
    train_metrics["LogRMSE"] = log_rmse_train
    test_metrics["LogMAE"] = log_mae_test
    test_metrics["LogRMSE"] = log_rmse_test
 
    print(f"\n=== {model_name} — metrics {log_note} ===")
    header = f"{'Metric':<10}{'Train':>15}{'Test':>15}"
    print(header)
    for key in ["MAE", "RMSE", "MAPE", "LogMAE", "LogRMSE", "R2"]:
        t, v = train_metrics[key], test_metrics[key]
        t_str = f"{t:,.4f}" if pd.notna(t) else "NaN"
        v_str = f"{v:,.4f}" if pd.notna(v) else "NaN"
        print(f"{key:<10}{t_str:>15}{v_str:>15}")
 
    return {"train": train_metrics, "test": test_metrics}
 
 
def save_metrics2(model_name, metrics, filename=None):
    filename = filename or f"{model_name.lower().replace(' ', '_')}_metrics.json"
    path = METRICS_OUTPUT_PATH / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    print(f"\nSaved metrics to {path}")
    return path

def load_raw_dataset():
    print("Loading raw dataset...")
    try:
        return pd.read_csv(RAW_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}")

def load_cleaned_dataset():
    

    print("Loading processed dataset...")
    try:
        return pd.read_csv(CLEANED_PATH,parse_dates=["Date"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {CLEANED_PATH}")

def load_splits():
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test_df = pd.read_csv(TEST_PATH, parse_dates=["Date"])
    return train_df, test_df

def save_model(model, model_name: str):

    joblib.dump(model, MODEL_OUTPUT_PATH/model_name)

    print(f"Model saved to {MODEL_OUTPUT_PATH}/{model_name}")


def save_metrics(model_name: str, metrics: dict):
    # Ensure the target directory exists
    METRICS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Sanitize the model name to create a safe filename
    safe_filename = f"{model_name.replace('/', '_').replace(' ', '_')}.json"
    output_path = METRICS_OUTPUT_PATH / safe_filename

    # Save the specific model metrics to its own file
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics for {model_name} saved to {output_path}")