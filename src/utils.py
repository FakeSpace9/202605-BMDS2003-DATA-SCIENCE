

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
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