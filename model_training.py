import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import joblib
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# =============================================================================
# 1. ADVANCED METRICS & EVALUATION
# =============================================================================
def evaluate_advanced(name, y_true, y_pred):
    """Comprehensive evaluation with safety checks."""
    y_pred_safe = np.clip(y_pred, 0, None)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred_safe))
    
    # MAPE (Mean Absolute Percentage Error) - Business friendly metric
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred_safe[mask]) / y_true[mask])) * 100
    
    print(f"\n{'='*50}")
    print(f"📊 MODEL: {name}")
    print(f"{'='*50}")
    print(f"  R² Score:      {r2:.4f}")
    print(f"  RMSLE:         {rmsle:.4f}")
    print(f"  RMSE:          {rmse:.2f}")
    print(f"  MAE:           {mae:.2f}")
    print(f"  MAPE:          {mape:.2f}%")
    print(f"{'='*50}\n")
    
    return {"r2": r2, "rmsle": rmsle, "rmse": rmse, "mae": mae, "mape": mape}

# =============================================================================
# 2. FEATURE ENGINEERING PIPELINE
# =============================================================================
def build_advanced_pipeline():
    """
    Builds a pipeline with:
    1. Yeo-Johnson target transformation (superior to log1p for bike data)
    2. Interaction features for temporal/weather variables
    3. HistGradientBoosting with early stopping built-in
    """
    # Key numeric features that benefit from interactions
    interaction_features = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed(m/s)']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), interaction_features)
        ],
        remainder='passthrough'  # Keep all other engineered features from preprocessing
    )
    
    regressor = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        loss='squared_error',      # Works well with Yeo-Johnson transformed target
        early_stopping=True,       # CRITICAL: Prevents overfitting during search
        validation_fraction=0.1,   # Use 10% of training data for early stopping
        n_iter_no_change=10,       # Stop if no improvement for 10 iterations
        warm_start=False
    )
    
    pipeline = Pipeline([
        ('interactions', preprocessor),
        ('regressor', TransformedTargetRegressor(
            regressor=regressor,
            transformer=YeoJohnsonTransformer()  # Learns optimal lambda from data
        ))
    ])
    
    return pipeline

# =============================================================================
# 3. SUCCESSIVE HALVING SEARCH (More efficient than RandomizedSearch)
# =============================================================================
def get_search_space():
    """
    Focused search space based on HGB best practices.
    Note: Parameters are prefixed with 'regressor__regressor__' because
    TransformedTargetRegressor wraps the actual estimator.
    """
    return {
        'regressor__regressor__max_iter': [300, 500, 800, 1000],
        'regressor__regressor__learning_rate': [0.03, 0.05, 0.08, 0.1],
        'regressor__regressor__max_depth': [6, 8, 10, None],
        'regressor__regressor__min_samples_leaf': [15, 20, 30, 50],
        'regressor__regressor__l2_regularization': [0.0, 0.1, 0.5, 1.0],
        'regressor__regressor__max_bins': [128, 255],  # Lower bins = faster + regularization
    }

def run_advanced_tuning():
    # Load preprocessed data
    train_df = pd.read_csv("processed_bike_data_train.csv")
    test_df = pd.read_csv("processed_bike_data_test.csv")
    
    X_train = train_df.drop(columns=["Rented Bike Count"])
    y_train = train_df["Rented Bike Count"]
    X_test = test_df.drop(columns=["Rented Bike Count"])
    y_test = test_df["Rented Bike Count"]
    
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Features after interaction expansion will be ~{X_train.shape[1] + 10}")
    
    pipeline = build_advanced_pipeline()
    param_grid = get_search_space()
    
    # HalvingGridSearchCV: Starts with many candidates on small data subsets,
    # progressively eliminates poor performers and uses more data for survivors
    search = HalvingGridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        factor=3,              # Eliminate 2/3 of candidates each round
        min_resources='exhaust',  # Auto-calculate minimum resources
        scoring='neg_mean_squared_log_error',  # Optimize directly for RMSLE
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True             # Refit best model on full training set
    )
    
    print("\n🚀 Starting Successive Halving Search...")
    search.fit(X_train, y_train)
    
    # Clean parameter names for display
    best_params = {k.replace('regressor__regressor__', ''): v 
                   for k, v in search.best_params_.items()}
    
    print(f"\n✅ Best CV RMSLE: {-search.best_score_:.4f}")
    print(f"✅ Best Parameters: {best_params}")
    
    # Evaluate on held-out test set
    y_pred = search.best_estimator_.predict(X_test)
    metrics = evaluate_advanced("HistGradientBoosting (Advanced)", y_test, y_pred)
    
    # Save artifacts
    artifact_path = "best_bike_model_advanced.pkl"
    joblib.dump(search.best_estimator_, artifact_path)
    print(f"💾 Model saved to: {artifact_path}")
    
    # Save metrics for tracking
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("model_metrics_advanced.csv", index=False)
    
    return search.best_estimator_, metrics

if __name__ == "__main__":
    best_model, metrics = run_advanced_tuning()