"""
Tasks 10-11: Model training with hyperparameter tuning.

This script trains four models with TimeSeriesSplit cross-validation:
- Ridge Regression (Task 10)
- K-Nearest Neighbors (Task 10)
- Random Forest (Task 11)
- LightGBM (Task 11)

Each model is tuned with a quick hyperparameter grid and saved.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from lightgbm import LGBMRegressor

from src import config
from src.features import FEATURE_COLS, TARGET
from src.pipeline import build_pipeline
from src.metrics import calculate_metrics, print_metrics


# Create models directory
MODELS_DIR = config.ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    """
    Load train and validation data.
    
    Returns
    -------
    X_train, y_train, X_val, y_val : DataFrames and Series
    """
    print("Loading data...")
    train = pd.read_csv(config.PROCESSED_DIR / 'train.csv')
    val = pd.read_csv(config.PROCESSED_DIR / 'val.csv')
    
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_val = val[FEATURE_COLS]
    y_val = val[TARGET]
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Features: {len(FEATURE_COLS)}")
    
    return X_train, y_train, X_val, y_val


def train_ridge(X_train, y_train, X_val, y_val):
    """
    Train Ridge Regression with hyperparameter tuning.
    
    Ridge uses L2 regularization to prevent overfitting.
    Good for baseline linear model with many features.
    """
    print("\n" + "=" * 70)
    print("Training Ridge Regression (Task 10)")
    print("=" * 70)
    
    # Separate numeric and categorical features
    numeric_cols = [c for c in FEATURE_COLS if c != 'city']
    
    # Build pipeline
    pipeline = build_pipeline(Ridge(), numeric_cols)
    
    # Quick hyperparameter grid
    param_grid = {
        'model__alpha': [0.1, 1.0, 10.0]
    }
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    
    # Grid search
    print("Tuning hyperparameters with TimeSeriesSplit (5 folds)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.2f} µg/m³")
    
    # Evaluate on validation
    y_val_pred = grid_search.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    print_metrics(val_metrics, "Ridge (Validation)")
    
    # Save model
    model_path = MODELS_DIR / 'ridge_tuned.joblib'
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\n✓ Saved to: {model_path}")
    
    return grid_search.best_estimator_, val_metrics


def train_knn(X_train, y_train, X_val, y_val):
    """
    Train K-Nearest Neighbors with hyperparameter tuning.
    
    KNN is instance-based: predicts based on similar past days.
    Non-parametric and can capture complex patterns.
    """
    print("\n" + "=" * 70)
    print("Training K-Nearest Neighbors (Task 10)")
    print("=" * 70)
    
    # Separate numeric and categorical features
    numeric_cols = [c for c in FEATURE_COLS if c != 'city']
    
    # Build pipeline
    pipeline = build_pipeline(KNeighborsRegressor(), numeric_cols)
    
    # Quick hyperparameter grid
    param_grid = {
        'model__n_neighbors': [3, 5, 10, 15],
        'model__weights': ['uniform', 'distance']
    }
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    
    # Grid search
    print("Tuning hyperparameters with TimeSeriesSplit (5 folds)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.2f} µg/m³")
    
    # Evaluate on validation
    y_val_pred = grid_search.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    print_metrics(val_metrics, "KNN (Validation)")
    
    # Save model
    model_path = MODELS_DIR / 'knn_tuned.joblib'
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\n✓ Saved to: {model_path}")
    
    return grid_search.best_estimator_, val_metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest with hyperparameter tuning.
    
    Random Forest builds many decision trees and averages them.
    Handles non-linear relationships and feature interactions well.
    """
    print("\n" + "=" * 70)
    print("Training Random Forest (Task 11)")
    print("=" * 70)
    
    # Separate numeric and categorical features
    numeric_cols = [c for c in FEATURE_COLS if c != 'city']
    
    # Build pipeline
    pipeline = build_pipeline(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        numeric_cols
    )
    
    # Quick hyperparameter grid
    param_grid = {
        'model__n_estimators': [120],
        'model__max_depth': [12, 20],
        'model__min_samples_split': [5]
    }
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    
    # Grid search
    print("Tuning hyperparameters with TimeSeriesSplit (5 folds)...")
    print("(Optimized grid for robust training...)")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.2f} µg/m³")
    
    # Evaluate on validation
    y_val_pred = grid_search.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    print_metrics(val_metrics, "Random Forest (Validation)")
    
    # Save model
    model_path = MODELS_DIR / 'random_forest_tuned.joblib'
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\n✓ Saved to: {model_path}")
    
    return grid_search.best_estimator_, val_metrics


def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Train LightGBM with hyperparameter tuning.
    
    LightGBM is gradient boosting: builds trees sequentially,
    each correcting errors of previous trees. Fast and accurate.
    """
    print("\n" + "=" * 70)
    print("Training LightGBM (Task 11)")
    print("=" * 70)
    
    # Separate numeric and categorical features
    numeric_cols = [c for c in FEATURE_COLS if c != 'city']
    
    # Build pipeline
    pipeline = build_pipeline(
        LGBMRegressor(random_state=42, verbose=-1),
        numeric_cols
    )
    
    # Quick hyperparameter grid
    param_grid = {
        'model__n_estimators': [150, 300],
        'model__learning_rate': [0.03, 0.08],
        'model__num_leaves': [31, 45]
    }
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    
    # Grid search
    print("Tuning hyperparameters with TimeSeriesSplit (5 folds)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.2f} µg/m³")
    
    # Evaluate on validation
    y_val_pred = grid_search.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    print_metrics(val_metrics, "LightGBM (Validation)")
    
    # Save model
    model_path = MODELS_DIR / 'lightgbm_tuned.joblib'
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\n✓ Saved to: {model_path}")
    
    return grid_search.best_estimator_, val_metrics


def train_ensemble(ridge_model, rf_model, lgbm_model, X_train, y_train, X_val, y_val):
    """
    Train VotingRegressor ensemble combining Ridge, Random Forest, and LightGBM.
    """
    print("\n" + "=" * 70)
    print("Training Blended Ensemble (Ridge + RF + LightGBM)")
    print("=" * 70)
    
    ensemble = VotingRegressor(
        estimators=[
            ('ridge', ridge_model),
            ('rf', rf_model),
            ('lgbm', lgbm_model)
        ],
        weights=[0.2, 0.3, 0.5]
    )
    
    print("Fitting ensemble on training data...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate on validation
    y_val_pred = ensemble.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    print_metrics(val_metrics, "Ensemble (Validation)")
    
    # Save model
    model_path = MODELS_DIR / 'ensemble_tuned.joblib'
    joblib.dump(ensemble, model_path)
    print(f"\n✓ Saved to: {model_path}")
    
    return ensemble, val_metrics


def print_summary(all_metrics):
    """
    Print comparison table of all models.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 70)
    print(f"\n{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'AQI Acc':>10}")
    print("-" * 70)
    
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<20} "
              f"{metrics['mae']:>10.2f} "
              f"{metrics['rmse']:>10.2f} "
              f"{metrics['r2']:>8.4f} "
              f"{metrics['aqi_accuracy']:>9.1%}")
    
    print("-" * 70)
    
    # Find best model
    best_model = min(all_metrics.items(), key=lambda x: x[1]['mae'])
    print(f"\n✓ Best model by MAE: {best_model[0]}")
    print(f"  Validation MAE: {best_model[1]['mae']:.2f} µg/m³")


if __name__ == '__main__':
    """
    Train all models.
    Run: python -m src.train
    
    This will take approximately 20-25 minutes.
    """
    print("=" * 70)
    print("Tasks 10-11: Model Training with Hyperparameter Tuning")
    print("=" * 70)
    print("\nEstimated time: ~20-25 minutes")
    print("Models will be saved to: models/")
    
    # Load data
    X_train, y_train, X_val, y_val = load_data()
    
    # Store all results
    all_metrics = {}
    
    # Train each model
    best_ridge, ridge_metrics = train_ridge(X_train, y_train, X_val, y_val)
    all_metrics['Ridge'] = ridge_metrics
    
    _, knn_metrics = train_knn(X_train, y_train, X_val, y_val)
    all_metrics['KNN'] = knn_metrics
    
    best_rf, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    all_metrics['Random Forest'] = rf_metrics
    
    best_lgbm, lgbm_metrics = train_lightgbm(X_train, y_train, X_val, y_val)
    all_metrics['LightGBM'] = lgbm_metrics
    
    _, ensemble_metrics = train_ensemble(best_ridge, best_rf, best_lgbm, X_train, y_train, X_val, y_val)
    all_metrics['Ensemble'] = ensemble_metrics
    
    # Print summary
    print_summary(all_metrics)
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
