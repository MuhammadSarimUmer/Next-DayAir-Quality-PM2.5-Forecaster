"""
Task 12: Final model evaluation.

This script:
1. Loads all trained models
2. Compares performance on validation set
3. Selects the best model
4. Retrains winner on train+val combined
5. Evaluates ONCE on test set
6. Saves final pipeline
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src import config
from src.features import FEATURE_COLS, TARGET
from src.metrics import (
    calculate_metrics,
    calculate_per_city_metrics,
    print_metrics,
    print_per_city_metrics,
    pm25_to_aqi_category
)


MODELS_DIR = config.ROOT / 'models'

# Buffer zones around category boundaries (µg/m³)
BUFFER_SIZE = 3.0  # Conservative margin for uncertain predictions


def apply_buffer_zones(pm25_predictions):
    """
    Apply conservative buffer zones around AQI category boundaries.
    
    If prediction is within BUFFER_SIZE of a boundary, predict the
    worse (more conservative) category for safety.
    
    Boundaries:
    - 9.0: Good ↔ Moderate
    - 35.4: Moderate ↔ Unhealthy for Sensitive
    - 55.4: Unhealthy for Sensitive ↔ Unhealthy
    - 125.4: Unhealthy ↔ Very Unhealthy
    - 225.4: Very Unhealthy ↔ Hazardous
    
    Parameters
    ----------
    pm25_predictions : array-like
        Raw PM2.5 predictions from model
    
    Returns
    -------
    array
        Adjusted predictions with buffer zones applied
    """
    adjusted = np.array(pm25_predictions, dtype=float)
    
    # Near 35.4 boundary (Moderate ↔ Unhealthy for Sensitive)
    # If predicting 32.4-35.4, bump up to 35.5 (conservative)
    near_35 = (adjusted >= 35.4 - BUFFER_SIZE) & (adjusted <= 35.4)
    adjusted[near_35] = 35.5
    
    # Near 55.4 boundary (Unhealthy for Sensitive ↔ Unhealthy)
    # If predicting 52.4-55.4, bump up to 55.5 (conservative)
    near_55 = (adjusted >= 55.4 - BUFFER_SIZE) & (adjusted <= 55.4)
    adjusted[near_55] = 55.5
    
    # Near 125.4 boundary (Unhealthy ↔ Very Unhealthy)
    # If predicting 122.4-125.4, bump up to 125.5 (conservative)
    near_125 = (adjusted >= 125.4 - BUFFER_SIZE) & (adjusted <= 125.4)
    adjusted[near_125] = 125.5
    
    return adjusted


def load_trained_models():
    """Load all tuned models from Task 10-11."""
    models = {}
    
    model_files = {
        'Ridge': 'ridge_tuned.joblib',
        'KNN': 'knn_tuned.joblib',
        'Random Forest': 'random_forest_tuned.joblib',
        'LightGBM': 'lightgbm_tuned.joblib',
        'Ensemble': 'ensemble_tuned.joblib'
    }
    
    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    
    return models


def evaluate_all_models(models, X_val, y_val):
    """Evaluate all models on validation set."""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val, y_pred)
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics
        }
    
    return results


def print_comparison_table(results):
    """Print model comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 70)
    print(f"\n{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'AQI Acc':>10}")
    print("-" * 70)
    
    for name, data in results.items():
        m = data['metrics']
        print(f"{name:<20} {m['mae']:>10.2f} {m['rmse']:>10.2f} "
              f"{m['r2']:>8.4f} {m['aqi_accuracy']:>9.1%}")
    
    print("-" * 70)


def select_winner(results):
    """Select best model by validation MAE; prefer Ensemble when within 0.2 µg/m³."""
    winner_name = min(results.items(), key=lambda x: x[1]['metrics']['mae'])[0]
    best_mae = results[winner_name]["metrics"]["mae"]
    if "Ensemble" in results and results["Ensemble"]["metrics"]["mae"] <= best_mae + 0.2:
        winner_name = "Ensemble"
    winner_data = results[winner_name]
    
    print(f"\nSelected model: {winner_name}")
    print(f"Validation MAE: {winner_data['metrics']['mae']:.2f} µg/m³")
    
    return winner_name, winner_data['model']


def retrain_on_train_val(winner_model):
    """Retrain winner on train+val combined."""
    print("\n" + "=" * 70)
    print("Retraining on Train + Validation Combined")
    print("=" * 70)
    
    # Load train and val
    train = pd.read_csv(config.PROCESSED_DIR / 'train.csv')
    val = pd.read_csv(config.PROCESSED_DIR / 'val.csv')
    
    # Combine
    combined = pd.concat([train, val], ignore_index=True)
    X_combined = combined[FEATURE_COLS]
    y_combined = combined[TARGET]
    
    print(f"\nCombined training samples: {len(X_combined)}")
    
    # Retrain (winner_model is already a fitted pipeline, we use its structure)
    winner_model.fit(X_combined, y_combined)
    
    return winner_model


def evaluate_on_test(model, model_name):
    """Final evaluation on test set."""
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (Test Set - Run Once)")
    print("=" * 70)
    
    # Load test data
    test = pd.read_csv(config.PROCESSED_DIR / 'test.csv')
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET]
    
    print(f"\nTest samples: {len(X_test)}")
    
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, model_name)
    
    test['predictions'] = y_pred
    city_metrics = calculate_per_city_metrics(test, TARGET, 'predictions')
    print_per_city_metrics(city_metrics)
    
    # Apply buffer zones and re-evaluate AQI accuracy
    print("\n" + "-" * 70)
    print("With Buffer Zones (Conservative Boundaries)")
    print("-" * 70)
    
    y_pred_buffered = apply_buffer_zones(y_pred)
    metrics_buffered = calculate_metrics(y_test, y_pred_buffered)
    
    # Show improvement
    aqi_improvement = (metrics_buffered['aqi_accuracy'] - metrics['aqi_accuracy']) * 100
    print(f"\nAQI Accuracy: {metrics['aqi_accuracy']:.2%} → {metrics_buffered['aqi_accuracy']:.2%} "
          f"({aqi_improvement:+.1f} percentage points)")
    print(f"MAE:          {metrics['mae']:.2f} → {metrics_buffered['mae']:.2f} µg/m³")
    print(f"(Slight MAE increase is expected - conservative predictions)")
    
    return metrics, city_metrics, metrics_buffered


def compare_with_baseline():
    """Load and display baseline results for comparison."""
    print("\n" + "=" * 70)
    print("Baseline Comparison")
    print("=" * 70)
    
    # From baseline.py results (Task 9)
    baseline_test = {
        'mae': 14.54,
        'rmse': 26.98,
        'r2': 0.5700,
        'aqi_accuracy': 0.6999
    }
    
    print("\nBaseline (Persistence) - Test Set:")
    print(f"  MAE:          {baseline_test['mae']:.2f} µg/m³")
    print(f"  RMSE:         {baseline_test['rmse']:.2f} µg/m³")
    print(f"  R²:           {baseline_test['r2']:.4f}")
    print(f"  AQI Accuracy: {baseline_test['aqi_accuracy']:.2%}")
    
    return baseline_test


if __name__ == '__main__':
    """
    Run final evaluation.
    Run: python -m src.evaluate
    """
    print("=" * 70)
    print("Task 12: Final Model Evaluation")
    print("=" * 70)
    
    # Load validation data
    val = pd.read_csv(config.PROCESSED_DIR / 'val.csv')
    X_val = val[FEATURE_COLS]
    y_val = val[TARGET]
    
    # Load all trained models
    print("\nLoading trained models...")
    models = load_trained_models()
    print(f"Loaded {len(models)} models")
    
    # Evaluate all on validation
    print("\nEvaluating on validation set...")
    results = evaluate_all_models(models, X_val, y_val)
    
    # Print comparison
    print_comparison_table(results)
    
    # Select winner
    winner_name, winner_model = select_winner(results)
    
    # Retrain on train+val
    final_model = retrain_on_train_val(winner_model)
    
    # Evaluate on test (ONCE)
    test_metrics, test_city_metrics, test_metrics_buffered = evaluate_on_test(final_model, winner_name)
    
    # Compare with baseline
    baseline_metrics = compare_with_baseline()
    
    # Calculate improvements
    print("\n" + "=" * 70)
    print("Improvement Over Baseline")
    print("=" * 70)
    mae_improvement = (baseline_metrics['mae'] - test_metrics['mae']) / baseline_metrics['mae'] * 100
    rmse_improvement = (baseline_metrics['rmse'] - test_metrics['rmse']) / baseline_metrics['rmse'] * 100
    aqi_improvement = (test_metrics['aqi_accuracy'] - baseline_metrics['aqi_accuracy']) / baseline_metrics['aqi_accuracy'] * 100
    
    print(f"\nMAE:          {baseline_metrics['mae']:.2f} → {test_metrics['mae']:.2f} µg/m³ ({mae_improvement:+.1f}%)")
    print(f"RMSE:         {baseline_metrics['rmse']:.2f} → {test_metrics['rmse']:.2f} µg/m³ ({rmse_improvement:+.1f}%)")
    print(f"R²:           {baseline_metrics['r2']:.4f} → {test_metrics['r2']:.4f}")
    print(f"AQI Accuracy: {baseline_metrics['aqi_accuracy']:.1%} → {test_metrics['aqi_accuracy']:.1%} ({aqi_improvement:+.1f}%)")
    
    # Save final model
    final_model_path = MODELS_DIR / 'final_pipeline.joblib'
    joblib.dump(final_model, final_model_path)
    
    print("\n" + "=" * 70)
    print(f"Final model saved: {final_model_path}")
    print("=" * 70)
    
    # Summary
    print("\nSummary:")
    print(f"  - Selected model: {winner_name}")
    print(f"  - Test MAE: {test_metrics['mae']:.2f} µg/m³")
    print(f"  - Test AQI Accuracy (standard): {test_metrics['aqi_accuracy']:.1%}")
    print(f"  - Test AQI Accuracy (with buffers): {test_metrics_buffered['aqi_accuracy']:.1%}")
    print(f"  - Improvement over baseline: {mae_improvement:.1f}% (MAE)")
