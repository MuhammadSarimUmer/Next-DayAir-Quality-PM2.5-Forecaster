"""
Evaluate all 6 models on the held-out test set and generate comparison table.

Models:
1. Baseline (persistence)
2. Prophet (Karachi)
3. Prophet (Lahore)
4. ARIMAX (Karachi)
5. ARIMAX (Lahore)
6. Ensemble (VotingRegressor)

Run: python -m src.evaluate_all
"""

import pandas as pd
import joblib
import numpy as np
from src import config
from src.features import FEATURE_COLS, TARGET
from src.metrics import calculate_metrics, print_metrics

MODELS_DIR = config.ROOT / 'models'


def evaluate_prophet(test_df, city):
    """Evaluate Prophet model for one city."""
    from src.train_prophet import prepare_prophet_data
    
    model = joblib.load(MODELS_DIR / f'prophet_{city}.pkl')
    test_city, _ = prepare_prophet_data(test_df, city)
    
    forecast = model.predict(test_city)
    y_true = test_city['y'].values
    y_pred = forecast['yhat'].values
    
    return calculate_metrics(y_true, y_pred)


def evaluate_arimax(test_df, city):
    """Evaluate ARIMAX model for one city."""
    from src.train_arimax import prepare_arimax_data
    
    model_path = MODELS_DIR / f'arimax_{city}.pkl'
    if not model_path.exists():
        print(f"  ⚠️  ARIMAX model not found for {city} (training failed)")
        return None
    
    model = joblib.load(model_path)
    y_true, exog = prepare_arimax_data(test_df, city)
    
    y_pred = model.predict(n_periods=len(y_true), exogenous=exog)
    
    return calculate_metrics(y_true, y_pred)


def evaluate_ensemble(test_df):
    """Evaluate sklearn ensemble on full test set (both cities)."""
    model = joblib.load(MODELS_DIR / 'final_pipeline.joblib')
    
    X_test = test_df[FEATURE_COLS]
    y_true = test_df[TARGET]
    y_pred = model.predict(X_test)
    
    return calculate_metrics(y_true, y_pred)


def evaluate_baseline(test_df):
    """Persistence baseline: tomorrow = today."""
    y_true = test_df[TARGET].values
    y_pred = test_df['pm25_now'].values
    
    return calculate_metrics(y_true, y_pred)


def main():
    print("="*70)
    print("Evaluating All Models on Test Set")
    print("="*70)
    
    # Load test data
    test = pd.read_csv(config.PROCESSED_DIR / 'test.csv', parse_dates=['datetime_pkt'])
    
    results = {}
    
    # 1. Baseline
    print("\n[1/6] Baseline (Persistence)...")
    results['Baseline'] = evaluate_baseline(test)
    
    # 2-3. Prophet per city
    print("\n[2/6] Prophet (Karachi)...")
    results['Prophet (Karachi)'] = evaluate_prophet(test, 'karachi')
    
    print("\n[3/6] Prophet (Lahore)...")
    results['Prophet (Lahore)'] = evaluate_prophet(test, 'lahore')
    
    # Average Prophet
    prophet_avg = {
        k: (results['Prophet (Karachi)'][k] + results['Prophet (Lahore)'][k]) / 2
        for k in ['mae', 'rmse', 'r2', 'aqi_accuracy']
    }
    results['Prophet (Average)'] = prophet_avg
    
    # 4-5. ARIMAX per city
    print("\n[4/6] ARIMAX (Karachi)...")
    results['ARIMAX (Karachi)'] = evaluate_arimax(test, 'karachi')
    
    print("\n[5/6] ARIMAX (Lahore)...")
    results['ARIMAX (Lahore)'] = evaluate_arimax(test, 'lahore')
    
    # Average ARIMAX (only if both succeeded)
    if results['ARIMAX (Karachi)'] and results['ARIMAX (Lahore)']:
        arimax_avg = {
            k: (results['ARIMAX (Karachi)'][k] + results['ARIMAX (Lahore)'][k]) / 2
            for k in ['mae', 'rmse', 'r2', 'aqi_accuracy']
        }
        results['ARIMAX (Average)'] = arimax_avg
    else:
        results['ARIMAX (Average)'] = None
        print("  ⚠️  ARIMAX average skipped (one or both cities failed)")
    
    # 6. Ensemble
    print("\n[6/6] Ensemble (VotingRegressor)...")
    results['Ensemble'] = evaluate_ensemble(test)
    
    # Print comparison table
    print("\n" + "="*70)
    print("TEST SET RESULTS (All Models)")
    print("="*70)
    print(f"{'Model':<30} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'AQI Acc':>10}")
    print("-"*70)
    
    for name in ['Baseline', 'Prophet (Average)', 'ARIMAX (Average)', 'Ensemble']:
        m = results[name]
        if m is None:
            print(f"{name:<30} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>10}")
        else:
            print(f"{name:<30} {m['mae']:>10.2f} {m['rmse']:>10.2f} {m['r2']:>8.4f} {m['aqi_accuracy']:>9.1%}")
    
    print("="*70)
    
    # Save results
    import json
    output = {
        name: {k: float(v) for k, v in metrics.items()}
        for name, metrics in results.items()
    }
    
    output_path = config.ROOT / 'reports' / 'all_models_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved results: {output_path}")


if __name__ == '__main__':
    main()
