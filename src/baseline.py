"""
Task 9a: Baseline model for PM2.5 forecasting.

Baseline: "Tomorrow = Now" (persistence forecast)
Predicts that PM2.5 24 hours from now will equal current PM2.5.

This is a strong baseline for air quality because PM2.5 has high
24-hour persistence (correlation ~0.85). Any ML model must beat this.
"""

import pandas as pd
from src import config
from src.metrics import (
    calculate_metrics, 
    calculate_per_city_metrics,
    print_metrics,
    print_per_city_metrics
)


def persistence_forecast(df, current_pm25_col='pm25_now', target_col='target_pm25_t24h'):
    """
    Persistence baseline: predict target = current PM2.5.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with current PM2.5 and target columns.
    current_pm25_col : str, default='pm25_now'
        Column name for current PM2.5 (used as prediction).
    target_col : str, default='target_pm25_t24h'
        Column name for true 24h-ahead PM2.5.
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with added 'baseline_pred' column.
    """
    df = df.copy()
    df['baseline_pred'] = df[current_pm25_col]
    return df


def evaluate_baseline(split='val'):
    """
    Evaluate persistence baseline on a data split.
    
    Parameters
    ----------
    split : str, default='val'
        Which split to evaluate: 'train', 'val', or 'test'.
    
    Returns
    -------
    metrics : dict
        Overall metrics.
    city_metrics : dict
        Per-city metrics.
    """
    # Load data
    data_file = config.PROCESSED_DIR / f'{split}.csv'
    df = pd.read_csv(data_file)
    
    print(f"\nLoaded {split} split: {len(df)} rows")
    print(f"Cities: {df['city'].value_counts().to_dict()}")
    
    # Generate baseline predictions
    df = persistence_forecast(df)
    
    # Calculate overall metrics
    metrics = calculate_metrics(
        df['target_pm25_t24h'], 
        df['baseline_pred']
    )
    
    # Calculate per-city metrics
    city_metrics = calculate_per_city_metrics(
        df, 
        'target_pm25_t24h', 
        'baseline_pred'
    )
    
    return metrics, city_metrics


if __name__ == '__main__':
    """
    Run baseline evaluation on all splits.
    Run: python -m src.baseline
    """
    print("=" * 70)
    print("Task 9: Baseline Model Evaluation")
    print("Baseline: 'Tomorrow = Now' (Persistence Forecast)")
    print("=" * 70)
    
    # Evaluate on train, val, test
    for split in ['train', 'val', 'test']:
        print(f"\n{'=' * 70}")
        print(f"Evaluating on {split.upper()} set")
        print("=" * 70)
        
        metrics, city_metrics = evaluate_baseline(split)
        
        # Print results
        print_metrics(metrics, f"Baseline ({split.upper()})")
        print_per_city_metrics(city_metrics)
    
    print("\n" + "=" * 70)
    print("Baseline Evaluation Complete")
    print("=" * 70)
