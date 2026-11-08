"""
Task 9b: Evaluation metrics for PM2.5 forecasting.

This module provides:
- Regression metrics: MAE, RMSE, R²
- Per-city metrics breakdown
- AQI category accuracy (practical usefulness metric)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Parameters
    ----------
    y_true : array-like
        True PM2.5 values (µg/m³).
    y_pred : array-like
        Predicted PM2.5 values (µg/m³).
    
    Returns
    -------
    float
        MAE in µg/m³. Lower is better.
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    RMSE penalizes large errors more than MAE.
    Important for smog spike prediction.
    
    Parameters
    ----------
    y_true : array-like
        True PM2.5 values (µg/m³).
    y_pred : array-like
        Predicted PM2.5 values (µg/m³).
    
    Returns
    -------
    float
        RMSE in µg/m³. Lower is better.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true, y_pred):
    """
    Calculate R² (coefficient of determination).
    
    R² indicates the proportion of variance explained by the model.
    
    Parameters
    ----------
    y_true : array-like
        True PM2.5 values (µg/m³).
    y_pred : array-like
        Predicted PM2.5 values (µg/m³).
    
    Returns
    -------
    float
        R² score. Range: (-∞, 1]. Higher is better. 1 = perfect.
    """
    return r2_score(y_true, y_pred)


def pm25_to_aqi_category(pm25):
    """
    Convert PM2.5 concentration to AQI category.
    
    Uses US EPA 24-hour PM2.5 AQI breakpoints.
    
    Parameters
    ----------
    pm25 : float or array-like
        PM2.5 concentration in µg/m³.
    
    Returns
    -------
    str or list of str
        AQI category name.
    
    Categories
    ----------
    - Good: 0.0 - 9.0 µg/m³
    - Moderate: 9.1 - 35.4 µg/m³
    - Unhealthy for Sensitive Groups: 35.5 - 55.4 µg/m³
    - Unhealthy: 55.5 - 125.4 µg/m³
    - Very Unhealthy: 125.5 - 225.4 µg/m³
    - Hazardous: 225.5+ µg/m³
    
    Notes
    -----
    This is a simplified implementation for metrics calculation.
    Full implementation with health advice will be in aqi.py (Task 14).
    """
    # Handle single value
    if np.isscalar(pm25):
        if pm25 <= 9.0:
            return "Good"
        elif pm25 <= 35.4:
            return "Moderate"
        elif pm25 <= 55.4:
            return "Unhealthy for Sensitive"
        elif pm25 <= 125.4:
            return "Unhealthy"
        elif pm25 <= 225.4:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    # Handle array
    pm25 = np.asarray(pm25)
    categories = np.empty(pm25.shape, dtype=object)
    
    categories[pm25 <= 9.0] = "Good"
    categories[(pm25 > 9.0) & (pm25 <= 35.4)] = "Moderate"
    categories[(pm25 > 35.4) & (pm25 <= 55.4)] = "Unhealthy for Sensitive"
    categories[(pm25 > 55.4) & (pm25 <= 125.4)] = "Unhealthy"
    categories[(pm25 > 125.4) & (pm25 <= 225.4)] = "Very Unhealthy"
    categories[pm25 > 225.4] = "Hazardous"
    
    return categories.tolist()


def calculate_aqi_accuracy(y_true, y_pred):
    """
    Calculate AQI category accuracy.
    
    This measures practical usefulness: what % of predictions
    fall in the correct health category?
    
    Parameters
    ----------
    y_true : array-like
        True PM2.5 values (µg/m³).
    y_pred : array-like
        Predicted PM2.5 values (µg/m³).
    
    Returns
    -------
    accuracy : float
        Proportion of correct category predictions (0-1).
    confusion : dict
        Confusion matrix as dict: {(true_cat, pred_cat): count}
    
    Examples
    --------
    >>> y_true = [50.0, 60.0, 150.0]
    >>> y_pred = [52.0, 58.0, 140.0]
    >>> acc, conf = calculate_aqi_accuracy(y_true, y_pred)
    >>> print(f"Accuracy: {acc:.2%}")
    Accuracy: 100.00%
    """
    true_cats = pm25_to_aqi_category(y_true)
    pred_cats = pm25_to_aqi_category(y_pred)
    
    # Calculate accuracy
    accuracy = sum(t == p for t, p in zip(true_cats, pred_cats)) / len(y_true)
    
    # Build confusion matrix
    from collections import defaultdict
    confusion = defaultdict(int)
    for t, p in zip(true_cats, pred_cats):
        confusion[(t, p)] += 1
    
    return accuracy, dict(confusion)


def calculate_metrics(y_true, y_pred):
    """
    Calculate all metrics at once.
    
    Parameters
    ----------
    y_true : array-like
        True PM2.5 values (µg/m³).
    y_pred : array-like
        Predicted PM2.5 values (µg/m³).
    
    Returns
    -------
    dict
        Dictionary with all metrics:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - r2: R² score
        - aqi_accuracy: AQI category accuracy
    """
    aqi_acc, _ = calculate_aqi_accuracy(y_true, y_pred)
    
    return {
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'aqi_accuracy': aqi_acc
    }


def calculate_per_city_metrics(df, y_true_col, y_pred_col, city_col='city'):
    """
    Calculate metrics separately for each city.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predictions and city column.
    y_true_col : str
        Column name for true values.
    y_pred_col : str
        Column name for predictions.
    city_col : str, default='city'
        Column name for city.
    
    Returns
    -------
    dict
        Nested dict: {city_name: {metric_name: value}}
    
    Examples
    --------
    >>> results_df['predictions'] = model.predict(X_test)
    >>> city_metrics = calculate_per_city_metrics(
    ...     results_df, 'target_pm25_t24h', 'predictions'
    ... )
    >>> print(city_metrics['karachi']['mae'])
    """
    city_metrics = {}
    
    for city in df[city_col].unique():
        city_data = df[df[city_col] == city]
        y_true = city_data[y_true_col]
        y_pred = city_data[y_pred_col]
        
        city_metrics[city] = calculate_metrics(y_true, y_pred)
    
    return city_metrics


def print_metrics(metrics, model_name="Model"):
    """
    Pretty print metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary returned by calculate_metrics().
    model_name : str, default="Model"
        Name to display.
    """
    print(f"\n{model_name} Performance:")
    print(f"  MAE:          {metrics['mae']:.2f} µg/m³")
    print(f"  RMSE:         {metrics['rmse']:.2f} µg/m³")
    print(f"  R²:           {metrics['r2']:.4f}")
    print(f"  AQI Accuracy: {metrics['aqi_accuracy']:.2%}")


def print_per_city_metrics(city_metrics):
    """
    Pretty print per-city metrics.
    
    Parameters
    ----------
    city_metrics : dict
        Dictionary returned by calculate_per_city_metrics().
    """
    print("\nPer-City Breakdown:")
    for city, metrics in city_metrics.items():
        print(f"\n  {city.title()}:")
        print(f"    MAE:          {metrics['mae']:.2f} µg/m³")
        print(f"    RMSE:         {metrics['rmse']:.2f} µg/m³")
        print(f"    R²:           {metrics['r2']:.4f}")
        print(f"    AQI Accuracy: {metrics['aqi_accuracy']:.2%}")


if __name__ == '__main__':
    """
    Test metrics with dummy data.
    Run: python -m src.metrics
    """
    print("=" * 60)
    print("Task 9b: Testing Metrics")
    print("=" * 60)
    
    # Test data
    y_true = np.array([50.0, 60.0, 150.0, 30.0, 200.0])
    y_pred = np.array([52.0, 58.0, 140.0, 35.0, 210.0])
    
    print("\nTest predictions:")
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        true_cat = pm25_to_aqi_category(t)
        pred_cat = pm25_to_aqi_category(p)
        match = "✓" if true_cat == pred_cat else "✗"
        print(f"  {i+1}. True: {t:6.1f} ({true_cat:25s}) | Pred: {p:6.1f} ({pred_cat:25s}) {match}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Model")
    
    # Test per-city metrics
    print("\n" + "-" * 60)
    print("Testing per-city metrics:")
    
    test_df = pd.DataFrame({
        'city': ['karachi', 'karachi', 'lahore', 'lahore', 'lahore'],
        'true': y_true,
        'pred': y_pred
    })
    
    city_metrics = calculate_per_city_metrics(test_df, 'true', 'pred')
    print_per_city_metrics(city_metrics)
    
    print("\n" + "=" * 60)
    print("Metrics test complete")
    print("=" * 60)
