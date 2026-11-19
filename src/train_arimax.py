"""
Train ARIMAX models per city for classical time-series baseline.

ARIMAX = ARIMA with eXogenous variables (weather).
Uses pmdarima's auto_arima to find best (p,d,q) hyperparameters.

Run: python -m src.train_arimax
"""

import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')  # ARIMA prints many convergence warnings

from pmdarima import auto_arima
from src import config
from src.features import TARGET
from src.metrics import calculate_metrics, print_metrics

MODELS_DIR = config.ROOT / 'models'


def prepare_arimax_data(df, city):
    """
    Extract time-series + exogenous variables for one city.
    
    Returns
    -------
    y : array
        Target time-series (PM2.5 24h ahead)
    exog : DataFrame
        Exogenous variables (weather, CAMS)
    """
    city_df = df[df['city'] == city].copy().sort_values('datetime_pkt')
    
    y = city_df[TARGET].values
    
    # ARIMAX works best with 3-7 exogenous variables (too many = overfitting)
    exog_cols = [
        'temperature_2m',
        'relative_humidity_2m',
        'wind_speed_10m',
        'cams_pm2_5',
        'pm25_lag_24h'  # Recent PM2.5 as exog variable
    ]
    
    exog = city_df[exog_cols]
    
    return y, exog


def train_arimax_for_city(train_df, val_df, city):
    """
    Train ARIMAX with auto hyperparameter search.
    
    auto_arima tries different (p,d,q) combinations and picks best AIC.
    """
    print(f"\n{'='*70}")
    print(f"Training ARIMAX for {city.title()}")
    print('='*70)
    
    # Prepare data
    y_train, exog_train = prepare_arimax_data(train_df, city)
    y_val, exog_val = prepare_arimax_data(val_df, city)
    
    print(f"  Train: {len(y_train)} timesteps")
    print(f"  Val:   {len(y_val)} timesteps")
    print(f"  Exog:  {exog_train.shape[1]} variables")
    
    # Auto-search for best ARIMA orders
    print("  Searching for best ARIMA(p,d,q)(P,D,Q,24) orders...")
    print("  (This takes 2-3 minutes per city)")
    
    model = auto_arima(
        y_train,
        exogenous=exog_train,
        seasonal=True,
        m=12,  # ← Changed from 24 to 12 (half-day seasonality, uses 1/4 the RAM)
        start_p=0, max_p=2,  # ← Reduced from 3
        start_q=0, max_q=2,  # ← Reduced from 3
        d=None,
        start_P=0, max_P=1,  # ← Reduced from 2
        start_Q=0, max_Q=1,  # ← Reduced from 2
        D=None,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )
    
    print(f"\n  Best model: {model.order} x {model.seasonal_order}")
    print(f"  AIC: {model.aic():.2f}")
    
    # Predict on validation
    print("  Predicting on validation...")
    y_pred = model.predict(n_periods=len(y_val), exogenous=exog_val)
    
    # Calculate metrics
    val_metrics = calculate_metrics(y_val, y_pred)
    print_metrics(val_metrics, f"ARIMAX ({city.title()})")
    
    return model, val_metrics


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    train = pd.read_csv(config.PROCESSED_DIR / 'train.csv', parse_dates=['datetime_pkt'])
    val = pd.read_csv(config.PROCESSED_DIR / 'val.csv', parse_dates=['datetime_pkt'])
    
    results = {}
    
    # Train per city
    for city in ['karachi', 'lahore']:
        model, metrics = train_arimax_for_city(train, val, city)
        
        # Save
        model_path = MODELS_DIR / f'arimax_{city}.pkl'
        joblib.dump(model, model_path)
        print(f"  Saved: {model_path}")
        
        results[city] = metrics
    
    # Overall validation metrics (average both cities)
    overall_mae = (results['karachi']['mae'] + results['lahore']['mae']) / 2
    overall_r2 = (results['karachi']['r2'] + results['lahore']['r2']) / 2
    overall_aqi = (results['karachi']['aqi_accuracy'] + results['lahore']['aqi_accuracy']) / 2
    
    print("\n" + "="*70)
    print("ARIMAX Overall Validation Results (Average)")
    print("="*70)
    print(f"  MAE:          {overall_mae:.2f} µg/m³")
    print(f"  R²:           {overall_r2:.4f}")
    print(f"  AQI Accuracy: {overall_aqi:.2%}")
    print("\nNote: ARIMAX assumes linear relationships. Tree models (RF, LightGBM)")
    print("      capture non-linear interactions better for this 24h horizon.")


if __name__ == '__main__':
    main()
