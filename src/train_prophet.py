"""
Train Prophet models per city for time-series baseline comparison.

Prophet models trend + seasonality + regressors (weather).
One model per city since Prophet can't handle categorical city features.

Run: python -m src.train_prophet
"""

import pandas as pd
import joblib
from prophet import Prophet
from src import config
from src.features import TARGET
from src.metrics import calculate_metrics, print_metrics

MODELS_DIR = config.ROOT / 'models'


def prepare_prophet_data(df, city):
    """
    Convert sklearn format to Prophet format.
    
    Prophet expects:
    - 'ds': datetime column
    - 'y': target column
    - regressors: additional columns
    """
    city_df = df[df['city'] == city].copy()
    
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(city_df['datetime_pkt']),
        'y': city_df[TARGET]
    })
    
    # Add top weather/CAMS regressors (Prophet handles 5-10 best)
    regressors = [
        'temperature_2m',
        'relative_humidity_2m', 
        'wind_speed_10m',
        'precipitation',
        'cams_pm2_5',
        'pm25_lag_24h',  # Prophet can use this as a regressor
        'pm25_roll_mean_24h'
    ]
    
    for col in regressors:
        prophet_df[col] = city_df[col].values
    
    return prophet_df, regressors


def train_prophet_for_city(train_df, val_df, city):
    """
    Train one Prophet model for one city.
    
    Returns
    -------
    model : Prophet
        Fitted model
    val_metrics : dict
        Validation metrics
    """
    print(f"\n{'='*70}")
    print(f"Training Prophet for {city.title()}")
    print('='*70)
    
    # Prepare data
    train_prophet, regressors = prepare_prophet_data(train_df, city)
    val_prophet, _ = prepare_prophet_data(val_df, city)
    
    print(f"  Train: {len(train_prophet)} rows")
    print(f"  Val:   {len(val_prophet)} rows")
    print(f"  Regressors: {len(regressors)}")
    
    # Initialize Prophet
    model = Prophet(
        daily_seasonality=True,   # Captures hourly patterns
        weekly_seasonality=True,  # Captures weekday patterns
        yearly_seasonality=False, # Not enough data (only 16 months)
        changepoint_prior_scale=0.05,  # Conservative trend changes
        seasonality_prior_scale=10.0,  # Flexible seasonality
    )
    
    # Add regressors
    for col in regressors:
        model.add_regressor(col)
    
    # Fit (Prophet prints progress)
    print("  Fitting...")
    model.fit(train_prophet)
    
    # Predict on validation
    print("  Predicting on validation...")
    forecast = model.predict(val_prophet)
    
    # Extract predictions (Prophet returns many columns, we want 'yhat')
    y_true = val_prophet['y'].values
    y_pred = forecast['yhat'].values
    
    # Calculate metrics
    val_metrics = calculate_metrics(y_true, y_pred)
    print_metrics(val_metrics, f"Prophet ({city.title()})")
    
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
        model, metrics = train_prophet_for_city(train, val, city)
        
        # Save
        model_path = MODELS_DIR / f'prophet_{city}.pkl'
        joblib.dump(model, model_path)
        print(f"  Saved: {model_path}")
        
        results[city] = metrics
    
    # Overall validation metrics (average both cities)
    overall_mae = (results['karachi']['mae'] + results['lahore']['mae']) / 2
    overall_r2 = (results['karachi']['r2'] + results['lahore']['r2']) / 2
    overall_aqi = (results['karachi']['aqi_accuracy'] + results['lahore']['aqi_accuracy']) / 2
    
    print("\n" + "="*70)
    print("Prophet Overall Validation Results (Average)")
    print("="*70)
    print(f"  MAE:          {overall_mae:.2f} µg/m³")
    print(f"  R²:           {overall_r2:.4f}")
    print(f"  AQI Accuracy: {overall_aqi:.2%}")
    print("\nNote: Prophet typically has higher MAE than ensemble (linear + seasonality)")
    print("      but provides interpretable trend/seasonality components.")


if __name__ == '__main__':
    main()
