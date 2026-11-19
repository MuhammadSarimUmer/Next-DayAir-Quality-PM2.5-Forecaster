"""
Convert Kaggle historical datasets to match OpenAQ pipeline format.

Kaggle data is DAILY, we need HOURLY. Strategy:
1. Convert AQI to PM2.5 µg/m³
2. Interpolate daily values to hourly (repeat each day's value for 24 hours)
3. Convert weather units (F to C, mph to m/s, inches to hPa)
4. Save in same format as OpenAQ data

Run: python -m src.convert_kaggle
"""

import pandas as pd
import numpy as np
from src import config

KAGGLE_LAHORE = config.ROOT / "datasets" / "2" / "lahore_aqi_2019_to_2023.csv"


def aqi_to_pm25(aqi):
    """
    Convert US EPA AQI (PM2.5) to PM2.5 µg/m³.
    
    Uses reverse calculation of EPA AQI formula.
    """
    if pd.isna(aqi) or aqi < 0:
        return np.nan
    
    # EPA AQI breakpoints (AQI_low, AQI_high, PM_low, PM_high)
    breakpoints = [
        (0, 50, 0.0, 9.0),
        (51, 100, 9.1, 35.4),
        (101, 150, 35.5, 55.4),
        (151, 200, 55.5, 125.4),
        (201, 300, 125.5, 225.4),
        (301, 500, 225.5, 500.4),
    ]
    
    for aqi_low, aqi_high, pm_low, pm_high in breakpoints:
        if aqi_low <= aqi <= aqi_high:
            # Reverse formula: PM = ((AQI - I_low) / (I_high - I_low)) * (C_high - C_low) + C_low
            pm25 = ((aqi - aqi_low) / (aqi_high - aqi_low)) * (pm_high - pm_low) + pm_low
            return pm25
    
    # Above 500 AQI
    if aqi > 500:
        return 500.4 + (aqi - 500) * 2  # Extrapolate
    
    return np.nan


def fahrenheit_to_celsius(f):
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5/9


def mph_to_ms(mph):
    """Convert miles per hour to meters per second."""
    return mph * 0.44704


def inches_to_hpa(inches):
    """Convert inches of mercury to hectopascals."""
    return inches * 33.8639


def expand_daily_to_hourly(df, date_col='date'):
    """
    Expand daily data to hourly by repeating each day 24 times.
    
    This is a simple interpolation - each hour gets the same value as the daily average.
    Better than nothing for historical extension.
    """
    hourly_rows = []
    
    for _, row in df.iterrows():
        # Parse date (format: MM-DD-YY)
        date_str = row[date_col]
        try:
            # Try multiple date formats
            if '-' in date_str and len(date_str.split('-')[0]) == 2:
                # Format: MM-DD-YY or DD-MM-YY
                date = pd.to_datetime(date_str, format='%d-%m-%y')
            else:
                date = pd.to_datetime(date_str)
        except:
            continue
        
        # Create 24 hourly rows for this day
        for hour in range(24):
            hourly_row = row.copy()
            hourly_row['datetime_pkt'] = date + pd.Timedelta(hours=hour)
            hourly_rows.append(hourly_row)
    
    return pd.DataFrame(hourly_rows)


def convert_lahore_kaggle():
    """
    Convert Kaggle Lahore AQI dataset to match OpenAQ format.
    
    Output columns match clean_hourly.csv:
    - city, datetime_pkt, pm25, temperature_2m, relative_humidity_2m, 
      surface_pressure, wind_speed_10m, precipitation, cloud_cover, wind_direction_10m
    """
    print("Converting Kaggle Lahore dataset...")
    df = pd.read_csv(KAGGLE_LAHORE)
    
    print(f"  Loaded {len(df)} daily rows ({df['date'].min()} to {df['date'].max()})")
    
    # Expand to hourly
    print("  Expanding daily to hourly (24x rows)...")
    hourly = expand_daily_to_hourly(df)
    print(f"  Created {len(hourly)} hourly rows")
    
    # Convert AQI to PM2.5
    print("  Converting AQI to PM2.5 µg/m³...")
    hourly['pm25'] = hourly['aqi_pm2.5'].apply(aqi_to_pm25)
    
    # Convert weather units
    print("  Converting weather units...")
    hourly['temperature_2m'] = hourly['avg_temp_f'].apply(fahrenheit_to_celsius)
    hourly['relative_humidity_2m'] = hourly['avg_humidity_percent']
    hourly['surface_pressure'] = hourly['avg_pressure_in'].apply(inches_to_hpa)
    hourly['wind_speed_10m'] = hourly['avg_wind_speed_mph'].apply(mph_to_ms)
    
    # Missing columns (Kaggle doesn't have these)
    hourly['precipitation'] = 0.0  # Assume no rain (conservative)
    hourly['cloud_cover'] = 50.0   # Assume 50% cloud cover (neutral)
    hourly['wind_direction_10m'] = 180.0  # Assume south wind (neutral)
    
    # Format output
    output = pd.DataFrame({
        'city': 'lahore',
        'datetime_pkt': hourly['datetime_pkt'],
        'pm25': hourly['pm25'],
        'temperature_2m': hourly['temperature_2m'],
        'relative_humidity_2m': hourly['relative_humidity_2m'],
        'surface_pressure': hourly['surface_pressure'],
        'wind_speed_10m': hourly['wind_speed_10m'],
        'precipitation': hourly['precipitation'],
        'cloud_cover': hourly['cloud_cover'],
        'wind_direction_10m': hourly['wind_direction_10m'],
    })
    
    # Remove rows with missing PM2.5
    output = output.dropna(subset=['pm25'])
    
    # Filter to useful date range (2020-2022, before OpenAQ data starts)
    output = output[output['datetime_pkt'] < '2025-06-01']
    output = output[output['datetime_pkt'] >= '2020-01-01']
    
    print(f"  Final: {len(output)} hourly rows with valid PM2.5")
    print(f"  Date range: {output['datetime_pkt'].min()} to {output['datetime_pkt'].max()}")
    
    # Save
    output_path = config.PROCESSED_DIR / 'kaggle_lahore_hourly.csv'
    output.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    return output


def merge_with_openaq():
    """
    Merge Kaggle historical data with current OpenAQ data.
    
    Creates SEPARATE files to avoid breaking current pipeline:
    - clean_hourly_extended.csv (merged)
    - features_extended.csv (will be created by features.py)
    - train_extended.csv, val_extended.csv, test_extended.csv (will be created by split.py)
    
    Your original files stay untouched!
    """
    print("\nMerging Kaggle + OpenAQ data...")
    
    # Load both
    kaggle_lhr = pd.read_csv(config.PROCESSED_DIR / 'kaggle_lahore_hourly.csv', parse_dates=['datetime_pkt'])
    openaq = pd.read_csv(config.PROCESSED_DIR / 'clean_hourly.csv', parse_dates=['datetime_pkt'])
    
    # Kaggle only has Lahore, keep OpenAQ Karachi as-is
    openaq_karachi = openaq[openaq['city'] == 'karachi'].copy()
    openaq_lahore = openaq[openaq['city'] == 'lahore'].copy()
    
    # Combine Lahore: Kaggle (historical) + OpenAQ (recent)
    lahore_combined = pd.concat([kaggle_lhr, openaq_lahore], ignore_index=True)
    lahore_combined = lahore_combined.sort_values('datetime_pkt').drop_duplicates('datetime_pkt', keep='last')
    
    # Combine both cities
    combined = pd.concat([openaq_karachi, lahore_combined], ignore_index=True)
    combined = combined.sort_values(['city', 'datetime_pkt'])
    
    print(f"  Karachi: {len(openaq_karachi)} rows")
    print(f"  Lahore (Kaggle): {len(kaggle_lhr)} rows")
    print(f"  Lahore (OpenAQ): {len(openaq_lahore)} rows")
    print(f"  Lahore (combined): {len(lahore_combined)} rows")
    print(f"  Total: {len(combined)} rows")
    
    print(f"\n  Combined date range:")
    for city in ['karachi', 'lahore']:
        city_data = combined[combined['city'] == city]
        print(f"    {city.title()}: {city_data['datetime_pkt'].min()} to {city_data['datetime_pkt'].max()}")
    
    # Save with _extended suffix to keep original safe
    output_path = config.PROCESSED_DIR / 'clean_hourly_extended.csv'
    combined.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"\n  ✅ Your original clean_hourly.csv is UNTOUCHED")
    
    return combined


def main():
    print("="*70)
    print("Converting Kaggle Historical Data to OpenAQ Format")
    print("="*70)
    
    # Convert Kaggle Lahore data
    kaggle_data = convert_lahore_kaggle()
    
    # Merge with OpenAQ
    merged = merge_with_openaq()
    
    print("\n" + "="*70)
    print("✅ SAFE TESTING MODE - Original Pipeline Untouched")
    print("="*70)
    print("\nTo test with extended data:")
    print("  1. python -m src.features_extended")
    print("  2. python -m src.split_extended")
    print("  3. python -m src.train_extended")
    print("\nTo rollback:")
    print("  Just delete *_extended.csv files and continue using originals!")
    print("\n⚠️  NOTE: Create features_extended.py, split_extended.py scripts")
    print("    that read/write *_extended.csv files (I can create these)")
    print("\nLahore will now have 3 historical winters (2020-2022) in training!")


if __name__ == '__main__':
    main()
