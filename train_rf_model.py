#!/usr/bin/env python3
"""
Train and save the Random Forest model for Home Assistant AppDaemon.

This script:
1. Loads your CSV data
2. Trains the Random Forest model with thermal mass features
3. Saves the model to rf_model.pkl for AppDaemon to use

Usage:
    python train_rf_model.py

The model will be saved to /config/appdaemon/rf_model.pkl
(or the current directory if /config/appdaemon doesn't exist)
"""

import pandas as pd
import numpy as np
import math
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Constants
WINDOW_AZIMUTH = 292.5  # WNW orientation

# Solar position data for each hour (0-23)
SOLAR_POSITION = {
    0: {'azimuth': 0.0, 'elevation': -19.3},
    1: {'azimuth': 0.0, 'elevation': -18.1},
    2: {'azimuth': 0.0, 'elevation': -14.6},
    3: {'azimuth': 26.4, 'elevation': -9.1},
    4: {'azimuth': 59.5, 'elevation': -2.2},
    5: {'azimuth': 78.3, 'elevation': 5.9},
    6: {'azimuth': 89.8, 'elevation': 14.6},
    7: {'azimuth': 102.1, 'elevation': 23.8},
    8: {'azimuth': 116.1, 'elevation': 32.9},
    9: {'azimuth': 133.3, 'elevation': 41.6},
    10: {'azimuth': 154.8, 'elevation': 49.1},
    11: {'azimuth': 180.1, 'elevation': 54.5},
    12: {'azimuth': 205.5, 'elevation': 56.5},
    13: {'azimuth': 226.9, 'elevation': 54.5},
    14: {'azimuth': 244.0, 'elevation': 49.0},
    15: {'azimuth': 258.1, 'elevation': 41.5},
    16: {'azimuth': 270.4, 'elevation': 32.8},
    17: {'azimuth': 281.8, 'elevation': 23.7},
    18: {'azimuth': 264.6, 'elevation': 14.5},
    19: {'azimuth': 153.6, 'elevation': 5.8},
    20: {'azimuth': 0.0, 'elevation': -2.3},
    21: {'azimuth': 0.0, 'elevation': -9.2},
    22: {'azimuth': 0.0, 'elevation': -14.6},
    23: {'azimuth': 0.0, 'elevation': -18.1},
}


def calculate_cos_azimuth_diff(solar_azimuth, window_azimuth=WINDOW_AZIMUTH):
    """Calculate cosine of azimuth difference between sun and window."""
    diff = abs(solar_azimuth - window_azimuth)
    if diff > 180:
        diff = 360 - diff
    return math.cos(math.radians(diff))


def calculate_Q_vertical(Q, solar_azimuth, solar_elevation, window_azimuth=WINDOW_AZIMUTH):
    """Calculate radiation on vertical (window) surface."""
    if Q <= 0 or solar_elevation <= 0:
        return 0.0
    cos_azm_diff = calculate_cos_azimuth_diff(solar_azimuth, window_azimuth)
    if cos_azm_diff <= 0:
        return 0.0
    sin_elevation = math.sin(math.radians(solar_elevation))
    return Q * cos_azm_diff * sin_elevation


def add_features(df, max_delay=6):
    """Add all features for the Random Forest model."""
    df = df.copy()
    
    # Basic features
    df['T_celsius'] = df['T'] / 10.0  # Convert from 0.1°C to °C
    df['thermometer2_celsius'] = df['thermometer2'] / 10.0
    df['hour_0_23'] = df['HH'] - 1  # Convert from 1-24 to 0-23
    
    # Solar geometry
    df['solar_azimuth'] = df['hour_0_23'].map(lambda h: SOLAR_POSITION[h]['azimuth'])
    df['solar_elevation'] = df['hour_0_23'].map(lambda h: SOLAR_POSITION[h]['elevation'])
    df['cos_azimuth_diff'] = df['solar_azimuth'].apply(lambda az: calculate_cos_azimuth_diff(az))
    df['Q_vertical'] = df.apply(
        lambda row: calculate_Q_vertical(row['Q'], row['solar_azimuth'], row['solar_elevation']),
        axis=1
    )
    
    # Delayed features (thermal mass)
    for delay in range(1, max_delay + 1):
        df[f'Q_delay_{delay}'] = df.groupby('YYYYMMDD')['Q'].shift(delay)
        df[f'Q_vertical_delay_{delay}'] = df.groupby('YYYYMMDD')['Q_vertical'].shift(delay)
    
    # Fill NaN values with 0
    for delay in range(1, max_delay + 1):
        df[f'Q_delay_{delay}'] = df[f'Q_delay_{delay}'].fillna(0)
        df[f'Q_vertical_delay_{delay}'] = df[f'Q_vertical_delay_{delay}'].fillna(0)
    
    return df


def train_model(df):
    """Train the Random Forest model."""
    df = add_features(df, max_delay=6)
    
    # Features to use (must match AppDaemon app)
    features = [
        'T_celsius', 'Q', 'Q_vertical', 'cos_azimuth_diff', 'solar_elevation',
        'hour_0_23', 'Q_delay_1', 'Q_delay_2', 'Q_delay_3', 'Q_delay_4', 'Q_delay_5', 'Q_delay_6',
        'Q_vertical_delay_1', 'Q_vertical_delay_2', 'Q_vertical_delay_3'
    ]
    
    y = df['thermometer2_celsius']
    X = df[features].fillna(0)
    
    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Model trained successfully!")
    print(f"  MAE: {mae:.4f}°C")
    print(f"  R²: {r2:.4f}")
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in importances.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, features, mae, r2


def validate_on_hot_days(df, model, features):
    """Validate the model on hot days (>30°C)."""
    df = add_features(df, max_delay=6)
    
    # Find hot days
    hot_days = df[df['thermometer2_celsius'] >= 30]['YYYYMMDD'].unique()
    print(f"\nValidating on {len(hot_days)} hot days...")
    
    total_mae = 0
    total_count = 0
    max_error = 0
    
    for date in sorted(hot_days):
        day_data = df[df['YYYYMMDD'] == date]
        X = day_data[features].fillna(0)
        y = day_data['thermometer2_celsius']
        
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        day_max_error = max(abs(y - y_pred))
        
        total_mae += mae * len(day_data)
        total_count += len(day_data)
        max_error = max(max_error, day_max_error)
        
        print(f"  {date}: MAE={mae:.2f}°C, Max Error={day_max_error:.2f}°C")
    
    avg_mae = total_mae / total_count if total_count > 0 else 0
    print(f"\nAverage MAE on hot days: {avg_mae:.2f}°C")
    print(f"Maximum error on hot days: {max_error:.2f}°C")
    
    return avg_mae, max_error


def save_model(model, output_path):
    """Save the model to a file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    # Load data
    csv_path = 'clean data temperature hot days2.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        print("Please ensure the CSV file is in the same directory.")
        return
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Date range: {df['YYYYMMDD'].min()} to {df['YYYYMMDD'].max()}")
    print(f"Hot days (>=30°C): {len(df[df['thermometer2'] >= 300])}")
    
    # Train model
    model, features, mae, r2 = train_model(df)
    
    # Validate on hot days
    hot_mae, hot_max_error = validate_on_hot_days(df, model, features)
    
    # Save model
    # Try to save to /config/appdaemon/ first (Home Assistant default)
    output_path1 = '/config/appdaemon/rf_model.pkl'
    output_path2 = 'rf_model.pkl'  # Fallback to current directory
    
    try:
        save_model(model, output_path1)
        print(f"\nModel saved to Home Assistant config directory: {output_path1}")
    except PermissionError:
        print(f"Warning: Could not save to {output_path1} (permission denied)")
        save_model(model, output_path2)
        print(f"Model saved to current directory: {output_path2}")
        print(f"You will need to manually copy it to {output_path1}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Overall MAE: {mae:.4f}°C")
    print(f"Overall R²: {r2:.4f}")
    print(f"Hot days MAE: {hot_mae:.4f}°C")
    print(f"Hot days max error: {hot_max_error:.4f}°C")
    print(f"\nModel ready for AppDaemon!")
    print(f"Copy the model file to your Home Assistant server at:")
    print(f"  /config/appdaemon/rf_model.pkl")


if __name__ == "__main__":
    main()
