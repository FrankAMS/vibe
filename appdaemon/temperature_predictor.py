"""
AppDaemon application for predicting thermometer2 temperature using Random Forest.
This app loads a pre-trained Random Forest model and makes predictions based on
KNMI sensor data from Home Assistant.

Features used:
- T_celsius: Outdoor temperature in °C
- Q: Solar radiation (J/cm²)
- Q_vertical: Radiation on WNW-facing window surface
- cos_azimuth_diff: Cosine of angle between sun and window
- solar_elevation: Sun elevation angle
- hour_0_23: Hour of day (0-23)
- Q_delay_1 through Q_delay_6: Radiation from previous hours (thermal mass)
- Q_vertical_delay_1 through Q_vertical_delay_3: Vertical radiation from previous hours

Window orientation: WNW at 292.5° from North
"""

import appdaemon.plugins.hass.hassapi as hass
import pickle
import os
import math
import numpy as np
from datetime import datetime, timedelta


# Constants
WINDOW_AZIMUTH = 292.5  # WNW orientation
MODEL_FILE = "/config/appdaemon/rf_model.pkl"  # Path to trained model

# Solar position data for each hour (0-23)
# This is used when KNMI azimuth/elevation sensors are not available
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


class TemperaturePredictor(hass.Hass):
    """
    AppDaemon app for predicting indoor temperature using Random Forest model.
    """
    
    def initialize(self):
        """Initialize the app."""
        self.log(f"Initializing Temperature Predictor...")
        
        # Load the trained model
        self.model = self.load_model()
        if self.model is None:
            self.log("Failed to load model! Check model path.", level="ERROR")
            return
        
        # Feature names (must match training)
        self.features = [
            'T_celsius', 'Q', 'Q_vertical', 'cos_azimuth_diff', 'solar_elevation',
            'hour_0_23', 'Q_delay_1', 'Q_delay_2', 'Q_delay_3', 'Q_delay_4', 'Q_delay_5', 'Q_delay_6',
            'Q_vertical_delay_1', 'Q_vertical_delay_2', 'Q_vertical_delay_3'
        ]
        
        # Entity IDs - customize these to match your Home Assistant setup
        # Default KNMI integration entity IDs
        self.entity_config = {
            'temperature': 'sensor.knmi_temperature',
            'radiation': 'sensor.knmi_solar_radiation',
            'azimuth': 'sensor.knmi_solar_azimuth',
            'elevation': 'sensor.knmi_solar_elevation',
        }
        
        # Storage for delayed values (thermal mass)
        self.history = {
            'Q': [],
            'Q_vertical': [],
            'timestamp': []
        }
        self.max_history = 6  # Store last 6 hours
        
        # Create sensors for predictions
        self.create_sensors()
        
        # Set up listeners for KNMI sensors
        self.listen_state(self.on_sensor_change, self.entity_config['temperature'])
        self.listen_state(self.on_sensor_change, self.entity_config['radiation'])
        
        # Also listen on azimuth and elevation if available
        if self.entity_config['azimuth']:
            self.listen_state(self.on_sensor_change, self.entity_config['azimuth'])
        if self.entity_config['elevation']:
            self.listen_state(self.on_sensor_change, self.entity_config['elevation'])
        
        # Initial prediction
        self.update_prediction()
        
        # Schedule regular updates (every 5 minutes)
        self.run_every(self.update_prediction, datetime.now(), 300)
        
        self.log("Temperature Predictor initialized successfully!")
    
    def load_model(self):
        """Load the trained Random Forest model from file."""
        try:
            if not os.path.exists(MODEL_FILE):
                self.log(f"Model file not found: {MODEL_FILE}", level="ERROR")
                return None
            
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            
            self.log(f"Model loaded successfully from {MODEL_FILE}")
            return model
        except Exception as e:
            self.log(f"Error loading model: {e}", level="ERROR")
            return None
    
    def create_sensors(self):
        """Create the prediction sensors in Home Assistant."""
        # Main prediction sensor
        self.set_state(
            "sensor.thermometer2_rf_prediction",
            state="unknown",
            attributes={
                "friendly_name": "Thermometer2 RF Prediction",
                "unit_of_measurement": "°C",
                "icon": "mdi:thermometer"
            }
        )
        
        # Prediction error (if actual sensor available)
        self.set_state(
            "sensor.thermometer2_prediction_error",
            state="unknown",
            attributes={
                "friendly_name": "Prediction Error",
                "unit_of_measurement": "°C",
                "icon": "mdi:chart-line"
            }
        )
        
        # Model info
        self.set_state(
            "sensor.thermometer2_model_info",
            state="Random Forest",
            attributes={
                "friendly_name": "Model Info",
                "model_type": "Random Forest",
                "features": ", ".join(self.features),
                "window_azimuth": WINDOW_AZIMUTH
            }
        )
        
        # Last update time
        self.set_state(
            "sensor.thermometer2_prediction_updated",
            state="never",
            attributes={
                "friendly_name": "Last Prediction Update"
            }
        )
    
    def calculate_cos_azimuth_diff(self, solar_azimuth):
        """Calculate cosine of azimuth difference between sun and window."""
        diff = abs(solar_azimuth - WINDOW_AZIMUTH)
        if diff > 180:
            diff = 360 - diff
        return math.cos(math.radians(diff))
    
    def calculate_Q_vertical(self, Q, solar_azimuth, solar_elevation):
        """Calculate radiation on vertical (window) surface."""
        if Q <= 0 or solar_elevation <= 0:
            return 0.0
        cos_azm_diff = self.calculate_cos_azimuth_diff(solar_azimuth)
        if cos_azm_diff <= 0:
            return 0.0
        sin_elevation = math.sin(math.radians(solar_elevation))
        return Q * cos_azm_diff * sin_elevation
    
    def get_solar_position(self, hour):
        """Get solar azimuth and elevation for a given hour."""
        hour_int = int(hour) % 24
        return SOLAR_POSITION[hour_int]
    
    def get_delayed_values(self, current_time, value_type):
        """Get delayed values from history."""
        delayed = []
        for i in range(1, 7):  # Get delays 1-6
            delay_time = current_time - timedelta(hours=i)
            # Find the closest historical value
            closest_value = None
            closest_diff = timedelta(hours=24)
            
            for hist_time, hist_value in zip(self.history['timestamp'], self.history[value_type]):
                diff = abs((hist_time - delay_time).total_seconds())
                if diff < closest_diff.total_seconds():
                    closest_diff = timedelta(seconds=diff)
                    closest_value = hist_value
            
            delayed.append(closest_value if closest_value is not None else 0)
        
        return delayed
    
    def update_history(self, Q, Q_vertical):
        """Update the history with current values."""
        now = datetime.now()
        
        # Remove old entries (older than 6 hours)
        cutoff = now - timedelta(hours=self.max_history)
        self.history['timestamp'] = [t for t in self.history['timestamp'] if t >= cutoff]
        self.history['Q'] = self.history['Q'][:len(self.history['timestamp'])]
        self.history['Q_vertical'] = self.history['Q_vertical'][:len(self.history['timestamp'])]
        
        # Add current values
        self.history['timestamp'].append(now)
        self.history['Q'].append(Q)
        self.history['Q_vertical'].append(Q_vertical)
    
    def get_features(self):
        """Extract all features for prediction."""
        try:
            # Get current sensor values
            T = float(self.get_state(self.entity_config['temperature']))
            Q = float(self.get_state(self.entity_config['radiation']))
            
            # Try to get azimuth and elevation from sensors
            solar_azimuth = None
            solar_elevation = None
            
            if self.entity_config['azimuth']:
                azimuth_state = self.get_state(self.entity_config['azimuth'])
                if azimuth_state and azimuth_state != 'unknown' and azimuth_state != 'unavailable':
                    solar_azimuth = float(azimuth_state)
            
            if self.entity_config['elevation']:
                elevation_state = self.get_state(self.entity_config['elevation'])
                if elevation_state and elevation_state != 'unknown' and elevation_state != 'unavailable':
                    solar_elevation = float(elevation_state)
            
            # If sensors not available, use lookup table based on hour
            current_hour = datetime.now().hour
            if solar_azimuth is None or solar_elevation is None:
                solar_pos = self.get_solar_position(current_hour)
                if solar_azimuth is None:
                    solar_azimuth = solar_pos['azimuth']
                if solar_elevation is None:
                    solar_elevation = solar_pos['elevation']
            
            # Calculate derived features
            T_celsius = T  # Assuming KNMI provides in °C, if in 0.1°C, divide by 10
            hour_0_23 = current_hour
            cos_azimuth_diff = self.calculate_cos_azimuth_diff(solar_azimuth)
            Q_vertical = self.calculate_Q_vertical(Q, solar_azimuth, solar_elevation)
            
            # Get delayed values
            now = datetime.now()
            Q_delays = self.get_delayed_values(now, 'Q')
            Q_vertical_delays = self.get_delayed_values(now, 'Q_vertical')
            
            # Build feature array in the correct order
            features = [
                T_celsius,
                Q,
                Q_vertical,
                cos_azimuth_diff,
                solar_elevation,
                hour_0_23,
                Q_delays[0],  # Q_delay_1
                Q_delays[1],  # Q_delay_2
                Q_delays[2],  # Q_delay_3
                Q_delays[3],  # Q_delay_4
                Q_delays[4],  # Q_delay_5
                Q_delays[5],  # Q_delay_6
                Q_vertical_delays[0],  # Q_vertical_delay_1
                Q_vertical_delays[1],  # Q_vertical_delay_2
                Q_vertical_delays[2],  # Q_vertical_delay_3
            ]
            
            return features
            
        except Exception as e:
            self.log(f"Error getting features: {e}", level="ERROR")
            return None
    
    def on_sensor_change(self, entity, attribute, old, new, kwargs):
        """Called when a sensor changes."""
        self.log(f"Sensor {entity} changed: {old} -> {new}")
        self.update_prediction()
    
    def update_prediction(self, kwargs=None):
        """Update the temperature prediction."""
        try:
            features = self.get_features()
            if features is None:
                self.log("Could not get features for prediction", level="WARNING")
                return
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            
            # Round to 1 decimal place
            prediction = round(prediction, 1)
            
            # Update prediction sensor
            self.set_state(
                "sensor.thermometer2_rf_prediction",
                state=prediction,
                attributes={
                    "friendly_name": "Thermometer2 RF Prediction",
                    "unit_of_measurement": "°C",
                    "icon": "mdi:thermometer",
                    "features": dict(zip(self.features, features)),
                    "model": "Random Forest"
                }
            )
            
            # Calculate error if actual sensor is available
            actual_entity = "sensor.thermometer2"  # Change this to your actual sensor
            actual_state = self.get_state(actual_entity)
            if actual_state and actual_state != 'unknown' and actual_state != 'unavailable':
                try:
                    actual = float(actual_state)
                    error = round(prediction - actual, 1)
                    self.set_state(
                        "sensor.thermometer2_prediction_error",
                        state=error,
                        attributes={
                            "friendly_name": "Prediction Error",
                            "unit_of_measurement": "°C",
                            "icon": "mdi:chart-line",
                            "actual": actual,
                            "predicted": prediction
                        }
                    )
                except ValueError:
                    pass
            
            # Update last update time
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.set_state("sensor.thermometer2_prediction_updated", state=now)
            
            # Update history with current Q and Q_vertical
            if len(features) >= 2:
                Q = features[1]  # Q is the second feature
                Q_vertical = features[2]  # Q_vertical is the third feature
                self.update_history(Q, Q_vertical)
            
            self.log(f"Prediction updated: {prediction}°C")
            
        except Exception as e:
            self.log(f"Error updating prediction: {e}", level="ERROR")
