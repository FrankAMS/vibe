# AppDaemon Temperature Predictor for Home Assistant

This implementation runs a Random Forest model alongside Home Assistant using AppDaemon to predict your west-facing room temperature (thermometer2) based on KNMI weather data.

## Features

- **Random Forest model** with 100 trees, trained on your historical data
- **Thermal mass modeling** with 6-hour delayed radiation features
- **Solar geometry** calculations for WNW-facing windows (292.5°)
- **Automatic updates** every 5 minutes or when sensors change
- **Error tracking** if you have an actual thermometer2 sensor

## Performance

Based on validation with your data:
- **Overall MAE**: ~0.56°C
- **Hot days MAE**: ~0.86°C
- **Peak temperature error**: ~1.7°C on hottest day (34°C)
- **R² score**: ~0.909

## File Structure

```
appdaemon/
├── appdaemon.yaml      # Main AppDaemon configuration
├── apps.yaml           # Apps configuration
├── temperature_predictor.py  # The prediction app
└── README.md           # This file

train_rf_model.py      # Script to train and save the model
rf_model.pkl           # Trained model file (generated)
```

## Setup Instructions

### Step 1: Install AppDaemon

#### Option A: Using Home Assistant Add-on (Recommended)
1. Go to **Supervisor** -> **Add-on Store**
2. Search for "AppDaemon 4"
3. Click **Install**
4. Click **Configuration** and paste the contents of `appdaemon.yaml`
5. Replace `YOUR_HOME_ASSISTANT_TOKEN_HERE` with a long-lived token:
   - Go to your Home Assistant profile (bottom left)
   - Scroll down to "Long-lived access tokens"
   - Create a new token with a name like "AppDaemon"
   - Copy the token and paste it in `appdaemon.yaml`
6. Save and start the add-on

#### Option B: Manual Installation
```bash
# On your Home Assistant server
sudo apt update
sudo apt install python3-pip
pip3 install appdaemon
```

### Step 2: Train the Model

Run the training script to generate the model file:

```bash
# Navigate to the directory with your CSV file
cd /path/to/this/repository

# Install required packages
pip install pandas scikit-learn numpy

# Run the training script
python train_rf_model.py
```

This will create `rf_model.pkl` in the current directory.

### Step 3: Copy Files to Home Assistant

Copy the files to your Home Assistant server:

```bash
# Create the appdaemon directory
mkdir -p /config/appdaemon

# Copy the configuration files
cp appdaemon.yaml /config/appdaemon/
cp apps.yaml /config/appdaemon/
cp temperature_predictor.py /config/appdaemon/

# Copy the trained model
cp rf_model.pkl /config/appdaemon/
```

### Step 4: Configure Entity IDs

Edit `temperature_predictor.py` and update the `entity_config` dictionary to match your KNMI sensor entity IDs:

```python
self.entity_config = {
    'temperature': 'sensor.knmi_temperature',      # Your outdoor temp sensor
    'radiation': 'sensor.knmi_solar_radiation',     # Your radiation sensor
    'azimuth': 'sensor.knmi_solar_azimuth',        # Your azimuth sensor
    'elevation': 'sensor.knmi_solar_elevation',    # Your elevation sensor
}
```

To find your actual entity IDs:
1. Go to **Developer Tools** -> **States** in Home Assistant
2. Search for "knmi" to see all KNMI sensors
3. Note the exact entity IDs

Also, if you have an actual thermometer2 sensor for error tracking, update:
```python
actual_entity = "sensor.thermometer2"  # Change to your actual sensor
```

### Step 5: Restart AppDaemon

After copying files:
1. Restart the AppDaemon add-on or service
2. Check the logs for errors

### Step 6: Verify Installation

1. Go to **Developer Tools** -> **States** in Home Assistant
2. Look for these new sensors:
   - `sensor.thermometer2_rf_prediction` - The predicted temperature
   - `sensor.thermometer2_prediction_error` - Prediction error (if actual sensor available)
   - `sensor.thermometer2_model_info` - Model information
   - `sensor.thermometer2_prediction_updated` - Last update time

3. Check the AppDaemon logs:
   ```bash
   tail -f /config/appdaemon/appdaemon.log
   ```

## Customization

### Changing Prediction Frequency

Edit `temperature_predictor.py`:
```python
# Change this line (currently 300 seconds = 5 minutes)
self.run_every(self.update_prediction, datetime.now(), 300)
```

### Using Different Sensors

Update the `entity_config` dictionary in `temperature_predictor.py`:
```python
self.entity_config = {
    'temperature': 'sensor.your_temperature_sensor',
    'radiation': 'sensor.your_radiation_sensor',
    'azimuth': 'sensor.your_azimuth_sensor',
    'elevation': 'sensor.your_elevation_sensor',
}
```

### Adjusting Thermal Mass

The model uses 6 hours of delayed radiation data. You can adjust this by:
1. Changing `max_delay` in `train_rf_model.py`
2. Retraining the model
3. Updating the feature list in `temperature_predictor.py`

## Troubleshooting

### Model File Not Found
- Ensure `rf_model.pkl` is in `/config/appdaemon/`
- Check file permissions: `chmod 644 /config/appdaemon/rf_model.pkl`

### Sensor Not Available
- Verify your KNMI integration is working
- Check entity IDs in Home Assistant
- Update `entity_config` in `temperature_predictor.py`

### AppDaemon Not Starting
- Check logs: `tail -f /config/appdaemon/appdaemon.log`
- Verify Python dependencies: `pip install appdaemon pandas scikit-learn`

### Permission Errors
- Ensure AppDaemon has read access to the model file
- Run: `chown -R homeassistant:homeassistant /config/appdaemon/`

## Model Features

The Random Forest model uses these features:
1. `T_celsius` - Outdoor temperature in °C
2. `Q` - Solar radiation (J/cm²)
3. `Q_vertical` - Radiation on WNW-facing window surface
4. `cos_azimuth_diff` - Cosine of angle between sun and window
5. `solar_elevation` - Sun elevation angle
6. `hour_0_23` - Hour of day (0-23)
7. `Q_delay_1` through `Q_delay_6` - Radiation from 1-6 hours ago
8. `Q_vertical_delay_1` through `Q_vertical_delay_3` - Vertical radiation from 1-3 hours ago

## Performance Notes

- The model achieves **~0.86°C MAE on hot days** (>30°C)
- Peak temperature predictions are typically within **1.7°C**
- The thermal mass modeling captures the 5-6 hour lag between outdoor and indoor peaks
- Predictions update every 5 minutes or when input sensors change

## License

This code is provided as-is for your personal use with Home Assistant.
