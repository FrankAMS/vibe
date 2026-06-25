# AppDaemon Random Forest Temperature Predictor - Complete Setup

## ✅ What You Now Have

I've created a **complete AppDaemon implementation** that runs your Random Forest model on the same device as Home Assistant. Here's what was created:

### Files Created

```
FrankAMS__vibe/
├── appdaemon/
│   ├── appdaemon.yaml          # Main AppDaemon configuration
│   ├── apps.yaml               # Apps configuration
│   ├── temperature_predictor.py # The prediction application
│   ├── rf_model.pkl            # Trained Random Forest model (1.9MB)
│   └── README.md               # Detailed setup instructions
└── train_rf_model.py           # Training script (for future retraining)
```

## 🎯 Performance Summary

Based on validation with your data:

| Metric | Value |
|--------|-------|
| **Overall MAE** | 0.65°C |
| **Hot days MAE** | 0.70°C |
| **Peak error** | ~3.1°C on hottest day |
| **R² score** | 0.873 |
| **Model size** | 1.9 MB |

**Note:** The peak error of ~3.1°C on the hottest day (July 1, 34°C) is acceptable given the thermal mass complexity. The model captures the 5-6 hour lag between outdoor and indoor temperature peaks.

## 📋 Quick Start Guide

### Step 1: Install AppDaemon on Your NUC

**Using Home Assistant Add-on (Recommended):**
1. Open Home Assistant
2. Go to **Supervisor** → **Add-on Store**
3. Search for "AppDaemon 4"
4. Click **Install**
5. Wait for installation to complete

### Step 2: Generate a Home Assistant Token

1. Click your **profile icon** (bottom left in Home Assistant)
2. Scroll down to "Long-lived access tokens"
3. Click **Create Token**
4. Name it "AppDaemon" 
5. Copy the token (you won't see it again!)

### Step 3: Copy Files to Home Assistant

```bash
# On your NUC, run these commands:

# Create the appdaemon directory
mkdir -p /config/appdaemon

# Copy all files from this repository
cp /path/to/FrankAMS__vibe/appdaemon/* /config/appdaemon/

# Set correct permissions
chown -R homeassistant:homeassistant /config/appdaemon/
chmod 644 /config/appdaemon/*
```

**Alternative:** Use Samba share or File Editor add-on to copy files.

### Step 4: Configure AppDaemon

Edit `/config/appdaemon/appdaemon.yaml`:

```yaml
http:
  url: http://localhost:8123
  token: YOUR_ACTUAL_TOKEN_HERE  # Replace with the token you copied
```

### Step 5: Verify Entity IDs

Edit `/config/appdaemon/temperature_predictor.py` and update:

```python
self.entity_config = {
    'temperature': 'sensor.knmi_temperature',      # Verify this matches your sensor
    'radiation': 'sensor.knmi_solar_radiation',     # Verify this matches your sensor
    'azimuth': 'sensor.knmi_solar_azimuth',        # Verify this matches your sensor
    'elevation': 'sensor.knmi_solar_elevation',    # Verify this matches your sensor
}

# If you have an actual thermometer2 sensor:
actual_entity = "sensor.thermometer2"  # Change to your actual sensor entity ID
```

**To find your entity IDs:**
1. Go to **Developer Tools** → **States**
2. Search for "knmi" to see all KNMI sensors
3. Note the exact entity IDs

### Step 6: Start AppDaemon

1. Restart the AppDaemon add-on
2. Check the logs: **Supervisor** → **AppDaemon 4** → **Logs**

### Step 7: Verify It Works

1. Go to **Developer Tools** → **States**
2. Look for these new sensors:
   - `sensor.thermometer2_rf_prediction` - Predicted temperature
   - `sensor.thermometer2_prediction_error` - Error vs actual (if available)
   - `sensor.thermometer2_model_info` - Model information
   - `sensor.thermometer2_prediction_updated` - Last update time

## 🔧 Customization Options

### Change Update Frequency

In `temperature_predictor.py`, find:
```python
self.run_every(self.update_prediction, datetime.now(), 300)  # 300 = 5 minutes
```
Change `300` to your preferred interval in seconds.

### Use Different Sensors

Update the `entity_config` dictionary in `temperature_predictor.py`:
```python
self.entity_config = {
    'temperature': 'sensor.your_temp_sensor',
    'radiation': 'sensor.your_radiation_sensor',
    'azimuth': 'sensor.your_azimuth_sensor',
    'elevation': 'sensor.your_elevation_sensor',
}
```

### Adjust Thermal Mass

The model uses 6 hours of delayed radiation. To change this:
1. Edit `train_rf_model.py` and change `max_delay=6`
2. Retrain the model: `python train_rf_model.py`
3. Copy the new `rf_model.pkl` to `/config/appdaemon/`
4. Update the feature list in `temperature_predictor.py`

## 📊 Model Features

The Random Forest uses these 14 features:

| Feature | Description | Importance |
|---------|-------------|------------|
| `hour_0_23` | Hour of day (0-23) | 36.2% |
| `T_celsius` | Outdoor temperature (°C) | 19.6% |
| `Q_delay_6` | Radiation 6 hours ago | 11.3% |
| `Q_delay_5` | Radiation 5 hours ago | 4.9% |
| `Q` | Current radiation | 4.7% |
| `Q_vertical_delay_3` | Vertical radiation 3h ago | 4.1% |
| `cos_azimuth_diff` | Sun-window angle cosine | 3.4% |
| `solar_elevation` | Sun elevation | 3.2% |
| `Q_delay_4` | Radiation 4 hours ago | 3.0% |
| `Q_delay_1` | Radiation 1 hour ago | 2.5% |
| `Q_delay_2` | Radiation 2 hours ago | 2.0% |
| `Q_delay_3` | Radiation 3 hours ago | 1.9% |
| `Q_vertical` | Current vertical radiation | 1.8% |
| `Q_vertical_delay_1` | Vertical radiation 1h ago | 0.7% |
| `Q_vertical_delay_2` | Vertical radiation 2h ago | 0.7% |

## 🎯 Why This Works on Your NUC

Your specs (Celeron N3350, 4GB RAM) are **more than sufficient**:

- **CPU Usage**: Model prediction takes ~1-5ms per call
- **RAM Usage**: Model uses ~2MB in memory
- **Disk Usage**: Model file is ~1.9MB
- **Updates**: Every 5 minutes = 288 predictions/day
- **Total daily CPU time**: ~1-2 seconds

**No performance impact on Home Assistant!**

## 🔄 Retraining the Model

If you collect more data and want to retrain:

```bash
# On any machine with Python
pip install pandas scikit-learn numpy

# Run the training script
python train_rf_model.py

# Copy the new model to Home Assistant
cp rf_model.pkl /config/appdaemon/

# Restart AppDaemon
```

## 🛠️ Troubleshooting

### "Model file not found"
- Verify `rf_model.pkl` exists in `/config/appdaemon/`
- Check permissions: `ls -la /config/appdaemon/`
- Fix: `chmod 644 /config/appdaemon/rf_model.pkl`

### "Sensor not available"
- Verify KNMI integration is working
- Check entity IDs in Home Assistant
- Update `entity_config` in `temperature_predictor.py`

### AppDaemon won't start
- Check logs: `tail -f /config/appdaemon/appdaemon.log`
- Verify token is correct in `appdaemon.yaml`
- Ensure all files are in `/config/appdaemon/`

### Permission errors
```bash
chown -R homeassistant:homeassistant /config/appdaemon/
chmod 644 /config/appdaemon/*
```

## 📈 Expected Results

Once running, you'll see:

```
# Example sensor states:
sensor.thermometer2_rf_prediction: 28.5
  unit_of_measurement: °C
  friendly_name: Thermometer2 RF Prediction
  icon: mdi:thermometer

sensor.thermometer2_prediction_error: -0.3
  unit_of_measurement: °C
  friendly_name: Prediction Error
  actual: 28.8
  predicted: 28.5

sensor.thermometer2_prediction_updated: "2025-06-25 14:30:00"
```

## 🎉 You're Done!

Your Random Forest model is now running locally on your NUC alongside Home Assistant. It will:
- Predict temperature every 5 minutes
- Update when KNMI sensors change
- Track prediction error (if you have an actual sensor)
- Handle thermal mass with 6-hour delayed features
- Use solar geometry for your WNW-facing windows

**No cloud dependency, no external API, all local processing!**
