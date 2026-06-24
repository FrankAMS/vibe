import pandas as pd
from sklearn.linear_model import LinearRegression

def train_temperature_model(csv_path):
    # Load the data
    df = pd.read_csv(csv_path, sep=';')

    # Convert temperatures from 0.1°C to °C
    df['T'] = df['T'] / 10
    df['thermometer2'] = df['thermometer2'] / 10

    # Filter for warmer days (T >= 25)
    warmer_days = df[df['T'] >= 25]

    # Prepare features (T, Q) and target (thermometer2)
    X = warmer_days[['T', 'Q']]
    y = warmer_days['thermometer2']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Return a prediction function
    def predict_thermometer2(T, Q):
        return model.predict([[T, Q]])[0]

    return predict_thermometer2

# Example usage:
# predict_func = train_temperature_model('clean data temperaturs.csv')
# predicted_temp = predict_func(T=26, Q=100)
# print(predicted_temp)