from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv("lasso_regression.csv")
X = df[['Temperature', 'Humidity', 'WindSpeed', 'ApplianceUsage']]
y = df['EnergyConsumption']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Lasso model
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, "lasso_model.pkl")
joblib.dump(scaler, "scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind = float(request.form['windspeed'])
    appliance = float(request.form['appliance'])

    scaler = joblib.load("scaler.pkl")
    model = joblib.load("lasso_model.pkl")

    features_scaled = scaler.transform([[temp, humidity, wind, appliance]])
    prediction = model.predict(features_scaled)[0]

    return render_template('index.html', prediction_text=f'Predicted Energy Consumption: {prediction:.2f} kWh')

if __name__ == "__main__":
    app.run(debug=True)
