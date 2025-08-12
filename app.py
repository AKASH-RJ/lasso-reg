from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model and scaler
with open("lasso_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind = float(request.form['wind_speed'])

        scaled_input = scaler.transform([[temp, humidity, wind]])
        prediction = model.predict(scaled_input)[0]

        return render_template('index.html', prediction_text=f"Predicted Energy Consumption: {prediction:.2f} kWh")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
