#  Lasso Regression – Energy Consumption Forecasting

## Overview

This project demonstrates **Energy Consumption Forecasting** using the **Lasso Regression** algorithm. Lasso Regression is used here because it performs **feature selection** by shrinking less important feature coefficients to zero, reducing overfitting and improving generalization. The model is trained on a dataset containing **200 rows** of historical data, then deployed via a **Flask web app** with HTML & CSS for the user interface.

-----

## Features

  - **Lasso Regression** model for prediction.
  - **Flask backend** for deployment.
  - **HTML/CSS frontend** for data input & output display.
  - Dataset with 200 records for training.

-----

## Project Structure

```
lasso_regression_energy/
│
├── model.py             # Trains and saves the model
├── app.py               # Flask app for prediction
├── templates/
│   ├── index.html       # Input form
│   └── result.html      # Prediction result
├── static/
│   └── style.css        # Styles for the frontend
├── dataset.csv          # Dataset for training
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

-----

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
Flask==3.0.0
numpy==1.26.2
pandas==2.1.4
scikit-learn==1.3.2
```

-----

## Dataset

The dataset (`dataset.csv`) contains 200 rows with historical energy usage and weather-related features.

Example:

```
temperature,humidity,wind_speed,day_of_week,hour,energy_consumption
28,65,5,1,14,320
30,70,3,1,15,340
```

Columns:

  - `temperature`: Temperature in °C
  - `humidity`: Humidity percentage
  - `wind_speed`: Wind speed in km/h
  - `day_of_week`: Day of the week (0=Sunday, 6=Saturday)
  - `hour`: Hour of the day (0–23)
  - `energy_consumption`: Energy usage in kWh (target)

-----

## How It Works

### Model Training (`model.py`)

  - Loads and preprocesses the dataset.
  - Trains a Lasso Regression model.
  - Saves the model as `model.pkl`.

### Prediction (`app.py`)

  - Loads `model.pkl`.
  - Accepts input from the HTML form.
  - Predicts future energy consumption.
  - Displays the results.

-----

## Running the Project

1.  **Train the model:**
    ```bash
    python model.py
    ```
2.  **Start Flask app:**
    ```bash
    python app.py
    ```
3.  **Open in browser:**
    `http://127.0.0.1:5000/`

-----

## Screenshots
---

Home Page

<img width="423" height="419" alt="Screenshot 2025-08-12 122816" src="https://github.com/user-attachments/assets/a1c87911-40e5-4b31-bd8d-7b20309c4c58" />

---
Prediction Result

<img width="381" height="420" alt="Screenshot 2025-08-12 122824" src="https://github.com/user-attachments/assets/61cb5845-0df8-48ee-8370-3296cb315bf9" />
