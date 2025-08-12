import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("ds.csv")

# Features & Target
X = df[['temperature', 'humidity', 'wind_speed']]
y = df['energy_consumption']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso Regression model
model = Lasso(alpha=0.1)
model.fit(X_train_scaled, y_train)

# Save model & scaler
with open("lasso_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Lasso Regression Model Trained & Saved Successfully!")
