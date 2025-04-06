import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import os

# === Load ESG CSV files ===
paths = [
    "/Users/girishamalni/Downloads/ESG_CSV/ESGcountry-series.csv",
    "/Users/girishamalni/Downloads/ESG_CSV/ESGCountry.csv",
    "/Users/girishamalni/Downloads/ESG_CSV/ESGseries-time.csv",
    "/Users/girishamalni/Downloads/ESG_CSV/ESGanother.csv"  # Replace with the correct file name
]

dfs = [pd.read_csv(p) for p in paths]

# === Data Preprocessing (Example logic) ===
# For demonstration, let’s concatenate and clean the data
esg_data = pd.concat(dfs, ignore_index=True)
esg_data = esg_data.dropna()

# Rename columns if needed to match target labels
esg_data.columns = [col.strip() for col in esg_data.columns]

# Assuming numeric ESG columns and some textual metadata
feature_columns = ['Country Code', 'Year', 'Series Code']  # Adjust accordingly
target_columns = ['Environment', 'Social', 'Governance']

# Example encoding (you can do one-hot or embeddings too)
X = pd.get_dummies(esg_data[feature_columns])
y = esg_data[target_columns]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multi-output Regressor
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Save model
model_path = "esg_score_model.pkl"
joblib.dump((model, X.columns.tolist()), model_path)
print(f"Model saved to {model_path}")
