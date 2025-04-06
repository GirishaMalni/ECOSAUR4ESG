import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import os

paths = [
    "/Users/girishamalni/Downloads/ESG_CSV/ESGcountry-series.csv",
    "/Users/girishamalni/Downloads/ESG_CSV/ESGCountry.csv",
    "/Users/girishamalni/Downloads/ESG_CSV/ESGseries-time.csv",
    "/Users/girishamalni/Downloads/ESG_CSV/ESGanother.csv"  
]

dfs = [pd.read_csv(p) for p in paths]

esg_data = pd.concat(dfs, ignore_index=True)
esg_data = esg_data.dropna()

esg_data.columns = [col.strip() for col in esg_data.columns]

feature_columns = ['Country Code', 'Year', 'Series Code']  
target_columns = ['Environment', 'Social', 'Governance']

X = pd.get_dummies(esg_data[feature_columns])
y = esg_data[target_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

model_path = "esg_score_model.pkl"
joblib.dump((model, X.columns.tolist()), model_path)
print(f"Model saved to {model_path}")
