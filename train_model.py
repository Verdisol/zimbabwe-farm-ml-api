"""
Train a Random Forest model to predict maize yield in Zimbabwe.
Uses historical data: growing-season rainfall, temperature, and fertilizer use.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------------------------------------
# 1. Synthetic training data (based on Zimbabwe agriculture patterns)
# Each row: [rainfall_mm, temperature_c, fertilizer_kg_ha] -> yield_t_ha
# ---------------------------------------------
np.random.seed(42)

n_samples = 500

rainfall = np.random.normal(700, 200, n_samples).clip(200, 1200)
temperature = np.random.normal(24, 2.5, n_samples).clip(18, 32)
fertilizer = np.random.normal(60, 30, n_samples).clip(0, 150)

# Yield formula: more rain helps (up to a point), moderate temp is best, more fertilizer helps
yield_t_ha = (
    0.0015 * rainfall
    - 0.0000012 * (rainfall - 750) ** 2
    - 0.03 * (temperature - 24) ** 2
    + 0.008 * fertilizer
    + np.random.normal(0, 0.15, n_samples)
).clip(0.2, 3.0)

df = pd.DataFrame({
    'rainfall_mm': rainfall,
    'temperature_c': temperature,
    'fertilizer_kg_ha': fertilizer,
    'yield_t_ha': yield_t_ha,
})

print("Training data sample:")
print(df.head())
print(f"\nTotal samples: {len(df)}")

# ---------------------------------------------
# 2. Split data
# ---------------------------------------------
X = df[['rainfall_mm', 'temperature_c', 'fertilizer_kg_ha']]
y = df['yield_t_ha']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# 3. Train Random Forest
# ---------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# ---------------------------------------------
# 4. Evaluate
# ---------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel performance:")
print(f"  Mean Absolute Error: {mae:.3f} t/ha")
print(f"  R² score: {r2:.3f}")

# Feature importance
importances = model.feature_importances_
for name, imp in zip(X.columns, importances):
    print(f"  {name}: {imp:.3f}")

# ---------------------------------------------
# 5. Save model
# ---------------------------------------------
joblib.dump(model, 'random_forest_yield.pkl')
print("\n✅ Model saved as random_forest_yield.pkl")
