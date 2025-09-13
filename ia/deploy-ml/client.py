# client.py
from joblib import load
from pathlib import Path
import pandas as pd

artifact = load(Path("model/stroke-predictor-v1.joblib"))
model = artifact["model"]
features = artifact["features"]  # <- orden exacto esperado

# Ejemplo (ajústalo si quieres)
sample = {
    "Age": 72.0,
    "BMI": 29.3,
    "Avg_Glucose": 118.0,
    "Gender": "Male",         # texto crudo
    "SES": "Medium",          # texto crudo
    "Smoking_Status": "Former",
    "Hypertension": 1,
    "Heart_Disease": 0,
    "Diabetes": 0
}

df = pd.DataFrame([sample])

# Clave: asegurar columnas y orden EXACTO
X = df.reindex(columns=features)

# Predicción
proba = float(model.predict_proba(X)[:, 1][0])  # prob stroke=1
pred  = int(model.predict(X)[0])                # clase 0/1

print("Clase predicha:", pred)
print("Probabilidad Stroke=1:", round(proba, 4))

