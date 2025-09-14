
from joblib import load
from pathlib import Path
import pandas as pd

artifact = load(Path("model/stroke-predictor-v1.joblib"))
model = artifact["model"]
features = artifact["features"]  


sample = {
    "Age": 72.0,
    "BMI": 29.3,
    "Avg_Glucose": 118.0,
    "Gender": "Male",         
    "SES": "Medium",          
    "Smoking_Status": "Former",
    "Hypertension": 1,
    "Heart_Disease": 0,
    "Diabetes": 0
}

df = pd.DataFrame([sample])


X = df.reindex(columns=features)


proba = float(model.predict_proba(X)[:, 1][0])  
pred  = int(model.predict(X)[0])                

print("Clase predicha:", pred)
print("Probabilidad Stroke=1:", round(proba, 4))

