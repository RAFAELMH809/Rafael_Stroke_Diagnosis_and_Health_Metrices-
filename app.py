
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
from pathlib import Path
import os
import pandas as pd

app = FastAPI(title="Stroke Risk Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  
ARTIFACT = None  

@app.on_event("startup")
def load_model():
    global ARTIFACT
    model_path = Path("model") / "stroke-predictor-v1.joblib"
    ARTIFACT = load(model_path)
    if not (isinstance(ARTIFACT, dict) and "model" in ARTIFACT and "features" in ARTIFACT):
        raise RuntimeError("Artifact inválido: faltan 'model' y/o 'features'.")

class Input(BaseModel):
    Age: float
    Gender: str             # "Male" | "Female"
    SES: str                # "Low" | "Medium" | "High"
    Hypertension: int       # 0/1
    Heart_Disease: int      # 0/1
    BMI: float
    Avg_Glucose: float
    Diabetes: int           # 0/1
    Smoking_Status: str     


class LabelOutput(BaseModel):
    label: int

@app.post("/score", response_model=LabelOutput)
def score(data: Input):
    if ARTIFACT is None:
        raise HTTPException(500, "Modelo no cargado")

    model = ARTIFACT["model"]
    feats = ARTIFACT["features"]

    payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    df = pd.DataFrame([payload])
    X = df.reindex(columns=feats)

    try:
        
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[:, 1][0])  
            label = 1 if proba >= THRESHOLD else 0
            return {"label": label}
        
        pred = int(model.predict(X)[0])
        return {"label": pred}
    except Exception as e:
        raise HTTPException(500, detail=f"Inference error: {type(e).__name__}: {e}")

