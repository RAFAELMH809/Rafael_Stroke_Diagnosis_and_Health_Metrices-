# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
from pathlib import Path
import pandas as pd

app = FastAPI(title="Stroke Risk Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

ARTIFACT = None  # {'model': Pipeline, 'features': [...]}

@app.on_event("startup")
def load_model():
    global ARTIFACT
    model_path = Path("model") / "stroke-predictor-v1.joblib"
    ARTIFACT = load(model_path)
    if not (isinstance(ARTIFACT, dict) and "model" in ARTIFACT and "features" in ARTIFACT):
        raise RuntimeError("Artifact inválido: faltan 'model' y/o 'features'.")

# Entrada con campos crudos (como en el CSV)
class Input(BaseModel):
    Age: float
    Gender: str             # "Male" | "Female"
    SES: str                # "Low" | "Medium" | "High"
    Hypertension: int       # 0/1
    Heart_Disease: int      # 0/1
    BMI: float
    Avg_Glucose: float
    Diabetes: int           # 0/1
    Smoking_Status: str     # "Never" | "Current" | "Former"

class Output(BaseModel):
    score: float            # probabilidad de Stroke=1

@app.post("/score", response_model=Output)
def score(data: Input):
    if ARTIFACT is None:
        raise HTTPException(500, "Modelo no cargado")

    model = ARTIFACT["model"]
    feats = ARTIFACT["features"]

    # DataFrame de UNA fila con los campos crudos
    payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    df = pd.DataFrame([payload])
    

    # Asegura columnas y orden crudo esperado (pipeline se encarga del OneHot y escalado)
    X = df.reindex(columns=feats)

    try:
        proba = float(model.predict_proba(X)[:, 1][0])  # prob Stroke=1
        return {"score": proba}
    except Exception as e:
        raise HTTPException(500, detail=f"Inference error: {type(e).__name__}: {e}")
