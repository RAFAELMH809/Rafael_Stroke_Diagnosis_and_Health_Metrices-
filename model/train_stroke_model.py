
from pathlib import Path
from joblib import dump
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv(Path('../data/stroke_data.csv'))


TARGET = 'Stroke'
y = df[TARGET]
X = df.drop(columns=[TARGET])


NUM_COLS  = ["Age", "BMI", "Avg_Glucose"]
CAT_COLS  = ["Gender", "SES", "Smoking_Status"]
BOOL_COLS = ["Hypertension", "Heart_Disease", "Diabetes"]
SERVING_ORDER = NUM_COLS + CAT_COLS + BOOL_COLS


expected = set(NUM_COLS + CAT_COLS + BOOL_COLS)
missing = expected - set(X.columns)
if missing:
    raise ValueError(f"Faltan columnas en el CSV: {missing}")


numeric = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num",  numeric, NUM_COLS),
        ("cat",  categorical, CAT_COLS),
        ("bool", "passthrough", BOOL_COLS)
    ]
)


rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


clf = Pipeline(steps=[
    ("pre", preprocessor),
    ("model", rf)
])

X = X.reindex(columns=SERVING_ORDER)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)


print("Training model...")
clf.fit(X_train, y_train)


y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

print("\nEjemplo de probabilidades (10):", [round(p, 4) for p in y_proba[:10]])


artifact = {
    "model": clf,                          
    "features": NUM_COLS + CAT_COLS + BOOL_COLS  
}
out_path = Path('stroke-predictor-v1.joblib')
dump(artifact, out_path)
print(f"\n OK: modelo guardado en {out_path} con features: {artifact['features']}")

