
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/cicids2017_model_ready.csv")
MODEL_PATH = Path("models/nids_model.joblib")
METRICS_DIR = Path("outputs/metrics")

SAMPLE_SIZE = 300000
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH, low_memory=False)
target = "Attack Type"

df = df.dropna(subset=[target]).copy()
df[target] = df[target].astype(str).str.strip()
df = df[df[target].ne("") & df[target].ne("nan")]

feature_cols = [
    col for col in df.select_dtypes(include=np.number).columns
    if col != target
]

X = df[feature_cols]
y = df[target]

if len(y) > SAMPLE_SIZE:
    X, _, y, _ = train_test_split(
        X,
        y,
        train_size=SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
labels = model.named_steps["classifier"].classes_

accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    labels=labels,
    zero_division=0,
    output_dict=True,
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=labels,
    zero_division=0,
))
print("\nClass order:", list(labels))
print("\nConfusion matrix:")
print(matrix)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "feature_columns": feature_cols,
    "classes": list(labels),
}, MODEL_PATH)

# 7. Save evaluation metrics
METRICS_DIR.mkdir(parents=True, exist_ok=True)

with open(METRICS_DIR / "classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

pd.DataFrame(
    matrix,
    index=labels,
    columns=labels,
).to_csv(METRICS_DIR / "confusion_matrix.csv")

with open(METRICS_DIR / "summary.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "sample_size": len(X),
        "random_state": RANDOM_STATE,
        "model": "RandomForestClassifier",
        "features": feature_cols,
        "classes": list(labels),
    }, f, indent=4)

pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": predictions,
}).to_csv(METRICS_DIR / "test_predictions.csv", index=False)

print("\nSaved model:", MODEL_PATH)
print("Saved metrics:", METRICS_DIR.resolve())
