import pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json

BASE = Path(__file__).resolve().parents[2]  # repo root
DATA = BASE / "data" / "samples" / "samples.csv"
MODELS = BASE / "data" / "models"
EVID = BASE / "docs" / "evidence"

MODELS.mkdir(parents=True, exist_ok=True)
EVID.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
X = df["text"].astype(str).values
y = (df["label"] == "phishing").astype(int).values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=300))
])
pipe.fit(Xtr, ytr)
yhat = pipe.predict(Xte)

p, r, f1, _ = precision_recall_fscore_support(yte, yhat, average="binary", zero_division=0)
cm = confusion_matrix(yte, yhat).tolist()

joblib.dump(pipe, MODELS / "model.pkl")
(Path(EVID) / "metrics.json").write_text(json.dumps({
    "precision": float(p), "recall": float(r), "f1": float(f1), "confusion_matrix": cm
}, indent=2))

print("Model saved →", MODELS / "model.pkl")
print("Metrics saved →", EVID / "metrics.json")
