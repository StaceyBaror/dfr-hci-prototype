# eval/evaluate.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def normalize_label(v):
    """Map common string/boolean-ish labels to {0,1}."""
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().lower()
    # positive (phishing)
    if s in {"1", "true", "phish", "phishing", "malicious", "spam", "attack"}:
        return 1
    # negative (benign)
    if s in {"0", "false", "ham", "benign", "legit", "legitimate", "normal"}:
        return 0
    # fallback: treat anything else as 0
    return 0

def main():
    parser = argparse.ArgumentParser(description="Evaluate detection model on samples.csv")
    parser.add_argument("--samples", default="data/samples/samples.csv", help="Path to CSV dataset")
    parser.add_argument("--model", default="data/models/model.pkl", help="Path to joblib model pipeline")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("THRESHOLD", 0.5)),
                        help="Decision threshold for positive class (default 0.5 or $THRESHOLD)")
    args = parser.parse_args()

    MODELS = Path(args.model).resolve().parent
    DOCS = Path("docs/evidence")
    DOCS.mkdir(parents=True, exist_ok=True)

    # ---- Load data
    df = pd.read_csv(args.samples)

    # Accept common text column names
    text_col_candidates = ["text", "message", "content", "body"]
    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    if text_col is None:
        raise SystemExit(f"No text column found. Expected one of: {text_col_candidates}")

    # Accept common label column names
    label_col_candidates = ["label", "y", "target", "class"]
    label_col = next((c for c in label_col_candidates if c in df.columns), None)
    if label_col is None:
        raise SystemExit(f"No label column found. Expected one of: {label_col_candidates}")

    # Normalize labels
    df["_y"] = df[label_col].apply(normalize_label)
    if df["_y"].isna().any():
        raise SystemExit("Some labels could not be normalized—please check your CSV.")
    texts = df[text_col].astype(str).tolist()
    y_true = df["_y"].astype(int).values

    # ---- Load model and predict
    model = joblib.load(args.model)
    y_prob = model.predict_proba(texts)[:, 1]
    thr = float(args.threshold)
    y_pred = (y_prob >= thr).astype(int)

    # ---- Metrics (robust to div-by-zero)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    out = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "threshold": thr,
        "n": int(len(df)),
        "confusion_matrix": cm,
    }

    (DOCS / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
