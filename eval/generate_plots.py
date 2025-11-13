#!/usr/bin/env python3
"""
Generate ROC, PR, Confusion Matrix, and a SHAP placeholder as PNGs for the thesis.
Defaults match evaluate.py (samples/model paths). Headless-safe (Agg backend).
"""

import os, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import argparse, joblib, pandas as pd

def normalize_label(v):
    if pd.isna(v): return None
    if isinstance(v, (int, float)): return int(v)
    s = str(v).strip().lower()
    if s in {"1","true","phish","phishing","malicious","spam","attack"}: return 1
    if s in {"0","false","ham","benign","legit","legitimate","normal"}: return 0
    return 0

def load_predictions(samples_csv: Path, model_path: Path, threshold: float):
    df = pd.read_csv(samples_csv)
    text_col = next((c for c in ["text","message","content","body"] if c in df.columns), None)
    label_col = next((c for c in ["label","y","target","class"] if c in df.columns), None)
    if text_col is None or label_col is None:
        raise SystemExit("CSV must have one text column (text/message/content/body) and one label column (label/y/target/class).")
    y_true = df[label_col].apply(normalize_label).astype(int).values
    texts  = df[text_col].astype(str).tolist()
    model  = joblib.load(model_path)
    y_prob = model.predict_proba(texts)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_true, y_pred, y_prob

def save_confusion(y_true, y_pred, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix"); plt.xticks([0,1], ["Benign","Incident"]); plt.yticks([0,1], ["Benign","Incident"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    return str(path), cm.tolist()

def save_roc(y_true, y_prob, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={roc_auc:.3f})"); plt.tight_layout()
    path = out_dir / "roc.png"
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    return str(path), roc_auc, fpr.tolist(), tpr.tolist()

def save_pr(y_true, y_prob, out_dir: Path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AUC={pr_auc:.3f})"); plt.tight_layout()
    path = out_dir / "pr.png"
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    return str(path), pr_auc, rec.tolist(), prec.tolist()

def save_shap_placeholder(out_dir: Path):
    plt.figure()
    plt.title("SHAP placeholder — replace when model explainer is wired")
    plt.text(0.5, 0.5, "SHAP Figure", ha="center", va="center")
    plt.axis("off")
    path = out_dir / "shap-example.png"
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    return str(path)

def main():
    p = argparse.ArgumentParser(description="Generate thesis plots (ROC, PR, CM, SHAP placeholder).")
    p.add_argument("--samples", default="data/samples/samples.csv", help="Path to CSV dataset")
    p.add_argument("--model",   default="data/models/model.pkl",   help="Joblib model pipeline")
    p.add_argument("--threshold", type=float, default=float(os.getenv("THRESHOLD", 0.5)),
                   help="Decision threshold (default 0.5 or $THRESHOLD)")
    p.add_argument("--out", default="docs/figures", help="Output folder for images")
    p.add_argument("--manifest", default="docs/evidence/plots_manifest.json", help="Where to write a JSON manifest")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob = load_predictions(Path(args.samples), Path(args.model), float(args.threshold))

    cm_path, cm_json   = save_confusion(y_true, y_pred, out_dir)
    roc_path, roc_auc, fpr, tpr = save_roc(y_true, y_prob, out_dir)
    pr_path,  pr_auc, rec, prec = save_pr(y_true, y_prob, out_dir)
    shap_path = save_shap_placeholder(out_dir)

    manifest = {
        "confusion_matrix_png": cm_path,
        "roc_png": roc_path, "roc_auc": roc_auc, "fpr": fpr, "tpr": tpr,
        "pr_png": pr_path,   "pr_auc": pr_auc,   "recall": rec, "precision": prec,
        "shap_png": shap_path,
        "threshold": float(args.threshold),
        "n": int(len(y_true))
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()