from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json, time, base64, joblib
from typing import Optional

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "data" / "artifacts"
EVID = BASE / "docs" / "evidence"
MODELS = BASE / "data" / "models"
ART.mkdir(parents=True, exist_ok=True)
EVID.mkdir(parents=True, exist_ok=True)

MODEL = joblib.load(MODELS / "model.pkl")  # Pipeline(tfidf->clf)

app = FastAPI(title="DFR-HCI Prototype", version="0.1.0")

def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256_hex(b: bytes):
    import hashlib; return hashlib.sha256(b).hexdigest()

def save_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

class UploadIn(BaseModel):
    message_id: str
    text: str
    dept: str = "General"
    user: str = "unknown@org"
    ts: Optional[str] = None

class AnalyzeIn(BaseModel):
    artifact_id: str

class DetectIn(BaseModel):
    artifact_id: str
    threshold: float = 0.80

@app.get("/")
def root():
    return {"service": "dfr-hci-gateway", "status": "ok", "time": now_utc()}

@app.post("/upload")
def upload(inp: UploadIn):
    raw = inp.text.encode("utf-8")
    h = sha256_hex(raw)
    artifact_id = f"{inp.message_id}-{h[:8]}"
    rec = {
        "artifact_id": artifact_id, "sha256": h, "dept": inp.dept, "user": inp.user,
        "ts": inp.ts or now_utc(), "text": inp.text
    }
    save_json(ART / f"{artifact_id}.json", rec)
    return {"artifact_id": artifact_id, "sha256": h}

@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    p = ART / f"{inp.artifact_id}.json"
    if not p.exists():
        return {"error": "artifact not found"}
    # In this MVP, TF-IDF lives in the model pipeline, so we just stash a feature ref
    save_json(ART / f"{inp.artifact_id}-feat.json", {"ref": inp.artifact_id})
    return {"artifact_id": inp.artifact_id, "feature_ref": f"{inp.artifact_id}-feat"}

@app.post("/detect")
def detect(inp: DetectIn):
    base = ART / f"{inp.artifact_id}.json"
    if not base.exists():
        return {"error": "artifact not found"}
    text = json.loads(base.read_text())["text"]
    proba = float(MODEL.predict_proba([text])[0,1])
    yhat = int(proba >= inp.threshold)
    out = {
        "artifact_id": inp.artifact_id,
        "y_hat": "phishing" if yhat==1 else "benign",
        "confidence": proba,
        "threshold": inp.threshold,
        "alert": bool(yhat),
        "model_version": "v0.1.0",
        "ts": now_utc()
    }
    save_json(EVID / f"{inp.artifact_id}-detect.json", out)
    return out

@app.get("/xai/{artifact_id}")
def xai(artifact_id: str):
    # Minimal SHAP bar plot over the pipeline
    try:
        import shap, matplotlib.pyplot as plt
        base = ART / f"{artifact_id}.json"
        if not base.exists(): return {"error": "artifact not found"}
        text = json.loads(base.read_text())["text"]
        def f(X): return MODEL.predict_proba(X)[:,1]
        explainer = shap.Explainer(f)
        sv = explainer([text])
        plt.figure()
        shap.plots.bar(sv, show=False)
        out_png = EVID / f"{artifact_id}-shap.png"
        plt.savefig(out_png, bbox_inches="tight"); plt.close()
        b64 = base64.b64encode(out_png.read_bytes()).decode("ascii")
        return {"artifact_id": artifact_id, "shap_png_base64": b64}
    except Exception as e:
        return {"artifact_id": artifact_id, "error": str(e)}
