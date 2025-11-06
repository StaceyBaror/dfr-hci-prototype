# gateway/main.py
from __future__ import annotations

import os
import json
import time
import uuid
import base64
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import joblib
import jwt
from fastapi import (
    FastAPI, HTTPException, Request, Form, UploadFile, File,
    Depends, Header, Body
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Paths & globals
# ----------------------------
BASE = Path(__file__).resolve().parents[1]
ART = BASE / "data" / "artifacts"      # raw uploads + base records
EVID = BASE / "docs" / "evidence"      # detections, xai, reports, audit
MODELS = BASE / "data" / "models"      # model + provenance
ART.mkdir(parents=True, exist_ok=True)
EVID.mkdir(parents=True, exist_ok=True)

# Model + provenance
try:
    MODEL = joblib.load(MODELS / "model.pkl")  # Pipeline(tfidf->clf)
except Exception as e:
    MODEL = None
    logging.error("Failed to load model: %s", e)

MODEL_VERSION = (MODELS / "VERSION").read_text().strip() if (MODELS / "VERSION").exists() else "v0.1.0"
DATASET_SHA256 = (MODELS / "DATASET.SHA256").read_text().strip() if (MODELS / "DATASET.SHA256").exists() else "unknown"

# Auth
AUTH_SECRET = os.getenv("AUTH_SECRET") or os.getenv("DFR_SECRET") or "dev-secret"
ALGO = "HS256"
TOKEN_TTL = int(os.getenv("AUTH_TTL_SECONDS", "3600"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="DFR-HCI Prototype", version="0.1.0")

# CORS (relaxed for dev; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # e.g., ["https://your-ui.example.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UI support (templates & static) ---
TEMPLATES = Jinja2Templates(directory=str(BASE / "templates"))
STATIC_DIR = BASE / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ----------------------------
# Helpers
# ----------------------------
def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def append_audit(event: Dict[str, Any]) -> None:
    """Append-only JSONL audit log."""
    event.setdefault("ts", now_utc())
    (EVID / "audit.log.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (EVID / "audit.log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def issue_jwt(role: str, ttl_seconds: int = TOKEN_TTL) -> str:
    now = int(time.time())
    return jwt.encode({"role": role, "iat": now, "exp": now + ttl_seconds}, AUTH_SECRET, algorithm=ALGO)


def require_role(allowed: set[str]):
    """FastAPI DI-friendly dependency. Returns JWT claims if authorized."""
    def _inner(authorization: Optional[str] = Header(default=None)):
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, AUTH_SECRET, algorithms=[ALGO])
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
        role = payload.get("role")
        if role not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return payload
    return _inner


def create_artifact_record(message_id: str, text: str, dept: str, user: str, ts: Optional[str] = None) -> Dict[str, Any]:
    """Core creator used by API + UI to avoid auth coupling."""
    raw = text.encode("utf-8")
    h = sha256_hex(raw)
    artifact_id = f"{message_id}-{h[:8]}"
    rec = {
        "artifact_id": artifact_id,
        "sha256": h,
        "dept": dept,
        "user": user,
        "ts": ts or now_utc(),
        "text": text,
        "model_version_at_upload": MODEL_VERSION
    }
    save_json(ART / f"{artifact_id}.json", rec)
    append_audit({"event": "upload", "artifact_id": artifact_id, "sha256": h, "dept": dept, "user": user})
    return rec


def _iter_recent_artifacts(limit: int):
    """Yield most recent base artifact JSON paths (skip '-feat.json')."""
    items = []
    for p in ART.glob("*.json"):
        if p.name.endswith("-feat.json"):
            continue
        try:
            items.append((p.stat().st_mtime, p))
        except Exception:
            continue
    items.sort(reverse=True)
    for _, p in items[:limit]:
        yield p


# ----------------------------
# Middleware: request id + timing (robust)
# ----------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logging.exception("Unhandled error")
        append_audit({
            "rid": rid,
            "path": request.url.path,
            "method": request.method,
            "latency_ms": round(dt_ms, 2),
            "error": str(e),
        })
        response = JSONResponse({"detail": "Internal Server Error"}, status_code=500)
    else:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        append_audit({
            "rid": rid,
            "path": request.url.path,
            "method": request.method,
            "latency_ms": round(dt_ms, 2),
        })
    finally:
        logging.info(f"rid={rid} method={request.method} path={request.url.path} latency_ms={dt_ms:.2f}")
        response.headers["X-Request-ID"] = rid
    return response


# ----------------------------
# Schemas
# ----------------------------
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


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "service": "dfr-hci-gateway",
        "status": "ok",
        "time": now_utc(),
        "model_version": MODEL_VERSION,
        "dataset_sha256": DATASET_SHA256
    }


@app.get("/favicon.ico")
def favicon():
    # Silence browser 404 noise
    return JSONResponse(content={}, status_code=204)


@app.get("/auth/token")
def get_token(role: str = "user"):
    if role not in {"user", "analyst"}:
        raise HTTPException(status_code=400, detail="role must be 'user' or 'analyst'")
    return {"role": role, "token": issue_jwt(role)}


# -------- Uploads --------
@app.post("/upload")
def upload(inp: UploadIn, _claims=Depends(require_role({"user", "analyst"}))):
    rec = create_artifact_record(inp.message_id, inp.text, inp.dept, inp.user, inp.ts)
    return {"artifact_id": rec["artifact_id"], "sha256": rec["sha256"], "rid": str(uuid.uuid4())}


@app.post("/upload_file")
async def upload_file(
    message_id: str = Form(...),
    dept: str = Form("General"),
    user: str = Form("unknown@org"),
    file: UploadFile = File(...),
    _claims=Depends(require_role({"user", "analyst"})),
):
    raw = await file.read()
    text = raw.decode("utf-8", errors="replace")
    rec = create_artifact_record(message_id, text, dept, user, None)
    # keep filename in a sidecar
    side = rec.copy()
    side["filename"] = file.filename
    save_json(ART / f"{rec['artifact_id']}-meta.json", side)
    append_audit({"event": "upload_file", "artifact_id": rec["artifact_id"], "sha256": rec["sha256"], "filename": file.filename})
    return {"artifact_id": rec["artifact_id"], "sha256": rec["sha256"]}


@app.post("/analyze")
def analyze(inp: AnalyzeIn, _claims=Depends(require_role({"user", "analyst"}))):
    p = ART / f"{inp.artifact_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    base = load_json(p)
    feat = {
        "artifact_id": inp.artifact_id,
        "feature_ref": f"{inp.artifact_id}-feat",
        "feature_hash": base["sha256"],   # placeholder linkage
        "ts": now_utc()
    }
    save_json(ART / f"{inp.artifact_id}-feat.json", feat)
    append_audit({"event": "analyze", "artifact_id": inp.artifact_id})
    return feat


# -------- Detect --------
@app.post("/detect")
def detect(inp: DetectIn, _claims=Depends(require_role({"analyst"}))):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    base_p = ART / f"{inp.artifact_id}.json"
    if not base_p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    base = load_json(base_p)
    text: str = base["text"]

    # Verify checksum integrity before scoring
    checksum_verified = (sha256_hex(text.encode("utf-8")) == base["sha256"])

    # Predict
    proba = float(MODEL.predict_proba([text])[0, 1])
    yhat = int(proba >= inp.threshold)
    out = {
        "artifact_id": inp.artifact_id,
        "y_hat": "phishing" if yhat == 1 else "benign",
        "confidence": round(proba, 6),
        "threshold": inp.threshold,
        "alert": bool(yhat),
        "checksum_verified": bool(checksum_verified),
        "model_version": MODEL_VERSION,
        "dataset_sha256": DATASET_SHA256,
        "ts": now_utc()
    }
    save_json(EVID / f"{inp.artifact_id}-detect.json", out)
    append_audit({"event": "detect", "artifact_id": inp.artifact_id, "alert": bool(yhat), "checksum_verified": bool(checksum_verified)})
    return out


@app.post("/detect_batch")
def detect_batch(
    limit: int = 10,
    threshold: float = 0.80,
    _claims=Depends(require_role({"analyst"})),
):
    """Detect on the last N artifacts (by mtime)."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    art_files = sorted(
        (p for p in ART.glob("*.json") if not p.name.endswith("-feat.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    items, alerts = [], 0
    for p in art_files[: max(1, int(limit))]:
        aid = load_json(p)["artifact_id"]
        res = detect(DetectIn(artifact_id=aid, threshold=threshold))  # reuse logic
        alerts += 1 if res.get("alert") else 0
        items.append({"artifact_id": aid, "alert": bool(res.get("alert")), "confidence": res.get("confidence")})
    return {"count": len(items), "alerts": alerts, "items": items}


def _iter_recent_artifacts(limit: int):
    """Yield most recent base artifact JSON paths (skip '-feat.json')."""
    items = []
    for p in ART.glob("*.json"):
        if p.name.endswith("-feat.json"):
            continue
        try:
            items.append((p.stat().st_mtime, p))
        except Exception:
            continue
    items.sort(reverse=True)
    for _, p in items[:limit]:
        yield p


@app.post("/detect/latest")
def detect_latest(
    limit: int = Body(5, embed=True),
    threshold: float = Body(0.80, embed=True),
    _claims=Depends(require_role({"analyst"})),
):
    """Detect the latest N artifacts using the standard detect() path."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for p in _iter_recent_artifacts(limit):
        aid = load_json(p)["artifact_id"]
        res = detect(DetectIn(artifact_id=aid, threshold=threshold))  # reuse logic
        results.append(res)
    return {"count": len(results), "items": results}


# -------- XAI / Report --------
@app.get("/xai/{artifact_id}")
def xai(artifact_id: str, _claims=Depends(require_role({"analyst"}))):
    base_p = ART / f"{artifact_id}.json"
    if not base_p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = load_json(base_p)["text"]
    try:
        import shap, matplotlib.pyplot as plt

        def f(X: List[str]):  # probability for class 1
            return MODEL.predict_proba(X)[:, 1]

        explainer = shap.Explainer(f)
        sv = explainer([text])

        # Save bar plot
        out_png = EVID / f"{artifact_id}-shap.png"
        plt.figure()
        shap.plots.bar(sv, show=False)
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()

        # Minimal JSON summary
        shap_json = {"artifact_id": artifact_id, "explanation_type": "bar", "ts": now_utc()}
        save_json(EVID / f"{artifact_id}-shap.json", shap_json)

        b64 = base64.b64encode(out_png.read_bytes()).decode("ascii")
        append_audit({"event": "xai", "artifact_id": artifact_id, "png": out_png.name})
        return {"artifact_id": artifact_id, "shap_png_base64": b64}
    except Exception as e:
        err = {"artifact_id": artifact_id, "error": str(e), "ts": now_utc()}
        save_json(EVID / f"{artifact_id}-shap.error.json", err)
        raise HTTPException(status_code=500, detail=f"XAI failure: {e}")


@app.get("/report/{artifact_id}")
def report(artifact_id: str, _claims=Depends(require_role({"analyst"}))):
    base_p = ART / f"{artifact_id}.json"
    det_p = EVID / f"{artifact_id}-detect.json"
    if not base_p.exists() or not det_p.exists():
        raise HTTPException(status_code=404, detail="artifact or detection not found")

    base = load_json(base_p)
    det = load_json(det_p)
    shap_png = EVID / f"{artifact_id}-shap.png"
    shap_img_b64 = base64.b64encode(shap_png.read_bytes()).decode("ascii") if shap_png.exists() else None

    # Simple HTML report (for Chapter 10 screenshots / appendix)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DFR-HCI Report {artifact_id}</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;margin:24px}} code,pre{{background:#f6f8fa;padding:2px 4px}}</style>
</head><body>
<h2>Potential Cybercrime Incident Report</h2>
<p><b>Artifact:</b> {artifact_id}</p>
<p><b>Dept:</b> {base.get('dept')}</p>
<p><b>User:</b> {base.get('user')}</p>
<p><b>Uploaded:</b> {base.get('ts')}</p>
<p><b>SHA-256:</b> <code>{base.get('sha256')}</code></p>
<hr/>
<h3>Detection</h3>
<ul>
<li><b>Label:</b> {det.get('y_hat')}</li>
<li><b>Confidence:</b> {det.get('confidence')}</li>
<li><b>Threshold:</b> {det.get('threshold')}</li>
<li><b>Checksum verified:</b> {det.get('checksum_verified')}</li>
<li><b>Model version:</b> {det.get('model_version')}</li>
<li><b>Dataset hash:</b> {det.get('dataset_sha256')}</li>
<li><b>Scored at:</b> {det.get('ts')}</li>
</ul>
{"<h3>Explanation (SHAP)</h3><img style='max-width:720px' src='data:image/png;base64," + shap_img_b64 + "'/>" if shap_img_b64 else "<p><i>No SHAP image available.</i></p>"}
<hr/>
<h3>Original Text</h3>
<pre>{base.get('text')}</pre>
</body></html>"""
    out_html = EVID / f"{artifact_id}-report.html"
    out_html.write_text(html, encoding="utf-8")
    append_audit({"event": "report", "artifact_id": artifact_id, "report_file": out_html.name})
    return FileResponse(path=out_html, media_type="text/html", filename=out_html.name)


# -------- Viz API --------
@app.get("/viz/summary")
def viz_summary(_claims=Depends(require_role({"analyst"}))):
    detects = list(EVID.glob("*-detect.json"))
    total = 0
    alerts = 0
    by_dept: Dict[str, Dict[str, int]] = {}
    for p in detects:
        d = load_json(p)
        total += 1
        dept = "Unknown"
        a = ART / f"{d['artifact_id']}.json"
        if a.exists():
            dept = load_json(a).get("dept", "Unknown")
        by_dept.setdefault(dept, {"total": 0, "alerts": 0})
        by_dept[dept]["total"] += 1
        if d.get("alert"):
            alerts += 1
            by_dept[dept]["alerts"] += 1
    return {"total": total, "alerts": alerts, "by_dept": by_dept}


@app.get("/viz/recent")
def viz_recent(limit: int = 10, _claims=Depends(require_role({"analyst"}))):
    items = []
    files = sorted(EVID.glob("*-detect.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    for p in files:
        d = load_json(p)
        a = ART / f"{d['artifact_id']}.json"
        dept = "Unknown"
        user = "Unknown"
        if a.exists():
            aj = load_json(a)
            dept = aj.get("dept", dept)
            user = aj.get("user", user)
        items.append({
            "artifact_id": d["artifact_id"],
            "y_hat": d["y_hat"],
            "confidence": d.get("confidence"),
            "alert": d.get("alert"),
            "dept": dept,
            "user": user,
            "ts": d.get("ts"),
        })
    return {"items": items}


# -------- UI (no auth; UI calls protected APIs with token) --------
@app.get("/ui/upload")
def ui_upload(request: Request):
    return TEMPLATES.TemplateResponse(
        "upload.html",
        {"request": request, "now": now_utc()}
    )


@app.post("/ui/upload")
def ui_upload_post(
    request: Request,
    message_id: str = Form(...),
    text: str = Form(...),
    dept: str = Form("General"),
    user: str = Form("unknown@org"),
):
    rec = create_artifact_record(message_id, text, dept, user, None)
    result = {"artifact_id": rec["artifact_id"], "sha256": rec["sha256"]}
    return TEMPLATES.TemplateResponse(
        "upload.html",
        {"request": request, "now": now_utc(), "result": result,
         "prefill": {"message_id": message_id, "text": text, "dept": dept, "user": user}}
    )


@app.get("/ui/dashboard")
def ui_dashboard(request: Request):
    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {"request": request, "now": now_utc()}
    )
