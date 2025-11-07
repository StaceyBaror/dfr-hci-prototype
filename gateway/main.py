# gateway/main.py
from __future__ import annotations
import os
import json
import time
import uuid
import base64
import hashlib
from fastapi import Query, Cookie, Response
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import joblib
import jwt
from fastapi import (
    FastAPI, Header, HTTPException, Request, Form,
    UploadFile, File, Depends, Body, Cookie, Response, Query
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# ----------------------------
# Paths & globals
# ----------------------------
BASE = Path(__file__).resolve().parents[1]
ART = BASE / "data" / "artifacts"      # raw uploads + base records
EVID = BASE / "docs" / "evidence"      # detections, xai, reports, audit
MODELS = BASE / "data" / "models"      # model + provenance
ART.mkdir(parents=True, exist_ok=True)
EVID.mkdir(parents=True, exist_ok=True)

# Model + provenance (safe load)
MODEL_VERSION_FILE = MODELS / "VERSION"
DATASET_SHA_FILE = MODELS / "DATASET.SHA256"
MODEL_FILE = MODELS / "model.pkl"

MODEL_VERSION = MODEL_VERSION_FILE.read_text().strip() if MODEL_VERSION_FILE.exists() else "v0.1.0"
DATASET_SHA256 = DATASET_SHA_FILE.read_text().strip() if DATASET_SHA_FILE.exists() else "unknown"

try:
    MODEL = joblib.load(MODEL_FILE)  # e.g., Pipeline(tfidf -> clf)
except Exception as e:
    MODEL = None
    logging.warning("Model not loaded: %s", e)

# Auth
AUTH_SECRET = os.getenv("AUTH_SECRET", "change-me")  # set in .env for prod
ALGO = "HS256"
TOKEN_TTL = int(os.getenv("TOKEN_TTL", "3600"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="DFR-HCI Prototype", version="0.1.0")

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
    """
    Dependency that authorizes via Bearer token header OR 'auth_token' cookie.
    Returns decoded payload on success.
    """
    def _inner(
        request: Request,
        authorization: Optional[str] = Header(default=None),
        auth_token_cookie: Optional[str] = Cookie(default=None, alias="auth_token"),
    ):
        token: Optional[str] = None

        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1]
        elif auth_token_cookie:
            token = auth_token_cookie

        if not token:
            raise HTTPException(status_code=401, detail="Missing bearer token")

        try:
            payload = jwt.decode(token, AUTH_SECRET, algorithms=[ALGO])
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")

        role = payload.get("role")
        if role not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return payload
    return _inner

def _extract_token(
    authorization: Optional[str],
    token_q: Optional[str],
    token_cookie: Optional[str],
) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1]
    if token_q:
        return token_q
    if token_cookie:
        return token_cookie
    return None


def require_role_any(allowed: set[str]):
    """Dependency: find token in Authorization header, ?token=, or cookie; then check role."""
    def _dep(
        authorization: Optional[str] = Header(default=None),
        token_q: Optional[str] = Query(default=None),
        auth_token: Optional[str] = Cookie(default=None),
    ) -> None:
        token = _extract_token(authorization, token_q, auth_token)
        if not token:
            raise HTTPException(status_code=401, detail="Missing bearer token")
        try:
            payload = jwt.decode(token, AUTH_SECRET, algorithms=[ALGO])
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
        if payload.get("role") not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
    return _dep
# --- helper: extract token from header OR ?token OR cookie (for IMG/PDF links) ---
def _extract_token(authorization: Optional[str], token_q: Optional[str], auth_token: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1]
    if token_q:
        return token_q
    if auth_token:
        return auth_token
    return None

METRICS_JSON = EVID / "metrics.json"

def _metrics_load() -> Dict[str, Any]:
    if METRICS_JSON.exists():
        try:
            return load_json(METRICS_JSON)
        except Exception as e:
            logging.warning(f"Unable to parse metrics.json: {e}")
    return {}

# ---------- Raw metrics as JSON (e.g., precision/recall/F1) ----------
@app.get("/viz/metrics")
def viz_metrics(authorization: Optional[str] = Header(default=None)):
    require_role({"analyst"})(authorization)
    m = _metrics_load()
    if not m:
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return m

# ---------- Confusion matrix as JSON ----------
@app.get("/viz/confusion")
def viz_confusion(authorization: Optional[str] = Header(default=None)):
    require_role({"analyst"})(authorization)
    m = _metrics_load()
    # Expected structure from your evaluate.py:
    # { "labels": ["benign","phishing"], "confusion_matrix": [[tn, fp],[fn, tp]], ... }
    labels = m.get("labels") or m.get("classes") or ["benign", "phishing"]
    cm = m.get("confusion_matrix")
    if not cm:
        # fall back: empty matrix
        cm = [[0, 0], [0, 0]]
    return {"labels": labels, "matrix": cm}

# ---------- Confusion matrix heatmap PNG (nice for thesis screenshots) ----------
@app.get("/viz/confusion.png")
def viz_confusion_png(
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Cookie(default=None),
):
    # accept header OR ?token= OR cookie
    tok = _extract_token(authorization, token, auth_token)
    require_role({"analyst"})(f"Bearer {tok}" if tok else None)

    m = _metrics_load()
    labels = m.get("labels") or m.get("classes") or ["benign", "phishing"]
    cm = m.get("confusion_matrix") or [[0, 0], [0, 0]]

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=200)
        cm_arr = np.array(cm, dtype=float)
        im = ax.imshow(cm_arr, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm_arr):
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=0)
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        logging.warning(f"confusion.png generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to render confusion matrix: {e}")

# ---------- XAI PNG (for <img> thumbnails) ----------
@app.get("/xai/png/{artifact_id}")
def xai_png(
    artifact_id: str,
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Cookie(default=None),
):
    tok = _extract_token(authorization, token, auth_token)
    require_role({"analyst"})(f"Bearer {tok}" if tok else None)

    # If PNG exists, serve; otherwise, generate via existing /xai path logic
    png_path = EVID / f"{artifact_id}-shap.png"
    if not png_path.exists():
        # trigger XAI generation (will raise if fails)
        _ = xai(artifact_id, authorization=f"Bearer {tok}")
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="XAI PNG not found")
    return FileResponse(png_path, media_type="image/png", filename=png_path.name)

# ---------- Time series counts for chart (last N days) ----------
@app.get("/viz/series")
def viz_series(days: int = 30, authorization: Optional[str] = Header(default=None)):
    require_role({"analyst"})(authorization)
    import datetime as dt
    today = dt.datetime.utcnow().date()
    start = today - dt.timedelta(days=max(1, days) - 1)
    # initialize buckets
    series = { (start + dt.timedelta(days=i)).isoformat(): {"total": 0, "alerts": 0} for i in range(days) }

    for p in EVID.glob("*-detect.json"):
        d = load_json(p)
        ts = d.get("ts")
        try:
            day = ts[:10]  # YYYY-MM-DD
        except Exception:
            continue
        if day in series:
            series[day]["total"] += 1
            if d.get("alert"):
                series[day]["alerts"] += 1

    # emit arrays sorted by day
    days_sorted = sorted(series.keys())
    totals = [series[d]["total"] for d in days_sorted]
    alerts = [series[d]["alerts"] for d in days_sorted]
    return {"days": days_sorted, "totals": totals, "alerts": alerts}

# ---------- PDF report (HTML -> PDF via WeasyPrint if available) ----------
@app.get("/report_pdf/{artifact_id}")
def report_pdf(
    artifact_id: str,
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Cookie(default=None),
):
    tok = _extract_token(authorization, token, auth_token)
    require_role({"analyst"})(f"Bearer {tok}" if tok else None)

    # Ensure HTML exists (calls your existing /report logic to generate it)
    html_path = EVID / f"{artifact_id}-report.html"
    if not html_path.exists():
        # call existing /report to (re)build
        _ = report(artifact_id, authorization=f"Bearer {tok}")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Report HTML not found")

    pdf_path = EVID / f"{artifact_id}-report.pdf"
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)
    except Exception as e:
        # Fall back: send HTML if PDF not possible
        logging.warning(f"PDF generation failed: {e}")
        return FileResponse(html_path, media_type="text/html", filename=html_path.name)

# ===== Confusion matrix / metrics helpers =====
METRICS_JSON = EVID / "metrics.json"

def _metrics_load() -> Dict[str, Any]:
    if METRICS_JSON.exists():
        try:
            return load_json(METRICS_JSON)
        except Exception as e:
            logging.warning(f"Unable to parse metrics.json: {e}")
    return {}

# ---------- Raw metrics as JSON (e.g., precision/recall/F1) ----------
@app.get("/viz/metrics")
def viz_metrics(authorization: Optional[str] = Header(default=None)):
    require_role({"analyst"})(authorization)
    m = _metrics_load()
    if not m:
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return m

# ---------- Confusion matrix as JSON ----------
@app.get("/viz/confusion")
def viz_confusion(authorization: Optional[str] = Header(default=None)):
    require_role({"analyst"})(authorization)
    m = _metrics_load()
    # Expected structure from your evaluate.py:
    # { "labels": ["benign","phishing"], "confusion_matrix": [[tn, fp],[fn, tp]], ... }
    labels = m.get("labels") or m.get("classes") or ["benign", "phishing"]
    cm = m.get("confusion_matrix")
    if not cm:
        # fall back: empty matrix
        cm = [[0, 0], [0, 0]]
    return {"labels": labels, "matrix": cm}

# ---------- Confusion matrix heatmap PNG (nice for thesis screenshots) ----------
@app.get("/viz/confusion.png")
def viz_confusion_png(
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Cookie(default=None),
):
    # accept header OR ?token= OR cookie
    tok = _extract_token(authorization, token, auth_token)
    require_role({"analyst"})(f"Bearer {tok}" if tok else None)

    m = _metrics_load()
    labels = m.get("labels") or m.get("classes") or ["benign", "phishing"]
    cm = m.get("confusion_matrix") or [[0, 0], [0, 0]]

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=200)
        cm_arr = np.array(cm, dtype=float)
        im = ax.imshow(cm_arr, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm_arr):
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=0)
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        logging.warning(f"confusion.png generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to render confusion matrix: {e}")

# ----------------------------
# Middleware: request id + timing
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
@app.get("/health")
def health():
    return {"ok": True, "time": now_utc()}


@app.get("/")
def root():
    return {
        "service": "dfr-hci-gateway",
        "status": "ok",
        "time": now_utc(),
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
        "dataset_sha256": DATASET_SHA256
    }


@app.get("/favicon.ico")
def favicon():
    # Silence browser 404 noise
    return JSONResponse(content={}, status_code=204)


@app.get("/auth/token")
def get_token(role: str = "user", set_cookie: bool = True, response: Response = None):
    if role not in {"user", "analyst"}:
        raise HTTPException(status_code=400, detail="role must be 'user' or 'analyst'")
    tok = issue_jwt(role)
    # Optionally set HttpOnly cookie so browser UI calls work without manually adding headers
    if set_cookie and response is not None:
        response.set_cookie(
            "auth_token",
            tok,
            httponly=True,
            samesite="lax",
            secure=False,  # set True if serving over HTTPS
            max_age=60 * 60,
        )
    return {"role": role, "token": tok, "cookie_set": bool(set_cookie)}

@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie("auth_token")
    return {"detail": "logged out"}


@app.post("/upload")
def upload(inp: UploadIn, authorization: Optional[str] = Header(default=None)):
    # Optional: enforce auth → require_role({"user", "analyst"})(request, authorization)
    rid = str(uuid.uuid4())
    raw = inp.text.encode("utf-8")
    h = sha256_hex(raw)
    artifact_id = f"{inp.message_id}-{h[:8]}"
    rec = {
        "artifact_id": artifact_id,
        "sha256": h,
        "dept": inp.dept,
        "user": inp.user,
        "ts": inp.ts or now_utc(),
        "text": inp.text,
        "model_version_at_upload": MODEL_VERSION
    }
    save_json(ART / f"{artifact_id}.json", rec)
    append_audit({"rid": rid, "event": "upload", "artifact_id": artifact_id, "sha256": h, "dept": inp.dept, "user": inp.user})
    return {"artifact_id": artifact_id, "sha256": h, "rid": rid}


@app.post("/upload_file")
async def upload_file(
    message_id: str = Form(...),
    dept: str = Form("General"),
    user: str = Form("unknown@org"),
    file: UploadFile = File(...),
):
    raw = await file.read()
    h = sha256_hex(raw)
    artifact_id = f"{message_id}-{h[:8]}"
    text = raw.decode("utf-8", errors="replace")
    rec = {
        "artifact_id": artifact_id,
        "sha256": h,
        "dept": dept,
        "user": user,
        "ts": now_utc(),
        "text": text,
        "filename": file.filename,
        "model_version_at_upload": MODEL_VERSION,
    }
    save_json(ART / f"{artifact_id}.json", rec)
    append_audit({"event": "upload_file", "artifact_id": artifact_id, "sha256": h, "filename": file.filename})
    return {"artifact_id": artifact_id, "sha256": h}

@app.post("/analyze")
def analyze(inp: AnalyzeIn, authorization: Optional[str] = Header(default=None)):
    # Optional: enforce auth → require_role({"user", "analyst"})(request, authorization)
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


@app.post("/detect")
def detect(
    inp: DetectIn,
    _payload = Depends(require_role({"analyst"})),
):
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
    out_p = EVID / f"{inp.artifact_id}-detect.json"
    save_json(out_p, out)
    append_audit({"event": "detect", "artifact_id": inp.artifact_id, "alert": bool(yhat), "checksum_verified": bool(checksum_verified)})
    return out

@app.post("/detect_batch")
def detect_batch(
    limit: int = Body(10, embed=True),
    threshold: float = Body(0.80, embed=True),
    _= Depends(require_role({"analyst"})),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    art_files = sorted(
        [p for p in ART.glob("*.json") if not p.name.endswith("-feat.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    processed = 0
    alerts = 0
    items = []
    for p in art_files[: max(1, int(limit))]:
        aj = load_json(p)
        aid = aj["artifact_id"]
        text = aj["text"]
        proba = float(MODEL.predict_proba([text])[0, 1])
        yhat = int(proba >= threshold)

        out = {
            "artifact_id": aid,
            "y_hat": "phishing" if yhat == 1 else "benign",
            "confidence": round(proba, 6),
            "threshold": threshold,
            "alert": bool(yhat),
            "model_version": MODEL_VERSION,
            "dept": aj.get("dept", "General"),
            "user": aj.get("user", "unknown@org"),
            "ts": now_utc(),
        }
        save_json(EVID / f"{aid}-detect.json", out)
        processed += 1
        if out["alert"]:
            alerts += 1
        items.append(out)

    append_audit({"event": "detect_batch", "count": processed, "alerts": alerts})
    return {"count": processed, "alerts": alerts, "items": items}

@app.get("/xai/{artifact_id}")
def xai(
    artifact_id: str,
    _= Depends(require_role({"analyst"})),
):
    base_p = ART / f"{artifact_id}.json"
    if not base_p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    text = load_json(base_p)["text"]

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # SHAP for callable probability; works for many text pipelines.
    try:
        import shap, matplotlib.pyplot as plt  # heavy deps; import inside

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
        # Persist the error for traceability
        err = {"artifact_id": artifact_id, "error": str(e), "ts": now_utc()}
        save_json(EVID / f"{artifact_id}-shap.error.json", err)
        raise HTTPException(status_code=500, detail=f"XAI failure: {e}")

@app.get("/report/{artifact_id}")
def report(
    artifact_id: str,
    _payload = Depends(require_role({"analyst"})),
):
    base_p = ART / f"{artifact_id}.json"
    det_p = EVID / f"{artifact_id}-detect.json"
    if not base_p.exists() or not det_p.exists():
        raise HTTPException(status_code=404, detail="artifact or detection not found")

    base = load_json(base_p)
    det = load_json(det_p)
    shap_png = EVID / f"{artifact_id}-shap.png"
    shap_img_b64 = base64.b64encode(shap_png.read_bytes()).decode("ascii") if shap_png.exists() else None

    # Simple HTML report for Chapter 10 screenshots / appendix
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

@app.get("/viz/summary")
def viz_summary(_= Depends(require_role({"analyst"}))):
    # Visual summary for dashboard widgets (counts + by dept)
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


# ---------- UI: Upload ----------
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
    payload = UploadIn(message_id=message_id, text=text, dept=dept, user=user)
    result = upload(payload)  # no auth enforced here; adjust if needed
    return TEMPLATES.TemplateResponse(
        "upload.html",
        {"request": request, "now": now_utc(), "result": result, "prefill": payload.dict()}
    )

# ----- helpers for recent artifacts -----
def _iter_recent_artifacts(limit: int):
    items = []
    for p in ART.glob("*.json"):
        if p.name.endswith("-feat.json"):
            continue
        try:
            stat = p.stat()
            items.append((stat.st_mtime, p))
        except Exception:
            continue
    items.sort(reverse=True)
    for _, p in items[:limit]:
        yield p

@app.post("/detect/latest")
def detect_latest(
    limit: int = Body(5, embed=True),
    threshold: float = Body(0.80, embed=True),
    _= Depends(require_role({"analyst"})),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for p in _iter_recent_artifacts(limit):
        base = load_json(p)
        artifact_id = base["artifact_id"]
        text = base["text"]
        proba = float(MODEL.predict_proba([text])[0, 1])
        yhat = int(proba >= threshold)
        out = {
            "artifact_id": artifact_id,
            "y_hat": "phishing" if yhat == 1 else "benign",
            "confidence": round(proba, 6),
            "threshold": threshold,
            "alert": bool(yhat),
            "model_version": MODEL_VERSION,
            "dept": base.get("dept", "General"),
            "user": base.get("user", "unknown@org"),
            "ts": now_utc(),
        }
        save_json(EVID / f"{artifact_id}-detect.json", out)
        results.append(out)
    append_audit({"event": "detect_latest", "count": len(results)})
    return {"count": len(results), "items": results}

# ---------- UI: Dashboard ----------

@app.get("/ui/dashboard")
def ui_dashboard(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    token_q: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Cookie(default=None),
):
    token = _extract_token(authorization, token_q, auth_token)
    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {"request": request, "now": now_utc(), "token": token}
    )

# ---------- JSON helper: recent detections (for dashboard list) ----------
@app.get("/viz/recent")
def viz_recent(
    limit: int = 10,
    _= Depends(require_role_any({"analyst"})),
):
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