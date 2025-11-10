# gateway/main.py
from __future__ import annotations

import os
import io
import json
import time
import uuid
import base64
import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, Any, List

import joblib
import jwt
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    Form,
    UploadFile,
    File,
    Depends,
    Body,
    Cookie,
    Query,
    Response,
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
EVID = BASE / "docs" / "evidence"      # detections, xai, reports, audit, metrics
MODELS = BASE / "data" / "models"      # model + provenance

ART.mkdir(parents=True, exist_ok=True)
EVID.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS / "model.pkl"
MODEL_VERSION_FILE = MODELS / "VERSION"
DATASET_SHA_FILE = MODELS / "DATASET.SHA256"

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
    log_path = EVID / "audit.log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def issue_jwt(role: str, ttl_seconds: int = TOKEN_TTL) -> str:
    now = int(time.time())
    return jwt.encode(
        {"role": role, "iat": now, "exp": now + ttl_seconds},
        AUTH_SECRET,
        algorithm=ALGO,
    )


def _decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, AUTH_SECRET, algorithms=[ALGO])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def _pick_token(
    authorization: Optional[str],
    token_q: Optional[str],
    auth_cookie: Optional[str],
) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1]
    if token_q:
        return token_q
    if auth_cookie:
        return auth_cookie
    return None


def require_role(allowed: set[str]):
    """
    Dependency for JSON/API routes.
    Looks at Authorization header or auth_token cookie.
    """
    def _dep(
        authorization: Optional[str] = Header(default=None),
        auth_token: Optional[str] = Cookie(default=None, alias="auth_token"),
    ) -> Dict[str, Any]:
        token = _pick_token(authorization, None, auth_token)
        if not token:
            raise HTTPException(status_code=401, detail="Missing bearer token")
        payload = _decode_token(token)
        if payload.get("role") not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return payload
    return _dep


def require_role_link(allowed: set[str]):
    """
    Dependency for link-based routes (Report / PDF / XAI PNG / confusion PNG).
    Accepts Authorization header OR ?token= OR auth_token cookie.
    """
    def _dep(
        authorization: Optional[str] = Header(default=None),
        token: Optional[str] = Query(default=None),
        auth_token: Optional[str] = Cookie(default=None, alias="auth_token"),
    ) -> Dict[str, Any]:
        tok = _pick_token(authorization, token, auth_token)
        if not tok:
            raise HTTPException(status_code=401, detail="Missing bearer token")
        payload = _decode_token(tok)
        if payload.get("role") not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return payload
    return _dep


# ===== Confusion matrix / metrics helpers =====
METRICS_JSON = EVID / "metrics.json"


def _metrics_load() -> Dict[str, Any]:
    if METRICS_JSON.exists():
        try:
            return load_json(METRICS_JSON)
        except Exception as e:
            logging.warning(f"Unable to parse metrics.json: {e}")
    return {}


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
# Basic routes
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
        "dataset_sha256": DATASET_SHA256,
    }


@app.get("/favicon.ico")
def favicon():
    # Silence browser 404 noise
    return JSONResponse(content={}, status_code=204)


# ----------------------------
# Auth routes
# ----------------------------
@app.get("/auth/token")
def get_token(role: str = "user"):
    if role not in {"user", "analyst"}:
        raise HTTPException(status_code=400, detail="role must be 'user' or 'analyst'")
    tok = issue_jwt(role)
    return {"role": role, "token": tok}


@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie("auth_token")
    return {"detail": "logged out"}


# ----------------------------
# Upload / analyze / detect
# ----------------------------
@app.post("/upload")
def upload(inp: UploadIn):
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
        "model_version_at_upload": MODEL_VERSION,
    }
    save_json(ART / f"{artifact_id}.json", rec)
    append_audit({
        "rid": rid,
        "event": "upload",
        "artifact_id": artifact_id,
        "sha256": h,
        "dept": inp.dept,
        "user": inp.user,
    })
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
    append_audit({
        "event": "upload_file",
        "artifact_id": artifact_id,
        "sha256": h,
        "filename": file.filename,
    })
    return {"artifact_id": artifact_id, "sha256": h}


@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    p = ART / f"{inp.artifact_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    base = load_json(p)
    feat = {
        "artifact_id": inp.artifact_id,
        "feature_ref": f"{inp.artifact_id}-feat",
        "feature_hash": base["sha256"],   # placeholder linkage
        "ts": now_utc(),
    }
    save_json(ART / f"{inp.artifact_id}-feat.json", feat)
    append_audit({"event": "analyze", "artifact_id": inp.artifact_id})
    return feat


@app.post("/detect")
def detect(
    inp: DetectIn,
    _payload=Depends(require_role({"analyst"})),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    base_p = ART / f"{inp.artifact_id}.json"
    if not base_p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    base = load_json(base_p)
    text: str = base["text"]
    checksum_verified = (sha256_hex(text.encode("utf-8")) == base["sha256"])

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
        "ts": now_utc(),
    }
    out_p = EVID / f"{inp.artifact_id}-detect.json"
    save_json(out_p, out)
    append_audit({
        "event": "detect",
        "artifact_id": inp.artifact_id,
        "alert": bool(yhat),
        "checksum_verified": bool(checksum_verified),
    })
    return out


@app.post("/detect_batch")
def detect_batch(
    limit: int = Body(10, embed=True),
    threshold: float = Body(0.80, embed=True),
    _payload=Depends(require_role({"analyst"})),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    art_files = sorted(
        [p for p in ART.glob("*.json") if not p.name.endswith("-feat.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
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
    _payload=Depends(require_role({"analyst"})),
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


# ----------------------------
# XAI (simple term-frequency) & PNG
# ----------------------------
@app.get("/xai/{artifact_id}")
def xai(
    artifact_id: str,
    _payload=Depends(require_role({"analyst"})),
):
    """
    Simple, robust XAI endpoint without SHAP.

    Creates a bar chart of the top tokens in the message as a
    lightweight explanation and stores it as <artifact>-shap.png.
    """
    base_p = ART / f"{artifact_id}.json"
    if not base_p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    base = load_json(base_p)
    text = (base.get("text") or "").strip()
    if not text:
        meta = {
            "artifact_id": artifact_id,
            "explanation_type": "none",
            "reason": "empty_text",
            "ts": now_utc(),
        }
        save_json(EVID / f"{artifact_id}-shap.json", meta)
        return {
            "artifact_id": artifact_id,
            "shap_png_base64": None,
            "note": "No text available for explanation.",
        }

    try:
        import matplotlib.pyplot as plt

        words = [w.lower() for w in text.split()]
        counts = Counter(words)
        top = counts.most_common(10)
        if not top:
            top = [("no-tokens", 1)]

        labels, values = zip(*top)
        out_png = EVID / f"{artifact_id}-shap.png"

        fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Top terms (lightweight explanation)")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

        meta = {
            "artifact_id": artifact_id,
            "explanation_type": "term_freq",
            "ts": now_utc(),
        }
        save_json(EVID / f"{artifact_id}-shap.json", meta)

        b64 = base64.b64encode(out_png.read_bytes()).decode("ascii")
        append_audit({"event": "xai_term_freq", "artifact_id": artifact_id})
        return {
            "artifact_id": artifact_id,
            "shap_png_base64": b64,
            "note": "Simple token-frequency explanation.",
        }
    except Exception as e:
        err = {
            "artifact_id": artifact_id,
            "error": str(e),
            "ts": now_utc(),
        }
        save_json(EVID / f"{artifact_id}-shap.error.json", err)
        raise HTTPException(status_code=500, detail="XAI unavailable; see logs.")


@app.get("/xai/png/{artifact_id}")
def xai_png(
    artifact_id: str,
    _payload=Depends(require_role_link({"analyst"})),
):
    """
    Lightweight PNG-serving endpoint.

    If the PNG does not exist yet, it calls /xai to generate it.
    """
    png_path = EVID / f"{artifact_id}-shap.png"
    if not png_path.exists():
        # generate via same logic (without auth dependency)
        _ = xai(artifact_id, _payload={})  # _payload not used inside

    if not png_path.exists():
        raise HTTPException(status_code=404, detail="XAI PNG not found")

    return FileResponse(png_path, media_type="image/png", filename=png_path.name)


# ----------------------------
# Reports (HTML + PDF)
# ----------------------------
@app.get("/report/{artifact_id}")
def report(
    artifact_id: str,
    _payload=Depends(require_role_link({"analyst"})),
):
    base_p = ART / f"{artifact_id}.json"
    det_p = EVID / f"{artifact_id}-detect.json"
    if not base_p.exists() or not det_p.exists():
        raise HTTPException(status_code=404, detail="artifact or detection not found")

    base = load_json(base_p)
    det = load_json(det_p)
    shap_png = EVID / f"{artifact_id}-shap.png"
    shap_img_b64 = base64.b64encode(shap_png.read_bytes()).decode("ascii") if shap_png.exists() else None

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DFR-HCI Report {artifact_id}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:24px}}
code,pre{{background:#f6f8fa;padding:2px 4px}}
</style>
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
{("<h3>Explanation (XAI)</h3><img style='max-width:720px' src='data:image/png;base64," + shap_img_b64 + "'/>") if shap_img_b64 else "<p><i>No SHAP image available.</i></p>"}
<hr/>
<h3>Original Text</h3>
<pre>{base.get('text')}</pre>
</body></html>"""
    out_html = EVID / f"{artifact_id}-report.html"
    out_html.write_text(html, encoding="utf-8")
    append_audit({"event": "report", "artifact_id": artifact_id, "report_file": out_html.name})
    return FileResponse(path=out_html, media_type="text/html", filename=out_html.name)


@app.get("/report_pdf/{artifact_id}")
def report_pdf(
    artifact_id: str,
    _payload=Depends(require_role_link({"analyst"})),
):
    """
    PDF wrapper around the HTML report.
    Falls back to HTML if WeasyPrint is not available.
    """
    html_path = EVID / f"{artifact_id}-report.html"
    if not html_path.exists():
        _ = report(artifact_id, _payload={})  # build HTML

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Report HTML not found")

    pdf_path = EVID / f"{artifact_id}-report.pdf"
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)
    except Exception as e:
        logging.warning(f"PDF generation failed, returning HTML instead: {e}")
        return FileResponse(html_path, media_type="text/html", filename=html_path.name)


# ----------------------------
# Metrics / viz endpoints
# ----------------------------
@app.get("/viz/metrics")
def viz_metrics(_payload=Depends(require_role({"analyst"}))):
    """Full metrics.json – precision, recall, F1, etc."""
    m = _metrics_load()
    if not m:
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return m


@app.get("/viz/confusion")
def viz_confusion(_payload=Depends(require_role({"analyst"}))):
    """Confusion matrix as JSON."""
    m = _metrics_load()
    labels = m.get("labels") or m.get("classes") or ["benign", "phishing"]
    cm = m.get("confusion_matrix") or [[0, 0], [0, 0]]
    return {"labels": labels, "matrix": cm}


@app.get("/viz/confusion.png")
def viz_confusion_png(
    _payload=Depends(require_role_link({"analyst"})),
):
    """Confusion matrix as PNG – useful for thesis figures."""
    m = _metrics_load()
    labels = m.get("labels") or m.get("classes") or ["benign", "phishing"]
    cm = m.get("confusion_matrix") or [[0, 0], [0, 0]]

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=200)
        cm_arr = np.array(cm, dtype=float)
        ax.imshow(cm_arr, cmap="Blues")
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


@app.get("/viz/summary")
def viz_summary(_payload=Depends(require_role({"analyst"}))):
    """Summary counts for dashboard cards."""
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
def viz_recent(
    limit: int = 10,
    _payload=Depends(require_role({"analyst"})),
):
    """Recent detections for the dashboard list."""
    items = []
    files = sorted(
        EVID.glob("*-detect.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
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


@app.get("/viz/series")
def viz_series(
    days: int = 30,
    _payload=Depends(require_role({"analyst"})),
):
    """Simple time series (per-day totals and alerts) for charts if needed."""
    import datetime as dt

    days = max(1, int(days))
    today = dt.datetime.utcnow().date()
    start = today - dt.timedelta(days=days - 1)
    series = {(start + dt.timedelta(days=i)).isoformat(): {"total": 0, "alerts": 0} for i in range(days)}

    for p in EVID.glob("*-detect.json"):
        d = load_json(p)
        ts = d.get("ts")
        if not ts:
            continue
        day = ts[:10]
        if day in series:
            series[day]["total"] += 1
            if d.get("alert"):
                series[day]["alerts"] += 1

    days_sorted = sorted(series.keys())
    totals = [series[d]["total"] for d in days_sorted]
    alerts = [series[d]["alerts"] for d in days_sorted]
    return {"days": days_sorted, "totals": totals, "alerts": alerts}


# ----------------------------
# UI: Upload
# ----------------------------
@app.get("/ui/upload")
def ui_upload(request: Request):
    return TEMPLATES.TemplateResponse(
        "upload.html",
        {"request": request, "now": now_utc()},
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
    result = upload(payload)
    return TEMPLATES.TemplateResponse(
        "upload.html",
        {"request": request, "now": now_utc(), "result": result, "prefill": payload.dict()},
    )


# ----------------------------
# UI: Dashboard
# ----------------------------
@app.get("/ui/dashboard")
def ui_dashboard(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    token_q: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Cookie(default=None, alias="auth_token"),
):
    token = _pick_token(authorization, token_q, auth_token)
    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {"request": request, "now": now_utc(), "token": token},
    )