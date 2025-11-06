# eval/latency_bench.py
import os, time, json, statistics, requests
from pathlib import Path

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
N = int(os.getenv("N", "30"))             # total detect calls to run
LAST = int(os.getenv("LAST_N", "10"))     # how many recent artifacts to use

def get_token(role="analyst"):
    r = requests.get(f"{BASE}/auth/token", params={"role": role}, timeout=10)
    r.raise_for_status()
    return r.json()["token"]

def ensure_recent_artifacts(token, last_n):
    """Fetch recent artifacts. If none, create one (upload->analyze) and return again."""
    hdr = {"Authorization": "Bearer " + token}
    r = requests.get(f"{BASE}/viz/recent", headers=hdr, params={"limit": last_n}, timeout=10)
    if r.status_code == 401:
        raise SystemExit("Unauthorized. Did you pass the analyst token?")
    r.raise_for_status()
    items = r.json().get("items", [])
    if items:
        return [it["artifact_id"] for it in items]

    # Create one synthetic artifact
    u = requests.post(f"{BASE}/upload",
                      json={"message_id":"bench",
                            "text":"Urgent: reset your password immediately.",
                            "dept":"IT","user":"bench@corp"},
                      timeout=10).json()
    aid = u["artifact_id"]
    requests.post(f"{BASE}/analyze", json={"artifact_id": aid}, timeout=10)
    return [aid]

def run_bench(token, artifact_ids, total_n=30, threshold=0.8):
    hdr = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    lat = []
    idx = 0
    while len(lat) < total_n:
        aid = artifact_ids[idx % len(artifact_ids)]
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/detect",
                          headers=hdr,
                          json={"artifact_id": aid, "threshold": threshold},
                          timeout=30)
        r.raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000.0)  # ms
        idx += 1
    # p95: nearest-rank on sorted latencies
    lat_sorted = sorted(lat)
    p95 = lat_sorted[max(0, int(round(0.95 * len(lat_sorted))) - 1)]
    return {
        "base_url": BASE,
        "n": total_n,
        "artifact_count": len(artifact_ids),
        "mean_ms": round(sum(lat)/len(lat), 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(min(lat), 2),
        "max_ms": round(max(lat), 2),
    }

def main():
    token = get_token("analyst")
    arts = ensure_recent_artifacts(token, LAST)
    res = run_bench(token, arts, total_n=N)
    outdir = Path("docs/evidence"); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "latency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
