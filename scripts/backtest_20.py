"""Reproducible 20-farmer memory backtest. Private artifacts live outside git.

All Langfuse access is GET-only. The backtest owns an isolated memory collection.
Reply replay uses recorded Amul agent requests and frozen tool results, never live
operational tools. No requests are sent to farmers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.getenv("AMUL_BACKTEST_WORK", str(ROOT.parent / "amul-memory-backtest" / "20260909-20-farmers")))
WORK.mkdir(parents=True, mode=0o700, exist_ok=True)
os.chmod(WORK, 0o700)
sys.path.insert(0, str(ROOT / "dreamer"))
sys.path.insert(0, str(ROOT / "service"))
from runtime_config import required, qdrant_headers
import lf


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    path.chmod(0o600)


def substantive(query):
    query = query.strip()
    return len(query) >= 10 and sum(c.isalpha() for c in query) >= 6


def candidates():
    path = WORK / "candidates.json"
    if path.exists():
        return json.loads(path.read_text())
    client = lf.LangfuseRO()
    found = {}
    # Deterministic recent sampling, selected before looking at memory outcomes.
    for day in range(7, 1, -1):
        start = f"2026-09-{day:02d}T00:00:00Z"
        stop = f"2026-09-{day+1:02d}T00:00:00Z"
        for page in range(1, 6):
            try:
                data = client._get("/api/public/traces", {"name": lf.CHAT_TRACE_NAME,
                    "fromTimestamp": start, "toTimestamp": stop, "limit": 100, "page": page})
            except TimeoutError:
                # A small session sample can still identify candidates without a huge trace scan.
                data = client._get("/api/public/traces", {"name": lf.CHAT_TRACE_NAME,
                    "fromTimestamp": start, "toTimestamp": start[:11] + "06:00:00Z", "limit": 100, "page": page})
            for row in data.get("data", []):
                farmer = str(row.get("userId") or "")
                if farmer and farmer.lower() not in {"anonymous", "undefined", "none"} and not farmer.startswith(("zz-", "__")):
                    found[farmer] = True
            if page >= (data.get("meta", {}).get("totalPages") or 1):
                break
        print(f"Candidate scan {day}: {len(found)} farmers", flush=True)
        if len(found) >= 100:
            break
    ids = sorted(found, key=lambda fid: hashlib.sha256(("amul-memory-20-v1:" + fid).encode()).hexdigest())
    save(path, ids)
    return ids


def fetch_farmer(farmer):
    name = hashlib.sha256(farmer.encode()).hexdigest()[:16]
    path = WORK / f"source-{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = lf.LangfuseRO().farmer_turns(farmer, days_back=40, until="2026-09-09T00:00:00Z")
    rows = [r for r in rows if (r.get("query") or "").strip()]
    save(path, rows)
    return rows


def select():
    if (WORK / "cohort.json").exists():
        print("Cohort already frozen; leaving it unchanged")
        return
    ids = candidates()
    cohort, rejected = [], []
    # Fetch small batches concurrently; inspect in stable selection order.
    for start in range(0, len(ids), 8):
        batch = ids[start:start+8]
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {fid: pool.submit(fetch_farmer, fid) for fid in batch}
            for fid in batch:
                try:
                    turns = jobs[fid].result()
                except Exception as exc:
                    rejected.append({"id_hash": hashlib.sha256(fid.encode()).hexdigest()[:12], "reason": str(exc)[:150]})
                    continue
                sessions = list(dict.fromkeys(t.get("session_id") for t in turns))
                if len(turns) < 15 or len(sessions) < 3:
                    continue
                # Latest feasible session boundary with >=5 later substantive questions
                # and at least 10 earlier turns. Different histories yield distinct cutoffs.
                options = []
                for i in range(10, len(turns)):
                    if turns[i].get("session_id") == turns[i-1].get("session_id"):
                        continue
                    later = [t for t in turns[i:] if substantive(t["query"])]
                    if len(later) >= 5:
                        options.append((i, later))
                if not options:
                    continue
                split, later = options[-1]
                cutoff = datetime.fromisoformat(turns[split]["ts"].replace("Z", "+00:00")) - timedelta(milliseconds=1)
                lower = cutoff - timedelta(days=30)
                training = [t for t in turns[:split] if datetime.fromisoformat(t["ts"].replace("Z", "+00:00")) >= lower]
                if len(training) < 10:
                    continue
                alias = f"farmer_{len(cohort)+1:02d}"
                item = {"alias": alias, "farmer_id": fid, "cutoff": cutoff.isoformat(),
                        "training_turns": len(training), "training_sessions": len({t.get('session_id') for t in training}),
                        "source_path": str(WORK / ("source-" + hashlib.sha256(fid.encode()).hexdigest()[:16] + ".json")),
                        "heldout_candidates": later}
                save(WORK / f"{alias}-training.json", training)
                cohort.append(item)
                print(f"Selected {alias}: {len(training)} earlier turns, {len(later)} eligible later queries; cutoff {cutoff.isoformat()}", flush=True)
                if len(cohort) == 20:
                    break
        if len(cohort) == 20:
            break
    save(WORK / "selection-errors.json", rejected)
    if len(cohort) != 20:
        save(WORK / "partial-cohort.json", cohort)
        raise RuntimeError(f"Only {len(cohort)} eligible farmers found; extend candidate sampling")
    config = {"qdrant": required("QDRANT_URL"),
              "service_env": {k: v for k, v in os.environ.items() if k.startswith(("MARQO_", "EMBED_"))},
              "vllm": required("DREAMER_BASE_URL"), "model": required("DREAMER_MODEL"),
              "collection": required("BACKTEST_COLLECTION"), "api": required("BACKTEST_API_URL")}
    save(WORK / "config.json", config)
    save(WORK / "cohort.json", cohort)
    print(f"Frozen 20-farmer cohort in {WORK}", flush=True)


def start():
    import socket
    import requests
    config = json.loads((WORK / "config.json").read_text())
    from urllib.parse import urlparse
    port = urlparse(config["api"]).port
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            response = requests.get(config["api"] + "/health", timeout=10).json()
            if response.get("collection") != config["collection"]:
                raise RuntimeError("Backtest port belongs to a different service")
            print("Backtest API already running")
            return
    env = dict(os.environ, **config["service_env"], QDRANT_URL=config["qdrant"],
               QDRANT_COLLECTION=config["collection"], PYTHONUNBUFFERED="1")
    with (WORK / "service.log").open("a") as output:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "memory_api:app", "--host", "127.0.0.1", "--port", str(port)],
                                   cwd=ROOT / "service", env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    config["service_pid"] = process.pid
    save(WORK / "config.json", config)
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("Backtest API stopped; see service.log")
        try:
            response = requests.get(config["api"] + "/health", timeout=2)
            response.raise_for_status()
            assert response.json()["collection"] == config["collection"]
            print(f"Isolated backtest API ready on {port}", flush=True)
            return
        except requests.RequestException:
            time.sleep(.1)
    raise RuntimeError("Backtest API did not become ready")


def train():
    config = json.loads((WORK / "config.json").read_text())
    cohort = json.loads((WORK / "cohort.json").read_text())
    env = dict(os.environ, DREAMER_PROVIDER="vllm", DREAMER_MODEL=config["model"],
               DREAMER_BASE_URL=config["vllm"], MEMORY_API_URL=config["api"], PYTHONUNBUFFERED="1")
    def worker(farmer):
        alias = farmer["alias"]
        status = WORK / f"{alias}-training-status.json"
        if status.exists() and json.loads(status.read_text()).get("complete"):
            return alias, 0
        started = time.monotonic()
        save(status, {"complete": False, "started": datetime.now(timezone.utc).isoformat()})
        print(f"Training {alias}: {farmer['training_turns']} frozen turns", flush=True)
        with (WORK / f"{alias}-dreamer.log").open("w") as output:
            result = subprocess.run([sys.executable, str(ROOT / "dreamer/dreamer.py"),
                "--farmer", farmer["farmer_id"], "--until", farmer["cutoff"],
                "--turns-file", str(WORK / f"{alias}-training.json")],
                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        save(status, {"complete": result.returncode == 0, "returncode": result.returncode,
                      "seconds": time.monotonic() - started})
        print(f"Finished {alias}: code={result.returncode}, {time.monotonic()-started:.1f}s", flush=True)
        return alias, result.returncode
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(worker, cohort))
    if any(code for _, code in results):
        raise RuntimeError(f"Training failures: {[alias for alias, code in results if code]}")
    print(f"All {len(cohort)} farmers trained", flush=True)


def prepare():
    from replay_records import prepare as prepare_record
    cohort = json.loads((WORK / "cohort.json").read_text())
    case_dir = WORK / "cases"
    case_dir.mkdir(mode=0o700, exist_ok=True)
    cache = WORK / "traces"
    cache.mkdir(mode=0o700, exist_ok=True)
    def worker(farmer):
        rows, failures = [], []
        client = lf.LangfuseRO()
        # A fixed spread across eligible later queries, then deterministic fallbacks
        # for recordings with incomplete telemetry. Selection never reads outcomes.
        candidates = farmer["heldout_candidates"]
        primary = [i * (len(candidates)-1) // 4 for i in range(5)]
        indices = list(dict.fromkeys(primary + list(range(len(candidates)))))
        for index in indices:
            turn = candidates[index]
            path = cache / (turn["trace_id"] + ".json")
            try:
                if path.exists():
                    trace = json.loads(path.read_text())
                else:
                    trace = client._get("/api/public/traces/" + turn["trace_id"])
                    save(path, trace)
                replay = prepare_record(trace, turn["query"])
            except Exception as exc:
                failures.append({"trace_id": turn["trace_id"], "reason": str(exc)[:200]})
                continue
            case_id = farmer["alias"] + f"_q{len(rows)+1:02d}"
            record = dict(replay, case_id=case_id, farmer=farmer["alias"], farmer_id=farmer["farmer_id"],
                          cutoff=farmer["cutoff"], query=turn["query"], query_ts=turn["ts"], trace_id=turn["trace_id"],
                          session_id=turn.get("session_id"))
            save(case_dir / (case_id + ".json"), record)
            rows.append(case_id)
            if len(rows) == 5:
                break
        save(WORK / (farmer["alias"] + "-fixture-errors.json"), failures)
        print(f"Prepared {farmer['alias']}: {len(rows)} paired questions", flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(worker, cohort))
    ids = [case for group in groups for case in group]
    save(WORK / "case-index.json", ids)
    if len(ids) != 100:
        raise RuntimeError(f"Only {len(ids)} complete replay fixtures; inspect missing farmers")
    print("Prepared 100 frozen Amul agent requests", flush=True)


def snapshot():
    import requests
    config = json.loads((WORK / "config.json").read_text())
    cohort = json.loads((WORK / "cohort.json").read_text())
    farmers = {row["farmer_id"]: row for row in cohort}
    points, offset = [], None
    while True:
        body = {"limit": 200, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        response = requests.post(config["qdrant"] + "/collections/" + config["collection"] + "/points/scroll", json=body,
                                 headers=qdrant_headers(config["qdrant"]), timeout=30)
        response.raise_for_status()
        result = response.json()["result"]
        points += result["points"]
        offset = result.get("next_page_offset")
        if offset is None:
            break
    save(WORK / "memory-snapshot.json", points)
    rows = []
    sys.path.insert(0, str(ROOT / "service"))
    import episode
    for fid, farmer in farmers.items():
        source = {t["trace_id"]: t for t in json.loads((WORK / (farmer["alias"] + "-training.json")).read_text())}
        owned = [p for p in points if p["payload"].get("farmer_id") == fid]
        episodes = [p for p in owned if p["payload"].get("type") == "memory" and not p["payload"].get("superseded_at")]
        for point in owned:
            payload = point["payload"]
            if payload.get("type") != "memory":
                continue
            assert set(payload.get("source_traces") or []) <= set(source), "non-training source in memory"
            text, chunks, contents = episode.layout(payload)
            episode.validate_contents(contents, len(chunks))
            assert "".join(text[c["start"]:c["end"]] for c in chunks) == text
            assert all(c["end"] - c["start"] <= episode.CHUNK_MAX_CHARS for c in chunks)
        rows.append({"farmer": farmer["alias"], "cutoff": farmer["cutoff"],
                     "training_turns": farmer["training_turns"], "episodes": len(episodes),
                     "chunks": sum(p["payload"].get("chunk_count", 0) for p in episodes),
                     "standing_records": sum(p["payload"].get("type") == "profile_record" for p in owned)})
    manifest = {"farmers": rows, "points": len(points), "cutoff_audit": "passed",
                "snapshot_sha256": hashlib.sha256((WORK / "memory-snapshot.json").read_bytes()).hexdigest(),
                "bot_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "amul-oan-api", text=True).strip(),
                "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for folder in ("dreamer", "service", "scripts") for path in (ROOT / folder).glob("*.py")}}
    save(WORK / "manifest.json", manifest)
    print(json.dumps({"farmers": len(rows), "episodes": sum(r["episodes"] for r in rows),
                      "standing_records": sum(r["standing_records"] for r in rows), "audit": "passed"}), flush=True)


def reply():
    config = json.loads((WORK / "config.json").read_text())
    service = config["service_env"]
    env = dict(os.environ, QDRANT_API_KEY=qdrant_headers(config["qdrant"]).get("api-key", ""))
    command = ["docker", "run", "--rm", "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}",
               "-e", "AMUL_OAN_API_PATH=/test-bot", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "MEMORY_ENABLED=true", "-e", "QDRANT_API_KEY",
               "-e", "QDRANT_URL=" + config["qdrant"], "-e", "QDRANT_COLLECTION=" + config["collection"],
               "-e", "MARQO_ENDPOINT_URL=" + service["MARQO_URL"],
               "-e", "MARQO_INDEX_NAME=" + service["MARQO_INDEX"],
               "-e", "MEMORY_EMBED_DIM=" + str(service.get("EMBED_DIM", "1024")),
               "-e", "AMUL_BACKTEST_LIMIT=" + os.getenv("AMUL_BACKTEST_LIMIT", "0"),
               "-v", f"{ROOT}:/test-memory:ro", "-v", f"{ROOT.parent / 'amul-oan-api'}:/test-bot:ro",
               "-v", f"{WORK}:/backtest", "--entrypoint", "python", "amul-oan-api_app:latest",
               "/test-memory/tests/backtest_reply.py"]
    result = subprocess.run(command, env=env)
    sys.exit(result.returncode)


def batch_sizes():
    from followup_study import batch_sizes as check
    check(WORK)

def gate():
    from followup_study import gate as check
    check(WORK)

def policy():
    from followup_study import policy as check
    check(WORK)

def followup_prepare():
    from followup_study import prepare as prepare_followups
    prepare_followups(WORK)


def check_qdrant():
    """Read-only connectivity check; never prints credentials or collection data."""
    import requests
    response = requests.get(required("QDRANT_URL").rstrip("/") + "/collections",
                            headers=qdrant_headers(), timeout=30)
    response.raise_for_status()
    save(WORK / "qdrant-connectivity.json", {"ok": True,
         "url": required("QDRANT_URL"), "collections": len(response.json()["result"]["collections"])})
    print("Authenticated Qdrant collection listing succeeded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["select", "start", "train", "prepare", "reply", "snapshot", "policy", "gate", "batch_sizes", "followup_prepare", "check_qdrant"])
    parser.add_argument("--work", type=Path, help="Separate experiment directory")
    args = parser.parse_args()
    if args.work:
        WORK = args.work.resolve()
        WORK.mkdir(parents=True, mode=0o700, exist_ok=True)
    globals()[args.action]()
