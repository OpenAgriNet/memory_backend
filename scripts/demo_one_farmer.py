"""Explicit one-farmer demo operations. Backups/logs stay outside the git checkout.

inspect is read-only. reset backs up the memory collection, then removes only
non-settings points from that collection. Other Qdrant collections are untouched.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
from runtime_config import required, qdrant_headers
WORK = ROOT.parent / "amul-memory-demo"
WORK.mkdir(mode=0o700, exist_ok=True)
STATE = WORK / "current.json"


def request(method, url, **kwargs):
    response = requests.request(method, url, timeout=90, headers=qdrant_headers(url), **kwargs)
    response.raise_for_status()
    return response.json()


def scan(base, collection):
    points, offset = [], None
    while True:
        body = {"limit": 200, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        result = request("POST", f"{base}/collections/{collection}/points/scroll", json=body)["result"]
        points += result["points"]
        offset = result.get("next_page_offset")
        if offset is None:
            return points


def inspect():
    details = json.loads(subprocess.check_output(["docker", "inspect", "amul-memory-api"], text=True))[0]
    env = dict(item.split("=", 1) for item in details["Config"].get("Env", []) if "=" in item)
    collection = env.get("QDRANT_COLLECTION", "amul_memory")
    if collection != "amul_memory":
        raise RuntimeError("Unexpected collection; inspect before changing the reset target")
    base = required("QDRANT_URL").rstrip("/")
    points = scan(base, collection)
    settings = {p["payload"]["farmer_id"]: p["payload"].get("memory_enabled", True)
                for p in points if p["payload"].get("type") == "settings"}
    farmers = collections.defaultdict(lambda: {"traces": set(), "sessions": set(), "points": 0})
    for point in points:
        pl = point["payload"]
        farmer = pl.get("farmer_id")
        if not farmer or farmer.startswith(("__", "zz-")) or pl.get("type") in ("settings", "key_doc", "profile_record", "memory_chunk"):
            continue
        stats = farmers[farmer]
        stats["traces"].update(pl.get("source_traces") or [])
        stats["sessions"].update(pl.get("source_sessions") or [pl.get("source_session_id")])
        stats["points"] += 1
    eligible = [fid for fid in farmers if settings.get(fid, True) is True]
    if not eligible:
        raise RuntimeError("No existing enabled demo farmer found")
    chosen = max(eligible, key=lambda fid: (len(farmers[fid]["sessions"] - {None}), len(farmers[fid]["traces"])))
    vllm = required("DREAMER_BASE_URL").rstrip("/")
    model = request("GET", vllm + "/models")["data"]
    state = {"farmer_id": chosen, "qdrant": base, "collection": collection,
             "model": model[0]["id"], "vllm": vllm,
             "service_env": {k: v for k, v in env.items() if k.startswith(("MARQO_", "EMBED_"))},
             "demo_api": required("MEMORY_API_URL")}
    STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps({"collection": collection, "points": len(points),
                      "record_kinds": dict(collections.Counter(p["payload"].get("type") for p in points)),
                      "farmers": [{"suffix": fid[-4:], "sessions": len(v["sessions"] - {None}),
                                   "traces": len(v["traces"]), "enabled": settings.get(fid, True)}
                                  for fid, v in farmers.items()],
                      "selected_suffix": chosen[-4:], "model": state["model"],
                      "state_file": str(STATE)}, indent=2))


def reset():
    state = json.loads(STATE.read_text())
    points = scan(state["qdrant"], state["collection"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = WORK / f"before-reset-{timestamp}.json"
    backup.write_text(json.dumps({"collection": state["collection"], "points": points}, ensure_ascii=False))
    os.chmod(backup, 0o600)
    # Read back the complete backup before any destructive request.
    assert len(json.loads(backup.read_text())["points"]) == len(points)
    keep = [p for p in points if p["payload"].get("type") == "settings"]
    request("POST", f"{state['qdrant']}/collections/{state['collection']}/points/delete?wait=true", json={
        "filter": {"must_not": [{"key": "type", "match": {"value": "settings"}}]}})
    remaining = scan(state["qdrant"], state["collection"])
    assert {p["id"]: p["payload"] for p in remaining} == {p["id"]: p["payload"] for p in keep}
    state["backup"] = str(backup)
    STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps({"backup": str(backup), "removed_memory_points": len(points) - len(keep),
                      "preserved_settings": len(keep)}))


def source():
    state = json.loads(STATE.read_text())
    sys.path.insert(0, str(ROOT / "dreamer"))
    import lf
    turns = lf.LangfuseRO().farmer_turns(state["farmer_id"], days_back=40,
                                      until="2026-09-09T00:00:00Z")
    turns = [t for t in turns if t.get("query", "").strip()]
    if not turns:
        raise RuntimeError("No source turns for the selected farmer")
    sessions = list(dict.fromkeys(t.get("session_id") for t in turns))
    held_sessions = set(sessions[-max(1, min(2, len(sessions) // 5)):])
    split = next(i for i, t in enumerate(turns) if t.get("session_id") in held_sessions)
    if split < 2:
        raise RuntimeError("Insufficient earlier conversations for a held-out test")
    from datetime import timedelta
    cutoff = datetime.fromisoformat(turns[split]["ts"].replace("Z", "+00:00")) - timedelta(milliseconds=1)
    state["cutoff"] = cutoff.isoformat()
    state["source_file"] = str(WORK / "source_turns.json")
    Path(state["source_file"]).write_text(json.dumps(turns, ensure_ascii=False, indent=2))
    os.chmod(state["source_file"], 0o600)
    state["held_out_file"] = str(WORK / "held_out.json")
    Path(state["held_out_file"]).write_text(json.dumps(turns[split:], ensure_ascii=False, indent=2))
    os.chmod(state["held_out_file"], 0o600)
    STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps({"farmer_suffix": state["farmer_id"][-4:], "total_turns": len(turns),
                      "sessions": len(sessions), "cutoff": state["cutoff"],
                      "earlier_turns": split, "held_out_turns": len(turns) - split,
                      "first_source": turns[0]["ts"], "last_source": turns[-1]["ts"]}, indent=2))


def start():
    state = json.loads(STATE.read_text())
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8101)) == 0:
            raise RuntimeError("Port 8101 already in use; leaving that process untouched")
    env = dict(os.environ, **state["service_env"], QDRANT_URL=state["qdrant"],
               QDRANT_COLLECTION=state["collection"], PYTHONUNBUFFERED="1")
    with (WORK / "service.log").open("a") as output:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "memory_api:app", "--host", "127.0.0.1", "--port", "8101"],
                                   cwd=ROOT / "service", env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    state["service_pid"] = process.pid
    STATE.write_text(json.dumps(state, indent=2))
    import time
    for _ in range(40):
        if process.poll() is not None:
            raise RuntimeError("Demo service failed; inspect service.log")
        try:
            response = request("GET", state["demo_api"] + "/health")
            print(json.dumps({"demo_api": state["demo_api"], "pid": process.pid, "health": response}))
            return
        except requests.RequestException:
            time.sleep(.25)
    raise RuntimeError("Demo service did not become healthy")


def restart():
    import signal
    import time
    state = json.loads(STATE.read_text())
    pid = state["service_pid"]
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    if b"uvicorn memory_api:app" not in command or b"8101" not in command:
        raise RuntimeError("Recorded PID is not the local demo API; leaving it untouched")
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(.1)
    start()


def schema():
    result = subprocess.run(["docker", "run", "--rm", "--network", "none",
                            "-e", "AMUL_OAN_API_PATH=/test-bot", "-e", "PYTHONDONTWRITEBYTECODE=1",
                            "-v", f"{ROOT}:/test-memory:ro", "-v", f"{ROOT.parent / 'amul-oan-api'}:/test-bot:ro",
                            "--entrypoint", "python", "amul-oan-api_app:latest", "/test-memory/tests/check_tool_schema.py"])
    sys.exit(result.returncode)


def mini():
    state = json.loads(STATE.read_text())
    service = state["service_env"]
    env = dict(os.environ, QDRANT_API_KEY=qdrant_headers(state["qdrant"]).get("api-key", ""))
    previous = WORK / "mini-test.json"
    if previous.exists():
        previous.rename(WORK / ("mini-test-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"))
    result = subprocess.run(["docker", "run", "--rm", "--network", "host",
                            "--user", f"{os.getuid()}:{os.getgid()}", "-e", "AMUL_OAN_API_PATH=/test-bot", "-e", "PYTHONDONTWRITEBYTECODE=1",
                            "-e", "MEMORY_ENABLED=true", "-e", "QDRANT_API_KEY",
                            "-e", "QDRANT_URL=" + state["qdrant"], "-e", "QDRANT_COLLECTION=" + state["collection"],
                            "-e", "MARQO_ENDPOINT_URL=" + service["MARQO_URL"],
                            "-e", "MARQO_INDEX_NAME=" + service["MARQO_INDEX"],
                            "-e", "MEMORY_EMBED_DIM=" + str(service.get("EMBED_DIM", "1024")),
                            "-v", f"{ROOT}:/test-memory:ro", "-v", f"{ROOT.parent / 'amul-oan-api'}:/test-bot:ro",
                            "-v", f"{WORK}:/demo", "--entrypoint", "python", "amul-oan-api_app:latest",
                            "/test-memory/tests/demo_agent.py"], env=env)
    sys.exit(result.returncode)


def export():
    state = json.loads(STATE.read_text())
    points = scan(state["qdrant"], state["collection"])
    path = WORK / "memories.json"
    path.write_text(json.dumps(points, ensure_ascii=False, indent=2))
    os.chmod(path, 0o600)
    current = [p for p in points if p["payload"].get("type") == "memory"
               and not p["payload"].get("superseded_at")]
    print(json.dumps({"points": len(points), "current_episodes": len(current),
                      "episodes": [{"id": p["id"], **{k: p["payload"].get(k) for k in
                                    ("headline", "contents", "chunk_count", "times_raised", "status")}}
                                   for p in current]}, ensure_ascii=False, indent=2))


def consolidate():
    state = json.loads(STATE.read_text())
    env = dict(os.environ, DREAMER_PROVIDER="vllm", DREAMER_MODEL=state["model"],
               DREAMER_BASE_URL=state["vllm"], MEMORY_API_URL=state["demo_api"], PYTHONUNBUFFERED="1")
    with (WORK / "consolidation.log").open("w") as output:
        result = subprocess.run([sys.executable, str(ROOT / "dreamer/dreamer.py"),
                                 "--farmer", state["farmer_id"], "--consolidate-only"],
                                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(f"Consolidation exited {result.returncode}; log {WORK / 'consolidation.log'}")
    sys.exit(result.returncode)


def dream():
    state = json.loads(STATE.read_text())
    env = dict(os.environ, DREAMER_PROVIDER="vllm", DREAMER_MODEL=state["model"],
               DREAMER_BASE_URL=state["vllm"], MEMORY_API_URL=state["demo_api"], PYTHONUNBUFFERED="1")
    # The established demo data comes from August. Read a fixed historical window
    # and reserve later questions for the one-farmer held-out mini-test.
    cutoff = state.get("cutoff", "2026-09-01T00:00:00Z")
    command = [sys.executable, str(ROOT / "dreamer/dreamer.py"), "--farmer", state["farmer_id"],
               "--days", "30", "--until", cutoff]
    log = WORK / "dreamer.log"
    print(f"Running configured vLLM for farmer …{state['farmer_id'][-4:]}; cutoff {cutoff}; log {log}", flush=True)
    with log.open("w") as output:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(f"Dreamer exited {result.returncode}; log {log}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "source", "start", "reset", "dream", "schema", "export", "mini", "restart", "consolidate"])
    args = parser.parse_args()
    globals()[args.action]()
