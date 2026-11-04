"""Periodic memory writer. The only state is one high-water mark, kept in Qdrant.

Each cycle reads farmer turns from Langfuse between the mark and now minus
MEMORY_SETTLE_SECONDS (so a conversation in progress is left for the next cycle), runs
the dreamer once per farmer over their new turns, then moves the mark forward. One
worker, one window at a time: no queue, no leases.

A farmer whose run fails is logged and skipped; the mark still advances, so that
window is not retried for them.

  python worker.py            # loop every MEMORY_SCAN_INTERVAL_SECONDS
  python worker.py --once     # one cycle (cron, manual runs)
  python worker.py --once --dry-run   # decide, write nothing, leave the mark alone
"""
import argparse
import datetime as dt
import os
import re
import time
import traceback
import uuid
from types import SimpleNamespace

import requests

import dreamer
import lf
from runtime_config import qdrant_headers, required

QDRANT_URL = required("QDRANT_URL").rstrip("/")
STATE_COLLECTION = os.getenv("MEMORY_WORKER_STATE_COLLECTION", "amul_memory_worker_state")
STATE_POINT = str(uuid.uuid5(uuid.NAMESPACE_URL, "amul-memory-worker:high-water-mark"))
SETTLE = dt.timedelta(seconds=int(os.getenv("MEMORY_SETTLE_SECONDS", str(40 * 60))))
INTERVAL = int(os.getenv("MEMORY_SCAN_INTERVAL_SECONDS", str(30 * 60)))
FIRST_LOOKBACK = dt.timedelta(seconds=int(os.getenv("MEMORY_FIRST_LOOKBACK_SECONDS", str(24 * 3600))))
# Comma-separated phones; empty means every eligible farmer.
ALLOWLIST = {p.strip()[-10:] for p in os.getenv("MEMORY_FARMER_ALLOWLIST", "").split(",") if p.strip()}


def _qdrant(method, path, body=None):
    r = requests.request(method, f"{QDRANT_URL}{path}", json=body, timeout=30,
                         headers=qdrant_headers(f"{QDRANT_URL}{path}"))
    r.raise_for_status()
    return r.json()


def read_mark():
    try:
        got = _qdrant("POST", f"/collections/{STATE_COLLECTION}/points",
                      {"ids": [STATE_POINT], "with_payload": True})["result"]
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            _qdrant("PUT", f"/collections/{STATE_COLLECTION}", {"vectors": {"size": 1, "distance": "Dot"}})
            return None
        raise
    return dt.datetime.fromisoformat(got[0]["payload"]["mark"]) if got else None


def write_mark(mark):
    _qdrant("PUT", f"/collections/{STATE_COLLECTION}/points?wait=true", {"points": [
        {"id": STATE_POINT, "vector": [0.0], "payload": {"mark": mark.isoformat()}}]})


def _phone(value):
    digits = re.sub(r"\D", "", value) if isinstance(value, str) else ""
    return digits[-10:] if len(digits) >= 10 and len(set(digits[-10:])) > 1 else None


def eligible(row):
    """(farmer_id, turn) for a successful farmer turn.v1 trace, else None.

    Voice turns count only when the caller was signed in; the farmer id must match the
    trace's userId.
    """
    md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    pipeline = md.get("amul.pipeline")
    if (md.get("amul.schema_version") != lf.TURN_SCHEMA_VERSION or md.get("amul.service") != "amul-oan-api"
            or md.get("amul.persona") != "farmer" or md.get("amul.outcome") != "success"
            or pipeline not in ("webapp", "whatsapp", "telephony")):
        return None
    if pipeline == "telephony" and str(md.get("amul.signed_in")).lower() != "true":
        return None
    farmer = _phone(md.get("amul.farmer_id"))
    query, answer = row.get("input"), row.get("output")
    if (not farmer or farmer != md.get("amul.farmer_id") or _phone(row.get("userId")) != farmer
            or not isinstance(query, str) or not query.strip()
            or not isinstance(answer, str) or not answer.strip()):
        return None
    return farmer, {"trace_id": row["id"], "ts": row.get("timestamp"), "session_id": row.get("sessionId"),
                    "query": query.strip(), "answer": answer.strip()}


def cycle(dry_run=False):
    now = dt.datetime.now(dt.timezone.utc)
    start = read_mark() or now - FIRST_LOOKBACK
    stop = now - SETTLE
    if stop <= start:
        print(f"nothing settled since {start.isoformat()}")
        return
    rows = lf.LangfuseRO().traces(start, stop, name=lf.TURN_TRACE_NAME)
    turns = {}
    for row in rows:
        hit = eligible(row)
        if hit and (not ALLOWLIST or hit[0] in ALLOWLIST):
            turns.setdefault(hit[0], {})[hit[1]["trace_id"]] = hit[1]
    print(f"window {start.isoformat()} .. {stop.isoformat()}: {len(rows)} turn traces, "
          f"{sum(len(t) for t in turns.values())} eligible from {len(turns)} farmer(s)")
    args = SimpleNamespace(dry_run=dry_run, pipeline=False, no_headline_pass=False,
                           no_consolidate=False, no_profile_pass=False)
    failed = 0
    for farmer, by_id in turns.items():
        ordered = sorted(by_id.values(), key=lambda t: t["ts"] or "")
        try:
            dreamer.dream({farmer: dreamer.chunk_turns(ordered)}, args)
        except Exception:  # noqa: BLE001 - one farmer must not stop the others
            failed += 1
            print(f"farmer ...{farmer[-4:]} failed:\n{traceback.format_exc()}")
    if not dry_run:
        write_mark(stop)
    print(f"cycle done: {len(turns) - failed} ok, {failed} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    while True:
        try:
            cycle(dry_run=args.dry_run)
        except Exception:  # noqa: BLE001 - Langfuse/Qdrant blips: retry next cycle, mark unchanged
            print(f"cycle failed, mark unchanged:\n{traceback.format_exc()}")
        if args.once:
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
