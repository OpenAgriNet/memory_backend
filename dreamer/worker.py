"""One-shot scheduled worker for completed farmer text-chat sessions."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "service")]

import agent
import lf
import llm
import memory_client

if __package__:
    from dreamer import dreamer
else:
    import dreamer

MOBILE = re.compile(r"^[0-9]{10}$")
ENVIRONMENT = os.environ.get("MEMORY_WORKER_LANGFUSE_ENVIRONMENT", "chat-production")
INACTIVITY = dt.timedelta(minutes=int(os.environ.get("MEMORY_WORKER_INACTIVITY_MINUTES", "30")))
INCREMENTAL = dt.timedelta(hours=int(os.environ.get("MEMORY_WORKER_INCREMENTAL_HOURS", "2")))
RECONCILE = dt.timedelta(hours=int(os.environ.get("MEMORY_WORKER_RECONCILE_HOURS", "48")))
CONCURRENCY = max(1, int(os.environ.get("MEMORY_WORKER_CONCURRENCY", "1")))
LEASE_RENEW_SECONDS = max(1, int(os.environ.get("MEMORY_WORKER_LEASE_RENEW_SECONDS", "300")))


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def parse_time(value):
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or dt.timezone.utc).astimezone(dt.timezone.utc)


def trace_turn(trace):
    """Return a canonical worker turn, or None for every ineligible trace shape."""
    metadata = trace.get("metadata") or {}
    farmer_id = metadata.get("memory_farmer_id")
    inp, output = trace.get("input") or {}, trace.get("output")
    if (
        trace.get("name") != lf.CHAT_TRACE_NAME
        or trace.get("environment") != ENVIRONMENT
        or metadata.get("persona") != "farmer"
        or not isinstance(farmer_id, str)
        or not MOBILE.fullmatch(farmer_id)
        or not trace.get("id")
        or not trace.get("sessionId")
        or not trace.get("timestamp")
        or not isinstance(inp, dict)
        or not str(inp.get("query") or "").strip()
        or not isinstance(output, str)
        or not output.strip()
    ):
        return None
    return {
        "trace_id": trace["id"], "ts": trace["timestamp"],
        "source_version": trace.get("updatedAt") or trace.get("version"),
        "session_id": trace["sessionId"], "farmer_id": farmer_id,
        "query": str(inp["query"]), "answer": output,
    }


def fingerprint(turns):
    markers = sorted(
        (turn["trace_id"], turn.get("source_version") or turn["ts"]) for turn in turns
    )
    raw = json.dumps(markers, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def processed_trace(turn):
    row = {"trace_id": turn["trace_id"]}
    if turn.get("source_version"):
        row["source_version"] = str(turn["source_version"])
    else:
        row["source_timestamp"] = turn["ts"]
    return row


def marker(row):
    if row.get("source_version"):
        return str(row["source_version"])
    value = row.get("source_timestamp") or row.get("ts")
    return parse_time(value).isoformat() if value else ""


class Checkpoints:
    def __init__(self, base=dreamer.MEMORY_API):
        self.base = base.rstrip("/")

    def call(self, method, path, body=None):
        response = memory_client.request(method, self.base + path, json=body, timeout=60)
        response.raise_for_status()
        return response.json()

    def capabilities(self):
        return self.call("GET", "/internal/checkpoints/capabilities")

    def claim(self, kind, subject, farmer, candidate_fingerprint, last_activity, run_id):
        return self.call("POST", "/internal/checkpoints/claim", {
            "kind": kind, "subject_id": subject, "farmer_id": farmer,
            "candidate_fingerprint": candidate_fingerprint,
            "last_activity_at": last_activity, "run_id": run_id,
        })

    def transition(self, action, lease, **values):
        return self.call("POST", f"/internal/checkpoints/{action}", {
            "kind": lease.kind, "subject_id": lease.subject,
            "lease_token": lease.token, "checkpoint_version": lease.version, **values,
        })

    def due(self):
        rows, cursor = [], None
        while True:
            body = {"limit": 100}
            if cursor:
                body["cursor"] = cursor
            page = self.call("POST", "/internal/checkpoints/due", body)
            rows.extend(page.get("checkpoints") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                return rows


class Lease:
    """Renew a claim while model work is running; serialize renewal and completion."""
    def __init__(self, client, kind, subject, claim):
        self.client, self.kind, self.subject = client, kind, subject
        self.token, self.version = claim["lease_token"], claim["checkpoint_version"]
        self.stop_event, self.error = threading.Event(), None
        self.thread = threading.Thread(target=self._renew, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def _renew(self):
        while not self.stop_event.wait(LEASE_RENEW_SECONDS):
            try:
                result = self.client.transition("renew", self)
                self.version = result["checkpoint_version"]
            except Exception as exc:  # noqa: BLE001
                self.error = exc
                self.stop_event.set()

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        if self.error:
            raise RuntimeError("checkpoint lease renewal failed") from self.error


def discover(reader, start, stop, *, inactive_before):
    grouped = defaultdict(dict)
    rows = reader.trace_window(
        start, stop, name=lf.CHAT_TRACE_NAME, environment=ENVIRONMENT,
    )
    for raw in rows:
        turn = trace_turn(raw)
        if turn:
            grouped[(turn["farmer_id"], turn["session_id"])][turn["trace_id"]] = turn
    sessions = {}
    for key, by_id in grouped.items():
        turns = sorted(by_id.values(), key=lambda turn: turn["ts"])
        if parse_time(turns[-1]["ts"]) <= inactive_before:
            sessions[key] = turns
    return sessions


def session_turns(reader, session_id, farmer_id, now, last_activity=None):
    anchor = parse_time(last_activity) if last_activity else now
    rows = reader.session_traces(session_id, anchor - RECONCILE, min(now, anchor + dt.timedelta(hours=1)))
    turns = [turn for turn in map(trace_turn, rows) if turn and turn["farmer_id"] == farmer_id]
    return sorted({turn["trace_id"]: turn for turn in turns}.values(), key=lambda turn: turn["ts"])


def chunks_with_context(turns, previous):
    old = {row["trace_id"]: marker(row) for row in previous}
    pending = [i for i, turn in enumerate(turns) if old.get(turn["trace_id"]) != marker(turn)]
    new_per_chunk = min(10, max(3, dreamer.CHUNK_TURNS)) - 2
    for offset in range(0, len(pending), new_per_chunk):
        indexes = pending[offset:offset + new_per_chunk]
        first = indexes[0]
        context = [i for i in range(max(0, first - 2), first) if i not in indexes]
        yield [dict(turns[i], context_only=i in context) for i in context + indexes]


def run_dreamer(farmer_id, chunks, *, dry_run):
    known_keys, wrote = dreamer._known_keys(), False
    for chunk in chunks:
        ctx = agent.dream(
            dreamer.MEMORY_API, farmer_id, chunk, source_ts=chunk[-1].get("ts"),
            session_id=chunk[-1].get("session_id"), known_keys=known_keys,
            dry_run=dry_run, verbose=False,
        )
        if ctx.error:
            raise RuntimeError("dreamer model call failed") from ctx.error
        if ctx.blocked and not any(row["action"] in ("write", "update", "fold") for row in ctx.writes):
            raise RuntimeError("dreamer refused every proposed write")
        wrote |= any(row["action"] in ("write", "update", "fold") for row in ctx.writes)
        if wrote and not dry_run:
            known_keys = dreamer._known_keys()
    return "completed" if wrote else "no_op"


def safe_failure(client, lease, exc):
    try:
        result = client.transition(
            "fail", lease, error_code=type(exc).__name__,
            error_summary=f"{type(exc).__name__} while processing {lease.kind}",
        )
        return (result.get("checkpoint") or {}).get("status")
    except Exception:  # A stale/expired lease must not hide the original failure.
        return None


def process_session(reader, client, run_id, farmer_id, session_id, candidate, now, dry_run):
    seed, last_activity = candidate["turns"], candidate.get("last_activity")
    try:
        turns = session_turns(reader, session_id, farmer_id, now, last_activity) or seed
    except Exception:
        if not seed:
            raise
        turns = seed
    if not turns:
        return Counter(skipped=1), None
    candidate_fingerprint = fingerprint(turns)
    if dry_run:
        status = run_dreamer(farmer_id, chunks_with_context(turns, []), dry_run=True)
        return Counter({status: 1}), candidate_fingerprint
    claim = client.claim(
        "session_ingestion", session_id, farmer_id, candidate_fingerprint,
        turns[-1]["ts"], run_id,
    )
    if not claim.get("claimed"):
        return Counter(skipped=1), candidate_fingerprint
    lease = Lease(client, "session_ingestion", session_id, claim)
    try:
        chunks = list(chunks_with_context(turns, claim.get("processed_traces") or []))
        if not chunks:
            status = "no_op"
        else:
            with lease:
                status = run_dreamer(farmer_id, chunks, dry_run=False)
        client.transition(
            "complete", lease, status=status,
            processed_traces=[processed_trace(turn) for turn in turns],
        )
        return Counter({status: 1}), candidate_fingerprint
    except Exception as exc:  # noqa: BLE001
        state = safe_failure(client, lease, exc)
        return Counter({"needs_review" if state == "needs_review" else "failed": 1}), candidate_fingerprint


def process_maintenance(client, run_id, farmer_id, kind, candidate_fingerprint, dry_run, asks=()):
    fn = dreamer.consolidate if kind == "farmer_consolidation" else dreamer.update_profile
    kwargs = {"dry_run": dry_run, "strict": True}
    if kind == "farmer_profile":
        kwargs["recent_asks"] = asks
    if dry_run:
        return "completed" if fn(farmer_id, **kwargs) else "no_op"
    claim = client.claim(kind, farmer_id, farmer_id, candidate_fingerprint, None, run_id)
    if not claim.get("claimed"):
        return "skipped"
    lease = Lease(client, kind, farmer_id, claim)
    try:
        with lease:
            changed = fn(farmer_id, **kwargs)
        status = "completed" if changed else "no_op"
        client.transition("complete", lease, status=status, processed_traces=[])
        return status
    except Exception as exc:  # noqa: BLE001
        state = safe_failure(client, lease, exc)
        return "needs_review" if state == "needs_review" else "failed"


def process_farmer(reader, client, run_id, farmer_id, sessions, maintenance, now, dry_run):
    counts, fingerprints, asks = Counter(), [], []
    for session_id, candidate in sorted(
            sessions.items(), key=lambda item: item[1].get("last_activity") or ""):
        try:
            result, value = process_session(
                reader, client, run_id, farmer_id, session_id, candidate, now, dry_run,
            )
        except Exception:  # One unreadable due session must not stop other farmers.
            result, value = Counter(failed=1), None
        counts.update(result)
        if value and (result["completed"] or result["no_op"]):
            fingerprints.append((session_id, value))
        asks = ([turn["query"][:200] for turn in candidate["turns"] if turn.get("query")] + asks)[:10]
    maintenance_fingerprint = hashlib.sha256(
        json.dumps(sorted(fingerprints), separators=(",", ":")).encode()
    ).hexdigest()
    for kind in ("farmer_consolidation", "farmer_profile"):
        if fingerprints or kind in maintenance:
            candidate_fingerprint = (
                maintenance_fingerprint if fingerprints else maintenance[kind]
            )
            counts[process_maintenance(
                client, run_id, farmer_id, kind, candidate_fingerprint, dry_run, asks,
            )] += 1
    return counts


def preflight(reader, client):
    now = utc_now()
    reader.trace_window(now - dt.timedelta(minutes=1), now,
                        name=lf.CHAT_TRACE_NAME, environment=ENVIRONMENT)
    capability = client.capabilities()
    if not capability.get("atomic_mutations"):
        raise RuntimeError("atomic checkpoint mutations are unavailable")
    models = list(llm.client.models.list())
    if models and llm.MODEL not in {model.id for model in models}:
        raise RuntimeError("configured Dreamer model is unavailable")
    return capability


def run(mode, dry_run=False, *, reader=None, client=None, now=None):
    started, now, run_id = time.monotonic(), now or utc_now(), uuid.uuid4().hex
    reader, client = reader or lf.LangfuseRO(), client or Checkpoints()
    if mode == "preflight":
        capability = preflight(reader, client)
        print(json.dumps({"mode": mode, "ok": True, "checkpoint": capability}))
        return 0
    window = INCREMENTAL if mode == "incremental" else RECONCILE
    sessions = discover(reader, now - window, now, inactive_before=now - INACTIVITY)
    farmers, maintenance = defaultdict(dict), defaultdict(dict)
    for (farmer_id, session_id), turns in sessions.items():
        farmers[farmer_id][session_id] = {"turns": turns, "last_activity": turns[-1]["ts"]}
    backlog_times = [parse_time(turns[-1]["ts"]) for turns in sessions.values()]
    if not dry_run:
        for checkpoint in client.due():
            farmer_id, kind = checkpoint.get("farmer_id"), checkpoint.get("kind")
            if not farmer_id:
                continue
            if kind == "session_ingestion":
                farmers[farmer_id].setdefault(checkpoint["subject_id"], {
                    "turns": [], "last_activity": checkpoint.get("last_activity_at"),
                })
            elif kind in ("farmer_consolidation", "farmer_profile"):
                maintenance[farmer_id][kind] = checkpoint.get("candidate_fingerprint") or "due"
            if checkpoint.get("last_activity_at"):
                backlog_times.append(parse_time(checkpoint["last_activity_at"]))
    ids = sorted(set(farmers) | set(maintenance))
    counts = Counter(
        candidates=len(sessions),
        eligible=sum(map(len, farmers.values())) + sum(map(len, maintenance.values())),
    )
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(
            process_farmer, reader, client, run_id, farmer_id,
            farmers[farmer_id], maintenance[farmer_id], now, dry_run,
        ) for farmer_id in ids]
        for future in futures:
            counts.update(future.result())
    summary = {
        "mode": mode, "dry_run": dry_run, "window_hours": int(window.total_seconds() / 3600),
        **counts,
        "oldest_backlog_hours": round((now - min(backlog_times)).total_seconds() / 3600, 2)
        if backlog_times else 0,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("incremental", "reconcile", "preflight"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(args.mode, args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"mode": args.mode, "ok": False, "error": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
