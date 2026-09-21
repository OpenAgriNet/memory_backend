"""Read-only Langfuse client for the configured Amul environment.

Uses one URL and one public/secret key pair from .env or process environment.
GET only; nothing is written back to Langfuse.
"""
import datetime as dt
import os
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

CHAT_TRACE_NAME = "chat.translation"


class LangfuseRO:
    def __init__(self):
        config = {**dotenv_values(os.environ.get("AMUL_MEMORY_ENV_FILE", Path(__file__).resolve().parents[1] / ".env")),
                  **os.environ}
        names = ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
        if not all(config.get(key) for key in names):
            raise ValueError("Set LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY together")
        self.base = config[names[0]].rstrip("/")
        self.auth = (config[names[1]], config[names[2]])
        self.s = requests.Session()

    def _get(self, path, params=None, retries=4):
        for attempt in range(retries):
            try:
                r = self.s.get(f"{self.base}{path}", auth=self.auth, params=params, timeout=180)
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 422:
                # Langfuse asking for a narrower window; caller splits the range.
                raise TimeoutError(r.text[:200])
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
        raise RuntimeError(f"GET {path} failed after {retries} attempts")

    def farmer_turns(self, farmer_id, days_back=30, until=None):
        """Every chat turn for one farmer, oldest first.

        Returns dicts of {trace_id, ts, session_id, query, answer} — the farmer-facing text
        only. `until` (ISO) caps the window, which is what the backtest cutoff uses
        so a replay never sees anything the farmer hadn't said yet.
        """
        end = (dt.datetime.fromisoformat(until.replace("Z", "+00:00"))
               if until else dt.datetime.now(dt.timezone.utc))
        if end.tzinfo is None:
            end = end.replace(tzinfo=dt.timezone.utc)
        def read_window(start, stop):
            params = {"userId": str(farmer_id), "name": CHAT_TRACE_NAME,
                      "fromTimestamp": iso(start), "toTimestamp": iso(stop), "limit": 100,
                      "fields": "core,io"}  # Avoid unrelated observation/score/metric joins.

            rows, page = [], 1
            try:
                while True:
                    params["page"] = page
                    data = self._get("/api/public/traces", params)
                    items = data.get("data", [])
                    rows.extend(items)
                    if not items or page >= (data.get("meta", {}).get("totalPages") or 1):
                        return rows
                    page += 1
            except TimeoutError:
                if stop - start <= dt.timedelta(hours=1):
                    raise
                middle = start + (stop - start) / 2
                return read_window(start, middle) + read_window(middle, stop)

        out = {}
        for t in read_window(end - dt.timedelta(days=days_back), end):
            inp = t.get("input") or {}
            trace_id = t.get("id")
            if not trace_id:
                continue
            # Adjacent inclusive windows may share a boundary trace; keep it once.
            out[trace_id] = {
                "trace_id": trace_id, "ts": t.get("timestamp"), "session_id": t.get("sessionId"),
                "query": (inp.get("query") if isinstance(inp, dict) else None) or "",
                "answer": t.get("output") if isinstance(t.get("output"), str) else "",
            }
        return sorted(out.values(), key=lambda row: row["ts"] or "")


def iso(d: dt.datetime) -> str:
    return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def group_sessions(turns):
    """Group adjacent session runs without reordering interleaved conversations."""
    from itertools import groupby
    ordered = sorted(turns, key=lambda t: t.get("ts") or "")
    return [(sid, list(run)) for sid, run in groupby(
        ordered, key=lambda t: t.get("session_id") or "unknown")]
