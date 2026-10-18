"""The tools the dreaming agent can call.

Two rules shape this file.

**The farmer is never a parameter.** Every tool takes the farmer from the context the
agent was launched with, never from the model. A dreaming agent that could name a
farmer could write one farmer's memory into another's, and no prompt instruction is
worth relying on for that.

**Writes are judged, not trusted.** write_memory and update_memory run the judges in
judges.py before anything reaches Qdrant, and a block is returned to the agent as a
tool result it has to deal with. The deterministic identifier strip runs too, after
the judges, because that one must not depend on a model call succeeding.
"""
from __future__ import annotations

import re
from copy import copy

import requests

if __package__:
    from . import dates, writer
else:
    import dates
    import writer


class Ctx:
    """What the agent is working on. Not visible to the model."""

    def __init__(self, api, farmer_id, turns=None, source_ts=None, session_id=None,
                 known_keys=(), dry_run=False, trace_ids=None, before_mutation=None):
        self.api = api.rstrip("/")
        self.farmer_id = str(farmer_id)
        self.turns = turns or []
        self.source_ts = source_ts
        self.session_id = session_id
        self.known_keys = list(known_keys)
        self.dry_run = dry_run
        # The turns behind this run. Written onto every entry, and the unit of
        # idempotency: the service refuses an update whose traces it has already
        # recorded, so re-running the job over the same window is a no-op.
        self.trace_ids = list(trace_ids or [t.get("trace_id") for t in (turns or [])])
        self.writes: list[dict] = []      # what actually happened, for the caller
        self.blocked: list[dict] = []     # what the judges refused, and why
        # Worker-only outcome signals.  They contain no exception text, and leave
        # the existing command-line reporting unchanged.
        self.error_code: str | None = None
        self.mutation_attempted = False
        self.uncertain_mutation = False
        self.tool_failures: list[str] = []
        self.before_mutation = before_mutation
        self._mutation_started = False
        self.seen_headlines: set[str] = set()  # idempotency within one agent run
        self.refusals: dict[str, int] = {}     # per-entry refusal count, to stop retry loops

    def start_mutation(self) -> None:
        """Persist uncertainty before the first tool can mutate farmer memory."""
        if self._mutation_started:
            return
        if self.before_mutation is not None:
            self.before_mutation()
        self._mutation_started = True
        self.mutation_attempted = True

    def for_sources(self, numbers):
        """Keep shared run bookkeeping, but restrict evidence to selected turns."""
        if not self.turns:
            return self  # Trusted consolidation uses stored evidence, not new turns.
        if (not isinstance(numbers, list) or not numbers or any(
                type(n) is not int or n < 1 or n > len(self.turns) for n in numbers)):
            raise ValueError("source_turns must list supporting turn numbers from this chunk")
        selected = copy(self)
        selected.turns = [self.turns[n - 1] for n in sorted(set(numbers))]
        selected.trace_ids = [t["trace_id"] for t in selected.turns if t.get("trace_id")]
        stamps = sorted(t["ts"] for t in selected.turns if t.get("ts"))
        selected.source_ts = stamps[-1] if stamps else None
        selected.session_id = selected.turns[-1].get("session_id")
        return selected

    def _url(self, path):
        return f"{self.api}/internal/memory/{self.farmer_id}{path}"


_norm = writer._norm
_strip_ids = writer.strip_identifiers


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------

def search_memory(ctx: Ctx, query: str, detail: bool = False, limit: int = 8, memory_ref: str | None = None) -> dict:
    path = "/search_expanded" if detail else "/search"
    r = requests.post(ctx._url(path),
                      json={"query": query, "limit": min(limit, 20), "only_current": True, "memory_ref": memory_ref},
                      timeout=60)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    return {"found": len(hits), "entries": [{
        "entry_id": h["id"], "status": h.get("status"),
        "times_raised": h.get("times_raised"),
        "first_came_up": dates.fmt(h.get("first_source_ts")),
        "last_mentioned": dates.fmt(h.get("source_ts")),
        "end_date": dates.fmt(h.get("expires_at")) if h.get("expires_at") else None,
        "headline": h.get("headline"),
        "detail": (h.get("passage") or h.get("expanded") or "") if detail else None,
        "chunk_number": h.get("chunk_number"), "chunk_count": h.get("chunk_count"),
        "contents": h.get("contents"),
        "metadata": h.get("metadata"),
    } for h in hits]}


def list_memory(ctx: Ctx, status: str | None = None,
                metadata_key: str | None = None, metadata_value: str | None = None,
                limit: int = 50, cursor: str | None = None) -> dict:
    """List recorded field matches. Missing tags are not evidence of absence."""
    params = {"limit": min(limit, 100), "only_current": "true"}
    for k, v in (("status", status),
                 ("metadata_key", metadata_key), ("metadata_value", metadata_value)):
        if v:
            params[k] = v
    if cursor:
        params["cursor"] = cursor
    r = requests.get(ctx._url("/entries"), params=params, timeout=60)
    r.raise_for_status()
    body = r.json()
    if body.get("memory_enabled") is False:
        return {"available": False, "reason": "memory disabled; do not infer absence"}
    ents = [e for e in body.get("entries", [])
            if e.get("type") not in ("profile_record", "settings", "key_doc")]
    return {"found": len(ents), "truncated": bool(body.get("truncated")),
            "next_cursor": body.get("next_cursor"),
            "note": "Only recorded values matching the filter; not proof that nothing else happened.", "entries": [{
        "entry_id": e["id"], "status": e.get("status"),
        "times_raised": e.get("times_raised"),
        "first_came_up": dates.fmt(e.get("first_source_ts")),
        "last_mentioned": dates.fmt(e.get("source_ts")),
        "end_date": dates.fmt(e.get("expires_at")) if e.get("expires_at") else None,
        "headline": e.get("headline"), "metadata": e.get("metadata"),
    } for e in ents]}


def read_memory(ctx: Ctx, entry_id: str, chunk_ids: list[int] | None = None) -> dict:
    """Read selected evidence, following explicit continuation before rewriting."""
    response = requests.post(ctx._url(f"/entries/{entry_id}/read"), json={
        "chunk_ids": chunk_ids, "whole_chunks": True}, timeout=60)
    response.raise_for_status()
    body = response.json()
    return {"entry_id": entry_id, "current": body["current"], "chunks": body["chunks"],
            "continuation": body.get("continuation"), "metadata": body["entry"].get("metadata")}


def known_keys(ctx: Ctx) -> dict:
    r = requests.get(f"{ctx.api}/keys", timeout=30)
    r.raise_for_status()
    ks = r.json().get("keys", [])
    ctx.known_keys = [{"key": k["key"], "description": k.get("description")} for k in ks]
    # Share vocabulary, never other farmers' values with this farmer's agent.
    return {"keys": [{"key": k["key"], "means": k.get("description")} for k in ks]}


# --------------------------------------------------------------------------
# writes — thin adapters. All policy lives in writer.apply().
# --------------------------------------------------------------------------

def write_memory(ctx: Ctx, headline: str, expanded: str = "",
                 status: str = "n/a",
                 expires_at: str | None = None, metadata: dict | None = None,
                 source_turns: list[int] | None = None) -> dict:
    try:
        ctx = ctx.for_sources(source_turns)
    except ValueError as exc:
        return {"written": False, "error": str(exc)}
    res = writer.apply(ctx, {"op": "write", "headline": headline,
                             "expanded": expanded, "status": status,
                             "expires_at": expires_at, "metadata": metadata})
    return {"written": res.pop("ok"), **res}


def update_memory(ctx: Ctx, entry_id: str, headline: str, expanded: str | None = None,
                  status: str | None = None, metadata: dict | None = None,
                  source_turns: list[int] | None = None) -> dict:
    try:
        ctx = ctx.for_sources(source_turns)
    except ValueError as exc:
        return {"updated": False, "error": str(exc)}
    res = writer.apply(ctx, {"op": "update", "entry_id": entry_id, "headline": headline,
                             "expanded": expanded, "status": status, "metadata": metadata})
    return {"updated": res.pop("ok"), **res}


def fold_memory(ctx: Ctx, entry_id: str, merged_into: str, reason: str) -> dict:
    if entry_id == merged_into:
        return {"folded": False, "error": "an entry cannot be folded into itself"}
    res = writer.apply(ctx, {"op": "fold", "entry_id": entry_id,
                             "merged_into": merged_into, "reason": reason})
    return {"folded": res.pop("ok"), **res}


def document_key(ctx: Ctx, key: str, description: str) -> dict:
    if ctx.dry_run:
        return {"documented": True, "dry_run": True}
    try:
        requests.post(f"{ctx.api}/keys/doc",
                      json={"key": key, "description": description},
                      timeout=30).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return {"documented": False, "error": str(exc)[:150]}
    return {"documented": True}


def nothing_to_remember(ctx: Ctx, why: str) -> dict:
    """Record why these turns add no useful supported memory."""
    if ctx.tool_failures:
        return {"acknowledged": False, "error": "earlier_tool_failure"}
    ctx.writes.append({"action": "nothing", "notes": [why]})
    return {"acknowledged": True, "notes": [why]}


# --------------------------------------------------------------------------
# schemas handed to the model
# --------------------------------------------------------------------------

def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


SCHEMAS = [
    _fn("search_memory",
        "Search what is already remembered about this farmer, by meaning. Use this "
        "BEFORE writing anything, to find out whether the situation is already on "
        "record. Several searches with different wording are better than one.",
        {"query": {"type": "string", "description": "What you are looking for, in plain words."},
         "detail": {"type": "boolean", "description": "Search the full detail text rather than the short headlines."},
         "limit": {"type": "integer"}, "memory_ref": {"type": "string"}},
        ["query"]),
    _fn("list_memory",
        "List this farmer's memories by exact field, with no searching. Use when the "
        "question is about recorded state or a particular tagged value. This reports "
        "matching records only; extraction and tags can be incomplete.",
        {"status": {"type": "string", "enum": ["open", "pending", "resolved", "n/a"]},
         "metadata_key": {"type": "string"}, "metadata_value": {"type": "string"},
         "limit": {"type": "integer"}, "cursor": {"type": "string"}}),
    _fn("read_memory", "Read the selected entry before rewriting it. Follow continuation for more detail.",
        {"entry_id": {"type": "string"},
         "chunk_ids": {"type": "array", "items": {"type": "integer"}}}, ["entry_id"]),
    _fn("known_keys",
        "The metadata keys already in use across the system, and what they mean. "
        "Check this before inventing a key.", {}),
    _fn("write_memory",
        "Write a NEW memory. Only after searching and finding that this situation is "
        "not already on record. The write is checked before it is stored and may be "
        "refused, in which case you will be told why.",
        {"source_turns": {"type": "array", "items": {"type": "integer"}, "description": "Numbers of the turns supporting this memory; omit unrelated turns."},
         "headline": {"type": "string", "description": "1-2 self-contained sentences, third person, injected into every future conversation."},
         "expanded": {"type": "string", "description": "The fuller detail: what was tried, what was advised, figures the farmer gave."},
         "status": {"type": "string", "enum": ["open", "pending", "resolved", "n/a"]},
         "expires_at": {"type": "string", "description": "ISO date, only when a real end date is knowable (a milk withdrawal period, a deadline)."},
         "metadata": {"type": "object", "description": "Extra filterable detail. An animal's ear tag goes here under animal_id and NEVER in the text."}},
        ["headline", "status", "source_turns"]),
    _fn("update_memory",
        "Update an existing memory because this conversation moved it along. Use this "
        "rather than writing a second entry about the same situation. The previous "
        "version is kept as history.",
        {"source_turns": {"type": "array", "items": {"type": "integer"}, "description": "Numbers of the turns supporting the new information."},
         "entry_id": {"type": "string"},
         "headline": {"type": "string", "description": "Rewritten to cover the whole situation, including that it has come up again."},
         "expanded": {"type": "string", "description": "The previous detail plus what is new, in time order. Do not drop what was already there."},
         "status": {"type": "string", "enum": ["open", "pending", "resolved", "n/a"]},
         "metadata": {"type": "object"}},
        ["entry_id", "headline", "source_turns"]),
    _fn("fold_memory",
        "Retire a duplicate entry into another that now covers it, when you find two "
        "entries recording the same situation. Nothing is deleted.",
        {"entry_id": {"type": "string", "description": "The duplicate to retire."},
         "merged_into": {"type": "string", "description": "The entry that now covers it."},
         "reason": {"type": "string"}},
        ["entry_id", "merged_into", "reason"]),
    _fn("document_key",
        "Record what a metadata key you have just invented means, so later runs reuse "
        "it instead of coining a synonym.",
        {"key": {"type": "string"}, "description": {"type": "string"}},
        ["key", "description"]),
    _fn("nothing_to_remember",
        "Call this when these turns add no useful supported memory. Explain why; "
        "do not assume either writing or skipping is the expected outcome.",
        {"why": {"type": "string"}}, ["why"]),
]

IMPL = {
    "search_memory": search_memory, "list_memory": list_memory, "read_memory": read_memory, "known_keys": known_keys,
    "write_memory": write_memory, "update_memory": update_memory, "fold_memory": fold_memory,
    "document_key": document_key, "nothing_to_remember": nothing_to_remember,
}

# Tools with no side effects. Safe to run concurrently with anything.
READ_ONLY = {"search_memory", "list_memory", "read_memory", "known_keys"}
