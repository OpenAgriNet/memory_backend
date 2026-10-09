"""The single way anything is written to memory.

Every mutation — from the agent's tools, from the consolidation pass, from anything
added later — goes through `apply()`. Nothing calls the memory service's write
endpoints directly.

WHY THIS EXISTS

Because the alternative was tried and it failed in a specific, traceable way. There
were three write paths (the agent's tools, the consolidation merge, the standing
record), and each enforced its own subset of the rules. The consolidation path never
ran the judges — so the highest-stakes entries in the store, the merged ones that get
injected every turn thereafter, were the only unjudged writes in the system. Nobody
decided that; it was just what happened when a rule lived in one path and not another.

A single funnel makes each invariant true by construction rather than by everyone
remembering it:

  1. Idempotency      a turn already recorded against an entry is never applied twice
  2. Judging          sensitive content, key vocabulary, honesty — see judges.py
  3. Identifier strip deterministic, runs after the judges, cannot fail open
  4. Provenance       every write carries the trace ids and session behind it

Callers PROPOSE. This decides. The agent's tools are adapters that build a change and
hand it over; they contain no policy of their own.
"""
from __future__ import annotations

import re

import memory_client as requests

import judges
import llm
import prompts
import episode

_ID_RUN = re.compile(r"\b\d{8,}\b")

def _level2(entry):
    """Describe the final detail without condensing or deleting it."""
    entry = dict(entry)
    text = entry.get("expanded") or entry["headline"]
    chunks = episode.split_text(text)
    if len(chunks) == 1 and len(entry["headline"]) <= episode.SUMMARY_MAX_CHARS:
        entry["contents"] = episode.extractive_contents(text, chunks)
    else:
        groups = episode.contents_groups(len(chunks))
        for attempt in range(2):
            try:
                result = llm.ask_json(prompts.CONTENTS_SYSTEM,
                                      prompts.build_contents_prompt(entry, chunks, groups),
                                      prompts.contents_schema(groups), max_tokens=1800)
                headline = result["headline"].strip()
                if not 1 <= len(headline) <= episode.SUMMARY_MAX_CHARS:
                    raise ValueError("Level 2 summary exceeds its budget")
                descriptions = result["descriptions"]
                if len(descriptions) != len(groups):
                    raise ValueError("Expected one description per chunk group")
                rows = [{"start_chunk": first, "end_chunk": last, "text": description}
                        for (first, last), description in zip(groups, descriptions)]
                entry["contents"] = episode.validate_contents(rows, len(chunks))
                entry["headline"] = headline
                break
            except Exception:
                if attempt == 1:
                    raise
    entry["chunk_size"] = episode.CHUNK_MAX_CHARS
    return entry


def _checked_episode(entry, **evidence):
    """Check the actual Level 2 and Level 3 text together; rebuild after redaction."""
    notes = []
    for _ in range(2):
        entry, moved = strip_identifiers(entry)
        notes += moved
        try:
            entry = _level2(entry)
        except Exception as exc:
            return "block", entry, notes + [f"retry needed: contents generation failed: {exc}"]
        before = entry.get("expanded")
        decision, entry, checks = judges.review(entry, **evidence)
        notes += checks
        if decision == "block":
            return decision, entry, notes
        clean, moved = strip_identifiers(entry)
        notes += moved
        if clean.get("expanded") == before:
            return "allow", clean, notes
        entry = clean  # Text positions changed: rebuild the contents and check again.
    return "block", entry, notes + ["retry needed: redaction kept changing the chunk layout"]


def _norm(text):
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def _tidy(text):
    """Clean up what removing an identifier leaves behind.

    Stripping two tags from "for animals 101... and 105..." leaves "for animals and
    .", which is confusing when read back. A fallback path — the model was told not
    to put tags in prose — so readable is the bar, not perfect.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+(?:and|or|,)\s*(?=[.;,]|$)", "", text)
    text = re.sub(r"\s+(?:and|or)\s+(?:and|or)\s+", " and ", text)
    text = re.sub(r"\s+([.;,])", r"\1", text)
    return text.strip()


def strip_identifiers(entry):
    """Deterministic backstop. Runs after the judges and cannot fail open."""
    out, moved = dict(entry), []
    # animal_id may be a single tag or a list, when one situation covers two animals.
    raw = (out.get("metadata") or {}).get("animal_id")
    aids = [str(a) for a in (raw if isinstance(raw, (list, tuple)) else [raw]) if a]
    for field in ("headline", "expanded"):
        text = out.get(field) or ""
        if not text:
            continue
        if any(a in text for a in aids):
            for a in aids:
                text = text.replace(a, "")
            text = _tidy(text)
            moved.append(f"known animal id removed from {field}")
        stripped = _tidy(_ID_RUN.sub("", text)) if _ID_RUN.search(text) else text
        if stripped != text:
            moved.append(f"unregistered identifier removed from {field}")
            text = stripped
        out[field] = text
    return out, moved


# --------------------------------------------------------------------------

def apply(ctx, change):
    """Apply one proposed change. Returns a plain dict describing what happened.

    change["op"] is "write", "update" or "fold". Everything else in the dict is the
    proposal. Nothing here raises: a caller — including a model reading a tool
    result — gets told what happened and why.
    """
    op = change.get("op")
    if op == "write":
        return _write(ctx, change)
    if op == "update":
        return _update(ctx, change)
    if op == "fold":
        return _fold(ctx, change)
    if op == "profile":
        return _profile(ctx, change)
    return {"ok": False, "error": f"unknown change op {op!r}"}


def _provenance(ctx, extra_traces=(), extra_sessions=()):
    return {
        "source_ts": ctx.source_ts,
        "first_source_ts": min((t["ts"] for t in ctx.turns if t.get("ts")), default=ctx.source_ts),
        "source_session_id": ctx.session_id,
        "source_traces": sorted({t for t in (list(ctx.trace_ids) + list(extra_traces)) if t}),
        "source_sessions": sorted({s for s in (
            [ctx.session_id] + [t.get("session_id") for t in ctx.turns] + list(extra_sessions)) if s}) or None,
    }


def _write(ctx, change):
    entry = {k: change.get(k) for k in
             ("headline", "expanded", "status", "metadata")}
    entry["metadata"] = entry.get("metadata") or {}

    # Cheap local guard first: an agent loop re-proposing the same entry should be
    # stopped before spending three judge calls on it.
    key = _norm(entry.get("headline"))
    if key and key in ctx.seen_headlines:
        return {"ok": False, "already_written_in_this_run": True,
                "what_to_do": "You have already written this in this run. Do not repeat "
                              "it. Handle something else, or stop."}

    decision, entry, notes = _checked_episode(
        entry, turns=ctx.turns, known_keys=ctx.known_keys)
    if decision == "block":
        ctx.blocked.append({"entry": entry, "why": notes})
        return {"ok": False, "refused": True, "reasons": notes,
                "what_to_do": "Do not retry this as written. Say only what the "
                              "conversation actually shows, or leave it out."}
    entry, moved = strip_identifiers(entry)
    notes += moved

    body = dict(entry)
    body.update(_provenance(ctx))
    if change.get("expires_at"):
        body["expires_at"] = change["expires_at"]

    if ctx.dry_run:
        ctx.seen_headlines.add(_norm(entry["headline"]))
        ctx.writes.append({"action": "would-write", "entry": body, "notes": notes})
        return {"ok": True, "dry_run": True, "entry_id": "dry-run", "notes": notes}

    try:
        r = requests.post(f"{ctx.api}/memory/{ctx.farmer_id}/entries", json=body, timeout=90)
        r.raise_for_status()
        res = r.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}

    # The service refuses a write this farmer already holds, so re-running the job
    # over the same window is safe.
    if res.get("duplicate_of_existing"):
        return {"ok": False, "already_on_record": True, "entry_id": res["id"],
                "reason": res.get("reason"),
                "what_to_do": "This is already on record. If this conversation moved it "
                              "along, update that entry instead."}
    ctx.seen_headlines.add(_norm(entry["headline"]))
    ctx.writes.append({"action": "write", "entry_id": res["id"], "entry": body, "notes": notes})
    return {"ok": True, "entry_id": res["id"], "notes": notes}


def _update(ctx, change):
    entry_id = change.get("entry_id")
    existing = change.get("_existing") or _fetch(ctx, entry_id)
    if not existing:
        return {"ok": False, "error": "no current entry with that id for this farmer"}

    key = _norm(change.get("headline"))
    if key and key in ctx.seen_headlines:
        return {"ok": False, "already_written_in_this_run": True,
                "what_to_do": "You have already made this change in this run."}
    if ctx.refusals.get(entry_id, 0) >= 2:
        return {"ok": False, "give_up": True,
                "what_to_do": "This update has been refused repeatedly. Leave the entry "
                              "as it is and move on — do not try another wording."}

    entry = {"headline": change.get("headline"),
             "expanded": (change["expanded"] if change.get("expanded") is not None
                          else existing.get("expanded", "")),
             "status": change.get("status") or existing.get("status"),
             "metadata": (change["metadata"] if change.get("metadata") is not None
                          else existing.get("metadata", {}))}

    # The evidence for an update is the conversation PLUS what is already on record
    # PLUS the record's own history — the last of which is what licenses phrasing like
    # "he has raised this repeatedly", which no single conversation can evidence.
    provenance = _provenance(ctx, extra_traces=change.get("extra_traces", ()),
                             extra_sessions=change.get("extra_sessions", ()))
    sessions = set(existing.get("source_sessions") or []) | set(provenance["source_sessions"] or [])
    first_dates = [d for d in (existing.get("first_source_ts"), provenance["first_source_ts"]) if d]
    last_dates = [d for d in (existing.get("source_ts"), provenance["source_ts"]) if d]
    record = dict(existing, times_raised=len(sessions) or existing.get("times_raised", 1),
                  first_source_ts=min(first_dates, default=None), source_ts=max(last_dates, default=None))
    decision, entry, notes = _checked_episode(
        entry, turns=ctx.turns, known_keys=ctx.known_keys,
        existing=change.get("_evidence") or existing,
        record=change.get("_record") or record)
    if decision == "block":
        ctx.refusals[entry_id] = ctx.refusals.get(entry_id, 0) + 1
        ctx.blocked.append({"entry": entry, "why": notes})
        return {"ok": False, "refused": True, "reasons": notes,
                "what_to_do": "Do not reword and retry. Say only what these turns plus "
                              "the existing entry support, or leave it alone."}
    entry, moved = strip_identifiers(entry)
    notes += moved

    body = {"headline": entry["headline"], "expanded": entry["expanded"],
            "status": entry["status"], "metadata": entry["metadata"],
            "contents": entry["contents"], "chunk_size": entry["chunk_size"]}
    # NOT times_raised: the service derives it from the distinct conversations behind
    # the entry. Incrementing per write is what made an entry claim a farmer had
    # raised something three times when they had said it once.
    body.update(provenance)
    if change.get("merge_from"):
        body["merge_from"] = change["merge_from"]

    if ctx.dry_run:
        ctx.seen_headlines.add(_norm(entry["headline"]))
        ctx.writes.append({"action": "would-update", "entry_id": entry_id,
                           "entry": body, "notes": notes})
        return {"ok": True, "dry_run": True, "notes": notes}

    try:
        r = requests.patch(f"{ctx.api}/memory/{ctx.farmer_id}/entries/{entry_id}",
                           json=body, timeout=120)
        r.raise_for_status()
        res = r.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}

    if res.get("applied") is False:
        return {"ok": False, "already_applied": True, "reason": res.get("reason"),
                "what_to_do": "These turns are already recorded on that entry. Nothing "
                              "to do."}
    new_id = res["new"]["id"]
    ctx.seen_headlines.add(_norm(entry["headline"]))
    ctx.writes.append({"action": "update", "entry_id": new_id, "replaced": entry_id,
                       "entry": body, "notes": notes})
    return {"ok": True, "entry_id": new_id, "notes": notes}


def _fold(ctx, change):
    if ctx.dry_run:
        ctx.writes.append({"action": "would-fold", "entry_id": change.get("entry_id"),
                           "into": change.get("merged_into"),
                           "notes": [change.get("reason", "")]})
        return {"ok": True, "dry_run": True}
    try:
        r = requests.post(
            f"{ctx.api}/memory/{ctx.farmer_id}/entries/{change['entry_id']}/fold",
            json={"merged_into": change["merged_into"], "reason": change.get("reason", "")},
            timeout=60)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}
    if r.status_code >= 300:
        return {"ok": False, "error": r.text[:200]}
    ctx.writes.append({"action": "fold", "entry_id": change["entry_id"],
                       "into": change["merged_into"], "notes": [change.get("reason", "")]})
    return {"ok": True}


def _fetch(ctx, entry_id):
    try:
        r = requests.get(f"{ctx.api}/internal/memory/{ctx.farmer_id}/entries/{entry_id}", timeout=60)
        r.raise_for_status()
        body = r.json()
        return body.get("entry") if body.get("current") else None
    except Exception:  # noqa: BLE001
        return None


def _profile(ctx, change):
    """Check the standing record through the same write boundary as other memory."""
    try:
        fields = prompts.profile_schema.validate(change.get("fields") or {})
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    cleaned = {}
    for name, value in fields.items():
        safe, _ = strip_identifiers({"headline": value})
        if safe["headline"]:
            cleaned[name] = safe["headline"]
    text = "\n".join(f"{name}: {value}" for name, value in cleaned.items())
    proposal = {"headline": text, "expanded": "", "status": "n/a", "metadata": {}}
    decision, checked, notes = judges.review(
        proposal, turns=ctx.turns, existing=change.get("_evidence"))
    # A free-text redaction cannot safely be mapped back to program-owned fields.
    # Leave the existing record alone; regenerate the proposal instead.
    if decision == "block" or checked.get("headline") != text:
        ctx.blocked.append({"entry": proposal, "why": notes})
        return {"ok": False, "refused": True, "reasons": notes or ["standing record needs a safe rewrite"]}
    body = {"fields": cleaned, "source_ts": ctx.source_ts}
    if ctx.dry_run:
        return {"ok": True, "dry_run": True, "fields": cleaned}
    try:
        response = requests.put(f"{ctx.api}/memory/{ctx.farmer_id}/profile", json=body, timeout=90)
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}
    return {"ok": True, "unchanged": bool(result.get("unchanged")), "fields": cleaned}
