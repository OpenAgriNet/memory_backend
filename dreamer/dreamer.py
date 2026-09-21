"""The dreamer: read farmers' real conversations, decide what to remember, write it.

Runs entirely in the background, off any live turn, so it has no latency budget to
respect (memory-overview.md Section 2, principle 1). Nothing here is called during a
conversation.

HOW WORK IS DIVIDED

A prompt never gets more than CHUNK_TURNS (10) turns. Whole conversations are packed
into a chunk while they fit; a single conversation longer than 10 turns is split.
Keeping conversations intact where possible matters — extraction is much better when
it can see a problem raised and answered in the same window.

A batch holds one chunk from each of many farmers, never two chunks of the same
farmer. Farmer state (what is already remembered) changes as their chunks are
processed, so their chunks have to be strictly ordered and cannot run beside each
other; different farmers are independent. Chunks are handed out round-robin, so a
farmer with a long history is simply finished in a later batch. Run daily and almost
everyone completes in the first batch or two.

THE THREE LAYERS, IN ORDER

Per chunk:
  1. Extract  (Layer 3)  one call decides what is worth remembering at all.
  2. Match               for each candidate, is this the same situation we already
                         track, or genuinely new? Defaults to "new" — the costs are
                         not symmetric (prompts.MATCH_SYSTEM).
  3. Write    (Layer 3)  PATCH an existing entry (supersedes, keeping history) or
                         POST a new one.
  4. Headline (Layer 2)  a second focused call writes the short version that gets
                         injected into every future turn. Separate because after a
                         merge the entry covers more than its old headline said, and
                         because this is the highest-stakes text in the system.
Once a farmer's chunks are all done:
  5. Profile  (Layer 1)  re-derive the standing record — fixed schema, same fields
                         for every farmer, only the populated ones stored.

Processing chronologically matters: by chunk N the store contains exactly what it
would have contained had this been running live all along, which is what makes the
backtest in demo-plan.md meaningful rather than circular.

Usage:
  python dreamer.py --farmer 9999999999 --days 30
  python dreamer.py --farmers 9999999999,8888888888 --days 30
  python dreamer.py --farmer 9999999999 --until 2026-08-21T00:00:00Z   # backtest cutoff
  python dreamer.py --farmer 9999999999 --dry-run                      # decide, write nothing
"""
import argparse
import json
import os
import re
import sys

import requests

import agent
import dates
import lf
import llm
import prompts
import tools
import writer
from runtime_config import required

MEMORY_API = required("MEMORY_API_URL").rstrip("/")
MATCH_CANDIDATES = int(os.environ.get("DREAMER_MATCH_CANDIDATES", "5"))
# How many hits to pull per query formulation before reranking. Wide on purpose —
# recall matters here and precision is the reranker's job.
CANDIDATE_POOL = int(os.environ.get("DREAMER_CANDIDATE_POOL", "15"))
# Cap on how many of a farmer's memories go into one consolidation prompt.
CLUSTER_BATCH = int(os.environ.get("DREAMER_CLUSTER_BATCH", "40"))

# Hard cap on how many turns go into one prompt. Not a tuning knob to raise casually:
# a small model's extraction quality falls off well before its context limit does.
CHUNK_TURNS = int(os.environ.get("DREAMER_CHUNK_TURNS", "10"))

# How many of the farmer's own headlines the Layer-1 pass gets to look at.
PROFILE_CONTEXT = int(os.environ.get("DREAMER_PROFILE_CONTEXT", "20"))


# --------------------------------------------------------------------------
# Chunking and scheduling
# --------------------------------------------------------------------------

def chunk_turns(turns, cap=CHUNK_TURNS):
    """Chronological chunks of at most `cap` turns, preferring whole conversations.

    Sessions are packed in until the next one would overflow the cap. A session
    longer than the cap is split across consecutive chunks — it stays in order, so
    the later part still sees what the earlier part wrote.
    """
    chunks, current = [], []
    for _sid, sturns in lf.group_sessions(turns):
        sturns = [t for t in sturns if (t.get("query") or "").strip()]
        if not sturns:
            continue
        if len(sturns) > cap:
            if current:
                chunks.append(current)
                current = []
            for i in range(0, len(sturns), cap):
                chunks.append(sturns[i:i + cap])
            continue
        if len(current) + len(sturns) > cap:
            chunks.append(current)
            current = []
        current.extend(sturns)
    if current:
        chunks.append(current)
    return chunks


def batches(queues):
    """Round-robin: each batch takes at most one chunk per farmer.

    queues: {farmer_id: [chunk, ...]} in chronological order.
    Yields [(farmer_id, chunk_index, chunk), ...] per batch.
    """
    depth = max((len(q) for q in queues.values()), default=0)
    for i in range(depth):
        batch = [(fid, i, q[i]) for fid, q in queues.items() if i < len(q)]
        if batch:
            yield batch


# --------------------------------------------------------------------------
# Memory service
# --------------------------------------------------------------------------

def _known_keys():
    try:
        r = requests.get(f"{MEMORY_API}/keys", timeout=30)
        r.raise_for_status()
        return [{"key": k["key"], "description": k.get("description")} for k in r.json()["keys"]]
    except Exception as exc:  # noqa: BLE001 - first run has no keys yet
        print(f"  ! could not read key vocabulary ({exc}); proceeding without it", file=sys.stderr)
        return []


def _candidates(farmer_id, entry):
    """Everything that might be the same situation. Deliberately WIDE.

    Three deliberate changes from the cheap version this replaces, all affordable
    because this runs nightly rather than during a turn:

    - **No type filter.** The old version only compared an entry against existing
      entries of the SAME type, which made the most common real duplication
      impossible to detect: one grievance recorded as a `complaint` one week and a
      `tracker` the next. On real data that alone produced six live entries for a
      single situation.
    - **Both vectors, several phrasings.** The headline and the expanded text are
      searched, then the results unioned. A
      situation described differently in two conversations is often only reachable
      from one of them.
    - **Wide, then reranked.** Scores over one farmer's memory are nearly
      uninformative (measured: everything lands in 0.74-0.88 whether or not it is
      the same situation), so ordering by score and taking 3 was close to arbitrary.
      Take many, then let a model that can read the text do the ordering.
    """
    seen, out = set(), []
    queries = [entry["headline"]]
    if (entry.get("expanded") or "").strip():
        queries.append(entry["expanded"][:400])

    for q in queries:
        for path in ("search", "search_expanded"):
            try:
                r = requests.post(
                    f"{MEMORY_API}/internal/memory/{farmer_id}/{path}",
                    json={"query": q, "limit": CANDIDATE_POOL, "only_current": True},
                    timeout=60,
                )
                r.raise_for_status()
                for hit in r.json().get("hits", []):
                    if hit["id"] not in seen:
                        seen.add(hit["id"])
                        out.append(hit)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! candidate lookup failed ({exc})", file=sys.stderr)
    return out


def _rerank(entry, candidates):
    """Reorder candidates by same-situation likelihood, dropping implausible ones."""
    if len(candidates) <= 1:
        return candidates
    try:
        out = llm.ask_json(
            prompts.RERANK_SYSTEM,
            prompts.build_rerank_prompt(entry, candidates),
            prompts.RERANK_SCHEMA,
            max_tokens=500,
        )
        order = [i for i in (out.get("ranked") or []) if 1 <= i <= len(candidates)]
    except Exception as exc:  # noqa: BLE001
        print(f"  ! rerank failed ({exc}); using search order", file=sys.stderr)
        return candidates[:MATCH_CANDIDATES]
    # An empty ranking is a real answer: nothing here is the same situation.
    return [candidates[i - 1] for i in order][:MATCH_CANDIDATES]


def _decide(entry, candidates):
    """Update an existing entry, or write a new one? Defaults to new when unclear."""
    if not candidates:
        return "new", None, "nothing existing of this kind"
    verdict = llm.ask_json(
        prompts.MATCH_SYSTEM,
        prompts.build_match_prompt(entry, candidates),
        prompts.MATCH_SCHEMA,
        max_tokens=400,
    )
    decision = verdict.get("decision")
    target = verdict.get("updates_entry_id")
    if decision == "update" and any(c["id"] == target for c in candidates):
        return "update", target, verdict.get("reason", "")
    # An "update" verdict naming an id that isn't in the candidate list is not
    # trustworthy — fall back to the safe side rather than guessing which it meant.
    return "new", None, verdict.get("reason", "")


def _strip_ids_from_prose(entry):
    """Enforce store-but-never-speak structurally, not by trusting the instruction.

    An animal id may live in metadata. If the model put one in the prose anyway, move
    it: the prose is what gets read back to the farmer.
    """
    meta = entry.get("metadata") or {}
    # May be a single tag or a list when one situation covers two animals.
    raw = meta.get("animal_id")
    aids = [str(a) for a in (raw if isinstance(raw, (list, tuple)) else [raw]) if a]
    for field in ("headline", "expanded"):
        text = entry.get(field) or ""
        if not text:
            continue
        if any(a in text for a in aids):
            for a in aids:
                text = text.replace(a, "")
            text = re.sub(r"\s+", " ", text).strip()
            print(f"  ! moved a known animal id out of {field}", file=sys.stderr)
        # Backstop for tags the model wrote into the prose WITHOUT also putting them
        # in metadata — the check above would miss those entirely. Nothing a farmer
        # should hear read back has an 8+ digit run in it: amounts are four or five
        # digits, dates are punctuated. Ear tags are twelve.
        stripped = re.sub(r"\s+", " ", re.sub(r"\b\d{8,}\b", "", text)).strip()
        if stripped != text:
            print(f"  ! stripped an unregistered identifier from {field}", file=sys.stderr)
            text = stripped
        if text != (entry.get(field) or ""):
            entry[field] = text.strip()
    return entry


# --------------------------------------------------------------------------
# Layer 2 — the always-injected short version
# --------------------------------------------------------------------------

def _write_headline(entry, first_ts=None, last_ts=None, times_raised=None):
    """Re-derive the short version from the full detail. Falls back to what we have."""
    if not (entry.get("expanded") or "").strip():
        return entry["headline"]  # nothing more to synthesise from
    try:
        out = llm.ask_json(
            prompts.HEADLINE_SYSTEM,
            prompts.build_headline_prompt(entry, first_ts, last_ts, times_raised),
            prompts.HEADLINE_SCHEMA,
            max_tokens=300,
        )
        new = (out.get("headline") or "").strip()
        # A headline pass that returns something absurd must not replace a working one.
        if 15 < len(new) < 400:
            return new
    except Exception as exc:  # noqa: BLE001
        print(f"  ! headline pass failed ({exc}); keeping extracted headline", file=sys.stderr)
    return entry["headline"]


# --------------------------------------------------------------------------
# One chunk
# --------------------------------------------------------------------------

def process_chunk_agent(farmer_id, turns, known_keys, dry_run=False):
    """Hand one chunk to the dreaming agent (the default path).

    The agent decides for itself how to bring memory up to date: it searches, often
    several ways at once, checks whether a situation is already tracked under
    different wording, and updates or merges rather than
    always writing new. Every write it proposes is judged before it lands.

    The fixed pipeline in process_chunk below is kept as a baseline to compare
    against, not as a fallback — they make different mistakes and it is worth being
    able to measure which.
    """
    payload = [t for t in turns if (t.get("query") or "").strip()]
    if not payload:
        return [], known_keys

    ctx = agent.dream(
        MEMORY_API, farmer_id, payload,
        source_ts=payload[-1].get("ts"), session_id=payload[0].get("session_id"),
        known_keys=known_keys, dry_run=dry_run,
    )
    written = []
    for w in ctx.writes:
        if w["action"] == "nothing":
            continue
        e = w.get("entry") or {}
        written.append({"action": w["action"],
                        "entry": {"status": e.get("status", "-"),
                                  "headline": e.get("headline") or ""},
                        "reason": "; ".join(w.get("notes") or [])})
    for b in ctx.blocked:
        written.append({"action": "REFUSED",
                        "entry": {"status": b["entry"].get("status", "-"),
                                  "headline": b["entry"].get("headline") or ""},
                        "reason": "; ".join(b.get("why") or [])})
    # A key the agent documented mid-run should be visible to the next chunk.
    if any(w["action"] in ("write", "update") for w in ctx.writes) and not dry_run:
        known_keys = _known_keys()
    return written, known_keys


def process_chunk(farmer_id, turns, known_keys, dry_run=False, headline_pass=True):
    written = []
    payload_turns = [t for t in turns if (t.get("query") or "").strip()]
    if not payload_turns:
        return written, known_keys

    result = llm.ask_json(
        prompts.EXTRACT_SYSTEM,
        prompts.build_extract_prompt(payload_turns, known_keys, f"...{str(farmer_id)[-4:]}"),
        prompts.EXTRACT_SCHEMA,
    )
    entries = result.get("entries") or []
    chunk_ts = payload_turns[-1].get("ts")
    session_id = payload_turns[0].get("session_id")

    # Document any newly invented keys so the next chunk can reuse them instead of
    # coining a synonym (memory-overview.md: reuse-first, no human gate).
    new_keys = result.get("new_keys") or []
    for nk in new_keys:
        if dry_run:
            continue
        try:
            requests.post(f"{MEMORY_API}/keys/doc", json={
                "key": nk["key"], "description": nk["description"]}, timeout=30).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not document key {nk.get('key')!r}: {exc}", file=sys.stderr)

    for entry in entries:
        pctx = tools.Ctx(MEMORY_API, farmer_id, turns=payload_turns,
                         known_keys=known_keys, dry_run=dry_run)
        try:
            pctx = pctx.for_sources(entry.get("source_turns"))
        except ValueError as exc:
            written.append({"action": "REFUSED", "entry": entry, "reason": str(exc)})
            continue
        chunk_ts, session_id = pctx.source_ts, pctx.session_id
        entry = _strip_ids_from_prose(entry)
        cands = _rerank(entry, _candidates(farmer_id, entry)) if not dry_run else []
        action, target, reason = _decide(entry, cands) if cands else ("new", None, "")

        old = next((c for c in cands if c["id"] == target), None) if action == "update" else None
        if old:
            # A merge means the entry now covers more than either version did. Carry
            # the earlier detail forward instead of replacing it — the point of an
            # update is the thread, and PATCH keeps the old version as history anyway.
            prev = (old.get("expanded") or "").strip()
            now = (entry.get("expanded") or "").strip()
            if prev and now and now not in prev:
                entry["expanded"] = f"{prev}\n\n[{dates.fmt(chunk_ts)}] {now}"
            elif prev and not now:
                entry["expanded"] = prev
            merged_meta = dict(old.get("metadata") or {})
            merged_meta.update(entry.get("metadata") or {})
            entry["metadata"] = merged_meta
            times = len(set((old.get("source_sessions") or []) +
                            [t.get("session_id") for t in pctx.turns]) - {None}) or 1
            entry = _strip_ids_from_prose(entry)
        else:
            times = 1

        if headline_pass and not dry_run:
            entry["headline"] = _write_headline(
                entry,
                first_ts=(old or {}).get("first_source_ts") or chunk_ts,
                last_ts=chunk_ts,
                times_raised=times,
            )
            entry = _strip_ids_from_prose(entry)

        body = {
            "headline": entry["headline"],
            "expanded": entry.get("expanded", ""),
            "status": entry.get("status", "n/a"),
            "source_session_id": session_id,
            "source_ts": chunk_ts,
            "metadata": entry.get("metadata") or {},
        }
        if entry.get("expires_at"):
            body["expires_at"] = entry["expires_at"]

        if dry_run:
            written.append({"action": "would-" + action, "entry": body, "reason": reason})
            continue

        # Through the same funnel as the agent. The baseline is meant to differ in
        # how it DECIDES, not in which invariants it is allowed to skip — a second
        # write path is what let the judges be bypassed in the first place.
        if action == "update":
            res = writer.apply(pctx, {
                "op": "update", "entry_id": target, "headline": body["headline"],
                "expanded": body["expanded"], "status": body["status"],
                "metadata": body["metadata"]})
        else:
            res = writer.apply(pctx, {
                "op": "write", "headline": body["headline"],
                "expanded": body["expanded"], "status": body["status"],
                "expires_at": body.get("expires_at"),
                "metadata": body["metadata"]})
        if res.get("ok"):
            written.append({"action": action, "entry": body, "reason": reason})
        else:
            why = "; ".join(res.get("reasons") or []) or res.get("reason") or res.get("error", "not applied")
            written.append({"action": "REFUSED", "entry": body, "reason": why})

    if new_keys and not dry_run:
        known_keys = _known_keys()
    return written, known_keys


# --------------------------------------------------------------------------
# Consolidation — the maximal pass. Runs after a farmer's chunks, before Layer 1.
# --------------------------------------------------------------------------
# The per-chunk match sees a handful of candidates and defaults to "new" when
# unsure, which is right for that moment: it only ever sees one conversation, and a
# wrong merge there is silent and permanent. The cost is over-splitting — on real
# data, six live entries for a single grievance, every one with times_raised=1,
# which makes "he has raised this repeatedly" undetectable.
#
# This pass is the other half of that deal. It has no latency budget, it sees the
# farmer's ENTIRE store at once, and its whole job is to put the pieces back
# together. Because it exists, over-splitting during a chunk is recoverable — which
# is what makes the cautious online default safe rather than merely convenient.

def _current_entries(farmer_id):
    """Read all current memories, following storage pagination."""
    entries, cursor, seen = [], None, set()
    while True:
        params = {"limit": 100, "only_current": "true"}
        if cursor:
            params["cursor"] = cursor
        response = requests.get(f"{MEMORY_API}/internal/memory/{farmer_id}/entries",
                                params=params, timeout=60)
        response.raise_for_status()
        body = response.json()
        entries.extend(body["entries"])
        cursor = body.get("next_cursor")
        if not cursor:
            return entries
        if cursor in seen:
            raise ValueError("memory listing returned a repeated cursor")
        seen.add(cursor)


def consolidate(farmer_id, dry_run=False, min_confidence="medium", known_keys=()):
    """Find entries that are the same situation and merge them into one.

    Returns a list of (situation, kept_id, folded_ids).
    """
    try:
        entries = _current_entries(farmer_id)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not read entries for consolidation ({exc})", file=sys.stderr)
        return []
    if len(entries) < 2:
        return []

    # Ordered oldest-first so groups read chronologically in the prompt, and so the
    # entry we keep is the one that has been on record longest.
    entries.sort(key=lambda e: str(e.get("first_source_ts") or e.get("source_ts") or ""))

    groups = []
    for i in range(0, len(entries), CLUSTER_BATCH):
        window = entries[i:i + CLUSTER_BATCH]
        if len(window) < 2:
            continue
        try:
            out = llm.ask_json(
                prompts.CLUSTER_SYSTEM,
                prompts.build_cluster_prompt(window, f"...{str(farmer_id)[-4:]}"),
                prompts.CLUSTER_SCHEMA,
                max_tokens=1500,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! clustering failed ({exc}); leaving entries as they are", file=sys.stderr)
            continue
        allowed = {"high", "medium"} if min_confidence == "medium" else {"high"}
        for g in out.get("groups") or []:
            idx = sorted({n for n in (g.get("members") or []) if 1 <= n <= len(window)})
            if len(idx) < 2:
                continue
            if (g.get("confidence") or "low") not in allowed:
                print(f"      skipped a {g.get('confidence')}-confidence group: "
                      f"{(g.get('situation') or '')[:60]}")
                continue
            groups.append(([window[n - 1] for n in idx], g.get("situation") or "",
                           g.get("reason") or ""))

    done = []
    for frags, situation, reason in groups:
        # Keep the longest-standing fragment, so first_source_ts and the thread's
        # history survive the merge rather than restarting today.
        frags.sort(key=lambda e: str(e.get("first_source_ts") or e.get("source_ts") or ""))
        keep, rest = frags[0], frags[1:]

        if dry_run:
            done.append((situation, keep["id"], [f["id"] for f in rest]))
            continue

        try:
            merged = llm.ask_json(
                prompts.MERGE_SYSTEM,
                prompts.build_merge_prompt(frags, situation),
                prompts.MERGE_SCHEMA,
                max_tokens=1800,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! merge synthesis failed ({exc}); fragments left alone", file=sys.stderr)
            continue

        meta = {}
        for f in frags:
            meta.update(f.get("metadata") or {})
        meta.update(merged.get("metadata") or {})

        entry = {"status": merged.get("status", keep.get("status", "n/a")),
                 "headline": merged["headline"], "expanded": merged.get("expanded", ""),
                 "metadata": meta}

        # The union of the conversations the fragments came from. The service turns
        # this into times_raised, counting DISTINCT conversations — summing the
        # fragments' own counts was wrong, because two fragments written from two
        # chunks of ONE conversation would sum to 2 and claim a repetition that
        # never happened.
        frag_sessions = sorted({s for f in frags for s in (
            (f.get("source_sessions") or []) + [f.get("source_session_id")]) if s})
        frag_traces = sorted({t for f in frags for t in (f.get("source_traces") or [])})

        # Merged entries were the ONLY writes bypassing the judges, and they are the
        # highest-stakes text in the store — a merge is what gets injected every turn
        # from then on. The evidence base for a merge is the fragments themselves
        # (there is no single conversation behind it), plus the aggregate history,
        # which is what licenses the "raised repeatedly" phrasing.
        frag_evidence = {
            "status": keep.get("status"),
            "headline": " | ".join(f.get("headline") or "" for f in frags),
            "expanded": "\n\n".join(
                f"[{dates.fmt(f.get('source_ts'))}] {f.get('expanded') or f.get('headline')}"
                for f in frags),
        }
        agg = {"times_raised": max(1, len(frag_sessions)),
               "first_source_ts": keep.get("first_source_ts") or keep.get("source_ts"),
               "source_ts": max((str(f.get("source_ts") or "") for f in frags), default="")}
        latest = agg["source_ts"] or None
        ctx = tools.Ctx(MEMORY_API, farmer_id, turns=None,
                        source_ts=latest, session_id=None,
                        known_keys=known_keys or (), dry_run=dry_run,
                        trace_ids=frag_traces)
        res = writer.apply(ctx, {
            "op": "update", "entry_id": keep["id"],
            "headline": entry["headline"],
            "expanded": entry.get("expanded", ""), "status": entry.get("status"),
            "metadata": entry["metadata"],
            "extra_sessions": frag_sessions,
            "merge_from": [fragment["id"] for fragment in rest],
            # The evidence for a merge is the fragments themselves — there is no
            # single conversation behind it — plus their aggregate history, which is
            # what licenses "raised repeatedly".
            "_existing": keep, "_evidence": frag_evidence, "_record": agg,
        })
        for n in res.get("notes") or []:
            print(f"      judge: {n[:120]}")
        if not res.get("ok"):
            why = "; ".join(res.get("reasons") or []) or res.get("reason") or res.get("error", "refused")
            print(f"      merge NOT APPLIED: {why[:140]} — fragments left alone")
            continue
        new_id = res["entry_id"]

        folded = []
        for f in rest:
            fr = writer.apply(ctx, {"op": "fold", "entry_id": f["id"],
                                    "merged_into": new_id, "reason": situation[:200]})
            if fr.get("ok"):
                folded.append(f["id"])
            else:
                print(f"  ! could not fold {f['id']}: {fr.get('error')}", file=sys.stderr)
        done.append((situation, new_id, folded))
    return done


# --------------------------------------------------------------------------
# Layer 1 — once per farmer, after their chunks are done
# --------------------------------------------------------------------------

def update_profile(farmer_id, dry_run=False, recent_asks=()):
    """Re-derive the standing record from what is now remembered about the farmer.

    `recent_asks` is a small sample of the farmer's own words, carried over from the
    chunks just processed. Without it the pass can never fill in what the farmer is
    called or how they want to be spoken to — that evidence exists only in their own
    phrasing, and the Layer-2 headlines have paraphrased it away.
    """
    try:
        cur = requests.get(f"{MEMORY_API}/internal/memory/{farmer_id}/profile", timeout=30)
        cur.raise_for_status()
        current = cur.json().get("profile") or {}
        headlines = sorted(_current_entries(farmer_id),
                           key=lambda e: e.get("source_ts") or "", reverse=True)[:PROFILE_CONTEXT]
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not read state for the standing record ({exc}); skipping",
              file=sys.stderr)
        return None

    # A farmer with no situation memories at all can still have a name and a way
    # they like to be spoken to, so their own words alone are enough to run on.
    if not headlines and not current and not recent_asks:
        return None

    try:
        out = llm.ask_json(
            prompts.PROFILE_SYSTEM,
            prompts.build_profile_prompt(
                current, headlines, f"...{str(farmer_id)[-4:]}", recent_asks=recent_asks),
            prompts.PROFILE_SCHEMA,
            max_tokens=600,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! standing record pass failed ({exc}); leaving it as it was", file=sys.stderr)
        return None

    fields = {
        k: v.strip() for k, v in (out.get("fields") or {}).items()
        if k in prompts.PROFILE_FIELDS and isinstance(v, str) and v.strip()
    }
    if not fields:
        return None
    if dry_run:
        return fields

    evidence = {"headline": "; ".join(f"{k}: {v}" for k, v in current.items()),
                "expanded": "\n".join(h.get("headline") or "" for h in headlines)}
    ctx = tools.Ctx(MEMORY_API, farmer_id, turns=[{"query": ask} for ask in recent_asks])
    result = writer.apply(ctx, {"op": "profile", "fields": fields, "_evidence": evidence})
    if not result.get("ok"):
        print("  ! standing record not saved: " + str(result.get("reasons") or result.get("error")), file=sys.stderr)
        return None
    if result.get("unchanged"):
        return None
    return result["fields"]


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farmer", help="one farmer id")
    ap.add_argument("--farmers", help="comma-separated farmer ids")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--turns-file", help="Frozen turn JSON for a one-farmer replay; requires --until")
    ap.add_argument("--until", help="ISO cutoff — ignore anything the farmer said after this")
    ap.add_argument("--dry-run", action="store_true", help="decide but write nothing")
    ap.add_argument("--max-chunks", type=int, default=0, help="per farmer; 0 = all")
    ap.add_argument("--no-headline-pass", action="store_true",
                    help="skip the Layer-2 rewrite and keep the extracted headline")
    ap.add_argument("--no-profile-pass", action="store_true",
                    help="skip the Layer-1 standing record")
    ap.add_argument("--no-consolidate", action="store_true",
                    help="skip the whole-store merge pass")
    ap.add_argument("--consolidate-only", action="store_true",
                    help="run only the merge pass over what is already stored")
    ap.add_argument("--pipeline", action="store_true",
                    help="use the fixed extract/match/write pipeline instead of the agent")
    args = ap.parse_args()

    ids = [f.strip() for f in (args.farmers or args.farmer or "").split(",") if f.strip()]
    if not ids:
        ap.error("give --farmer or --farmers")

    print(f"model: {llm.describe()}")
    print(f"memory api: {MEMORY_API}")
    print(f"turn cap per prompt: {CHUNK_TURNS}")
    print(f"mode: {'fixed pipeline' if args.pipeline else 'agent with tools'}")

    if args.consolidate_only:
        for fid in ids:
            print(f"\nfarmer ...{str(fid)[-4:]}")
            merges = consolidate(fid, dry_run=args.dry_run)
            for situation, kept, folded in merges:
                print(f"  {'would-' if args.dry_run else ''}merged {len(folded) + 1} -> 1: {situation[:70]}")
            if not merges:
                print("  nothing to merge")
        return

    if args.turns_file and (len(ids) != 1 or not args.until):
        ap.error("--turns-file requires one farmer and an explicit --until cutoff")
    ro = lf.LangfuseRO() if not args.turns_file else None
    queues = {}
    for fid in ids:
        if args.turns_file:
            from datetime import datetime
            with open(args.turns_file) as source:
                turns = json.load(source)
            cutoff = datetime.fromisoformat(args.until.replace("Z", "+00:00"))
            if any(datetime.fromisoformat(t["ts"].replace("Z", "+00:00")) > cutoff for t in turns):
                raise ValueError("Frozen training turns contain data after the cutoff")
        else:
            turns = ro.farmer_turns(fid, days_back=args.days, until=args.until)
        chunks = chunk_turns(turns)
        if args.max_chunks:
            chunks = chunks[: args.max_chunks]
        queues[fid] = chunks
        print(f"farmer ...{str(fid)[-4:]}: {len(turns)} turns -> {len(chunks)} chunk(s)"
              + (f", cut off at {args.until}" if args.until else ""))
    queues = {k: v for k, v in queues.items() if v}
    if not queues:
        print("nothing to dream about")
        return

    known_keys = _known_keys()
    remaining = {fid: len(q) for fid, q in queues.items()}
    asks = {fid: [] for fid in queues}
    total = 0

    for bno, batch in enumerate(batches(queues), 1):
        print(f"\n=== batch {bno}: {len(batch)} farmer(s), one chunk each ===")
        for fid, ci, chunk in batch:
            span = f"{(chunk[0].get('ts') or '')[:10]}"
            # Keep the most recent handful of the farmer's own questions for the
            # Layer-1 pass. Bounded on both counts — a standing record derived from
            # a huge sample is no better, and the prompt has to stay small.
            asks[fid] = ([(t.get("query") or "").strip()[:200]
                          for t in chunk if (t.get("query") or "").strip()]
                         + asks[fid])[:10]
            if args.pipeline:
                written, known_keys = process_chunk(
                    fid, chunk, known_keys,
                    dry_run=args.dry_run, headline_pass=not args.no_headline_pass,
                )
            else:
                written, known_keys = process_chunk_agent(
                    fid, chunk, known_keys, dry_run=args.dry_run)
            # Count only what actually landed. Refusals are useful to see but they
            # are not writes, and reporting them as such overstates the work done.
            total += sum(1 for w in written if w["action"] in ("write", "update", "fold",
                                                               "would-write", "would-update"))
            landed = sum(1 for w in written if w["action"] not in ("REFUSED", "nothing"))
            refused = sum(1 for w in written if w["action"] == "REFUSED")
            print(f"...{str(fid)[-4:]} chunk {ci + 1}/{remaining[fid]} {span} "
                  f"{len(chunk):>2} turns -> {landed} written"
                  + (f", {refused} refused" if refused else ""))
            for w in written:
                e = w["entry"]
                print(f"      {w['action']:12} [{e['status']}] {e['headline'][:88]}")
                if w.get("reason") and w["action"] == "REFUSED":
                    print(f"        reason: {w['reason']}")

            # The farmer's queue is drained. Consolidate BEFORE the standing record,
            # so Layer 1 is derived from whole situations rather than fragments.
            if ci + 1 == remaining[fid] and not args.no_consolidate:
                merges = consolidate(fid, dry_run=args.dry_run, known_keys=known_keys)
                for situation, kept, folded in merges:
                    print(f"      {'would-' if args.dry_run else ''}merged {len(folded) + 1} "
                          f"entries -> 1: {situation[:64]}")
                if not merges:
                    print("      consolidation: nothing to merge")
            if ci + 1 == remaining[fid] and not args.no_profile_pass:
                fields = update_profile(fid, dry_run=args.dry_run, recent_asks=asks[fid])
                if fields:
                    print(f"      {'would-' if args.dry_run else ''}standing record: "
                          + "; ".join(f"{k}={v}" for k, v in fields.items()))

    print(f"\ntotal entries {'that would be ' if args.dry_run else ''}written: {total}")


if __name__ == "__main__":
    main()
