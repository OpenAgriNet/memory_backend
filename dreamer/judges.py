"""Judges that gate every write into memory.

Deliberately placed in the write path rather than offered to the agent as advice. An
agent that can be talked out of a rule does not have that rule; these run on every
write no matter how the agent got there, and a block is a block.

Three independent questions, so they run CONCURRENTLY — nothing here is on a live
turn, but a write that waits for three sequential model calls slows a backfill by
three, and the questions genuinely don't depend on each other:

  pii      Is there anything in this text that must not be stored at all?
  keys     Do these metadata keys duplicate vocabulary that already exists?
  honesty  Does it assert as settled something the conversation left unconfirmed?

This is the "smaller model with several focused prompts and judges" preference from
oan-brain/style/swe-style.md applied where it matters most: the write is the
irreversible step. Everything before it is a proposal.
"""
from concurrent.futures import ThreadPoolExecutor

if __package__:
    from . import dates, llm
else:
    import dates
    import llm

# --------------------------------------------------------------------------

PII_SYSTEM = """You are the last check before a piece of text is written into a dairy farmer's long-term memory. Decide whether it is safe to store.

BLOCK when the proposal actually contains the following restricted information. Do not infer a restriction merely because it mentions health, payments, records or a cooperative:
- Selling milk outside the society, adulteration, or any banned practice.
- Admissions of financial distress or being unable to afford feed.
- Family conflict, human illness, or personal distress. This does not include livestock illness.
- Anything about a person other than this farmer.
- Anything that reads as an accusation against a named individual, rather than a description of a problem.
- Identifiers in metadata other than the permitted animal_id. Text redaction cannot remove metadata; block that proposal for correction.

REDACT (do not block) if it is otherwise fine but contains an identifier that should not be in stored prose:
- An account number, transaction id, phone number, or registration number.
- An animal ear tag written into the text. Ear tags may be stored in metadata under animal_id, but never in prose, because prose is read back to the farmer.
Return the same text with the identifier replaced by a plain description ("her cow", "a deduction of around 1200 rupees").

An unexplained deduction, missing record, or request for correction is an ordinary support concern, not by itself an admission of financial distress or wrongdoing.

ALLOW ordinary livestock-health information: animal symptoms, illness, veterinary advice, treatment history and recovery are central to dairy support. Do not block these merely because they concern health. Still apply the restrictions above if the proposal also contains a prohibited admission or private human information. Whether an animal diagnosis or treatment is supported is checked separately by the honesty judge.

ALLOW everything else. A farmer's ordinary grievance about a cooperative's process is exactly what this memory is for — a complaint that the society is not registering calves is a legitimate, storable problem, not sensitive content. Do not block a farmer from having their own problem remembered.

Be specific about what you found. Return strict JSON only."""

PII_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["allow", "redact", "block"]},
        "problems": {"type": "array", "items": {"type": "string"}},
        "redacted_headline": {"type": "string"},
        "redacted_expanded": {"type": "string"},
    },
    "required": ["verdict"],
}

KEYS_SYSTEM = """You are checking the metadata keys attached to a new memory entry against the keys already in use for this system.

Two keys that mean the same thing are a real cost: both become useless for filtering, because a query on one silently misses everything filed under the other.

For each key on the new entry, decide:
- "reuse" — an existing key already means this. Name it. Wording differences do not matter; meaning does.
- "keep" — genuinely new, and worth having.
- "drop" — not worth storing as a key at all: it restates a system field (type, status, dates), it is a one-off value nobody will ever filter on, or it is free text masquerading as a key.

Be strict about reuse and strict about drop. A key is only worth having if someone would plausibly filter on it later.

Return strict JSON only."""

KEYS_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "action": {"type": "string", "enum": ["reuse", "keep", "drop"]},
                    "reuse_existing": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["key", "action"],
            },
        },
    },
    "required": ["decisions"],
}

HONESTY_SYSTEM = """You are checking whether a memory entry is honest about what is actually known, against the conversation it was drawn from.

The failure this exists to catch is specific and it has really happened: a farmer was left believing a vet visit had been arranged when it never was. A memory that records "a visit was booked" when the conversation only shows a visit was OFFERED will cause the assistant to confirm something false, months later, with confidence.

BLOCK the entry if it:
- States as done, booked, arranged or resolved anything the conversation only shows as offered, requested, or attempted.
- States a cause or diagnosis the conversation does not establish.
- Attributes to the farmer something they did not say.
- Turns an absence of evidence into a negative fact — "no visit happened" instead of "no visit was confirmed in this conversation". The farmer may well have arranged one elsewhere.

ALLOW it if every claim is either something the farmer said, something the assistant said, already established in what is on record, or is explicitly marked as unconfirmed.

If a section headed ALREADY ON RECORD is present, this entry is an UPDATE to an ongoing situation. Everything in that section is established fact from earlier conversations and must be treated as supported — an update is meant to carry it forward. Judge only what the update ADDS beyond it.

If a section headed WHAT THE RECORD ITSELF SHOWS is present, use it for claims about REPETITION and DURATION. "He has raised this repeatedly", "this has been going on since August", "asked about four times" are SUPPORTED when the record shows several occasions or a date span — they are statements about the stored history, and no single conversation could ever evidence them. Do not block a claim of repetition that the record supports. Only block it when the record shows a single occasion.

Quote the specific phrase that is not supported. Return strict JSON only."""

HONESTY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["allow", "block"]},
        "problems": {"type": "array", "items": {"type": "string"}},
        "suggested_headline": {"type": "string"},
    },
    "required": ["verdict"],
}


# --------------------------------------------------------------------------

def _pii(entry):
    return llm.ask_json(PII_SYSTEM, f"""HEADLINE:
{entry.get('headline')}

DETAIL:
{entry.get('expanded') or '(none)'}

CONTENTS PAGE (also stored and shown to the reply agent):
{entry.get('contents') or []}

STRUCTURED DETAILS (identifiers outside animal_id must not be retained here either):
{entry.get('metadata') or {}}
""", PII_SCHEMA, max_tokens=700)


def _keys(entry, known_keys):
    meta = entry.get("metadata") or {}
    if not meta:
        return {"decisions": []}
    existing = "\n".join(
        f"- {k['key']}: {k.get('description') or '(no description recorded)'}"
        for k in known_keys
    ) or "(none yet)"
    return llm.ask_json(KEYS_SYSTEM, f"""KEYS ALREADY IN USE:
{existing}

NEW ENTRY
  headline: {entry.get('headline')}
  metadata: {meta}
""", KEYS_SCHEMA, max_tokens=700)


def _honesty(entry, turns, existing=None, record=None):
    convo = "\n\n".join(
        f"FARMER: {t.get('query')}\nASSISTANT: {t.get('answer')}"
        for t in (turns or []) if t.get("query")
    ) or "(the source conversation was not provided)"

    # On an UPDATE the evidence is the conversation *plus* what is already on record.
    # Without this the judge blocks every legitimate carry-forward: the entry being
    # updated is where the earlier facts live, and demanding that this week's turns
    # re-establish them means an ongoing situation can never be moved along.
    prior = ""
    if existing:
        prior = f"""

ALREADY ON RECORD (established in earlier conversations — treat as supported evidence,
this is what the entry is being updated FROM):
  status:   {existing.get('status')}
  headline: {existing.get('headline')}
  detail:   {existing.get('expanded') or '(none)'}"""

    # A merged or repeatedly-updated entry legitimately says "he has raised this
    # several times" — justified by the RECORD, not by any single conversation. Without
    # this the judge blocks every consolidated entry, because one conversation can
    # never show repetition.
    hist = ""
    if record:
        times = record.get("times_raised") or 1
        first = dates.fmt(record.get("first_source_ts"))
        last = dates.fmt(record.get("source_ts"))
        hist = f"""

WHAT THE RECORD ITSELF SHOWS (supported evidence — these are facts about the stored
history, not claims from the conversation):
  this situation has been recorded on {times} separate occasion(s)"""
        if first != dates.UNKNOWN:
            hist += f"\n  first came up {first}" + (f", last mentioned {last}" if last != first else "")

    return llm.ask_json(HONESTY_SYSTEM, f"""PROPOSED ENTRY
  status:   {entry.get('status')}
  headline: {entry.get('headline')}
  detail:   {entry.get('expanded') or '(none)'}
  contents: {entry.get('contents') or []}

THE CONVERSATION IT CAME FROM:
{convo}{prior}{hist}
""", HONESTY_SCHEMA, max_tokens=800)


def _checked_call(fn, schema, *args, attempts=2):
    """Retry unavailable or malformed checks; never interpret them as approval."""
    for attempt in range(attempts):
        try:
            result = fn(*args)
            valid, reason = llm._shape_ok(result, schema)
            if not valid:
                raise ValueError(reason)
            return result
        except Exception as exc:
            if attempt == attempts - 1:
                return {"_error": str(exc)}


def review(entry, *, turns=None, known_keys=(), skip=(), existing=None, record=None):
    """Run every judge concurrently and combine them into one decision.

    Returns (decision, entry, notes):
      decision "allow"  -> write it; `entry` may have been redacted or had keys dropped
      decision "block"  -> do not write; `notes` says why, for the agent to act on

    Unavailable checks are retried. If they still fail, the proposal stays unwritten
    and the caller gets an explicit retry reason. Existing memory is unchanged.
    """
    jobs = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        if "pii" not in skip:
            jobs["pii"] = pool.submit(_checked_call, _pii, PII_SCHEMA, entry)
        if "keys" not in skip:
            jobs["keys"] = pool.submit(_checked_call, _keys, KEYS_SCHEMA, entry, list(known_keys))
        if "honesty" not in skip:
            jobs["honesty"] = pool.submit(_checked_call, _honesty, HONESTY_SCHEMA, entry, turns, existing, record)
        results = {}
        for name, fut in jobs.items():
            try:
                results[name] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[name] = {"_error": str(exc)}

    entry = dict(entry)
    notes = []

    unavailable = [name for name, result in results.items() if result.get("_error")]
    if unavailable:
        return "block", entry, [
            "retry needed: check unavailable after retry: " + ", ".join(unavailable)]

    pii = results.get("pii", {})
    if pii.get("verdict") == "block":
        return "block", entry, [f"must not be stored: {p}" for p in (pii.get("problems") or ["sensitive content"])]
    elif pii.get("verdict") == "redact":
        for src, dst in (("redacted_headline", "headline"), ("redacted_expanded", "expanded")):
            new = (pii.get(src) or "").strip()
            if new and new != (entry.get(dst) or ""):
                entry[dst] = new
        notes.append("identifiers redacted from the text: "
                     + "; ".join(pii.get("problems") or ["unspecified"]))

    hon = results.get("honesty", {})
    if hon.get("verdict") == "block":
        return "block", entry, ["overstates what is known: " + p for p in
                                (hon.get("problems") or ["unsupported claim"])] + (
            [f"try instead: {hon['suggested_headline']}"] if hon.get("suggested_headline") else [])

    keys = results.get("keys", {})
    if keys.get("_error"):
        notes.append(f"key judge unavailable ({keys['_error'][:80]})")
    else:
        meta = dict(entry.get("metadata") or {})
        for d in keys.get("decisions") or []:
            k, action = d.get("key"), d.get("action")
            if k not in meta:
                continue
            if action == "drop":
                meta.pop(k, None)
                notes.append(f"dropped key {k!r}: {d.get('reason', 'not worth filtering on')}")
            elif action == "reuse" and d.get("reuse_existing"):
                target = d["reuse_existing"]
                if target != k:
                    meta[target] = meta.pop(k)
                    notes.append(f"renamed key {k!r} -> {target!r} (already in use)")
        entry["metadata"] = meta

    if pii.get("verdict") == "redact":
        # Redaction is another generated rewrite. Check its final claims and safety.
        safe = _checked_call(_pii, PII_SCHEMA, entry)
        honest = _checked_call(_honesty, HONESTY_SCHEMA, entry, turns, existing, record)
        if safe.get("_error") or honest.get("_error"):
            return "block", entry, notes + ["retry needed: redacted text could not be checked"]
        if safe.get("verdict") != "allow" or honest.get("verdict") != "allow":
            return "block", entry, notes + ["redacted text needs correction before saving"]
    return "allow", entry, notes
