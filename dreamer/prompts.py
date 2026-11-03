import json
import sys
from pathlib import Path

# The service image and background job use the same program-owned schema file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
if __package__:
    from service import profile_schema
    from . import dates
else:
    import profile_schema
    import dates
"""The instruction sheets that decide what gets remembered.

These are the heart of the dreamer — no tool decides what matters, this does.
Everything here is a direct translation of
oan_horizon/future_work/memory_amul/memory-overview.md (Section 2 principles,
Section 6 Decisions 1 and 2) into rules a model can follow.

Four separate small prompts rather than one big one, per the standing engineering
preference (oan-brain/style/swe-style.md): a small model split into narrow steps with
validation on top beats one large call, and each step can be graded on its own.

  EXTRACT  (Layer 3)  what happened in these turns, in detail
  MATCH               is this the same situation we already track, or new?
  HEADLINE (Layer 2)  the two sentences that get injected into every future turn
  PROFILE  (Layer 1)  the standing record — a fixed schema, identical for every farmer
"""

# --------------------------------------------------------------------------
# Shared rules — stated once, pasted into the prompts that need them, so the
# storage policy cannot drift between steps.
# --------------------------------------------------------------------------

_NEVER_STORE = """NEVER WRITE, UNDER ANY CIRCUMSTANCES:
- Anything the farmer could be uncomfortable with later, or that could harm them if the cooperative, a bank or an official saw it: selling milk outside the society, using banned practices, admitting they cannot afford feed, family or financial distress, anything about other people's affairs.
- Account numbers, transaction ids, phone numbers, registration numbers. Describe the thing instead — "a deduction of around 1200 rupees" — never the identifier.
- Anything you inferred rather than heard. If the farmer did not say it, it is not a memory."""

_ANIMAL_ID_RULE = """ANIMAL IDENTIFIERS — MAY BE STORED, NEVER SPOKEN:
An animal's ear tag or id is useful to keep: it is what makes "is this the same animal as last time" a certain answer rather than a guess. So you may record it — but ONLY inside "metadata", under the key "animal_id".
It must NEVER appear in "headline" or "expanded", because those are the words the assistant reads back to the farmer. In that prose, always describe the animal the way the farmer does: "her cow", "the older buffalo", "the one that calved in June"."""


# --------------------------------------------------------------------------
# Layer 3 — what is worth storing at all, in full detail
# --------------------------------------------------------------------------

USEFULNESS_RULE = """Decide from the evidence, with no target for how many memories to create.
Evaluate each farmer statement, not just the dominant topic of the batch. Routine
record lookups alongside a useful concern or intention do not make that concern
routine. Before skipping a batch, check its questions about problems, plans and
next steps individually.
Would retaining this help a later conversation understand the farmer's situation,
avoid repeating useful advice, or follow up on a concern, intention or outcome?
If yes, retain the supported context; if no, explain why and leave it out.

A stated goal or exploration of how to start something can be useful even without
a commitment or unresolved complaint. Remember it as an interest or discussion,
not as an action already taken. For example, asking how to start tracking each
animal's production can be retained without claiming the farmer started a register.

A complete answer is not proof that the farmer's situation is resolved. Keep an
unresolved concern even when the assistant gave advice or referred the farmer
elsewhere. Record what was reported, what was advised, and what remains unknown.
An ordinary livestock-health concern is eligible: symptoms, care discussed,
treatment reported by the farmer, and follow-up can provide useful continuity.
Do not turn a hypothetical question into an actual illness or advice into a
confirmed diagnosis, treatment, purchase or action.

Use live Amul records for current quantities, FAT/SNF, herd counts, prices, payments,
account links, village and union. A request to look up vaccination, deworming or
AI records is also a live-record request, not by itself a lasting care plan. Do not
retain merely that a lookup happened. Do not copy live values as current facts. A farmer's
unexplained missing entry or disputed deduction is a separate concern worth
considering; retain its context without assuming the record is wrong or resolved.
A general factual question without personal context, intention or useful follow-up
can be skipped. Repetition alone does not prove an ongoing problem.
"""

EXTRACT_SYSTEM = f"""You maintain the long-term memory of Sarlaben, Amul's assistant for dairy farmers in Gujarat.

You are given a short run of turns from one farmer's conversations. Decide what — if anything — is worth remembering for FUTURE conversations, and return it as structured entries.

Each entry must include source_turns: the numbers of only the turns supporting that situation. Do not attach unrelated turns.

{USEFULNESS_RULE}

Anything supported and useful may be remembered; there is no category list.
Store separate situations separately. Do not assign category labels.

{_NEVER_STORE}

{_ANIMAL_ID_RULE}

HOW TO WRITE EACH ENTRY:
- "headline": 1-2 plain sentences. Self-contained: it must make sense months later with no other context. Written about the farmer in the third person.
- "expanded": the fuller detail — what was tried, what was advised, figures the farmer gave, how long it has been going on. This is only read when the assistant deliberately looks it up, so it can be longer. Leave it empty if the headline already says everything.
- "status": "open" if unresolved, "pending" if something is in progress, "resolved" if closed, "n/a" for a plain fact.
- Be honest about what is NOT known. If a visit was offered but never confirmed, say "no booking was confirmed in this chat" — never "no visit happened", because the farmer may have arranged one elsewhere.
- "expires_at": only when a real end date is genuinely knowable from the conversation — a milk withdrawal period after treatment, an application deadline. ISO date. Otherwise leave it out. Do not guess.
- "metadata": extra detail worth keeping that does not fit the fields above.

USING METADATA KEYS — REUSE BEFORE INVENTING:
You will be shown the keys already in use, with what they mean. Before inventing a key, check that list and reuse an existing one if it fits, even if you would have phrased it differently. Only invent a key when nothing existing covers it, and when you do, describe it in "new_keys" so the next run can reuse yours. Two keys meaning the same thing is a real cost — it makes both useless for filtering.

Return strict JSON only."""


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_turns": {"type": "array", "items": {"type": "integer"}},
                    "headline": {"type": "string"},
                    "expanded": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "pending", "resolved", "n/a"]},
                    "expires_at": {"type": "string"},
                    "metadata": {"type": "object", "additionalProperties": True},
                },
                "required": ["headline", "status", "source_turns"],
            },
        },
        "new_keys": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["key", "description"],
            },
        },
        "nothing_worth_remembering": {"type": "boolean"},
    },
    "required": ["entries"],
}


def build_extract_prompt(turns, known_keys, farmer_label):
    """One chunk of at most CHUNK_TURNS turns, plus the key vocabulary already in use."""
    convo = "\n\n".join(
        f"Turn {n} [{dates.fmt(t.get('ts'))}] FARMER: {t['query']}\nASSISTANT: {t['answer']}"
        for n, t in enumerate(turns, 1) if t.get("query")
    )
    if known_keys:
        keys = "\n".join(
            f"- {k['key']}: {k.get('description') or '(no description recorded)'}"
            for k in known_keys
        )
    else:
        keys = "(none yet — you are the first to add any)"
    span = dates.span(turns[0].get("ts"), turns[-1].get("ts"))
    return f"""Farmer: {farmer_label}
These turns took place: {span}

METADATA KEYS ALREADY IN USE — reuse these before inventing new ones:
{keys}

TURNS:
{convo}
"""


# --------------------------------------------------------------------------
# Decision 2 — is this an update to something we already have, or new?
# --------------------------------------------------------------------------

MATCH_SYSTEM = """You are deciding whether a newly extracted memory about a dairy farmer is an UPDATE to something already on record, or a genuinely NEW thing.

You get one new entry and a few existing entries for the same farmer.

Say it is an update ONLY when it is clearly the same underlying situation — the same animal's same problem, the same complaint about the same thing, the same application. The wording will usually differ; that does not matter. What matters is whether a person reading both would say "that's the same issue, further along".

If both carry an animal id and the ids differ, they are different animals — answer "new".

When it is genuinely unclear, answer "new".

That default matters: a duplicate entry is mildly untidy and easy to clean up later. Wrongly merging two different problems makes a real, separate problem look like something already being tracked — so it gets quietly ignored. The costs are not symmetric.

Return strict JSON only."""

MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["update", "new"]},
        "updates_entry_id": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["decision"],
}


def build_match_prompt(new_entry, candidates):
    def _aid(meta):
        v = (meta or {}).get("animal_id")
        return f"  animal id: {v}\n" if v else ""

    cands = "\n\n".join(
        f"EXISTING ENTRY id={c['id']}\n"
        f"  status: {c.get('status')} | raised {c.get('times_raised')} time(s) | "
        f"first came up: {dates.fmt(c.get('first_source_ts'))}\n"
        + _aid(c.get("metadata"))
        + f"  headline: {c.get('headline')}\n"
        f"  detail:   {(c.get('expanded') or '')[:400]}"
        for c in candidates
    )
    return f"""NEW ENTRY
  status:   {new_entry.get('status')}
{_aid_line(new_entry)}  headline: {new_entry['headline']}
  detail:   {(new_entry.get('expanded') or '')[:400]}

EXISTING ENTRIES OF THE SAME KIND FOR THIS FARMER:
{cands}
"""


def _aid_line(entry):
    v = (entry.get("metadata") or {}).get("animal_id")
    return f"  animal id: {v}\n" if v else ""


# --------------------------------------------------------------------------
# Layer 2 — the two sentences that get injected into EVERY future turn
# --------------------------------------------------------------------------
# Written by its own call, not as a side effect of extraction. This is the
# highest-stakes text in the system: it is shown on every turn whether it is
# relevant or not, so it earns a focused prompt of its own. It also has to be
# rewritten after a merge, when the entry now covers more than it did.

HEADLINE_SYSTEM = f"""You write the one-line memory that Amul's assistant sees at the start of every future conversation with this farmer.

You are given the full detail of one remembered situation. Write the short version.

RULES:
- At most 2 sentences. Shorter is better. It is shown on every turn, relevant or not, so every wasted word costs.
- Self-contained. It will be read months later with nothing else around it.
- Third person, about the farmer, plain language.
- Lead with the thing that would change how the assistant responds. If something is unresolved, that goes first.
- If the situation has been raised several times or has run over weeks, say so — that is often the most useful part.
- State what is unconfirmed as unconfirmed. Never turn "no visit was booked in the chat" into "no visit happened".
- No numbers that will go stale (today's milk, this month's payment). Durable figures the farmer themselves gave are fine.
{_ANIMAL_ID_RULE}

Return strict JSON only."""

HEADLINE_SCHEMA = {
    "type": "object",
    "properties": {"headline": {"type": "string"}},
    "required": ["headline"],
}


def build_headline_prompt(entry, first_ts=None, last_ts=None, times_raised=None):
    hist = []
    if first_ts:
        hist.append(f"first came up {dates.fmt(first_ts)}")
    if last_ts and dates.fmt(last_ts) != dates.fmt(first_ts):
        hist.append(f"last mentioned {dates.fmt(last_ts)}")
    if times_raised and times_raised > 1:
        hist.append(f"raised {times_raised} times")
    return f"""SITUATION
  status:  {entry.get('status')}
  history: {', '.join(hist) or 'first time it has come up'}

FULL DETAIL:
{entry.get('expanded') or entry.get('headline')}

CURRENT SHORT VERSION (rewrite it; keep it if it is already right):
{entry.get('headline')}
"""


# --------------------------------------------------------------------------
# Layer 1 — the standing record. Fixed schema, same fields for every farmer.
# --------------------------------------------------------------------------
# Injected on every turn with no search at all, so it must stay small and must
# contain only things that are true across conversations rather than inside one.
#
# Deliberately excluded: village, district, union, herd counts, which accounts are
# linked, milk figures. All of those arrive from Amul's own systems on every turn
# via farmer_info. Copying them here would only produce a stale second version.

PROFILE_FIELDS = {name: f"{spec['description']} Maximum {spec['max_chars']} characters."
                  for name, spec in profile_schema.FIELDS.items()}

PROFILE_SYSTEM = f"""You maintain the standing record for one dairy farmer — the handful of facts Amul's assistant is told at the start of EVERY conversation with them.

You are given the record as it stands, plus what is currently remembered about this farmer. Return the record as it should now read.

THE FIELDS ARE FIXED. These are the only ones, for every farmer:
{chr(10).join(f'- "{k}": {v}' for k, v in PROFILE_FIELDS.items())}

RULES:
- Leave a field OUT entirely unless you have clear evidence for it. An empty field costs nothing; a guessed one is injected into every future conversation. Most farmers will have only two or three fields.
- Keep each field to a short phrase. This whole record is read on every turn.
- All populated values together must fit {profile_schema.MAX_TOTAL_CHARS} characters.
- Only put in things that hold ACROSS conversations. A problem from one chat belongs in the detailed memory, not here — it is already stored there.
- Never copy anything the assistant looks up live anyway: village, district, union, how many animals they have, which accounts are linked, milk quantities, payments. Those are fetched fresh every turn.
- Prefer what is already in the record. Change a field only when the new evidence actually contradicts it or clearly sharpens it. This record should be stable — it is not a running log.
- Never state a preference from a single instance. "Asked one question in English" is not a language preference.

{_NEVER_STORE}

Return strict JSON only, as {{"fields": {{...}}}}, containing only the fields you have evidence for."""

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in PROFILE_FIELDS},
            "additionalProperties": False,
        },
        "changed": {"type": "boolean"},
    },
    "required": ["fields"],
}


def build_profile_prompt(current, headlines, farmer_label, recent_asks=()):
    """Current standing record + the Layer-2 headlines + a sample of the farmer's words.

    Headlines rather than full detail on purpose: Layer 1 is about what holds across
    conversations, and the headlines are already the cross-conversation view. Feeding
    the full expanded text in would both blow up the prompt and tempt the model into
    copying one situation's specifics into the always-injected block.

    But the headlines alone are not enough. Three of the seven fields — what the
    farmer is called, how they want to be spoken to, and the words they use for their
    own animals — exist only in their own phrasing, which the headlines have already
    paraphrased away. So a bounded sample of what the farmer actually typed goes in
    too: their side only, since the assistant's replies say nothing about the farmer.
    """
    cur = (
        "\n".join(f'- {k}: {v}' for k, v in current.items())
        if current else "(empty — nothing recorded yet)"
    )
    mem = (
        "\n".join(
            f"- [{h.get('status')}] {h.get('headline')}" for h in headlines
        )
        if headlines else "(nothing remembered yet)"
    )
    asks = (
        "\n".join(f"- {a}" for a in recent_asks)
        if recent_asks else "(none available)"
    )
    return f"""Farmer: {farmer_label}

STANDING RECORD AS IT STANDS:
{cur}

WHAT IS CURRENTLY REMEMBERED ABOUT THIS FARMER:
{mem}

SOME THINGS THIS FARMER ACTUALLY SAID, IN THEIR OWN WORDS.
Use these only as evidence for how they speak, what they call themselves, and the
words they use for their animals. Do NOT turn the content of these questions into
standing facts — one question is not a pattern:
{asks}
"""


# --------------------------------------------------------------------------
# Consolidation — the maximal pass, run at dream time
# --------------------------------------------------------------------------
# Everything below runs nightly with NO latency budget, so it is deliberately
# thorough where the per-chunk match is cheap. That division is the point: the
# per-chunk match is allowed to over-split (it defaults to "new" when unsure,
# because a wrong merge during a chunk is invisible and permanent), and this pass
# then looks at the farmer's WHOLE store at once and puts the pieces back together.
#
# Over-splitting is only safe because this pass exists. Without it, "default to new"
# produced six separate entries for one grievance, all with times_raised=1 — which
# makes "he has raised this repeatedly" undetectable, the single most valuable thing
# memory could have told us (log-findings.md Cases 2 and 7).

CLUSTER_SYSTEM = """You are consolidating one dairy farmer's memory. You see EVERY memory currently held about them, numbered.

Your job: find the entries that are really the SAME underlying situation, recorded more than once.

This happens constantly, because entries are written conversation by conversation. The same grievance gets written down again each time the farmer raises it, in different words, sometimes filed under a different category — a complaint about the society not registering calves and a tracker for the calf-rearing subsidy he cannot get are usually ONE situation, not two.

WHAT MAKES ENTRIES THE SAME SITUATION:
- The same underlying problem, even if the wording, the category, or the emphasis differs.
- The same application, scheme, or request, at different stages.
- The same animal with the same condition, progressing over time.
- One entry being a more specific version of another (a general complaint plus a specific instance of it).

WHAT MAKES THEM DIFFERENT, AND MUST NOT BE MERGED:
- Different animals. If two entries carry different animal ids, they are different situations, full stop, however similar the words.
- The same KIND of problem at genuinely separate times, resolved in between.
- Two different schemes or applications, even if the frustration is identical.
- A general standing fact about the farmer and a specific incident.

Group by SITUATION, not by topic. "Anything to do with money" is not a situation. "The deduction he has asked about four times" is.

Return every group of 2 or more. Entries that stand alone need not be mentioned. Be decisive: you are looking at the whole picture at once, which is exactly the view the per-conversation step never had.

Return strict JSON only."""

CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "members": {"type": "array", "items": {"type": "integer"}},
                    "situation": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["members", "situation", "confidence"],
            },
        },
    },
    "required": ["groups"],
}


def build_cluster_prompt(entries, farmer_label):
    """Every current memory for one farmer, numbered from 1."""
    lines = []
    for i, e in enumerate(entries, 1):
        aid = (e.get("metadata") or {}).get("animal_id")
        bits = [f"{i}. [{e.get('status')}]"]
        if aid:
            bits.append(f"(animal {aid})")
        bits.append(f"({dates.span(e.get('first_source_ts'), e.get('source_ts'))}, "
                    f"raised in {e.get('times_raised') or 1} conversation(s))")
        lines.append(" ".join(bits) + f"\n   {e.get('headline')}")
        exp = (e.get("expanded") or "").strip()
        if exp:
            lines.append(f"   detail: {exp[:300]}")
    return f"""Farmer: {farmer_label}
All {len(entries)} memories currently held about them:

""" + "\n".join(lines) + "\n"


MERGE_SYSTEM = f"""You are merging several memory entries that were found to be the SAME situation, recorded separately over time, into one.

Write the single entry that should replace them. Nothing the farmer told us may be lost — you are combining, not summarising away.

RULES:
- "headline": 1-2 self-contained sentences. Several fragments can come from ONE conversation. Only describe repetition across conversations when the supplied distinct-conversation count is greater than one; never infer repetition from fragment count.
- "expanded": everything from all the fragments, in time order, deduplicated. Keep specifics: figures the farmer gave, what was advised, who they were told to contact, what they had already tried. If fragments disagree, keep both and say so rather than picking one.
- "status": the state it is in NOW, per the most recent fragment. If any part is still unresolved, it is not resolved.
- Never state as settled anything that is only unconfirmed. Never invent a resolution the fragments do not show.
- "metadata": the union of the fragments' metadata. Keep animal_id if any fragment has it.

{_ANIMAL_ID_RULE}

Return strict JSON only."""

MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "expanded": {"type": "string"},
        "status": {"type": "string", "enum": ["open", "pending", "resolved", "n/a"]},
        "metadata": {"type": "object", "additionalProperties": True},
    },
    "required": ["headline", "expanded", "status"],
}


def build_merge_prompt(fragments, situation):
    parts = []
    for i, e in enumerate(sorted(fragments, key=lambda x: str(x.get("source_ts") or "")), 1):
        parts.append(
            f"FRAGMENT {i} [{e.get('status')}] "
            f"({dates.fmt(e.get('source_ts'))}, raised in {e.get('times_raised') or 1} conversation(s))\n"
            f"  headline: {e.get('headline')}\n"
            f"  detail:   {e.get('expanded') or '(none)'}\n"
            f"  metadata: {json.dumps(e.get('metadata') or {}, ensure_ascii=False)}"
        )
    sessions = {sid for entry in fragments for sid in (
        (entry.get("source_sessions") or []) + [entry.get("source_session_id")]) if sid}
    return f"""SITUATION: {situation}
DISTINCT SUPPORTING CONVERSATIONS: {len(sessions)} (do not infer extra occasions)

{chr(10).join(parts)}
"""


# --------------------------------------------------------------------------
# Reranking candidates for the per-chunk match
# --------------------------------------------------------------------------
# Semantic search brings back candidates ordered by topical closeness, which on one
# farmer's memory is nearly uninformative — measured, everything in one farmer's
# store scores 0.74-0.88 regardless of whether it is the same situation. So the
# ordering is redone by a model that can read the text, over a WIDE candidate set
# rather than the top 3.

RERANK_SYSTEM = """You are given a newly extracted memory about a dairy farmer, and a list of memories already held about them.

Rank the existing memories by how likely each is to be THE SAME underlying situation as the new one — not by how similar the topic is. Two entries about money are not the same situation; two entries about the same unexplained deduction are.

Put the most likely first. Include only entries that are plausibly the same situation; leave out the rest entirely rather than padding the list. Returning an empty list is a valid and common answer.

If both carry an animal id and they differ, it is a different animal and must not be listed.

Return strict JSON only."""

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["ranked"],
}


def build_rerank_prompt(new_entry, candidates):
    lines = []
    for i, c in enumerate(candidates, 1):
        aid = (c.get("metadata") or {}).get("animal_id")
        lines.append(
            f"{i}. [{c.get('status')}]"
            + (f" (animal {aid})" if aid else "")
            + f" (raised in {c.get('times_raised') or 1} conversation(s), "
              f"first {dates.fmt(c.get('first_source_ts'))})\n"
            f"   {c.get('headline')}\n"
            f"   detail: {(c.get('expanded') or '')[:200]}"
        )
    aid_new = (new_entry.get("metadata") or {}).get("animal_id")
    return f"""NEW MEMORY
  status: {new_entry.get('status')}
""" + (f"  animal: {aid_new}\n" if aid_new else "") + f"""  headline: {new_entry.get('headline')}
  detail:   {(new_entry.get('expanded') or '')[:400]}

ALREADY HELD ABOUT THIS FARMER:
{chr(10).join(lines)}
"""


# --------------------------------------------------------------------------
# Describing a long-running episode's contents
# --------------------------------------------------------------------------
# Level 2 describes the bounded, lossless Level 3 chunks; it never rewrites them.
# --------------------------------------------------------------------------
if __package__:
    from service import episode
else:
    import episode

CONTENTS_SYSTEM = """Create a compact Level 2 summary and contents page for one remembered episode.
Keep the supplied detail intact: you are describing it, not replacing it.
Return one description for each supplied chunk group, in exactly that order.
Use specific supported keywords, treatments/schemes, actions and reported outcomes.
Do not invent facts, identifiers, category labels or certainty. Distinguish advice
and offers from confirmed actions. No extra keys."""


def contents_schema(groups):
    return {
        "type": "object", "properties": {
            "headline": {"type": "string", "minLength": 1,
                         "maxLength": episode.SUMMARY_MAX_CHARS},
            "descriptions": {"type": "array", "minItems": len(groups), "maxItems": len(groups),
                             "items": {"type": "string", "minLength": 1,
                                       "maxLength": episode.CONTENTS_ROW_MAX_CHARS}}},
        "required": ["headline", "descriptions"], "additionalProperties": False,
    }


def build_contents_prompt(entry, chunks, groups):
    text = entry.get("expanded") or entry["headline"]
    return (f"Summary limit: {episode.SUMMARY_MAX_CHARS} characters. "
            f"Exactly {len(groups)} descriptions, each at most {episode.CONTENTS_ROW_MAX_CHARS} characters.\n"
            f"Proposed summary: {entry['headline']}\n\n" + "\n\n".join(
                f"GROUP {i + 1}, chunks {first}-{last}:\n"
                + text[chunks[first - 1]["start"]:chunks[last - 1]["end"]]
                for i, (first, last) in enumerate(groups)))
