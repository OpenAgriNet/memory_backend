"""The dreaming agent: decides for itself how to bring a farmer's memory up to date.

This replaces a fixed extract -> match -> write pipeline. The pipeline worked, but it
could only ever do what it was told in the order it was told: one search, one
comparison, one decision. An agent can look, look again with different words, check
whether something is already tracked under different wording, and merge two
entries it discovers are the same thing — which is exactly the work that the fixed
pipeline got wrong, producing six live entries for one grievance.

It runs nightly, off any live turn, so it is allowed to take its time.

PARALLELISM, IN TWO PLACES

1. **Tool calls.** The model emits several tool calls in one message when they are
   independent (verified on this vLLM build), and read-only tools are executed
   concurrently. Three searches with different wording cost one round trip, not
   three, which is what makes "look properly before writing" affordable.
2. **Judges.** Every write is reviewed by three independent judges — sensitive
   content, key vocabulary, honesty about what is known — and those run concurrently
   inside the write tool. See judges.py.

Writes are deliberately NOT parallelised. They mutate the same farmer's store, and
two concurrent writes about the same situation would each search, each find nothing,
and each write — recreating the duplication this is meant to fix.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import sys

if __package__:
    from . import llm, prompts, tools
else:
    import llm
    import prompts
    import tools

MAX_STEPS = int(__import__("os").environ.get("DREAMER_AGENT_STEPS", "8"))

SYSTEM = """You maintain the long-term memory of Sarlaben, Amul's assistant for dairy farmers in Gujarat.

You are given trusted conversation turns and tools to read and write what is
remembered about the farmer. Bring their memory up to date.

HOW TO WORK

Look before you write. Search what is already remembered — more than once, with different wording — before deciding anything is new. You can issue several searches in a single step, and you should: the same situation is often recorded under wording you would not have guessed, and under a different description than you expect.

Before rewriting an existing account, use read_memory to inspect its detail, following the returned continuation when present. Search previews may be incomplete.

Search results can include record_kind=episode summaries. They are conversation
history, not stable farmer facts. Never update or fold them with the ordinary memory
tools; write_episode_summary is the only tool that may revise the current episode.

For every ordinary memory write, update, fold, or key documentation, provide
source_turns: the numbers of only the turns supporting that mutation. At least one
must be marked NEW EVIDENCE;
EARLIER CONTEXT can explain it but cannot independently cause another mutation.
Dates, source links, and session counts are derived by code. Never attach unrelated
turns.

Then decide, for each thing worth remembering:
- Already on record and this conversation moved it along -> update_memory. This is the case you should be actively looking for.
- Already on record twice under different entries -> first update one to preserve the combined detail, then fold the duplicate into the returned new entry id.
- Genuinely not on record -> write_memory.
- Not worth remembering -> leave it out.

[USEFULNESS_RULE]

Use a general summary and detail. Do not assign category labels.
When there is no useful supported memory to add or update, call nothing_to_remember
with a specific reason and stop.

BE HONEST ABOUT WHAT IS NOT KNOWN. If a visit was offered but never confirmed, write "no booking was confirmed in this conversation" — never "no visit happened", because the farmer may have arranged one elsewhere. A memory that overstates what is known will make the assistant confirm something false months later. Writes that do this are refused.

ANIMAL IDENTIFIERS: an ear tag may be stored in metadata under animal_id, so the same animal can be recognised later with certainty. It must NEVER appear in the headline or detail text, because that text is read back to the farmer. Describe the animal the way the farmer does — "her cow", "the one that calved in June".

METADATA KEYS: check known_keys before inventing one, and reuse an existing key if it fits even if you would have phrased it differently. Two keys meaning the same thing make both useless for filtering. If you do invent one, document_key it.

When you are done, say in one short line what you did. Do not call any more tools after that.""".replace("[USEFULNESS_RULE]", prompts.USEFULNESS_RULE)

EPISODE_SYSTEM = """EPISODE MODE

These turns are one numbered conversation episode, or one bounded part/revision of
it. Call write_episode_summary exactly once. It records what this conversation
discussed, not a standing fact. If an earlier episode summary is shown, rewrite the
complete summary using it plus the new evidence; do not discard earlier parts. The
program owns episode identity and provenance.

When no stable fact or ongoing matter should be changed, still write the episode
summary, then call nothing_to_remember with a specific reason."""


def _run_tools(ctx, calls):
    """Execute a batch of tool calls, read-only ones concurrently.

    Writes stay serial and in the order the model asked for them: they mutate the
    same store, and a write depends on what the writes before it did.
    """
    results = [None] * len(calls)
    reads = [(i, c) for i, c in enumerate(calls) if c["name"] in tools.READ_ONLY]
    writes = [(i, c) for i, c in enumerate(calls) if c["name"] not in tools.READ_ONLY]

    def call(c):
        fn = tools.IMPL.get(c["name"])
        if not fn:
            ctx.tool_failures.append("unknown_tool")
            return {"error": "unknown_tool"}
        try:
            if c["name"] not in tools.READ_ONLY and c["name"] != "nothing_to_remember":
                ctx.start_mutation()
            result = fn(ctx, **c["args"])
            if isinstance(result, dict) and result.get("error"):
                ctx.tool_failures.append("tool_returned_error")
                if c["name"] not in tools.READ_ONLY and c["name"] != "nothing_to_remember":
                    ctx.uncertain_mutation = True
            return result
        except TypeError:
            ctx.tool_failures.append("invalid_tool_arguments")
            if c["name"] not in tools.READ_ONLY and c["name"] != "nothing_to_remember" and ctx._mutation_started:
                ctx.uncertain_mutation = True
            return {"error": "invalid_tool_arguments"}
        except Exception:  # noqa: BLE001
            ctx.tool_failures.append("tool_exception")
            if c["name"] not in tools.READ_ONLY and c["name"] != "nothing_to_remember" and ctx._mutation_started:
                ctx.uncertain_mutation = True
            return {"error": "tool_exception"}

    if reads:
        with ThreadPoolExecutor(max_workers=min(len(reads), 6)) as pool:
            for (i, _), res in zip(reads, pool.map(lambda ic: call(ic[1]), reads)):
                results[i] = res
    for i, c in writes:
        results[i] = call(c)
    return results


def dream(api, farmer_id, turns, *, source_ts=None, session_id=None,
          known_keys=(), dry_run=False, verbose=True, trace_ids=None,
          before_mutation=None, pending_trace_ids=None, episode=None,
          previous_episode_summary=None):
    """Run the agent over one chunk of turns. Returns the Ctx, which records
    everything written and everything the judges refused."""
    ctx = tools.Ctx(api, farmer_id, turns=turns, source_ts=source_ts,
                    session_id=session_id, known_keys=known_keys, dry_run=dry_run,
                    trace_ids=trace_ids, before_mutation=before_mutation,
                    pending_trace_ids=pending_trace_ids, episode=episode)

    convo = "\n\n".join(
        f"Turn {n} [{'NEW EVIDENCE' if t.get('pending') else 'EARLIER CONTEXT'}] "
        f"[{(t.get('ts') or '')[:10]}] FARMER: {t['query']}\n"
        f"ASSISTANT: {t.get('answer') or ''}"
        for n, t in enumerate(turns, 1) if (t.get("query") or "").strip()
    )
    keys = "\n".join(f"- {k['key']}: {k.get('description') or '(undocumented)'}"
                     for k in ctx.known_keys) or "(none yet)"
    previous = "(none; this is the first part of the episode)"
    if previous_episode_summary:
        previous = (
            f"Headline: {previous_episode_summary.get('headline') or ''}\n"
            f"Detail: {previous_episode_summary.get('expanded') or ''}"
        )
    schemas = tools.SCHEMAS if episode else [
        schema for schema in tools.SCHEMAS
        if schema["function"]["name"] != "write_episode_summary"
    ]
    messages = [
        {"role": "system", "content": SYSTEM + ("\n\n" + EPISODE_SYSTEM if episode else "")},
        {"role": "user", "content": f"""Farmer: ...{str(farmer_id)[-4:]}

METADATA KEYS ALREADY IN USE:
{keys}

PREVIOUS EPISODE SUMMARY:
{previous}

TURNS:
{convo}
"""},
    ]

    for step in range(MAX_STEPS):
        try:
            msg = llm.chat(messages, tools=schemas, max_tokens=1500)
        except Exception as exc:  # noqa: BLE001
            ctx.error_code = "agent_model_failed"
            if verbose:
                print(f"  ! agent step failed ({exc}); stopping this chunk", file=sys.stderr)
            break

        raw_calls = getattr(msg, "tool_calls", None) or []
        if not raw_calls:
            if not ctx.writes and not ctx.blocked:
                ctx.error_code = "agent_stopped_without_outcome"
            if verbose and (msg.content or "").strip():
                print(f"      agent: {(msg.content or '').strip()[:160]}")
            break

        calls = []
        for tc in raw_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "args": args})

        if verbose:
            n_reads = sum(1 for c in calls if c["name"] in tools.READ_ONLY)
            names = ", ".join(c["name"] for c in calls)
            # Only claim parallelism when it actually happened: reads fan out, writes
            # stay serial because they mutate the same store.
            how = ""
            if n_reads > 1:
                how = f"  ({n_reads} reads in parallel)"
            elif len(calls) > 1:
                how = "  (writes run in order)"
            print(f"      step {step + 1}: {len(calls)} call(s) [{names}]{how}")

        results = _run_tools(ctx, calls)

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"],
                                         "arguments": json.dumps(c["args"])}} for c in calls],
        })
        for c, res in zip(calls, results):
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(res, ensure_ascii=False, default=str)})
            if verbose:
                for note in (res or {}).get("notes") or []:
                    print(f"        note: {note}")
                if (res or {}).get("refused"):
                    for why in res.get("reasons") or []:
                        print(f"        REFUSED: {why[:130]}")
    else:
        ctx.error_code = "agent_step_budget_exhausted"
        if verbose:
            print(f"      (hit the {MAX_STEPS}-step budget)")

    if ctx.tool_failures:
        ctx.error_code = "agent_tool_failure"
    return ctx
