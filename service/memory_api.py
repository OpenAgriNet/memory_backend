"""
Amul memory API — a thin testing/demo service over Qdrant.

Implements the memory design from
oan_horizon/future_work/memory_amul/memory-overview.md Section 6:

- One collection, one point per memory entry.
- Two named vectors per point: "headline" (short, searched every turn) and
  "expanded" (fuller detail, searched only via a deliberate tool call).
- Structured payload fields (type/status/times_raised/recorded_at/superseded_at) so
  "does this farmer have anything still open" is a direct filter, not an
  interpretation of free text.
- History kept by never overwriting: an update stamps the old entry with
  superseded_at and writes a new one.

Three different timelines, kept deliberately separate because conflating them is the
easiest way to get this wrong:
  * recorded_at / superseded_at  — RECORD time: when we wrote or replaced a record.
    Bookkeeping only. Never shown to the agent, never used for date filtering; after
    a backfill every entry shares the same recorded_at, so it says nothing about when
    anything happened.
  * source_ts / first_source_ts  — EVENT time: when the farmer actually said it, and
    when the thread first came up. This is the only thing date filters use.
  * expires_at                   — a known, actionable end date where the
    conversation genuinely gave one (a milk withdrawal period, a scheme deadline).
    SHOWN to the model, never used to hide a memory from it — see _farmer_filter for
    why no date decides visibility anywhere in this service.
- farmer_id is enforced in code on EVERY read path, never passed in a filter the
  caller controls — it is structurally impossible for a query to reach another
  farmer's memories.
- Account, transaction and phone numbers are never stored (principle 8): callers
  pass descriptions. An animal's ear tag is the one exception — it goes in
  metadata.animal_id so the same animal can be matched with certainty, and is
  stripped from the agent-facing key list so it can never reach a conversation.

Scope: this service creates and updates memory in Qdrant. That is all it does. It
does no ranking beyond Qdrant's own vector search, no reasoning, and no scheduling.
"""
import os
import hmac
import json
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
import profile_schema
import episode
import checkpoints
from runtime_config import required, qdrant_headers

QDRANT_URL = required("QDRANT_URL").rstrip("/")
COLLECTION = os.getenv("QDRANT_COLLECTION", "amul_memory")
MEMORY_INTERNAL_TOKEN = required("MEMORY_INTERNAL_TOKEN")

# ---------- embeddings ----------
# Qdrant searches vectors but does not create them, so one external call is
# unavoidable: text has to become numbers on the way in AND on the way out, with the
# same model both times or the numbers aren't comparable.
#
# Today that call goes to the Marqo instance Amul already runs, using the same model
# as Amul's knowledge base (multilingual-e5-large) — memory text is frequently
# Gujarati, so an English-first model would be a poor fit, and reusing what is
# already running means no second model to keep in sync.
#
# PLANNED: move to a dedicated, independently scalable embedding service. Marqo is a
# search engine being used purely as an embedding endpoint, so it scales on its own
# axis rather than ours. _embed_many below is the ONLY place text becomes vectors —
# that one function is the whole migration surface. Whatever replaces it must return
# EMBED_DIM-sized vectors from the same model family, or the collection has to be
# re-embedded (see the startup check).
MARQO_URL = required("MARQO_URL").rstrip("/")
MARQO_INDEX = os.getenv("MARQO_INDEX", "amul-veterinary-index")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
CHECKPOINT_LEASE_SECONDS = int(os.getenv("MEMORY_CHECKPOINT_LEASE_SECONDS", "3600"))
if CHECKPOINT_LEASE_SECONDS <= 0:
    raise RuntimeError("MEMORY_CHECKPOINT_LEASE_SECONDS must be positive")

# Stamped onto every entry, so a future migration is detectable instead of silent.
# The dangerous failure is not an outage — it is swapping the embedding service,
# getting vectors back that are the right SHAPE but a different meaning, and having
# every search quietly return nonsense with no error anywhere.
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", f"marqo:{MARQO_INDEX}/doc-query")

# Internal record separation only. Memories have no category requirement. Older
# records with category values remain readable and become plain memories on update.
MEMORY_TYPE = "memory"
CHUNK_TYPE = "memory_chunk"
WORKER_CHECKPOINT_TYPE = "worker_checkpoint"
INTERNAL_TYPES = ("settings", "profile_record", "key_doc", CHUNK_TYPE, WORKER_CHECKPOINT_TYPE)
EntryStatus = Literal["open", "resolved", "pending", "n/a"]

# A farmer's memory-on/off flag is stored here, in Qdrant, alongside their own
# memory — one source of truth, inspectable and listable in the same place as the
# data it governs, rather than a separate opaque flag store elsewhere.
SETTINGS_TYPE = "settings"
# Documentation of what a model-invented metadata key means. Also a placeholder-vector
# type, so it is excluded from search and from the embed-model check.
KEY_DOC_TYPE = "key_doc"
# The Layer-1 standing record is a singleton, excluded from memory search/listing.
PROFILE_TYPE = "profile_record"
# Metadata keys the agent must never be shown, even for the farmer's own data.
# An animal's ear tag is stored on purpose — it is what makes "the same animal as
# last time" a certain answer rather than a guess — but it is bookkeeping for the
# dreamer's matching step, not something to read back to a farmer. Enforced here, at
# the one endpoint that hands key names and values to a conversation, rather than
# left to a prompt instruction that a model may forget.
AGENT_HIDDEN_KEYS = {"animal_id", "account_no", "account_number", "transaction_id", "phone"}
# Reply use requires the bot's global switch. Per-farmer settings default on for
# now; an explicit off is respected. Background learning never changes settings.
DEFAULT_MEMORY_ENABLED = True

app = FastAPI(title="Amul memory API (test harness)")


@app.middleware("http")
async def _authenticate(request: Request, call_next):
    """Secure every current and future route except the public liveness check."""
    if request.method == "GET" and request.url.path == "/health":
        return await call_next(request)
    authorization = request.headers.get("Authorization", "")
    scheme, separator, supplied = authorization.partition(" ")
    valid = (
        bool(separator)
        and scheme.lower() == "bearer"
        and bool(supplied)
        and " " not in supplied
        and hmac.compare_digest(supplied, MEMORY_INTERNAL_TOKEN)
    )
    if not valid:
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


# ---------- models ----------

class NewEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    headline: str = Field(min_length=1, max_length=episode.SUMMARY_MAX_CHARS, description="Level 2 episode summary.")
    expanded: str = Field(default="", description="Fuller detail. Searched only on a tool call.")
    contents: list[dict[str, Any]] | None = None
    chunk_size: int = episode.CHUNK_MAX_CHARS
    status: EntryStatus = "n/a"
    times_raised: int = 1
    source_session_id: str | None = Field(
        default=None, description="Pointer back to the source conversation. Not a copy of it."
    )
    source_sessions: list[str] | None = Field(
        default=None,
        description="Every conversation this entry draws on. times_raised is derived from "
                    "the number of DISTINCT entries here, so consolidation passes the union "
                    "of its fragments' sessions.",
    )
    source_traces: list[str] | None = Field(
        default=None,
        description="The individual turns (Langfuse trace ids) behind this entry. The unit "
                    "of idempotency: a trace already recorded against an entry is never "
                    "applied to it twice, however often the job is re-run.",
    )
    source_ts: str | None = Field(
        default=None, description="When the farmer actually said this (event time), shown for context."
    )
    first_source_ts: str | None = Field(
        default=None, description="When this thread first came up. Preserved across updates."
    )
    expires_at: str | None = Field(
        default=None,
        description=(
            "Known-in-advance expiry (ISO timestamp). The one real-world end date we model. "
            "Distinct from superseded_at, which is only when a record was replaced. "
            "'this was always going to lapse on this date' — e.g. a milk withdrawal period "
            "after a treatment, a scheme window, a vaccination deadline. Shown for context; "
            "expiry does not remove the memory from current results."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional farmer-specific detail keys. Everything the model wants to record beyond the "
            "system fields goes in here, nested — never at the top level, so model-invented "
            "keys can never collide with or overwrite a field the system filters on "
            "(farmer_id / type / status / superseded_at / expires_at). Still queryable later via "
            "dotted paths (metadata.some_key) if a key turns out to be broadly useful."
        ),
    )


class SearchRequest(BaseModel):
    memory_ref: uuid.UUID | None = None
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    status: EntryStatus | None = None
    only_current: bool = True
    # No date parameters, deliberately — see _farmer_filter. Dates are shown to the
    # model, never used to hide a memory from it.
    metadata_key: str | None = Field(default=None, description="Filter on a metadata key, e.g. 'severity'.")
    metadata_value: Any | None = Field(default=None, description="Value that key must equal.")
    filters: dict[str, Any] = Field(default_factory=dict, description="Farmer-specific metadata key/value matches (AND).")


class UpdateEntry(BaseModel):
    merge_from: list[uuid.UUID] | None = Field(default=None, max_length=50)
    contents: list[dict[str, Any]] | None = None
    chunk_size: int = episode.CHUNK_MAX_CHARS
    status: EntryStatus | None = None
    times_raised: int | None = None
    headline: str | None = None
    expanded: str | None = None
    expires_at: str | None = None
    metadata: dict[str, Any] | None = None
    # Event time of the conversation that caused this update. Advancing it is what
    # makes "last mentioned" mean anything; first_source_ts keeps the start, so the
    # span of a long-running issue stays visible.
    source_ts: str | None = None
    first_source_ts: str | None = None
    source_session_id: str | None = None
    source_sessions: list[str] | None = Field(
        default=None,
        description="Full set of conversations this entry draws on. Used by consolidation "
                    "to pass the union of the fragments' sessions.",
    )
    source_traces: list[str] | None = Field(
        default=None,
        description="The turns behind this update. If every one is already recorded on "
                    "the entry, the update is rejected as already applied.",
    )



# ---------- helpers ----------

def _norm_text(text: str) -> str:
    """Loose normalisation, for spotting an entry we already hold."""
    return " ".join("".join(
        c if c.isalnum() or c.isspace() else " " for c in (text or "").lower()).split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _embed_many(texts: list[str], *, kind: str = "document") -> list[list[float]]:
    """Embed a batch of strings. The ONLY place text becomes vectors.

    `kind` is "document" for text being stored and "query" for text being searched
    with, and it is not cosmetic. multilingual-e5-large is trained asymmetrically —
    a stored passage and a question about it are meant to be encoded differently —
    and Marqo exposes that through content_type. Measured on a labelled set of real
    memory headlines and real farmer questions, using it lifted "did the search
    return the RIGHT memory" from 9/10 to 10/10.

    Note that "query" is Marqo's default: plain text embeds identically to
    content_type=query. So the thing that actually had to change was the write side,
    which was storing memories encoded as though they were questions.

    Both sides must agree forever. Changing this convention makes every stored
    vector incomparable with new queries, which is why it is part of EMBED_MODEL_ID
    and why the startup check watches for it.

    Batched rather than one-at-a-time because a write needs two vectors (headline and
    expanded) and a backfill needs thousands — one round trip per entry instead of
    two, and a dedicated embedding service is exactly the kind of thing that batches
    well. Order in equals order out.

    This function is the entire migration surface: swapping Marqo for a dedicated
    embedding service means changing this call and nothing else.
    """
    payload = [t if (t or "").strip() else " " for t in texts]
    if not payload:
        return []

    if kind not in ("document", "query"):
        raise HTTPException(500, f"bad embed kind {kind!r}")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{MARQO_URL}/indexes/{MARQO_INDEX}/embed",
                         json={"content": payload, "content_type": kind})
    if r.status_code != 200:
        raise HTTPException(502, f"embedding failed: {r.text[:200]}")
    vectors = r.json()["embeddings"]

    if len(vectors) != len(payload):
        raise HTTPException(
            502, f"embedding returned {len(vectors)} vectors for {len(payload)} inputs")
    bad = next((len(v) for v in vectors if len(v) != EMBED_DIM), None)
    if bad is not None:
        # Fail loudly rather than writing vectors the collection cannot hold — or,
        # worse, ones it can hold but that mean something different.
        raise HTTPException(
            502,
            f"embedding model returned {bad}-dim vectors but the collection is "
            f"{EMBED_DIM}-dim",
        )
    return vectors


async def _embed(text: str, *, kind: str = "document") -> list[float]:
    return (await _embed_many([text], kind=kind))[0]


def _farmer_filter(
    farmer_id: str,
    *,
    status: str | list[str] | None = None,
    only_current: bool = True,
    metadata_key: str | None = None,
    metadata_value: Any | None = None,
    filters: dict[str, Any] | None = None,
    include_chunks: bool = False,
) -> dict:
    """Build a Qdrant filter. farmer_id is ALWAYS applied here, in code — callers
    never supply it as part of a query they control.

    **No date is ever compared to "now" to decide whether a memory is visible.**
    Decided 2026-09-08, after getting this wrong three separate times on three
    different fields:

      - `since`/`until` range filters were built speculatively and never used by
        anything. Dead code on this axis.
      - `valid_from`/`valid_to` were record time wearing the name of real-world
        validity, and had to be renamed `recorded_at`/`superseded_at`.
      - `expires_at` was folded into `only_current`, which silently hid an OPEN
        booking whose scheme deadline had passed — the case most worth surfacing.

    Three failures on one axis is evidence the abstraction is wrong, not that the
    implementations were unlucky. The reasons it keeps failing:

      - A date cannot tell you whether something is resolved. `status` can.
      - `expires_at` is inferred from a conversation by a model, so it is a guess
        wearing the costume of a precise fact.
      - **Filtering is silent and destructive.** An entry removed by a filter
        produces no signal at all, so the assistant cannot reason about what it never
        saw. Showing the date instead lets it reason out loud — "that withdrawal
        period ended last week" — and say so to the farmer.

    What remains is `superseded_at`, and that is deliberately NOT a date comparison:
    it is a version-state check (set or unset), asking "has this entry been replaced
    by a newer one". Nothing is compared against the clock.

    Dates are still stored, still indexed, and still shown to the model. `GET /due`
    still uses them — there the date IS the question being asked, not a hidden filter
    on an unrelated query.
    """
    must: list[dict] = [{"key": "farmer_id", "match": {"value": farmer_id}}]
    if status:
        # A list, so "unresolved" (open OR pending) is one call rather than two plus
        # a union the caller has to compute.
        if isinstance(status, (list, tuple, set)):
            vals = [v for v in status if v]
            if vals:
                must.append({"key": "status", "match": {"any": list(vals)}})
        else:
            must.append({"key": "status", "match": {"value": status}})
    details = dict(filters or {})
    if metadata_key is not None and metadata_value is not None:
        details[metadata_key] = metadata_value
    for key, value in details.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise HTTPException(422, "Detail keys must be simple names such as feed_supplement")
        if not isinstance(value, (str, int, float, bool)):
            raise HTTPException(422, "Detail filter values must be text, numbers, or booleans")
        if isinstance(value, float) and not math.isfinite(value):
            raise HTTPException(422, "Numeric filters must be finite")
        condition = ({"range": {"gte": value, "lte": value}} if isinstance(value, float)
                     else {"match": {"value": value}})
        must.append({"key": f"metadata.{key}", **condition})

    if only_current:
        # Version state, not a date: has this entry been replaced by a newer version.
        must.append({"is_empty": {"key": "superseded_at"}})

    return {
        "must": must,
        # The settings entry (the memory on/off flag) and the Layer-1 standing
        # record share this collection with real memory, so both are excluded from
        # every normal read path — neither is a memory to be returned as one, and
        # the standing record is injected directly rather than searched.
        "must_not": [
            {"key": "type", "match": {"value": SETTINGS_TYPE}},
            {"key": "type", "match": {"value": PROFILE_TYPE}},
            {"key": "type", "match": {"value": KEY_DOC_TYPE}},
            {"key": "type", "match": {"value": WORKER_CHECKPOINT_TYPE}},
            *([] if include_chunks else [{"key": "type", "match": {"value": CHUNK_TYPE}}]),
        ],
    }


async def _qdrant(method: str, path: str, payload: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.request(method, f"{QDRANT_URL}{path}", json=payload,
                            headers=qdrant_headers(QDRANT_URL))
        if r.status_code >= 300:
            raise HTTPException(502, f"qdrant {r.status_code}: {r.text[:300]}")
        return r.json()


async def _checkpoint_qdrant(method: str, path: str, payload: dict | None = None) -> dict:
    """Translate unsupported conditional writes into a fail-closed 503."""
    try:
        return await _qdrant(method, path, payload)
    except HTTPException as exc:
        if method == "PUT":
            raise HTTPException(503, "atomic checkpoint storage is unavailable") from exc
        raise


async def _require_checkpoint_mutations() -> str:
    """Qdrant 1.17 introduced the update modes required for atomic leases."""
    try:
        info = await _qdrant("GET", "/")
        version = str(info.get("version", "0.0.0"))
        parts = tuple(int(part) for part in version.split(".")[:2])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "cannot verify atomic checkpoint support") from exc
    if parts < (1, 17):
        raise HTTPException(503, "Qdrant 1.17 or newer is required for atomic checkpoints")
    return version


async def _ensure_collection() -> None:
    """Create the collection and its payload indexes if they aren't there yet.

    Runs on startup so the store is reproducible from this file alone. It used to be
    a hand-run curl, which meant the schema lived only in someone's shell history —
    and a service that can't stand up its own store is a service that gets recreated
    slightly differently every time.

    Two named vectors per point: "headline" (short, searched every turn) and
    "expanded" (fuller detail, searched only via the recall tool). Nothing else —
    search here is semantic, so there is no keyword vector to maintain.

    Existing collections are never dropped, but missing payload indexes are repaired.
    """
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{QDRANT_URL}/collections/{COLLECTION}", headers=qdrant_headers(QDRANT_URL))
        exists = r.status_code == 200
        if not exists and r.status_code != 404:
            raise RuntimeError(f"Qdrant collection check failed: HTTP {r.status_code}; check endpoint and credentials")

    if not exists:
        await _qdrant("PUT", f"/collections/{COLLECTION}", {
            "vectors": {
                "headline": {"size": EMBED_DIM, "distance": "Cosine"},
                "expanded": {"size": EMBED_DIM, "distance": "Cosine"},
            },
        })
    # Indexed because every read path filters on them. source_ts and expires_at are
    # datetime so range filters work; the rest are exact-match keywords.
    for field, schema in (
        ("farmer_id", "keyword"),
        ("type", "keyword"),
        ("status", "keyword"),
        ("source_ts", "datetime"),
        ("episode_id", "keyword"),
        ("expires_at", "datetime"),
        ("checkpoint_kind", "keyword"),
        ("checkpoint_status", "keyword"),
        ("checkpoint_version", "integer"),
        ("checkpoint_next_retry_at", "datetime"),
        ("checkpoint_lease_expires_at", "datetime"),
        ("checkpoint_subject_id", "keyword"),
    ):
        await _qdrant("PUT", f"/collections/{COLLECTION}/index?wait=true",
                      {"field_name": field, "field_schema": schema})


async def _collection_embed_model() -> str | None:
    """Which model actually wrote the vectors already in this collection.

    Read from a stored point rather than tracked separately, so it reflects reality
    and not an intention.
    """
    try:
        got = await _qdrant("POST", f"/collections/{COLLECTION}/points/scroll", {
            # Placeholder-vector types (the settings flag, key documentation) are
            # never embedded, so they carry no stamp. Reading one of those would
            # report "unstamped" and silently disable the mismatch check.
            "filter": {"must_not": [
                {"key": "type", "match": {"value": SETTINGS_TYPE}},
                {"key": "type", "match": {"value": KEY_DOC_TYPE}},
                {"key": "type", "match": {"value": PROFILE_TYPE}},
                {"key": "type", "match": {"value": WORKER_CHECKPOINT_TYPE}},
            ]},
            "limit": 1, "with_payload": True,
        })
        points = got["result"]["points"]
    except Exception:  # noqa: BLE001
        return None
    if not points:
        return None
    # Points written before this stamp existed carry no marker; they came from the
    # original Marqo model, which is the only thing that has ever written here.
    return points[0]["payload"].get("embed_model") or "unstamped (pre-2026-09-08)"


@app.on_event("startup")
async def _startup() -> None:
    # Never fatal: if Qdrant is briefly unreachable the service should come up and
    # fail individual requests, not refuse to start.
    try:
        await _ensure_collection()
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not verify collection {COLLECTION}: {exc}")
        return

    # The migration trap, stated loudly at the one moment someone can act on it.
    # Swapping the embedding service and keeping the old collection produces vectors
    # of the right shape and the wrong meaning: nothing errors, and every search
    # quietly returns near-random neighbours. The fix is always to re-embed the
    # collection, never to hope it lines up.
    existing = await _collection_embed_model()
    if existing and not existing.startswith("unstamped") and existing != EMBED_MODEL_ID:
        print(
            f"WARNING: collection {COLLECTION} was written by embedding model "
            f"{existing!r} but this service is configured for {EMBED_MODEL_ID!r}. "
            "Stored vectors and new queries are not comparable — search results "
            "will be meaningless until the collection is re-embedded."
        )
    else:
        print(f"embeddings: {EMBED_MODEL_ID} ({EMBED_DIM}-dim) via {MARQO_URL}")


def _settings_point_id(farmer_id: str) -> str:
    """Deterministic id so a farmer has exactly one settings entry, never duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-settings:{farmer_id}"))


async def _memory_enabled(farmer_id: str) -> bool:
    """Default on for now; stored false disables reply use for this farmer.

    A failed lookup or malformed stored value never overrides an explicit off.
    Only the settings endpoint writes this record; the dreamer leaves it alone.
    """
    try:
        got = await _qdrant(
            "POST",
            f"/collections/{COLLECTION}/points",
            {"ids": [_settings_point_id(farmer_id)], "with_payload": True},
        )
    except (HTTPException, httpx.HTTPError):
        return False
    points = got.get("result") or []
    if not points:
        return DEFAULT_MEMORY_ENABLED
    return points[0].get("payload", {}).get("memory_enabled", DEFAULT_MEMORY_ENABLED) is True


# ---------- endpoints ----------

@app.get("/health")
async def health() -> dict:
    info = await _qdrant("GET", f"/collections/{COLLECTION}")
    return {
        "status": "ok",
        "collection": COLLECTION,
        "points": info["result"]["points_count"],
        "embed_model": EMBED_MODEL_ID,
        "embed_dim": EMBED_DIM,
        # Whatever actually wrote the vectors in this collection. If this disagrees
        # with embed_model above, search results are meaningless — see _startup.
        "collection_embed_model": await _collection_embed_model(),
    }


@app.post("/memory/{farmer_id}/entries")
async def add_entry(farmer_id: str, entry: NewEntry) -> dict:
    """Write a new memory entry. Never overwrites anything — updates go through
    PATCH, which closes out the old entry and writes a fresh one."""
    return await _store_entry(farmer_id, entry)


async def _store_entry(farmer_id: str, entry: NewEntry, *,
                       exclude_id: str | None = None, previous_version: str | None = None) -> dict:
    """Version links are generated only by the checked update path, never POST input."""
    sessions = sorted({s for s in (
        (entry.source_sessions or []) + [entry.source_session_id]) if s})
    traces = sorted({t for t in (entry.source_traces or []) if t})

    # Re-run protection. The dreamer is meant to be safe to run again over the same
    # window — nightly jobs get retried, windows overlap, and a crash mid-farmer has
    # to be resumable. Without this, every re-run writes the whole day again.
    mine = await _qdrant("POST", f"/collections/{COLLECTION}/points/scroll",
                         {"filter": _farmer_filter(farmer_id, only_current=True),
                          "limit": 400, "with_payload": True})
    want = _norm_text(entry.headline)
    for p in mine["result"]["points"]:
        if exclude_id and p["id"] == exclude_id:
            # The version being replaced is still current at this moment; it is
            # not a duplicate of its own successor.
            continue
        pl = p["payload"]
        if _norm_text(pl.get("headline") or "") == want:
            return {"id": p["id"], "payload": pl, "written": False,
                    "duplicate_of_existing": True,
                    "reason": "this farmer already holds an entry with the same headline"}
        # Sharing source turns is not enough to be a duplicate: one turn can
        # describe several independent situations. Semantic matching is the
        # writer's decision; a storage guard must not silently discard them.

    point_id = str(uuid.uuid4())
    payload = {
        "farmer_id": farmer_id,
        "type": MEMORY_TYPE,
        "status": entry.status,
        "headline": entry.headline,
        "expanded": entry.expanded,
        # Derived, never taken from the caller: the number of distinct conversations
        # this situation has come up in.
        "times_raised": max(1, len(sessions)) if sessions else entry.times_raised,
        "recorded_at": _now(),
        "superseded_at": None,
        "source_session_id": entry.source_session_id,
        "source_ts": entry.source_ts,
        "expires_at": entry.expires_at,
        # When the thread STARTED, carried forward across updates. Without this,
        # "how long has this been going on" would only live in the oldest
        # superseded version and be invisible without history traversal.
        "first_source_ts": entry.first_source_ts or entry.source_ts,
        # Which conversations this entry draws on. times_raised is DERIVED from this
        # rather than incremented per write, because a write is not an occasion: two
        # chunks of one conversation used to produce two entries, then a merge summed
        # them, and the count claimed a farmer had raised something three times when
        # they had said it once. That count is evidence for the honesty judge, so an
        # inflated one launders a false "raised repeatedly" claim past it.
        "source_sessions": sessions,
        "source_traces": traces,
        # Set on an update to point at the version this one replaced, so an
        # episode's history is walkable. Every superseded version already holds its
        # own full account and date; all that was missing was the link.
        "previous_version": previous_version,
        # Which model produced this point's vectors. Only ever read by the startup
        # check and /health — a mismatch here is the difference between "search is
        # broken" and "search silently returns nonsense".
        "embed_model": EMBED_MODEL_ID,
        # Nested on purpose: model-generated keys live here and can never shadow a
        # system field above. Qdrant's payload is schemaless either way — this
        # boundary is about protecting the fields we filter on, not about migrations.
        "metadata": entry.metadata or {},
    }
    if entry.chunk_size != episode.CHUNK_MAX_CHARS:
        raise HTTPException(422, "Writer and service chunk budgets differ; use the same MEMORY_CHUNK_MAX_CHARS")
    return await _store_layout(point_id, payload, entry.contents)


def _chunk_id(entry_id, number):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-chunk:{entry_id}:{number}"))


async def _store_layout(point_id, payload, contents=None):
    """Prepare every vector before writing; the original detailed account is retained."""
    text = payload.get("expanded") or payload["headline"]
    chunks = episode.split_text(text)
    try:
        rows = episode.validate_contents(contents, len(chunks)) if contents is not None else episode.extractive_contents(text, chunks)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    payload = {**payload, "chunks": chunks, "contents": rows, "chunk_count": len(chunks),
               "chunk_size": episode.CHUNK_MAX_CHARS, "embed_model": EMBED_MODEL_ID}
    texts = [episode.level2_text(payload["headline"], rows)] + [text[c["start"]:c["end"]] for c in chunks]
    vectors = await _embed_many(texts)
    points = [{"id": point_id, "vector": {"headline": vectors[0]}, "payload": payload}]
    for chunk, vector in zip(chunks, vectors[1:]):
        points.append({"id": _chunk_id(point_id, chunk["number"]), "vector": {"expanded": vector},
                       "payload": {"farmer_id": payload["farmer_id"], "type": CHUNK_TYPE,
                                   "episode_id": point_id, "chunk_number": chunk["number"],
                                   "start": chunk["start"], "end": chunk["end"],
                                   "status": payload.get("status", "n/a"),
                                   "metadata": payload.get("metadata", {}),
                                   "superseded_at": payload.get("superseded_at"),
                                   "embed_model": EMBED_MODEL_ID}})
    await _qdrant("PUT", f"/collections/{COLLECTION}/points?wait=true", {"points": points})
    return {"id": point_id, "payload": payload}


async def _retire(entry_id, old, changes):
    ids = [entry_id] + [_chunk_id(entry_id, n) for n in range(1, old.get("chunk_count", 0) + 1)]
    await _qdrant("POST", f"/collections/{COLLECTION}/points/payload?wait=true",
                  {"payload": changes, "points": ids})


@app.post("/internal/memory/{farmer_id}/entries/{entry_id}/index", include_in_schema=False)
async def index_existing_episode(farmer_id: str, entry_id: str):
    """Explicit background index upgrade. Preserve text, dates, history and settings."""
    got = await _qdrant("POST", f"/collections/{COLLECTION}/points", {"ids": [entry_id], "with_payload": True})
    points = got.get("result") or []
    if not points or points[0]["payload"].get("farmer_id") != farmer_id:
        raise HTTPException(404, "memory not found")
    payload = points[0]["payload"]
    if payload.get("type") in INTERNAL_TYPES or payload.get("superseded_at"):
        raise HTTPException(409, "only current episodes can be indexed")
    if payload.get("chunk_size") == episode.CHUNK_MAX_CHARS and payload.get("chunk_count"):
        return {"id": entry_id, "unchanged": True}
    return await _store_layout(entry_id, payload)


@app.post("/memory/{farmer_id}/search")
async def search_headlines(farmer_id: str, req: SearchRequest) -> dict:
    """Layer 2: the quick, every-turn search — headline vectors only."""
    return await _search(farmer_id, req, using="headline")


@app.post("/memory/{farmer_id}/search_expanded")
async def search_expanded(farmer_id: str, req: SearchRequest) -> dict:
    """Layer 3: the deeper search, meant to be reached via a deliberate tool call."""
    return await _search(farmer_id, req, using="expanded")


def _shape(point: dict, *, score: float | None = None, include_private: bool = False) -> dict:
    """One entry as a caller sees it.

    The point of the extra keys is that a caller always learns whether more detail
    EXISTS — which is the part that matters, because a reader who cannot tell whether
    they have the whole picture will assume they do.
    """
    payload = dict(point.get("payload") or {})
    payload.pop("type", None)  # Internal kind / legacy category is not conversation data.
    text, chunks, contents = episode.layout(payload)
    out = {"id": point["id"], **payload, "chunks": chunks, "contents": contents,
           "chunk_count": len(chunks), "chunk_indexed": bool(payload.get("chunk_count"))}
    out["visible_metadata"] = {
        key: value for key, value in (payload.get("metadata") or {}).items()
        if not _hidden_key(key)
    }
    if not include_private:
        # Safe by default, including callers that still read the metadata alias.
        out["metadata"] = dict(out["visible_metadata"])
    if score is not None:
        out["score"] = score
    n = payload.get("times_raised") or 1
    if payload.get("previous_version"):
        out["earlier_versions_available"] = True
        out["history_note"] = (
            f"This account has been rewritten as the situation developed, across {n} "
            "conversation(s). Earlier versions are kept and can be fetched — nothing "
            "was discarded to keep this short."
        )
    return out


def _hidden_key(key: str) -> bool:
    return (key in AGENT_HIDDEN_KEYS or key.endswith(("_id", "_number", "_no", "_tag"))
            or key in {"mobile", "mobile_number", "ear_tag", "phone_number"})


async def _search(farmer_id: str, req: SearchRequest, *, using: str,
                  respect_setting: bool = True) -> dict:
    """Semantic search, scoped to one farmer.

    Semantic only, by decision. A hybrid dense+keyword blend was built, measured and
    removed: without real inverse-document-frequency weighting the keyword side
    over-weighted moderately common words, and it ranked worse than plain semantic
    search. Don't bring it back without corpus IDF and a measurement.
    """
    # Enforced here, not in the caller: memory off for this farmer means nothing
    # comes back, whatever was asked for. A caller that forgets to check still
    # gets nothing.
    if respect_setting and not await _memory_enabled(farmer_id):
        return {"searched": using, "hits": [], "memory_enabled": False}

    flt = _farmer_filter(farmer_id, status=req.status, only_current=req.only_current,
                         metadata_key=req.metadata_key, metadata_value=req.metadata_value,
                         filters=req.filters, include_chunks=using == "expanded")
    if using == "expanded":
        # New episodes search chunks. Legacy episodes remain searchable until indexed.
        flt["must"].append({"should": [
            {"key": "type", "match": {"value": CHUNK_TYPE}},
            {"is_empty": {"key": "chunk_count"}}]})
    if req.memory_ref:
        ref = str(req.memory_ref)
        flt["must"].append({"should": [{"has_id": [ref]}, {"key": "episode_id", "match": {"value": ref}}]})
    vector = await _embed(req.query, kind="query")
    hits, offset, exhausted = [], 0, False
    # Parent validation prevents stale/orphan chunk hits from leaking after a partial write.
    for _ in range(4):
        size = req.limit if using == "headline" else min(100, max(10, req.limit * 2))
        res = await _qdrant("POST", f"/collections/{COLLECTION}/points/query", {
            "query": vector, "using": using, "filter": flt, "limit": size,
            "offset": offset, "with_payload": True})
        points = res["result"]["points"]
        parents = {}
        ids = list({p["payload"]["episode_id"] for p in points if p["payload"].get("type") == CHUNK_TYPE})
        if ids:
            found = await _qdrant("POST", f"/collections/{COLLECTION}/points", {"ids": ids, "with_payload": True})
            parents = {p["id"]: p for p in found.get("result") or []}
        for point in points:
            chunk = None
            if point["payload"].get("type") == CHUNK_TYPE:
                parent = parents.get(point["payload"]["episode_id"])
                if not parent:
                    continue
                pl = parent["payload"]
                if pl.get("farmer_id") != farmer_id or pl.get("type") in INTERNAL_TYPES:
                    continue
                if req.only_current and pl.get("superseded_at"):
                    continue
                number = point["payload"]["chunk_number"]
                layout = pl.get("chunks") or []
                if not 1 <= number <= len(layout):
                    continue
                chunk = layout[number - 1]
                if (chunk["start"], chunk["end"]) != (point["payload"]["start"], point["payload"]["end"]):
                    continue
            else:
                parent = point
            row = _shape(parent, score=round(point["score"], 4), include_private=not respect_setting)
            if using == "expanded":
                text = row.get("expanded") or row["headline"]
                if chunk:
                    row.update(chunk_number=chunk["number"], passage=text[chunk["start"]:chunk["end"]])
                else:
                    first = row["chunks"][0]
                    row.update(chunk_number=1, passage=text[first["start"]:first["end"]],
                               note="Legacy episode index: reindex in the background for passage-level search.")
            if respect_setting:
                row.pop("expanded", None)
                row.pop("chunks", None)
            hits.append(row)
            if len(hits) == req.limit:
                break
        exhausted = len(points) < size
        if len(hits) >= req.limit or exhausted or using == "headline":
            break
        offset += len(points)
    return {"searched": using, "hits": hits,
            "partial": not exhausted and len(hits) < req.limit}


@app.get("/memory/{farmer_id}/entries")
async def filter_entries(
    farmer_id: str,
    status: Annotated[list[EntryStatus] | None, Query()] = None,
    only_current: bool = True,
    metadata_key: str | None = None,
    metadata_value: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: str | None = None,
    filters: str | None = None,
) -> dict:
    """Exact matching over recorded fields, with explicit page continuation.

    Missing extraction/tags may still omit relevant conversations. Deadlines never
    silently remove an unresolved entry from this list.
    """
    if not await _memory_enabled(farmer_id):
        return {"entries": [], "memory_enabled": False}
    return await _list_entries(farmer_id, status, only_current, metadata_key,
                               metadata_value, limit, cursor, filters, include_private=False)


@app.get("/internal/memory/{farmer_id}/entries", include_in_schema=False)
async def _list_entries(
    farmer_id: str,
    status: Annotated[list[EntryStatus] | None, Query()] = None,
    only_current: bool = True,
    metadata_key: str | None = None,
    metadata_value: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: str | None = None,
    filters: str | None = None,
    include_private: bool = True,
) -> dict:
    try:
        details = json.loads(filters) if filters else {}
        if not isinstance(details, dict):
            raise ValueError("filters must be an object")
        if cursor:
            uuid.UUID(cursor)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "Invalid filters or page cursor") from exc
    flt = _farmer_filter(farmer_id, status=status, only_current=only_current,
                         metadata_key=metadata_key, metadata_value=metadata_value, filters=details)
    res = await _qdrant(
        "POST", f"/collections/{COLLECTION}/points/scroll",
        {"filter": flt, "limit": limit, "with_payload": True,
         **({"offset": cursor} if cursor else {})},
    )
    points = res["result"]["points"]
    next_cursor = res["result"].get("next_page_offset")
    truncated = next_cursor is not None
    out = {"entries": [_shape(p, include_private=include_private) for p in points],
           "returned": len(points), "truncated": truncated, "next_cursor": next_cursor}
    if truncated:
        out["note"] = (f"More than {limit} entries match. This is not the full list — "
                       "narrow the filter before concluding anything about totals.")
    return out


@app.get("/memory/{farmer_id}/entries/{entry_id}")
async def read_entry(farmer_id: str, entry_id: str) -> dict:
    """Open a selected memory directly. A second similarity search is unnecessary."""
    if not await _memory_enabled(farmer_id):
        return {"entry": None, "memory_enabled": False}
    return await _read_entry(farmer_id, entry_id, include_private=False)


@app.get("/internal/memory/{farmer_id}/entries/{entry_id}", include_in_schema=False)
async def _read_entry(farmer_id: str, entry_id: str, include_private: bool = True) -> dict:
    got = await _qdrant("POST", f"/collections/{COLLECTION}/points",
                        {"ids": [entry_id], "with_payload": True})
    points = got.get("result") or []
    if not points:
        raise HTTPException(404, "memory not found")
    payload = points[0]["payload"]
    if payload.get("farmer_id") != farmer_id or payload.get("type") in INTERNAL_TYPES:
        raise HTTPException(404, "memory not found")
    return {"entry": _shape(points[0], include_private=include_private), "current": not bool(payload.get("superseded_at"))}


@app.patch("/memory/{farmer_id}/entries/{entry_id}")
async def update_entry(farmer_id: str, entry_id: str, update: UpdateEntry) -> dict:
    """Update = write a new version, then retire the old one, so the history of how a
    fact changed is never lost AND a refused write leaves the entry untouched."""
    got = await _qdrant(
        "POST",
        f"/collections/{COLLECTION}/points",
        {"ids": [entry_id], "with_payload": True},
    )
    points = got["result"]
    if not points:
        raise HTTPException(404, "entry not found")
    old = points[0]["payload"]
    if old.get("farmer_id") != farmer_id:
        raise HTTPException(403, "entry does not belong to this farmer")

    if old.get("superseded_at") or old.get("type") in INTERNAL_TYPES:
        raise HTTPException(409, "only a current memory can be updated")

    # Idempotency, enforced here rather than in a caller: this is the authority, and
    # the bugs that made it necessary all came from a rule living in one write path
    # and not another. Deterministic on purpose — asking a model whether it had seen a
    # trace before would be slower, cost a call, and be able to get it wrong.
    #
    # Note this returns BEFORE anything is mutated. It used to run after the old
    # version had already been stamped superseded, so declining an update destroyed
    # the entry it declined to change.
    incoming = {t for t in (update.source_traces or []) if t}
    already = set(old.get("source_traces") or [])
    merge_ids = {str(value) for value in (update.merge_from or [])}
    if merge_ids:
        if entry_id in merge_ids:
            raise HTTPException(422, "A merge source cannot be the destination")
        source_points = (await _qdrant("POST", "/collections/" + COLLECTION + "/points", {
            "ids": sorted(merge_ids), "with_payload": True, "with_vector": False}))["result"]
        if {point["id"] for point in source_points} != merge_ids or any(
                point["payload"].get("farmer_id") != farmer_id
                or point["payload"].get("type") in INTERNAL_TYPES
                or point["payload"].get("superseded_at") for point in source_points):
            raise HTTPException(422, "Merge sources must be current memories belonging to this farmer")
        # Fragments from the same turns can contain different details. Their shared
        # trace IDs cannot reject a checked merge as though it were an ordinary retry.
        for point in source_points:
            source = point["payload"]
            incoming.update(source.get("source_traces") or [])
            update.source_sessions = sorted(set(update.source_sessions or []) | set(source.get("source_sessions") or []))
            update.source_ts = max(filter(None, [update.source_ts, source.get("source_ts")]), default=None)
            update.first_source_ts = min(filter(None, [update.first_source_ts,
                source.get("first_source_ts") or source.get("source_ts")]), default=None)
    if not merge_ids and incoming and already and incoming <= already:
        return {"applied": False,
                "reason": "every turn behind this update is already recorded on the entry",
                "entry_id": entry_id, "already_recorded": sorted(incoming),
                "unchanged": True}

    # Union of the conversations behind the old version and this update. A repeat
    # write from a session already counted does NOT raise the count.
    sessions = sorted({s for s in (
        (old.get("source_sessions") or [])
        + (update.source_sessions or [])
        + [update.source_session_id or old.get("source_session_id")]) if s})

    merged = NewEntry(
        headline=update.headline or old["headline"],
        expanded=update.expanded if update.expanded is not None else old.get("expanded", ""),
        contents=update.contents, chunk_size=update.chunk_size,
        status=update.status or old.get("status", "n/a"),
        # Derived from the distinct conversations, not incremented per write. A
        # caller-supplied times_raised is ignored when sessions are known.
        times_raised=(max(1, len(sessions)) if sessions
                      else (update.times_raised if update.times_raised is not None
                            else old.get("times_raised", 1))),
        source_sessions=sessions or None,
        source_traces=sorted(already | incoming) or None,
        source_session_id=update.source_session_id or old.get("source_session_id"),
        source_ts=max(filter(None, [update.source_ts, old.get("source_ts")]), default=None),
        first_source_ts=min(filter(None, [update.first_source_ts,
                            old.get("first_source_ts") or old.get("source_ts")]), default=None),
        expires_at=update.expires_at if update.expires_at is not None else old.get("expires_at"),
        metadata=update.metadata if update.metadata is not None else old.get("metadata", {}),
    )
    # ORDER MATTERS, and getting it wrong destroyed real memories. This used to stamp
    # the old version superseded FIRST — before even the idempotency check — and then
    # write the replacement. Any refusal after that point left the old entry retired
    # with nothing replacing it: the memory was silently gone. Five entries were lost
    # that way before it was caught, because an update rejected as already-applied
    # still retired the thing it declined to change.
    #
    # So: write the replacement first, and only retire the old version once the new
    # one actually exists. A refused write now leaves the entry exactly as it was.
    new = await _store_entry(farmer_id, merged, exclude_id=entry_id, previous_version=entry_id)
    if not new.get("written", True):
        return {"applied": False,
                "reason": new.get("reason", "the replacement was refused"),
                "entry_id": entry_id, "unchanged": True}

    now = _now()
    await _retire(entry_id, old, {"superseded_at": now})
    return {"closed": {"id": entry_id, "superseded_at": now}, "new": new}


@app.get("/memory/{farmer_id}/entries/{entry_id}/history")
async def entry_history(farmer_id: str, entry_id: str,
                        limit: Annotated[int, Query(ge=1, le=50)] = 20) -> dict:
    """How this episode's account developed, oldest first.

    Earlier stored accounts remain available. This is not a raw transcript and
    cannot recover detail that a generated proposal never stored.

    Two kinds of ancestor, both followed:
      - **replaced versions**, walked back through `previous_version`, which is how an
        account was rewritten as the situation developed;
      - **folded fragments**, which were separate entries until consolidation found
        they were one situation.
    """
    if not await _memory_enabled(farmer_id):
        return {"history": [], "memory_enabled": False}

    chain, seen, cur = [], set(), entry_id
    while cur and cur not in seen and len(chain) < limit:
        seen.add(cur)
        got = await _qdrant("POST", f"/collections/{COLLECTION}/points",
                            {"ids": [cur], "with_payload": True})
        points = got.get("result") or []
        if not points:
            break
        pl = points[0]["payload"]
        if pl.get("type") in INTERNAL_TYPES:
            raise HTTPException(404, "memory not found")
        if pl.get("farmer_id") != farmer_id:
            raise HTTPException(403, "entry does not belong to this farmer")
        chain.append({"id": cur, "kind": "version", **pl})
        cur = pl.get("previous_version")

    folded = await _qdrant("POST", f"/collections/{COLLECTION}/points/scroll",
                           {"filter": {"must_not": [{"key": "type", "match": {"any": list(INTERNAL_TYPES)}}], "must": [
                               {"key": "farmer_id", "match": {"value": farmer_id}},
                               {"key": "merged_into", "match": {"any": sorted(seen)}}]},
                            "limit": limit, "with_payload": True})
    for p in folded["result"]["points"]:
        chain.append({"id": p["id"], "kind": "folded_fragment", **p["payload"]})

    chain.sort(key=lambda e: str(e.get("source_ts") or ""))
    return {
        "history": [{k: e.get(k) for k in (
            "id", "kind", "status", "headline", "expanded", "source_ts",
            "first_source_ts", "times_raised", "merge_reason")} for e in chain],
        "count": len(chain),
        "truncated": bool(cur or folded["result"].get("next_page_offset") or any(
            p["payload"].get("previous_version") for p in folded["result"]["points"])),
        "note": "Oldest first. 'version' entries are earlier accounts of the same "
                "episode; 'folded_fragment' entries were separate memories until "
                "consolidation found they described one situation.",
    }


class FoldRequest(BaseModel):
    merged_into: str = Field(description="Id of the consolidated entry this was folded into.")
    reason: str | None = None


@app.post("/memory/{farmer_id}/entries/{entry_id}/fold")
async def fold_entry(farmer_id: str, entry_id: str, req: FoldRequest) -> dict:
    """Retire a fragment into a consolidated entry, without writing a replacement.

    PATCH is the wrong tool for consolidation: it supersedes the old version and
    writes a NEW one, which is right for "this fact changed" and wrong for "these six
    entries were always one situation" — that would leave six live entries again.

    This closes a fragment out and records where it went, so runtime search returns
    only the consolidated entry while the fragments stay readable for provenance.
    Nothing is deleted: the trail from a merged episode back to what it was built from
    has to survive, or a wrong merge becomes impossible to diagnose.
    """
    got = await _qdrant("POST", f"/collections/{COLLECTION}/points",
                        {"ids": [entry_id], "with_payload": True})
    points = got.get("result") or []
    if not points:
        raise HTTPException(404, "entry not found")
    if points[0]["payload"].get("farmer_id") != farmer_id:
        raise HTTPException(403, "entry does not belong to this farmer")
    source = points[0]["payload"]
    if source.get("superseded_at") or source.get("type") in INTERNAL_TYPES:
        raise HTTPException(409, "only a current memory can be folded")
    if entry_id == req.merged_into:
        raise HTTPException(400, "an entry cannot be folded into itself")

    # Retiring the source is safe only when its destination is a current memory
    # belonging to the same farmer. A guessed or stale id must leave it untouched.
    target = await _qdrant("POST", f"/collections/{COLLECTION}/points",
                           {"ids": [req.merged_into], "with_payload": True})
    targets = target.get("result") or []
    if not targets or targets[0]["payload"].get("farmer_id") != farmer_id:
        raise HTTPException(404, "merge destination not found for this farmer")
    destination = targets[0]["payload"]
    if destination.get("superseded_at") or destination.get("type") in INTERNAL_TYPES:
        raise HTTPException(409, "merge destination must be a current memory")

    now = _now()
    await _retire(entry_id, source, {"superseded_at": now, "merged_into": req.merged_into,
                                      "merge_reason": req.reason})
    return {"folded": entry_id, "into": req.merged_into, "superseded_at": now}


# ---------- per-farmer memory on/off, stored in Qdrant next to their memory ----------

class MemorySettings(BaseModel):
    memory_enabled: bool
    note: str | None = Field(default=None, description="Why it was set this way (e.g. 'shared account suspected').")


@app.put("/memory/{farmer_id}/settings")
async def put_settings(farmer_id: str, settings: MemorySettings) -> dict:
    """Turn memory on or off for one farmer.

    Stored as a single settings entry in the same collection as their memory, so
    there is exactly one place to look at (and list) who has memory enabled — no
    separate flag store to drift out of sync with the data.
    """
    payload = {
        "farmer_id": farmer_id,
        "type": SETTINGS_TYPE,
        "memory_enabled": settings.memory_enabled,
        "note": settings.note,
        "updated_at": _now(),
        # Excluded from every normal read path, which filters by real entry types.
        "status": "n/a",
        "recorded_at": _now(),
        "superseded_at": None,
    }
    zero = [0.0] * EMBED_DIM
    await _qdrant(
        "PUT",
        f"/collections/{COLLECTION}/points?wait=true",
        {"points": [{"id": _settings_point_id(farmer_id), "vector": {"headline": zero, "expanded": zero}, "payload": payload}]},
    )
    return {"farmer_id": farmer_id, "memory_enabled": settings.memory_enabled, "note": settings.note}


@app.get("/memory/{farmer_id}/settings")
async def get_settings(farmer_id: str) -> dict:
    return {"farmer_id": farmer_id, "memory_enabled": await _memory_enabled(farmer_id)}


@app.get("/settings")
async def list_settings(limit: int = 500) -> dict:
    """Every farmer with an explicit flag set, and what it's set to — one query,
    because the flags live with the data rather than scattered somewhere else.

    Since the default is ON, the interesting rows here are usually the ones turned
    OFF (an A/B comparison leg, or an account that looked shared). Farmers with no
    row at all are simply on the default.
    """
    res = await _qdrant(
        "POST",
        f"/collections/{COLLECTION}/points/scroll",
        {
            "filter": {"must": [{"key": "type", "match": {"value": SETTINGS_TYPE}}]},
            "limit": limit,
            "with_payload": True,
        },
    )
    rows = [
        {
            "farmer_id": p["payload"].get("farmer_id"),
            "memory_enabled": p["payload"].get("memory_enabled"),
            "note": p["payload"].get("note"),
            "updated_at": p["payload"].get("updated_at"),
        }
        for p in res["result"]["points"]
    ]
    return {
        "default_when_unset": DEFAULT_MEMORY_ENABLED,
        "explicitly_set": rows,
        "explicitly_off": [r["farmer_id"] for r in rows if r["memory_enabled"] is False],
    }


@app.get("/due")
async def coming_due(within_hours: int = 24, limit: int = 200) -> dict:
    """Entries lapsing soon — the trigger for proactive follow-ups.

    A reminder doesn't need its own separate store: it *is* a memory entry with an
    expires_at. "What should we follow up on" is therefore one range filter, e.g.
    a milk withdrawal period ending tomorrow, a scheme window closing, a
    vaccination deadline approaching.

    Note this deliberately does NOT filter by the per-farmer memory flag: it is a
    background/ops view, not a farmer-facing read. Anything acted on from here must
    re-check that farmer's flag before surfacing it to them.
    """
    now = datetime.now(timezone.utc)
    horizon = (now + timedelta(hours=within_hours)).isoformat()
    res = await _qdrant(
        "POST",
        f"/collections/{COLLECTION}/points/scroll",
        {
            "filter": {
                "must": [
                    {"is_empty": {"key": "superseded_at"}},
                    {"key": "expires_at", "range": {"gt": now.isoformat(), "lt": horizon}},
                ],
                "must_not": [
                    {"key": "type", "match": {"value": SETTINGS_TYPE}},
                    {"key": "type", "match": {"value": PROFILE_TYPE}},
                    {"key": "type", "match": {"value": WORKER_CHECKPOINT_TYPE}},
                ],
            },
            "limit": limit,
            "with_payload": True,
        },
    )
    return {
        "within_hours": within_hours,
        "due": [
            {
                "id": p["id"],
                "farmer_id": p["payload"].get("farmer_id"),
                "type": p["payload"].get("type"),
                "expires_at": p["payload"].get("expires_at"),
                "headline": p["payload"].get("headline"),
            }
            for p in res["result"]["points"]
        ],
    }


# ---------- key inventory: derived from the data, not a catalogue we maintain ----------



class KeyDoc(BaseModel):
    key: str = Field(description="The metadata key being introduced, e.g. 'severity'.")
    description: str = Field(description="What it means, in one line.")
    example_values: list[Any] = Field(default_factory=list)


@app.post("/keys/doc")
async def document_key(doc: KeyDoc) -> dict:
    """Record what a newly invented key means.

    Written by the model when it invents a key, as a byproduct of inventing — it
    gates nothing and blocks nothing. It exists so a human can later look at what
    the model has been making up and decide whether to keep it.
    """
    payload = {
        "farmer_id": "__system__",
        "type": KEY_DOC_TYPE,
        "status": "n/a",
        "key_name": doc.key,
        "description": doc.description,
        "example_values": doc.example_values,
        "first_seen": _now(),
        "recorded_at": _now(),
        "superseded_at": None,
    }
    zero = [0.0] * EMBED_DIM
    await _qdrant(
        "PUT",
        f"/collections/{COLLECTION}/points?wait=true",
        {"points": [{"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-keydoc:{doc.key}")),
                     "vector": {"headline": zero, "expanded": zero}, "payload": payload}]},
    )
    return {"key": doc.key, "documented": True}


@app.get("/keys")
async def list_keys(limit: int = 2000) -> dict:
    """Every metadata key actually in use, derived by scanning the entries.

    Nothing to keep in sync — this reads the data itself. Keys the model invented
    without documenting show up with description=None, which is itself the useful
    signal. Use this to review what has been invented, then DELETE /keys/{key} to
    drop anything that turns out to be junk.
    """
    res = await _qdrant(
        "POST",
        f"/collections/{COLLECTION}/points/scroll",
        {"limit": limit, "with_payload": True, "filter": {"must_not": [
            {"key": "type", "match": {"any": [CHUNK_TYPE, SETTINGS_TYPE, PROFILE_TYPE, WORKER_CHECKPOINT_TYPE]}}]}},
    )
    points = res["result"]["points"]

    docs: dict[str, dict] = {}
    usage: dict[str, dict] = {}
    for p in points:
        pl = p.get("payload", {})
        if pl.get("type") == KEY_DOC_TYPE:
            docs[pl.get("key_name")] = {
                "description": pl.get("description"),
                "example_values": pl.get("example_values", []),
                "first_seen": pl.get("first_seen"),
            }
            continue
        if pl.get("type") == SETTINGS_TYPE:
            continue
        for k, v in (pl.get("metadata") or {}).items():
            slot = usage.setdefault(k, {"count": 0, "farmers": set(), "entry_types": set(), "samples": []})
            slot["count"] += 1
            slot["farmers"].add(pl.get("farmer_id"))
            slot["entry_types"].add(pl.get("type"))
            if len(slot["samples"]) < 5 and v not in slot["samples"]:
                slot["samples"].append(v)

    keys = []
    for k, slot in sorted(usage.items(), key=lambda kv: -kv[1]["count"]):
        d = docs.get(k, {})
        keys.append({
            "key": k,
            "filter_path": f"metadata.{k}",
            "description": d.get("description"),
            "documented": bool(d),
            "used_on_entries": slot["count"],
            "distinct_farmers": len(slot["farmers"]),
            "entry_types": sorted(t for t in slot["entry_types"] if t),
            "sample_values": slot["samples"],
        })
    undocumented = [k["key"] for k in keys if not k["documented"]]
    documented_unused = [k for k in docs if k not in usage]
    return {
        "keys": keys,
        "undocumented": undocumented,
        "documented_but_unused": documented_unused,
    }


@app.delete("/keys/{key}")
async def delete_key(key: str) -> dict:
    """Drop a metadata key from every entry that carries it.

    The cleanup path: the model can invent freely, and anything that turns out to
    be junk can be removed here afterwards, rather than being blocked up front.
    """
    await _qdrant(
        "POST",
        f"/collections/{COLLECTION}/points/payload/delete?wait=true",
        {
            "keys": [f"metadata.{key}"],
            "filter": {"must_not": [
                {"key": "type", "match": {"value": SETTINGS_TYPE}},
                {"key": "type", "match": {"value": PROFILE_TYPE}},
                {"key": "type", "match": {"value": WORKER_CHECKPOINT_TYPE}},
            ]},
        },
    )
    await _qdrant(
        "POST",
        f"/collections/{COLLECTION}/points/delete?wait=true",
        {"filter": {"must": [
            {"key": "type", "match": {"value": KEY_DOC_TYPE}},
            {"key": "key_name", "match": {"value": key}},
        ]}},
    )
    return {"key": key, "removed": True}


@app.get("/memory/{farmer_id}/keys")
async def farmer_keys(farmer_id: str) -> dict:
    """Agent-safe key list: what can be filtered on for THIS farmer.

    Deliberately different from the global GET /keys, which is an ops-review view
    and crosses farmer boundaries (so it must never be handed to the agent). This
    one:
      - covers only this farmer's own entries,
      - returns key names and descriptions, plus this farmer's OWN distinct values
        (their data, safe to show them) — never values from anyone else,
      - respects the per-farmer memory flag like every other read path.
    Small and cheap enough to inject into every turn rather than making the agent
    ask for it.
    """
    if not await _memory_enabled(farmer_id):
        return {"keys": [], "memory_enabled": False}

    mine = await _qdrant(
        "POST",
        f"/collections/{COLLECTION}/points/scroll",
        {"filter": _farmer_filter(farmer_id, only_current=True), "limit": 500, "with_payload": True},
    )
    used: dict[str, set] = {}
    for p in mine["result"]["points"]:
        for k, v in (p.get("payload", {}).get("metadata") or {}).items():
            if _hidden_key(k):
                continue
            slot = used.setdefault(k, set())
            for value in (v if isinstance(v, list) else [v]):
                if isinstance(value, (str, int, float, bool)) and len(slot) < 6:
                    slot.add(value)

    if not used:
        return {"keys": []}

    docs = await _qdrant(
        "POST",
        f"/collections/{COLLECTION}/points/scroll",
        {"filter": {"must": [{"key": "type", "match": {"value": KEY_DOC_TYPE}}]},
         "limit": 500, "with_payload": True},
    )
    described = {
        p["payload"].get("key_name"): p["payload"].get("description")
        for p in docs["result"]["points"]
    }

    return {
        "keys": [
            {
                "filter_path": f"metadata.{k}",
                "description": described.get(k),
                "values_for_this_farmer": sorted(vals, key=str),
            }
            for k, vals in sorted(used.items())
        ]
    }


# ---------- Layer 1: one standing record per farmer, fixed schema ----------



# The same fields for every farmer. Deliberately only things learned from
# CONVERSATION — union, village, herd counts and the linked-account list are all
# re-fetched live from Amul's own systems every turn, so duplicating them here would
# only create stale copies.
PROFILE_FIELDS = tuple(profile_schema.FIELDS)


class ProfileRecord(BaseModel):
    fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Subset of PROFILE_FIELDS that actually have content. Empty ones are omitted.",
    )
    source_ts: str | None = None


def _profile_point_id(farmer_id: str) -> str:
    """Deterministic, so a farmer has exactly ONE standing record — never N
    accumulating profile entries that a limit=1 read would pick between arbitrarily."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-profile:{farmer_id}"))


@app.put("/memory/{farmer_id}/profile")
async def put_profile(farmer_id: str, rec: ProfileRecord) -> dict:
    """Replace the farmer's standing record, keeping the previous version as history.

    Only the fields with content are stored, so the block injected into a turn stays
    as short as what is actually known.
    """
    try:
        kept = profile_schema.validate(rec.fields)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    unknown = [k for k in rec.fields if k not in PROFILE_FIELDS]

    pid = _profile_point_id(farmer_id)
    existing = await _qdrant(
        "POST", f"/collections/{COLLECTION}/points", {"ids": [pid], "with_payload": True}
    )
    points = existing.get("result") or []

    # A standing record that has not changed must not be rewritten. It is re-derived
    # on every run, and without this each run superseded the previous version with an
    # identical copy — four identical "language register: Gujarati" versions
    # accumulated for one farmer across four runs. That is unbounded churn in the one
    # record that is read on every single turn, and it makes the version history
    # useless for seeing what actually changed about a farmer.
    if points and (points[0]["payload"].get("profile") or {}) == kept:
        return {"farmer_id": farmer_id, "stored_fields": list(kept),
                "ignored_unknown_fields": unknown, "unchanged": True}

    if points:
        # Keep the old version as history rather than overwriting it: copy it to a
        # fresh id, stamped superseded, then replace the singleton in place.
        old = dict(points[0]["payload"])
        old["superseded_at"] = _now()
        await _qdrant(
            "PUT",
            f"/collections/{COLLECTION}/points?wait=true",
            {"points": [{"id": str(uuid.uuid4()),
                         "vector": {"headline": [0.0] * EMBED_DIM,
                                    "expanded": [0.0] * EMBED_DIM},
                         "payload": old}]},
        )

    summary = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in kept.items())
    # A singleton read by id is never searched. It does not need an embedding.
    summary_vec = [0.0] * EMBED_DIM
    payload = {
        "farmer_id": farmer_id,
        "type": PROFILE_TYPE,
        "status": "n/a",
        "headline": summary,
        "expanded": "",
        "profile": kept,
        "recorded_at": _now(),
        "superseded_at": None,
        "source_ts": rec.source_ts,
        "first_source_ts": rec.source_ts,
        "times_raised": 1,
        "metadata": {},
    }
    await _qdrant(
        "PUT",
        f"/collections/{COLLECTION}/points?wait=true",
        {"points": [{"id": pid,
                     "vector": {"headline": summary_vec, "expanded": summary_vec},
                     "payload": payload}]},
    )
    return {"farmer_id": farmer_id, "stored_fields": list(kept), "ignored_unknown_fields": unknown}


@app.get("/memory/{farmer_id}/profile")
async def get_profile(farmer_id: str) -> dict:
    """The single standing record. Only populated fields come back."""
    if not await _memory_enabled(farmer_id):
        return {"profile": {}, "memory_enabled": False}
    return await _get_profile(farmer_id)


@app.get("/internal/memory/{farmer_id}/profile", include_in_schema=False)
async def _get_profile(farmer_id: str) -> dict:
    got = await _qdrant(
        "POST", f"/collections/{COLLECTION}/points",
        {"ids": [_profile_point_id(farmer_id)], "with_payload": True},
    )
    points = got.get("result") or []
    if not points:
        return {"profile": {}}
    pl = points[0]["payload"]
    visible = {k: v for k, v in (pl.get("profile") or {}).items() if k in PROFILE_FIELDS}
    return {"profile": visible, "summary": "; ".join(f"{k}: {v}" for k, v in visible.items()),
            "schema": list(PROFILE_FIELDS)}


# Trusted background reads are separate from conversation routes. Restrict the
# service (including its write routes) to trusted callers at deployment.
@app.post("/internal/memory/{farmer_id}/search", include_in_schema=False)
async def background_search(farmer_id: str, req: SearchRequest) -> dict:
    return await _search(farmer_id, req, using="headline", respect_setting=False)


@app.post("/internal/memory/{farmer_id}/search_expanded", include_in_schema=False)
async def background_search_expanded(farmer_id: str, req: SearchRequest) -> dict:
    return await _search(farmer_id, req, using="expanded", respect_setting=False)


class ChunkReadRequest(BaseModel):
    whole_chunks: bool = False  # Compatibility for older internal offset readers.
    start_chunk: int = Field(default=1, ge=1)
    chunk_ids: list[int] | None = Field(default=None, max_length=50)
    offset: int = Field(default=0, ge=0)
    max_chars: int = Field(default=2000, ge=1, le=12000)


@app.post("/memory/{farmer_id}/entries/{entry_id}/read")
async def read_episode_chunks(farmer_id: str, entry_id: str, req: ChunkReadRequest):
    if not await _memory_enabled(farmer_id):
        return {"chunks": [], "memory_enabled": False}
    return await _read_episode_chunks(farmer_id, entry_id, req, include_private=False)


@app.post("/internal/memory/{farmer_id}/entries/{entry_id}/read", include_in_schema=False)
async def _read_episode_chunks(farmer_id: str, entry_id: str, req: ChunkReadRequest,
                               include_private: bool = True):
    body = await _read_entry(farmer_id, entry_id, include_private=include_private)
    entry = body["entry"]
    try:
        if req.whole_chunks:
            if req.offset or req.start_chunk != 1:
                raise ValueError("Whole-chunk reading uses chunk_ids, without character offsets")
            chunks, next_ids = episode.whole_chunk_page(entry, req.chunk_ids)
            continuation = {"chunk_ids": next_ids} if next_ids else None
        else:
            chunks, continuation = episode.read_chunks(entry, req.chunk_ids, offset=req.offset, max_chars=req.max_chars, start_chunk=req.start_chunk)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    summary = {k: v for k, v in entry.items() if k not in ("expanded", "chunks")}
    return {"entry": summary, "current": body["current"], "chunks": chunks,
            "continuation": continuation, "returned_chars": sum(len(c["text"]) for c in chunks)}


# ---------- durable worker checkpoints ----------

@app.get("/internal/checkpoints/capabilities", include_in_schema=False)
async def checkpoint_capabilities() -> dict:
    version = await _require_checkpoint_mutations()
    return {"atomic_mutations": True, "qdrant_version": version, "minimum_version": "1.17"}

@app.post("/internal/checkpoints/lookup", include_in_schema=False)
async def checkpoint_lookup(req: checkpoints.CheckpointLookupRequest) -> dict:
    ids = [checkpoints.checkpoint_id(item.kind, item.subject_id) for item in req.subjects]
    result = await _qdrant("POST", f"/collections/{COLLECTION}/points", {
        "ids": ids, "with_payload": True, "with_vector": False,
    })
    found = {point["id"]: checkpoints.public_checkpoint(point) for point in result.get("result") or []}
    return {"checkpoints": [found.get(point_id) for point_id in ids]}


@app.post("/internal/checkpoints/claim", include_in_schema=False)
async def checkpoint_claim(req: checkpoints.CheckpointClaimRequest) -> dict:
    await _require_checkpoint_mutations()
    return await checkpoints.claim_checkpoint(
        _checkpoint_qdrant, COLLECTION, EMBED_DIM, CHECKPOINT_LEASE_SECONDS, req,
    )


@app.post("/internal/checkpoints/renew", include_in_schema=False)
async def checkpoint_renew(req: checkpoints.CheckpointRenewRequest) -> dict:
    await _require_checkpoint_mutations()
    return await checkpoints.renew_checkpoint(
        _checkpoint_qdrant, COLLECTION, EMBED_DIM, CHECKPOINT_LEASE_SECONDS, req,
    )


@app.post("/internal/checkpoints/complete", include_in_schema=False)
async def checkpoint_complete(req: checkpoints.CheckpointCompleteRequest) -> dict:
    await _require_checkpoint_mutations()
    return await checkpoints.complete_checkpoint(_checkpoint_qdrant, COLLECTION, EMBED_DIM, req)


@app.post("/internal/checkpoints/fail", include_in_schema=False)
async def checkpoint_fail(req: checkpoints.CheckpointFailRequest) -> dict:
    await _require_checkpoint_mutations()
    return await checkpoints.fail_checkpoint(_checkpoint_qdrant, COLLECTION, EMBED_DIM, req)


@app.post("/internal/checkpoints/due", include_in_schema=False)
async def checkpoint_due(req: checkpoints.CheckpointDueRequest) -> dict:
    return await checkpoints.due_checkpoints(_qdrant, COLLECTION, req)
