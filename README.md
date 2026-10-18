# amul-memory

Conversational memory for Amul's Sarlaben assistant. Current local design reviewed
with Gautam on 9 September 2026; no rollout is implied by these source changes.

- `service/`: FastAPI storage and retrieval over Qdrant.
- `dreamer/`: background agent, source selection, write checks, consolidation, and
  standing-record derivation from farmer chat traces.
- `tests/`: offline regression tests plus a separate live-service integration suite.

The reply-agent integration lives in `amul-oan-api` on branch `feat/memory_v0`.
It reads Qdrant directly and uses Marqo for query embeddings. The background job
independently consumes Langfuse and writes Qdrant through its private storage API;
the bot has no runtime dependency on that API or this repository.
Start with [the design](../oan_horizon/future_work/memory_amul/memory-overview.md),
[API contracts](../oan_horizon/future_work/memory_amul/api-and-integration.md), and
[the review ledger](../oan_horizon/future_work/memory_amul/design-review-2026-09-09.md).

## Environment and dev Qdrant

Copy [.env.example](.env.example) to `.env` and set the Qdrant API key and embedding
and model endpoints. `.env` is ignored by git; process environment takes precedence.
`AMUL_MEMORY_ENV_FILE` selects a different environment file. Containers should use
`--env-file` or Compose `env_file`; credentials are not copied into the image.

- `QDRANT_URL`: complete HTTP(S) REST endpoint, including a port only when needed.
- `QDRANT_API_KEY`: sent as `api-key` on Qdrant requests; leave empty only for an
  unauthenticated local instance. It is not sent to the embedding/model services.
- `QDRANT_COLLECTION`: collection used by the memory service and repair commands.
- `QDRANT_TASK_STATE_COLLECTION`: separate hash-only worker progress collection.
  It requires Qdrant 1.17+ and must not be set to `QDRANT_COLLECTION`.
- `MARQO_URL`, `MARQO_INDEX`, `EMBED_DIM`: embedding service and vector configuration.
- `DREAMER_BASE_URL`, `DREAMER_MODEL`: configured OpenAI-compatible model endpoint.
- `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`: the source
  project for background learning. Set all three together; partial configuration
  fails explicitly. There is one active configuration and no credential-file or
  environment-selection fallback. The local demo now selects dev
  Langfuse; historical production backtests remain frozen in their run directories.
- `MEMORY_API_URL`: private storage API called by the background job only. The bot
  uses `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` and its existing
  `MARQO_ENDPOINT_URL`/`MARQO_INDEX_NAME` instead; see the bot's deployment guide.
- `BACKTEST_API_URL`, `BACKTEST_COLLECTION`: isolated evaluation service and collection.

Qdrant REST over HTTPS is sufficient; this implementation does not need gRPC.
Service, repair and snapshot paths use the same environment-sourced API key.
Maintenance commands now use `QDRANT_URL`; the old `QDRANT_HTTP` setting is no
longer their connection setting. Authentication/server errors during the initial
collection check fail startup instead of attempting collection creation.

For a new experiment, endpoints come from the environment. Existing experiment
`config.json` files keep their frozen endpoints as provenance; they are not
silently retargeted to a different database. API keys are never saved in those files.
Use `python3 scripts/backtest_20.py check_qdrant --work <run-directory>` for a
read-only authenticated collection-list check. Run `start`, `train`, `snapshot`
and `reply` with that same directory after preparing its fixtures.

The synthetic diagnostic for the requested existing test farmer is reviewed in
`../amul-memory-backtest/20260909-synthetic-57046/`. Its source conversations are
invented and belong only in its dedicated test collection. They are not real
farmer facts or production Langfuse conversations. Authenticated dev connectivity
and the first 40-answer run completed. A separate replay through the bot's direct
Qdrant reader is in `../amul-memory-backtest/20260909-synthetic-57046-direct/`.

## Reading and writing

Use a small standing record every enabled turn, search relevant summaries and inline contents automatically,
and let the existing reply agent choose `search_memories`, `list_memories`, or
`read_memory` for more evidence. Memories have no category labels. Optional metadata
keys are flexible; exact filtering covers recorded values and can miss untagged entries.

Program teams control standing fields in [profile_fields.json](service/profile_fields.json).
The writer and service load the same configuration (`MEMORY_PROFILE_SCHEMA` can select
another file). The seven current names are optional-value starting fields, not a
mandatory taxonomy. Unknown fields are omitted/reported; bounds are enforced.

Only the administrative settings endpoint writes the separate Qdrant reply-use flag.
The global bot switch defaults off. Per-farmer settings currently default on; explicit
off and failed flag reads suppress reply memory. Background learning can continue via
internal reads without changing that flag, but every farmer-memory write, update,
fold, re-index, and profile update fails closed while memory is disabled or its
setting cannot be checked. The worker checks the same setting before invoking the
model. Keep the service's writes and internal routes restricted to trusted callers;
the service itself does not implement end-user authentication.

Queue tasks are normalized before they reach the worker. `process_task` handles one
session at a time and records only hashed progress in `QDRANT_TASK_STATE_COLLECTION`.
Claims persist a lease owner and expiry. Active leases defer redelivery; an expired
claim can be taken over only when no write tool had started. A crash after write work
started becomes `needs_review`, and replay requires the explicit
`administrative_recover(..., replay_confirmed_safe=True)` path after inspection.
There is deliberately no renewal thread or retry schedule here; the queue owns retry
timing.

Source turns are immutable. Reusing a trace ID with changed content sends the task to
review; it is never checkpointed as a successful correction. Previously processed
turns are not replayed as writable evidence: late arrivals are passed to the model by
themselves in chronological pending-only chunks. `python -m dreamer.worker <task.json>`
is the supported one-task composition for local/integration use; a later producer PR
supplies the real broker adapter and telemetry/session discovery.

The background agent proposes changes through `writer.apply`. Failed or malformed
checks retry once, then block saving and report the proposal for retry. The standing
record uses this boundary too. Source turns are selected per memory, and dates/session
counts are derived by code. Updates preserve omitted detail and save a replacement
before retiring its predecessor. Previous-version links are generated by update code,
not accepted from arbitrary create requests.

## Long episodes and reading budgets

Level 2 contains a summary plus an inline contents page with at most five rows.
Each row describes a range such as “chunks 3–4: registration attempts and missing
receipts.” Contents rows are navigation hints, not category labels or metadata keys.
Code partitions adjacent chunks into at most five groups; the model describes each
group. Code checks coverage and character limits, so the model need not count ranges.

The writer saves the full proposed Level 3 account without an automatic condensation
pass. Code splits it at paragraph/sentence/word boundaries where possible, preserving
every character. Defaults, shared by the writer and service:

| Setting | Default |
|---|---:|
| `MEMORY_CHUNK_MAX_CHARS` | 1,200 characters per stored chunk |
| `MEMORY_SUMMARY_MAX_CHARS` | 350 characters for the Level 2 summary |
| `MEMORY_CONTENTS_ROW_MAX_CHARS` | 120 characters per contents row |
| Contents rows | At most 5 |

Each chunk is separately embedded and searchable. The agent can search all this
farmer's Level 3 passages, optionally restrict to an episode, or directly read chunk
numbers from the contents page. Natural-language queries and specific keyword strings
both use dense semantic retrieval today; there is no separate exact keyword index.
The contents cannot mention every fact, so search is independent of its rows.

Latest simplification approved by Gautam: tools return complete chunks, never
character-truncated pieces. The Amul agent does not choose character budgets, offsets,
or result counts. Search accepts a query and optional episode reference; read accepts
an episode reference and optional chunk numbers. Listing retains optional state,
metadata filters and a next-page cursor. History reading remains optional.

Program defaults are three search chunks, three read chunks per page, eight list
summaries, and six requests per turn. The 12,000-character evidence guard stops
additional output rather than cutting a chunk. Memory tools serialize concurrent
calls so they share that guard. A response suggests the next whole chunk numbers.
Chunk numbers belong to one immutable episode version. Automatic context has a
separate 4,500-character limit and keeps a summary and its contents together when
dropping overflow. Low-level offset API reading remains for compatibility; neither
the reply agent nor the background read tool exposes character offsets.

This preserves the proposed account, not every fact in the source transcript:
extraction and merge proposals can still omit evidence. Earlier stored versions and
source references remain available for audit. Legacy entries get a readable layout
without a hidden write; an explicit internal index route can index their passages.

## Infrastructure and local commands

Qdrant owns the data and two named dense vectors (`headline`, `expanded`). Marqo supplies
embeddings using the existing multilingual model's document/query conventions.
Changing embedding models requires deliberate compatibility checking/re-embedding.

Build the service from its own directory:

```bash
docker build -t amul-memory-api:review ./service
```

Run it only against an intended collection/network. Startup creates missing schema;
it does not rebuild farmer memories. A local demo API serves this source on port
8101; the production bot has not been deployed or enabled.

Background job examples (these read source logs and may invoke a model; non-dry runs
write memory):

```bash
python3 dreamer/dreamer.py --farmer <id> --days 30 --dry-run
python3 dreamer/dreamer.py --farmer <id> --days 30 --until 2026-08-21T00:00:00Z
```

The default is the tool-using agent. `--pipeline` retains the older baseline for
comparison; it is not an automatic fallback. Credentials are resolved at runtime,
never copied into memory or these docs. Langfuse access is read-only.

### Queue worker boundary

New background orchestration must adapt source telemetry into the normalized
`dreamer.tasks.MemoryTask` contract before enqueueing it. The queue worker never
reads Langfuse field names directly. A task contains one farmer/session,
chronological turns, a source label, and a schema version. It derives opaque task
and subject keys for idempotency and safe logging without exposing farmer IDs or
conversation text.

The contract accepts only an authenticated canonical 10-digit farmer mobile. It
does not trim or repair identities and rejects anonymous, path-like, oversized, or
otherwise invalid values. A future telemetry adapter must derive that value only
from authenticated claims, require JWT `user_type=farmer`, and exclude doctor and
anonymous turns. It must never copy request-controlled `user_id`, `sub`, or a
fallback identity into `farmer_id`.

Tasks are bounded before model work: at most 256 KiB of canonical UTF-8 JSON, 100
turns, 8,000 query characters, 16,000 answer characters, and 128,000 aggregate
query/answer characters. Session and trace identifiers are bounded to 200 safe
identifier characters; source labels are allowlisted and bounded to 64. Timestamps
are normalized to UTC before keys are calculated. Log summaries hash the source
label rather than emitting it.

`dreamer.queue_worker.consume_one` defines the broker boundary: receive,
acknowledge, retry, and dead-letter. It acknowledges only after processing
succeeds. The concrete queue provider and session-closing telemetry adapter are
separate work; changing trace names or metadata must not change extraction or
storage code.

Telemetry discovery remains a later producer PR. That PR must inspect the then-live
`amul-oan-api/main` contract rather than assume field names. It owns current
root-turn mapping, farmer-only eligibility, authenticated identity extraction,
session-close policy, periodic scans, the concrete queue/visibility timeout/retry
and DLQ policy, allowlisted dry-run/replay rollout, and zero-eligible/failure alerts.
Per-phone erasure, restricted read-only bot Qdrant credentials, worker image,
schedule, and deployment configuration are also later integration/deployment work;
they are not implemented by this queue boundary.

Delivery is at-least-once. A task can be delivered again after writes succeed but
before acknowledgement, so the future processor must use the task key and source
trace IDs for idempotency. Retry and dead-letter reasons contain only fixed error
codes or exception class names, never exception text or task payloads.

## Verification

Offline, with no service/model calls:

```bash
python3 -m unittest discover -s tests -p 'test_unit.py'
python3 -m unittest discover -s tests -p 'test_review_regressions.py'
python3 -m unittest discover -s tests -p 'test_bot_tools.py'
python3 -m unittest discover -s tests -p 'test_queue_worker.py'
```

`tests/test_service.py` writes synthetic records only when `AMUL_MEMORY_INTEGRATION=1`.
`python3 tests/run_isolated.py` starts disposable Qdrant and fixture-embedding API
instances, runs chunk/HTTP contracts and stops them. It never uses the demo collection.
Ordinary unittest discovery skips service tests unless explicitly enabled.

Mechanical tests do not establish memory usefulness, merge correctness, privacy-check
recall, or live latency. Those require the documented held-out backtest. Scheduling,
durable retries, correction/deletion, voice parity, outreach, and rollout remain
separate work.

See [the one-farmer demo review](../oan_horizon/future_work/memory_amul/demo-review-2026-09-09.md) for measured results, failures and the bot diff review.


## Frozen 20-farmer backtest

`scripts/backtest_20.py` selects a deterministic cohort, freezes each farmer's
training turns and cutoff, starts an isolated service/collection, builds memories,
prepares recorded Amul agent requests, snapshots the database and generates paired
answers. The actions are `select`, `start`, `train`, `prepare`, `snapshot`, `reply`.
The run directory is `../amul-memory-backtest/20260909-20-farmers`; private source
traces, answers and CSVs stay outside git. Existing cohort and completed pairs are
reused, not replaced. Running the workflow requires the configured local services.

`dreamer/dreamer.py --farmer <id> --until <cutoff> --turns-file <frozen.json>`
accepts frozen training input and rejects any turn after the cutoff before writing.
Reply replay uses the recorded full Amul prompt/history and real memory tool
functions with frozen operational tool results; it does not perform live actions
or execute the HTTP/translation pipeline. Both arms use the configured Gemma model.

`scripts/review_backtest.py show <start> <end>` presents blind A/B pairs. Independent
assistant reviews are saved in `reviews.json`; `apply` requires complete pairs and
exports their scores/reasons to `memory_comparison.csv`. It does not generate scores
from answer length, tool use or the same model that wrote the answers.
See [the completed backtest review](../oan_horizon/future_work/memory_amul/backtest-review-2026-09-09.md).


## Retention-gate review and follow-up study

The shared usefulness rule evaluates each statement without a write/skip quota.
It separates unresolved record concerns and personal goals from routine lookups,
and explicitly allows ordinary livestock health while retaining the existing
human-privacy and honesty restrictions. Both agent and pipeline use the rule.

The replay runner accepts `--work <directory>` for a separate experiment. Each
experiment supplies its own frozen cohort, service port and collection. The
`policy` action exercises eight live-model allow/block examples. `gate` reviews
four recorded mixed/routine batches in dry-run mode; no proposals from that action
are stored. `followup_prepare` preserves recorded later-session questions and flags
missing telemetry without substituting questions based on the answer outcomes.
The final follow-up artifacts live in
`../amul-memory-backtest/20260909-followups-v2/`; the initial tuning attempt and the
original 20-farmer baseline remain separate.

`batch_sizes` compares caps of 10, 5 and 3 on the same four target examples,
using their actual chronological batch boundaries. It is a dry run and preserves
raw proposals, refusals and timings. The small study found no recovery gain from
smaller batches; the default remains 10. See the
[follow-up review](../oan_horizon/future_work/memory_amul/followup-review-2026-09-09.md)
for the scored answer CSV, memory inventory, evidence and remaining failures.
