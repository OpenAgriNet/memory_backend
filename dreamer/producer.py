"""Discover complete conversation episodes and enqueue schema-v2 memory tasks."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from dotenv import load_dotenv

if __package__:
    from .memory_client import memory_policy
    from .episode_ledger import EpisodeRecord, MemoryEpisodeLedger, RedisEpisodeLedger, SessionRecord, TraceRecord
    from .identity import content_fingerprint, farmer_key, opaque_key, trace_key
    from .lf import CHAT_SCHEMA_VERSION, CHAT_TRACE_NAME, TURN_SCHEMA_VERSION, TURN_TRACE_NAME, LangfuseRO
    from .redis_queue import RedisStreamTaskQueue
    from .tasks import MAX_ANSWER_CHARS, MAX_CONTENT_CHARS, MAX_QUERY_CHARS, MAX_TURNS, MemoryTask, TASK_SCHEMA_VERSION, TaskValidationError
else:
    from memory_client import memory_policy
    from episode_ledger import EpisodeRecord, MemoryEpisodeLedger, RedisEpisodeLedger, SessionRecord, TraceRecord
    from identity import content_fingerprint, farmer_key, opaque_key, trace_key
    from lf import CHAT_SCHEMA_VERSION, CHAT_TRACE_NAME, TURN_SCHEMA_VERSION, TURN_TRACE_NAME, LangfuseRO
    from redis_queue import RedisStreamTaskQueue
    from tasks import MAX_ANSWER_CHARS, MAX_CONTENT_CHARS, MAX_QUERY_CHARS, MAX_TURNS, MemoryTask, TASK_SCHEMA_VERSION, TaskValidationError


TASK_SOURCES = {"chat": "langfuse_chat_episode_v2", "voice": "langfuse_voice_episode_v2"}
_PHONE_DIGITS = re.compile(r"\D+")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


def _int_env(name: str, default: int, *, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def _farmer_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    digits = _PHONE_DIGITS.sub("", value)
    if len(digits) < 10:
        return None
    mobile = digits[-10:]
    return None if len(set(mobile)) == 1 else mobile


@dataclass(frozen=True)
class ProducerConfig:
    inactivity_seconds: int
    scan_seconds: int
    discovery_lookback_seconds: int
    reconcile_seconds: int
    reconcile_horizon_seconds: int
    event_not_before: datetime | None
    dry_run: bool
    allowlist_enabled: bool
    farmer_allowlist: frozenset[str]

    @classmethod
    def from_env(cls) -> "ProducerConfig":
        load_dotenv(os.environ.get("AMUL_MEMORY_ENV_FILE", Path(__file__).resolve().parents[1] / ".env"), override=False)
        inactivity = _int_env("MEMORY_EPISODE_INACTIVITY_SECONDS", 40 * 60, minimum=60)
        scan = _int_env("MEMORY_SCAN_INTERVAL_SECONDS", 30 * 60, minimum=60)
        lookback = _int_env("MEMORY_DISCOVERY_LOOKBACK_SECONDS", 24 * 60 * 60, minimum=scan)
        reconcile = _int_env("MEMORY_SESSION_RECONCILE_INTERVAL_SECONDS", 24 * 60 * 60, minimum=scan)
        horizon = _int_env("MEMORY_SESSION_RECONCILE_HORIZON_SECONDS", 30 * 24 * 60 * 60, minimum=reconcile)
        allowlist: set[str] = set()
        for item in os.environ.get("MEMORY_FARMER_ALLOWLIST", "").split(","):
            if not item.strip():
                continue
            identity = _farmer_id(item.strip())
            if identity is None:
                raise RuntimeError("MEMORY_FARMER_ALLOWLIST contains an invalid phone")
            allowlist.add(identity)
        raw_cutoff = os.environ.get("MEMORY_EVENT_NOT_BEFORE", "").strip()
        cutoff = _timestamp(raw_cutoff) if raw_cutoff else None
        if raw_cutoff and cutoff is None:
            raise RuntimeError("MEMORY_EVENT_NOT_BEFORE must be an ISO-8601 timestamp")
        dry_run = _bool_env("MEMORY_PRODUCER_DRY_RUN", True)
        if not dry_run and cutoff is None:
            raise RuntimeError("MEMORY_EVENT_NOT_BEFORE is required outside dry-run")
        return cls(
            inactivity, scan, lookback, reconcile, horizon, cutoff, dry_run,
            _bool_env("MEMORY_FARMER_ALLOWLIST_ENABLED", True), frozenset(allowlist),
        )


@dataclass(frozen=True)
class ObservedTrace:
    farmer_id: str
    surface: str
    session_id: str
    trace_id: str
    observed_at: datetime
    eligible: bool
    query: str | None
    answer: str | None
    exclusion: str | None

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(
            self.trace_id, _iso(self.observed_at), self.query or "", self.answer or "",
            "eligible" if self.eligible else self.exclusion or "ineligible",
        )


@dataclass(frozen=True)
class ScanResult:
    traces_seen: int
    sessions_seen: int
    episodes_closed: int
    turns_eligible: int
    late_revisions: int
    split_episodes: int
    zero_eligible: int
    restricted: int
    would_publish: int
    published: int
    duplicates: int
    skipped: Mapping[str, int]

    def safe_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "skipped": dict(sorted(self.skipped.items()))}


class EventSource(Protocol):
    def recent(self, start: datetime, stop: datetime) -> list[dict[str, Any]]: ...
    def session(self, session_id: str, start: datetime, stop: datetime) -> list[dict[str, Any]]: ...


class LangfuseEventSource:
    def __init__(self, client: LangfuseRO | None = None):
        self.client = client or LangfuseRO()

    def _rows(self, start: datetime, stop: datetime, **filters: Any) -> list[dict[str, Any]]:
        return [
            *self.client.traces(start, stop, name=CHAT_TRACE_NAME, fields="core,io,scores", **filters),
            *self.client.traces(start, stop, name=TURN_TRACE_NAME, fields="core,io,scores", **filters),
        ]

    def recent(self, start: datetime, stop: datetime) -> list[dict[str, Any]]:
        return self._rows(start, stop)

    def session(self, session_id: str, start: datetime, stop: datetime) -> list[dict[str, Any]]:
        return self._rows(start, stop, sessionId=session_id)


class CurrentMainTelemetryAdapter:
    """Separate trusted session activity from evidence eligible for memory."""

    @staticmethod
    def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
        value = row.get("metadata")
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _input(row: Mapping[str, Any]) -> Mapping[str, Any]:
        value = row.get("input")
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _text(value: Any) -> str | None:
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, Mapping) and isinstance(value.get("text"), str):
            return value["text"].strip() or None
        return None

    @classmethod
    def surface(cls, row: Mapping[str, Any]) -> str | None:
        if row.get("name") == CHAT_TRACE_NAME:
            return "chat"
        if row.get("name") == TURN_TRACE_NAME:
            pipeline = cls._metadata(row).get("amul.pipeline")
            if pipeline == "telephony":
                return "voice"
            if pipeline in {"webapp", "whatsapp"}:
                return "chat"
        return None

    @classmethod
    def trace(cls, row: Mapping[str, Any]) -> tuple[ObservedTrace | None, str]:
        surface = cls.surface(row)
        if surface is None:
            return None, "unknown_surface"
        metadata = cls._metadata(row)
        legacy_chat = row.get("name") == CHAT_TRACE_NAME and metadata.get("amul.schema_version") == CHAT_SCHEMA_VERSION
        current_turn = row.get("name") == TURN_TRACE_NAME and metadata.get("amul.schema_version") == TURN_SCHEMA_VERSION
        if not (legacy_chat or current_turn):
            return None, "schema_mismatch"
        session_id, trace_id = row.get("sessionId"), row.get("id")
        observed_at = _timestamp(row.get("timestamp"))
        if not isinstance(session_id, str) or not session_id:
            return None, "missing_session"
        if not isinstance(trace_id, str) or not trace_id or observed_at is None:
            return None, "invalid_trace"
        if legacy_chat:
            inputs = cls._input(row)
            if metadata.get("persona") != "farmer" or inputs.get("persona") != "farmer":
                return None, "not_farmer"
            identity = _farmer_id(metadata.get("user_id"))
            scores = row.get("scores") if isinstance(row.get("scores"), list) else []
            successful = any(
                isinstance(score, Mapping) and score.get("name") == "turn_outcome"
                and str(score.get("value") or "").lower() == "success"
                for score in scores
            )
            query, answer = cls._text(inputs.get("query")), cls._text(row.get("output"))
        else:
            if metadata.get("amul.service") != "amul-oan-api":
                return None, "wrong_service"
            if metadata.get("amul.persona") != "farmer":
                return None, "not_farmer"
            if surface == "voice" and str(metadata.get("amul.signed_in") or "").lower() != "true":
                return None, "not_signed_in"
            raw_identity = metadata.get("amul.farmer_id")
            identity = _farmer_id(raw_identity)
            if raw_identity != identity:
                return None, "invalid_identity"
            successful = str(metadata.get("amul.outcome") or "").lower() == "success"
            query, answer = cls._text(row.get("input")), cls._text(row.get("output"))
        if identity is None or _farmer_id(row.get("userId")) != identity:
            return None, "invalid_identity"
        if not successful:
            exclusion = "not_successful"
        elif not query or not answer:
            exclusion = "missing_text"
        elif len(query) > MAX_QUERY_CHARS or len(answer) > MAX_ANSWER_CHARS:
            exclusion = "oversized_turn"
        else:
            exclusion = None
        return ObservedTrace(
            identity, surface, session_id, trace_id, observed_at,
            exclusion is None, query, answer, exclusion,
        ), exclusion or "eligible"


def _ledger_key(farmer_id: str, surface: str, session_id: str) -> str:
    return opaque_key("session", farmer_id, surface, session_id)


def _distance(moment: datetime, episode: EpisodeRecord) -> float:
    start, end = _timestamp(episode.started_at), _timestamp(episode.ended_at)
    assert start is not None and end is not None
    if moment < start:
        return (start - moment).total_seconds()
    if moment > end:
        return (moment - end).total_seconds()
    return 0


def _assign(record: SessionRecord, trace: ObservedTrace, inactivity: int) -> EpisodeRecord:
    hashed_trace = trace_key(trace.trace_id)
    for episode in record.episodes:
        if hashed_trace in episode.traces:
            return episode
    candidates = [
        (distance, episode.number, episode)
        for episode in record.episodes
        if (distance := _distance(trace.observed_at, episode)) <= inactivity
    ]
    if candidates:
        episode = min(candidates, key=lambda item: (item[0], item[1]))[2]
    else:
        episode = EpisodeRecord(
            max((item.number for item in record.episodes), default=0) + 1,
            _iso(trace.observed_at), _iso(trace.observed_at),
        )
        record.episodes.append(episode)
    start = min(_timestamp(episode.started_at), trace.observed_at)
    end = max(_timestamp(episode.ended_at), trace.observed_at)
    assert start is not None and end is not None
    episode.started_at, episode.ended_at = _iso(start), _iso(end)
    return episode


def _task_payload(
    representative: ObservedTrace,
    episode: EpisodeRecord,
    revision: int,
    turns: list[ObservedTrace],
    pending: set[str],
    part_number: int,
    part_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "farmer_id": representative.farmer_id,
        "surface": representative.surface,
        "session_id": representative.session_id,
        "source": TASK_SOURCES[representative.surface],
        "episode_number": episode.number,
        "episode_revision": revision,
        "episode_started_at": episode.started_at,
        "episode_ended_at": episode.ended_at,
        "part_number": part_number,
        "part_count": part_count,
        "pending_trace_ids": [item.trace_id for item in turns if item.trace_id in pending],
        "turns": [
            {"trace_id": item.trace_id, "timestamp": _iso(item.observed_at), "query": item.query, "answer": item.answer}
            for item in turns
        ],
    }


def _build_tasks(episode: EpisodeRecord, traces: list[ObservedTrace]) -> tuple[list[MemoryTask], str | None]:
    eligible = sorted((trace for trace in traces if trace.eligible), key=lambda item: (item.observed_at, item.trace_id))
    pending = {
        trace.trace_id for trace in eligible
        if episode.traces[trace_key(trace.trace_id)].published_fingerprint != trace.fingerprint
    }
    if not pending:
        return [], None
    chunks: list[list[ObservedTrace]] = []
    current: list[ObservedTrace] = []
    characters = 0
    for trace in eligible:
        size = len(trace.query or "") + len(trace.answer or "")
        if current and (len(current) >= MAX_TURNS or characters + size > MAX_CONTENT_CHARS):
            chunks.append(current)
            current, characters = [], 0
        current.append(trace)
        characters += size
    if current:
        chunks.append(current)
    while True:
        affected = [chunk for chunk in chunks if any(item.trace_id in pending for item in chunk)]
        tasks: list[MemoryTask] = []
        split_index: int | None = None
        for index, chunk in enumerate(affected, 1):
            try:
                tasks.append(MemoryTask.from_dict(_task_payload(
                    chunk[0], episode, episode.revision + 1, chunk, pending, index, len(affected)
                )))
            except TaskValidationError as exc:
                if exc.code != "task_too_large" or len(chunk) == 1:
                    return [], exc.code
                split_index = chunks.index(chunk)
                break
        if split_index is None:
            return tasks, None
        chunk = chunks[split_index]
        middle = len(chunk) // 2
        chunks[split_index:split_index + 1] = [chunk[:middle], chunk[middle:]]


def _episode_traces(episode: EpisodeRecord, traces: list[ObservedTrace]) -> list[ObservedTrace]:
    keys = set(episode.traces)
    return [trace for trace in traces if trace_key(trace.trace_id) in keys]


def produce_once(
    source: EventSource,
    publisher: Any | None,
    config: ProducerConfig,
    *,
    ledger: Any | None = None,
    now: datetime | None = None,
    adapter: CurrentMainTelemetryAdapter | None = None,
) -> ScanResult:
    now = (now or _utc_now()).astimezone(timezone.utc)
    adapter = adapter or CurrentMainTelemetryAdapter()
    ledger = ledger or MemoryEpisodeLedger()
    cutoff = config.event_not_before or now - timedelta(seconds=config.discovery_lookback_seconds)
    watermark = publisher.watermark() if publisher is not None else None
    scan_start = max(cutoff, watermark - timedelta(seconds=config.scan_seconds) if watermark else cutoff)
    recent = source.recent(scan_start, now)
    skipped: Counter[str] = Counter()
    discovered: set[tuple[str, str, str]] = set()
    for row in recent:
        trace, reason = adapter.trace(row) if isinstance(row, Mapping) else (None, "invalid_trace")
        if trace is None:
            skipped[reason] += 1
        elif cutoff <= trace.observed_at <= now:
            discovered.add((trace.farmer_id, trace.surface, trace.session_id))
        else:
            skipped["outside_event_window"] += 1

    records_by_key = {}
    for key in set(ledger.due(now)) | set(ledger.reconcile_due(now)):
        raw = ledger.snapshot(key)
        if raw is not None:
            records_by_key[key] = SessionRecord.from_json(raw), raw
    for farmer_id, surface, session_id in discovered:
        key = _ledger_key(farmer_id, surface, session_id)
        if key not in records_by_key:
            raw = ledger.snapshot(key)
            record = SessionRecord.from_json(raw) if raw is not None else SessionRecord(surface, session_id)
            records_by_key[key] = record, raw

    sessions_seen = episodes_closed = turns_eligible = late_revisions = split_episodes = 0
    zero_eligible = restricted = would_publish = published = duplicates = 0
    stale_scan = False
    for expected_key, (record, snapshot) in records_by_key.items():
        context_start = min([cutoff, *(_timestamp(episode.started_at) for episode in record.episodes)])
        rows = source.session(record.session_id, context_start, now)
        trusted: list[ObservedTrace] = []
        for row in rows:
            trace, reason = adapter.trace(row) if isinstance(row, Mapping) else (None, "invalid_trace")
            if trace is None:
                skipped[reason] += 1
            elif trace.surface == record.surface and trace.session_id == record.session_id:
                trusted.append(trace)
        identities = {trace.farmer_id for trace in trusted}
        if len(identities) != 1:
            skipped["mixed_or_missing_identity"] += 1
            continue
        farmer_id = identities.pop()
        if _ledger_key(farmer_id, record.surface, record.session_id) != expected_key:
            skipped["ledger_identity_mismatch"] += 1
            continue
        sessions_seen += 1
        if config.allowlist_enabled and farmer_id not in config.farmer_allowlist:
            restricted += 1
            continue
        if not config.dry_run and isinstance(ledger, RedisEpisodeLedger):
            policy = memory_policy(farmer_id)
            if policy.erase_in_progress or not policy.enabled:
                skipped["memory_disabled"] += 1
                continue
            if policy.erase_cutoff is not None:
                trusted = [trace for trace in trusted if trace.observed_at > policy.erase_cutoff]
                if not trusted:
                    continue
        trusted = list({
            trace.trace_id: trace
            for trace in sorted(trusted, key=lambda item: (item.observed_at, item.trace_id))
        }.values())
        known_traces = {key for episode in record.episodes for key in episode.traces}
        dead_letters = []
        for trace in trusted:
            if trace.observed_at < cutoff and trace_key(trace.trace_id) not in known_traces:
                continue
            episode = _assign(record, trace, config.inactivity_seconds)
            hashed = trace_key(trace.trace_id)
            previous = episode.traces.get(hashed)
            episode.traces[hashed] = TraceRecord(
                _iso(trace.observed_at), trace.fingerprint, trace.eligible,
                previous.published_fingerprint if previous else None,
            )
            turns_eligible += int(trace.eligible)
            if trace.exclusion == "oversized_turn" and (
                previous is None or previous.published_fingerprint != trace.fingerprint
            ):
                skipped["oversized_turn"] += 1
                if publisher is not None and not config.dry_run:
                    dead_letters.append(
                        ("oversized_turn", opaque_key("trace", expected_key, trace.trace_id))
                    )
                    episode.traces[hashed].published_fingerprint = trace.fingerprint

        due_at: datetime | None = None
        tasks: list[MemoryTask] = []
        touched: list[tuple[EpisodeRecord, list[ObservedTrace]]] = []
        for episode in sorted(record.episodes, key=lambda item: item.number):
            last_activity = _timestamp(episode.ended_at)
            assert last_activity is not None
            closes_at = last_activity + timedelta(seconds=config.inactivity_seconds)
            if closes_at > now:
                due_at = min(due_at, closes_at) if due_at else closes_at
                continue
            episode_rows = _episode_traces(episode, trusted)
            pending_rows = [
                trace for trace in episode_rows if trace.eligible
                and episode.traces[trace_key(trace.trace_id)].published_fingerprint != trace.fingerprint
            ]
            if not pending_rows:
                if episode.revision == 0:
                    zero_eligible += 1
                continue
            available = {trace_key(trace.trace_id) for trace in episode_rows if trace.eligible}
            if any(
                saved.eligible and saved.published_fingerprint and key not in available
                for key, saved in episode.traces.items()
            ):
                skipped["missing_episode_context"] += 1
                due_at = min(due_at, now) if due_at else now
                continue
            built, error = _build_tasks(episode, episode_rows)
            if error:
                skipped["oversized_turn" if len(pending_rows) == 1 else error] += 1
                if publisher is not None and not config.dry_run:
                    dead_letters.append((error, opaque_key("episode", expected_key, str(episode.number))))
                continue
            episodes_closed += 1
            late_revisions += int(episode.revision > 0)
            split_episodes += int(len(built) > 1)
            would_publish += len(built)
            tasks.extend(built)
            touched.append((episode, pending_rows))

        reconcile_at = now + timedelta(seconds=config.reconcile_seconds)
        if config.dry_run:
            continue
        if publisher is None:
            raise RuntimeError("publisher is required when dry-run is disabled")
        farmer_hash = farmer_key(farmer_id)
        if tasks:
            for episode, pending_rows in touched:
                episode.revision += 1
                for trace in pending_rows:
                    episode.traces[trace_key(trace.trace_id)].published_fingerprint = trace.fingerprint
        else:
            latest = max((_timestamp(item.ended_at) for item in record.episodes), default=now)
            if latest < now - timedelta(seconds=config.reconcile_horizon_seconds):
                reconcile_at = datetime.max.replace(tzinfo=timezone.utc)
        result = ledger.commit(
            expected_key, farmer_hash, record, tasks, publisher, due_at, reconcile_at,
            expected=snapshot, dead_letters=dead_letters,
        )
        if result is None:
            stale_scan = True
            skipped["stale_episode_ledger"] += 1
            continue
        added, repeated = result
        published += added
        duplicates += repeated

    if not stale_scan and not config.dry_run and publisher is not None:
        publisher.advance_watermark(now)
    return ScanResult(
        len(recent), sessions_seen, episodes_closed, turns_eligible, late_revisions,
        split_episodes, zero_eligible, restricted, would_publish, published,
        duplicates, skipped,
    )


def _run_once(config: ProducerConfig) -> ScanResult:
    source = LangfuseEventSource()
    if config.dry_run:
        return produce_once(source, None, config, ledger=MemoryEpisodeLedger())
    queue = RedisStreamTaskQueue()
    return produce_once(source, queue, config, ledger=RedisEpisodeLedger(queue.client))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish completed conversation episodes.")
    parser.add_argument("--forever", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = ProducerConfig.from_env()
    except Exception as exc:
        print(json.dumps({"outcome": "configuration_error", "error": type(exc).__name__}))
        return 2
    while True:
        try:
            print(json.dumps({"outcome": "scan_completed", **_run_once(config).safe_dict()}, sort_keys=True), flush=True)
        except Exception as exc:
            print(json.dumps({"outcome": "scan_failed", "error": type(exc).__name__}, sort_keys=True), flush=True)
            if not args.forever:
                return 1
        if not args.forever:
            return 0
        time.sleep(config.scan_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
