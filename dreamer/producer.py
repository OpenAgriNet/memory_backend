"""Discover closed chat sessions and enqueue normalized memory tasks.

Only ``CurrentMainTelemetryAdapter`` knows the live telemetry field names.  The
rest of the producer and worker operate on ``MemoryTask``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from dotenv import load_dotenv

if __package__:
    from .lf import CHAT_SCHEMA_VERSION, CHAT_TRACE_NAME, LangfuseRO
    from .redis_queue import RedisStreamTaskQueue
    from .tasks import MemoryTask, TASK_SCHEMA_VERSION, TaskValidationError
else:
    from lf import CHAT_SCHEMA_VERSION, CHAT_TRACE_NAME, LangfuseRO
    from redis_queue import RedisStreamTaskQueue
    from tasks import MemoryTask, TASK_SCHEMA_VERSION, TaskValidationError


TASK_SOURCE = "langfuse_chat_turn_v1"
_PHONE_DIGITS = re.compile(r"\D+")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
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


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _farmer_id(value: Any) -> str | None:
    """Mirror current main's phone normalization: keep the last ten digits."""
    if not isinstance(value, str):
        return None
    digits = _PHONE_DIGITS.sub("", value)
    if len(digits) < 10:
        return None
    mobile = digits[-10:]
    if len(set(mobile)) == 1:
        return None
    return mobile


def _subject_key(farmer_id: str, session_id: str) -> str:
    return hashlib.sha256(f"{farmer_id}\0{session_id}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProducerConfig:
    close_seconds: int
    scan_seconds: int
    discovery_lookback_seconds: int
    session_lookback_days: int
    dry_run: bool
    allowlist_enabled: bool
    farmer_allowlist: frozenset[str]

    @classmethod
    def from_env(cls) -> "ProducerConfig":
        load_dotenv(
            os.environ.get(
                "AMUL_MEMORY_ENV_FILE", Path(__file__).resolve().parents[1] / ".env"
            ),
            override=False,
        )
        close_seconds = _int_env(
            "MEMORY_SESSION_CLOSE_SECONDS", 40 * 60, minimum=60
        )
        scan_seconds = _int_env(
            "MEMORY_SCAN_INTERVAL_SECONDS", 30 * 60, minimum=60
        )
        discovery_lookback_seconds = _int_env(
            "MEMORY_DISCOVERY_LOOKBACK_SECONDS", 24 * 60 * 60,
            minimum=close_seconds + scan_seconds,
        )
        raw_allowlist = os.environ.get("MEMORY_FARMER_ALLOWLIST", "")
        allowlist: set[str] = set()
        for item in raw_allowlist.split(","):
            if not item.strip():
                continue
            normalized = _farmer_id(item.strip())
            if normalized is None:
                raise RuntimeError("MEMORY_FARMER_ALLOWLIST contains an invalid phone")
            allowlist.add(normalized)
        return cls(
            close_seconds=close_seconds,
            scan_seconds=scan_seconds,
            discovery_lookback_seconds=discovery_lookback_seconds,
            session_lookback_days=_int_env(
                "MEMORY_SESSION_LOOKBACK_DAYS", 30, minimum=1
            ),
            dry_run=_bool_env("MEMORY_PRODUCER_DRY_RUN", True),
            allowlist_enabled=_bool_env(
                "MEMORY_FARMER_ALLOWLIST_ENABLED", True
            ),
            farmer_allowlist=frozenset(allowlist),
        )


@dataclass(frozen=True)
class Candidate:
    session_id: str
    farmer_id: str
    latest_at: datetime
    latest_trace_id: str


@dataclass(frozen=True)
class ScanResult:
    sessions_seen: int
    eligible: int
    restricted: int
    would_publish: int
    published: int
    duplicates: int
    skipped: Mapping[str, int]

    def safe_dict(self) -> dict[str, Any]:
        return {
            "sessions_seen": self.sessions_seen,
            "eligible": self.eligible,
            "restricted": self.restricted,
            "would_publish": self.would_publish,
            "published": self.published,
            "duplicates": self.duplicates,
            "skipped": dict(sorted(self.skipped.items())),
        }


class TaskPublisher(Protocol):
    def publish(self, task: MemoryTask) -> bool: ...
    def revision_published(self, subject_key: str, revision_key: str) -> bool: ...
    def mark_revision_published(self, subject_key: str, revision_key: str) -> None: ...


class LangfuseSessionSource:
    def __init__(self, client: LangfuseRO | None = None):
        self.client = client or LangfuseRO()

    def recent(self, start: datetime, stop: datetime) -> list[dict[str, Any]]:
        return self.client.traces(start, stop, fields="core,io,scores")

    def session(
        self, session_id: str, stop: datetime, *, lookback_days: int
    ) -> list[dict[str, Any]]:
        return self.client.traces(
            stop - timedelta(days=lookback_days),
            stop,
            sessionId=session_id,
            fields="core,io,scores",
        )


class CurrentMainTelemetryAdapter:
    """Map the telemetry contract currently released on amul-oan-api/main."""

    root_name = CHAT_TRACE_NAME
    schema_version = CHAT_SCHEMA_VERSION

    @staticmethod
    def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
        value = row.get("metadata")
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _input(row: Mapping[str, Any]) -> Mapping[str, Any]:
        value = row.get("input")
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _successful(row: Mapping[str, Any]) -> bool:
        scores = row.get("scores")
        if not isinstance(scores, list):
            return False
        return any(
            isinstance(score, Mapping)
            and score.get("name") == "turn_outcome"
            and str(score.get("value") or "").lower() == "success"
            for score in scores
        )

    def candidate(
        self, session_id: str, rows: Iterable[Mapping[str, Any]]
    ) -> tuple[Candidate | None, str | None]:
        identities: set[str] = set()
        latest: tuple[datetime, str] | None = None
        found = False
        found_success = False
        for row in rows:
            if row.get("sessionId") != session_id:
                continue
            found = True
            metadata = self._metadata(row)
            inputs = self._input(row)
            if metadata.get("amul.schema_version") != self.schema_version:
                return None, "schema_mismatch"
            if metadata.get("persona") != "farmer" or inputs.get("persona") != "farmer":
                return None, "not_farmer"
            identity = _farmer_id(metadata.get("user_id"))
            top_identity = _farmer_id(row.get("userId"))
            if identity is None or top_identity != identity:
                return None, "invalid_identity"
            identities.add(identity)
            trace_id = row.get("id")
            observed_at = _timestamp(row.get("timestamp"))
            if not isinstance(trace_id, str) or not trace_id or observed_at is None:
                return None, "invalid_trace"
            if latest is None or observed_at > latest[0]:
                latest = observed_at, trace_id
            found_success |= self._successful(row)
        if not found or latest is None:
            return None, "empty_session"
        if len(identities) != 1:
            return None, "mixed_identity"
        if not found_success:
            return None, "no_successful_turn"
        return Candidate(
            session_id=session_id,
            farmer_id=next(iter(identities)),
            latest_at=latest[0],
            latest_trace_id=latest[1],
        ), None

    def tasks(
        self, candidate: Candidate, rows: Iterable[Mapping[str, Any]]
    ) -> tuple[MemoryTask, ...]:
        turns: list[dict[str, str]] = []
        identities: set[str] = set()
        seen_trace_ids: set[str] = set()
        for row in rows:
            if row.get("sessionId") != candidate.session_id:
                continue
            metadata = self._metadata(row)
            inputs = self._input(row)
            if metadata.get("amul.schema_version") != self.schema_version:
                raise TaskValidationError("schema_mismatch")
            if metadata.get("persona") != "farmer" or inputs.get("persona") != "farmer":
                raise TaskValidationError("not_farmer")
            identity = _farmer_id(metadata.get("user_id"))
            top_identity = _farmer_id(row.get("userId"))
            if identity is None or top_identity != identity:
                raise TaskValidationError("invalid_identity")
            identities.add(identity)
            if not self._successful(row):
                continue
            query = inputs.get("query")
            answer = row.get("output")
            trace_id = row.get("id")
            timestamp = row.get("timestamp")
            # An error/cancelled trace normally has no final assistant output.
            # It still delayed closure above, but it is not writable evidence.
            if not all(isinstance(value, str) and value.strip() for value in (
                trace_id, timestamp, query, answer,
            )):
                continue
            if trace_id in seen_trace_ids:
                continue
            seen_trace_ids.add(trace_id)
            turns.append({
                "trace_id": trace_id,
                "timestamp": timestamp,
                "query": query,
                "answer": answer,
            })
        if identities != {candidate.farmer_id}:
            raise TaskValidationError("mixed_identity")
        turns.sort(key=lambda row: _timestamp(row["timestamp"]) or datetime.min.replace(
            tzinfo=timezone.utc
        ))

        def build(part: list[dict[str, str]]) -> MemoryTask:
            return MemoryTask.from_dict({
                "schema_version": TASK_SCHEMA_VERSION,
                "farmer_id": candidate.farmer_id,
                "session_id": candidate.session_id,
                "source": TASK_SOURCE,
                "turns": part,
            })

        parts: list[MemoryTask] = []
        current: list[dict[str, str]] = []
        splittable = {"invalid_turn_count", "task_content_too_large", "task_too_large"}
        for turn in turns:
            try:
                build([*current, turn])
            except TaskValidationError as exc:
                if not current or exc.code not in splittable:
                    raise
                parts.append(build(current))
                current = [turn]
                build(current)  # A single oversize turn remains invalid.
            else:
                current.append(turn)
        if current:
            parts.append(build(current))
        if not parts:
            raise TaskValidationError("invalid_turn_count")
        return tuple(parts)


def produce_once(
    source: LangfuseSessionSource,
    publisher: TaskPublisher | None,
    config: ProducerConfig,
    *,
    now: datetime | None = None,
    adapter: CurrentMainTelemetryAdapter | None = None,
) -> ScanResult:
    now = (now or _utc_now()).astimezone(timezone.utc)
    adapter = adapter or CurrentMainTelemetryAdapter()
    recent = source.recent(
        now - timedelta(seconds=config.discovery_lookback_seconds), now
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    skipped: Counter[str] = Counter()
    for row in recent:
        if not isinstance(row, Mapping):
            skipped["invalid_trace"] += 1
            continue
        session_id = row.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            skipped["missing_session"] += 1
            continue
        grouped.setdefault(session_id, []).append(row)

    eligible = restricted = would_publish = published = duplicates = 0
    close_before = now - timedelta(seconds=config.close_seconds)
    for session_id, rows in grouped.items():
        candidate, reason = adapter.candidate(session_id, rows)
        if candidate is None:
            skipped[reason or "invalid_session"] += 1
            continue
        if candidate.latest_at > close_before:
            skipped["session_open"] += 1
            continue
        eligible += 1
        if (
            config.allowlist_enabled
            and candidate.farmer_id not in config.farmer_allowlist
        ):
            restricted += 1
            continue

        revision_key = RedisStreamTaskQueue.revision_key(candidate.latest_trace_id)
        subject_key = _subject_key(candidate.farmer_id, candidate.session_id)
        if publisher is not None and publisher.revision_published(subject_key, revision_key):
            duplicates += 1
            continue
        try:
            session_rows = source.session(
                candidate.session_id,
                now,
                lookback_days=config.session_lookback_days,
            )
            tasks = adapter.tasks(candidate, session_rows)
        except TaskValidationError as exc:
            skipped[exc.code] += 1
            continue
        would_publish += len(tasks)
        if config.dry_run:
            continue
        if publisher is None:
            raise RuntimeError("publisher is required when dry-run is disabled")
        for task in tasks:
            if publisher.publish(task):
                published += 1
            else:
                duplicates += 1
        # Only mark the source revision after every fragment is durably present
        # or was already published. A partial failure is retried safely.
        publisher.mark_revision_published(subject_key, revision_key)

    return ScanResult(
        sessions_seen=len(grouped),
        eligible=eligible,
        restricted=restricted,
        would_publish=would_publish,
        published=published,
        duplicates=duplicates,
        skipped=skipped,
    )


def _run_once(config: ProducerConfig) -> ScanResult:
    source = LangfuseSessionSource()
    publisher = None if config.dry_run else RedisStreamTaskQueue()
    return produce_once(source, publisher, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find closed chat sessions and enqueue normalized memory tasks."
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="repeat every MEMORY_SCAN_INTERVAL_SECONDS; default is one scan",
    )
    args = parser.parse_args(argv)
    try:
        config = ProducerConfig.from_env()
    except Exception as exc:
        print(json.dumps({"outcome": "configuration_error", "error": type(exc).__name__}))
        return 2

    while True:
        try:
            result = _run_once(config)
            print(json.dumps(
                {"outcome": "scan_completed", **result.safe_dict()}, sort_keys=True
            ), flush=True)
        except Exception as exc:
            print(json.dumps(
                {"outcome": "scan_failed", "error": type(exc).__name__}, sort_keys=True
            ), flush=True)
            if not args.forever:
                return 1
        if not args.forever:
            return 0
        time.sleep(config.scan_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
