"""Discover closed chat and voice sessions and enqueue normalized memory tasks.

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
    from .lf import (
        CHAT_SCHEMA_VERSION,
        CHAT_TRACE_NAME,
        TURN_SCHEMA_VERSION,
        TURN_TRACE_NAME,
        LangfuseRO,
    )
    from .redis_queue import RedisStreamTaskQueue
    from .tasks import MemoryTask, TASK_SCHEMA_VERSION, TaskValidationError
else:
    from lf import (
        CHAT_SCHEMA_VERSION,
        CHAT_TRACE_NAME,
        TURN_SCHEMA_VERSION,
        TURN_TRACE_NAME,
        LangfuseRO,
    )
    from redis_queue import RedisStreamTaskQueue
    from tasks import MemoryTask, TASK_SCHEMA_VERSION, TaskValidationError


TASK_SOURCE = "langfuse_chat_turn_v1"
VOICE_TASK_SOURCE = "langfuse_voice_turn_v1"
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


def _subject_key(farmer_id: str, session_id: str, surface: str) -> str:
    # Preserve chat's existing keys so widening discovery never republishes it.
    value = f"{farmer_id}\0{session_id}"
    if surface != "chat":
        value += f"\0{surface}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    surface: str
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
    by_surface: Mapping[str, Mapping[str, int]]

    def safe_dict(self) -> dict[str, Any]:
        return {
            "sessions_seen": self.sessions_seen,
            "eligible": self.eligible,
            "restricted": self.restricted,
            "would_publish": self.would_publish,
            "published": self.published,
            "duplicates": self.duplicates,
            "skipped": dict(sorted(self.skipped.items())),
            "by_surface": {
                surface: dict(sorted(values.items()))
                for surface, values in sorted(self.by_surface.items())
            },
        }


class TaskPublisher(Protocol):
    def publish(self, task: MemoryTask) -> bool: ...
    def revision_published(self, subject_key: str, revision_key: str) -> bool: ...
    def mark_revision_published(self, subject_key: str, revision_key: str) -> None: ...


class LangfuseSessionSource:
    def __init__(self, client: LangfuseRO | None = None):
        self.client = client or LangfuseRO()

    def recent(self, start: datetime, stop: datetime) -> list[dict[str, Any]]:
        return [
            *self.client.traces(start, stop, name=CHAT_TRACE_NAME, fields="core,io,scores"),
            *self.client.traces(start, stop, name=TURN_TRACE_NAME, fields="core,io,scores"),
        ]

    def session(
        self, session_id: str, stop: datetime, *, lookback_days: int
    ) -> list[dict[str, Any]]:
        start = stop - timedelta(days=lookback_days)
        return [
            *self.client.traces(
                start, stop, name=CHAT_TRACE_NAME, sessionId=session_id,
                fields="core,io,scores",
            ),
            *self.client.traces(
                start, stop, name=TURN_TRACE_NAME, sessionId=session_id,
                fields="core,io,scores",
            ),
        ]


class CurrentMainTelemetryAdapter:
    """Map current chat plus PR #432's finalized voice telemetry contracts."""

    task_sources = {"chat": TASK_SOURCE, "voice": VOICE_TASK_SOURCE}

    @classmethod
    def surface(cls, row: Mapping[str, Any]) -> str | None:
        name = row.get("name")
        if name == CHAT_TRACE_NAME:
            return "chat"
        if name == TURN_TRACE_NAME:
            metadata = cls._metadata(row)
            pipeline = metadata.get("amul.pipeline")
            if pipeline == "telephony":
                return "voice"
            if pipeline in {"webapp", "whatsapp"}:
                return "chat"
        return None

    @classmethod
    def _contract(cls, row: Mapping[str, Any], surface: str) -> str | None:
        metadata = cls._metadata(row)
        if (surface == "chat" and row.get("name") == CHAT_TRACE_NAME
                and metadata.get("amul.schema_version") == CHAT_SCHEMA_VERSION):
            return "chat_current"
        if (row.get("name") == TURN_TRACE_NAME
                and metadata.get("amul.schema_version") == TURN_SCHEMA_VERSION
                and cls.surface(row) == surface):
            return "turn_v1"
        return None

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
        if isinstance(value, Mapping):
            text = value.get("text")
            return text.strip() if isinstance(text, str) and text.strip() else None
        return None

    @staticmethod
    def _chat_successful(row: Mapping[str, Any]) -> bool:
        scores = row.get("scores")
        if not isinstance(scores, list):
            return False
        return any(
            isinstance(score, Mapping)
            and score.get("name") == "turn_outcome"
            and str(score.get("value") or "").lower() == "success"
            for score in scores
        )

    @classmethod
    def _successful(cls, row: Mapping[str, Any], contract: str) -> bool:
        if contract == "chat_current":
            return cls._chat_successful(row)
        metadata = cls._metadata(row)
        return str(metadata.get("amul.outcome") or "").lower() == "success"

    @classmethod
    def _identity(
        cls, row: Mapping[str, Any], surface: str, contract: str
    ) -> tuple[str | None, str | None]:
        metadata = cls._metadata(row)
        if contract == "chat_current":
            inputs = cls._input(row)
            if metadata.get("persona") != "farmer" or inputs.get("persona") != "farmer":
                return None, "not_farmer"
            identity = _farmer_id(metadata.get("user_id"))
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
        top_identity = _farmer_id(row.get("userId"))
        if identity is None or top_identity != identity:
            return None, "invalid_identity"
        return identity, None

    @classmethod
    def _turn_text(cls, row: Mapping[str, Any], contract: str) -> tuple[str | None, str | None]:
        if contract == "chat_current":
            return cls._text(cls._input(row).get("query")), cls._text(row.get("output"))
        return cls._text(row.get("input")), cls._text(row.get("output"))

    def candidate(
        self, session_id: str, rows: Iterable[Mapping[str, Any]], surface: str = "chat"
    ) -> tuple[Candidate | None, str | None]:
        identities: set[str] = set()
        latest: tuple[datetime, str] | None = None
        found = False
        found_success = False
        for row in rows:
            if row.get("sessionId") != session_id or self.surface(row) != surface:
                continue
            found = True
            contract = self._contract(row, surface)
            if contract is None:
                return None, "schema_mismatch"
            identity, reason = self._identity(row, surface, contract)
            if identity is None:
                return None, reason
            identities.add(identity)
            trace_id = row.get("id")
            observed_at = _timestamp(row.get("timestamp"))
            if not isinstance(trace_id, str) or not trace_id or observed_at is None:
                return None, "invalid_trace"
            if latest is None or observed_at > latest[0]:
                latest = observed_at, trace_id
            found_success |= self._successful(row, contract)
        if not found or latest is None:
            return None, "empty_session"
        if len(identities) != 1:
            return None, "mixed_identity"
        if not found_success:
            return None, "no_successful_turn"
        return Candidate(
            session_id=session_id,
            farmer_id=next(iter(identities)),
            surface=surface,
            latest_at=latest[0],
            latest_trace_id=latest[1],
        ), None

    def tasks(
        self, candidate: Candidate, rows: Iterable[Mapping[str, Any]]
    ) -> tuple[MemoryTask, ...]:
        turns: list[dict[str, str]] = []
        identities: set[str] = set()
        seen_trace_ids: set[str] = set()
        task_source = self.task_sources[candidate.surface]
        for row in rows:
            if row.get("sessionId") != candidate.session_id or self.surface(row) != candidate.surface:
                continue
            contract = self._contract(row, candidate.surface)
            if contract is None:
                raise TaskValidationError("schema_mismatch")
            identity, reason = self._identity(row, candidate.surface, contract)
            if identity is None:
                raise TaskValidationError(reason or "invalid_identity")
            identities.add(identity)
            if not self._successful(row, contract):
                continue
            query, answer = self._turn_text(row, contract)
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
                "source": task_source,
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
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    skipped: Counter[str] = Counter()
    by_surface: dict[str, Counter[str]] = {
        surface: Counter() for surface in adapter.task_sources
    }
    for row in recent:
        if not isinstance(row, Mapping):
            skipped["invalid_trace"] += 1
            continue
        session_id = row.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            skipped["missing_session"] += 1
            continue
        surface = adapter.surface(row)
        if surface is None:
            skipped["unknown_surface"] += 1
            continue
        grouped.setdefault((session_id, surface), []).append(row)

    eligible = restricted = would_publish = published = duplicates = 0
    close_before = now - timedelta(seconds=config.close_seconds)
    for (session_id, surface), rows in grouped.items():
        candidate, reason = adapter.candidate(session_id, rows, surface)
        if candidate is None:
            skipped[reason or "invalid_session"] += 1
            by_surface[surface][reason or "invalid_session"] += 1
            continue
        if candidate.latest_at > close_before:
            skipped["session_open"] += 1
            by_surface[surface]["session_open"] += 1
            continue
        eligible += 1
        by_surface[surface]["eligible"] += 1
        if (
            config.allowlist_enabled
            and candidate.farmer_id not in config.farmer_allowlist
        ):
            restricted += 1
            by_surface[surface]["restricted"] += 1
            continue

        revision_key = RedisStreamTaskQueue.revision_key(candidate.latest_trace_id)
        subject_key = _subject_key(candidate.farmer_id, candidate.session_id, candidate.surface)
        if publisher is not None and publisher.revision_published(subject_key, revision_key):
            duplicates += 1
            by_surface[surface]["duplicates"] += 1
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
            by_surface[surface][exc.code] += 1
            continue
        would_publish += len(tasks)
        by_surface[surface]["would_publish"] += len(tasks)
        if config.dry_run:
            continue
        if publisher is None:
            raise RuntimeError("publisher is required when dry-run is disabled")
        for task in tasks:
            if publisher.publish(task):
                published += 1
                by_surface[surface]["published"] += 1
            else:
                duplicates += 1
                by_surface[surface]["duplicates"] += 1
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
        by_surface=by_surface,
    )


def _run_once(config: ProducerConfig) -> ScanResult:
    source = LangfuseSessionSource()
    publisher = None if config.dry_run else RedisStreamTaskQueue()
    return produce_once(source, publisher, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find closed chat and voice sessions and enqueue normalized memory tasks."
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
