"""Discover settled chat and voice traces and enqueue normalized memory tasks.

Only ``CurrentMainTelemetryAdapter`` knows the live telemetry field names.  The
rest of the producer and worker operate on ``MemoryTask``.
"""
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


@dataclass(frozen=True)
class ProducerConfig:
    settle_seconds: int
    scan_seconds: int
    discovery_lookback_seconds: int
    event_not_before: datetime | None
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
        settle_seconds = _int_env(
            "MEMORY_EVENT_SETTLE_SECONDS", 40 * 60, minimum=60
        )
        scan_seconds = _int_env(
            "MEMORY_SCAN_INTERVAL_SECONDS", 30 * 60, minimum=60
        )
        discovery_lookback_seconds = _int_env(
            "MEMORY_DISCOVERY_LOOKBACK_SECONDS", 24 * 60 * 60,
            minimum=settle_seconds + scan_seconds,
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
        not_before_raw = os.environ.get("MEMORY_EVENT_NOT_BEFORE", "").strip()
        not_before = _timestamp(not_before_raw) if not_before_raw else None
        if not_before_raw and not_before is None:
            raise RuntimeError("MEMORY_EVENT_NOT_BEFORE must be an ISO-8601 timestamp")
        dry_run = _bool_env("MEMORY_PRODUCER_DRY_RUN", True)
        if not dry_run and not_before is None:
            raise RuntimeError("MEMORY_EVENT_NOT_BEFORE is required outside dry-run")
        return cls(
            settle_seconds=settle_seconds,
            scan_seconds=scan_seconds,
            discovery_lookback_seconds=discovery_lookback_seconds,
            event_not_before=not_before,
            dry_run=dry_run,
            allowlist_enabled=_bool_env(
                "MEMORY_FARMER_ALLOWLIST_ENABLED", True
            ),
            farmer_allowlist=frozenset(allowlist),
        )


@dataclass(frozen=True)
class ScanResult:
    events_seen: int
    eligible: int
    restricted: int
    would_publish: int
    published: int
    duplicates: int
    skipped: Mapping[str, int]
    by_surface: Mapping[str, Mapping[str, int]]

    def safe_dict(self) -> dict[str, Any]:
        return {
            "events_seen": self.events_seen,
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
    def watermark(self) -> datetime | None: ...
    def advance_watermark(self, value: datetime) -> None: ...


class LangfuseEventSource:
    def __init__(self, client: LangfuseRO | None = None):
        self.client = client or LangfuseRO()

    def recent(self, start: datetime, stop: datetime) -> list[dict[str, Any]]:
        return [
            *self.client.traces(start, stop, name=CHAT_TRACE_NAME, fields="core,io,scores"),
            *self.client.traces(start, stop, name=TURN_TRACE_NAME, fields="core,io,scores"),
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

    def event(
        self, row: Mapping[str, Any]
    ) -> tuple[MemoryTask | None, str | None, str | None]:
        surface = self.surface(row)
        if surface is None:
            return None, "unknown_surface", None
        contract = self._contract(row, surface)
        if contract is None:
            return None, "schema_mismatch", surface
        session_id = row.get("sessionId")
        trace_id = row.get("id")
        timestamp = row.get("timestamp")
        if not isinstance(session_id, str) or not session_id:
            return None, "missing_session", surface
        if not isinstance(trace_id, str) or not trace_id or _timestamp(timestamp) is None:
            return None, "invalid_trace", surface
        identity, reason = self._identity(row, surface, contract)
        if identity is None:
            return None, reason or "invalid_identity", surface
        if not self._successful(row, contract):
            return None, "not_successful", surface
        query, answer = self._turn_text(row, contract)
        if not query or not answer:
            return None, "missing_text", surface
        try:
            task = MemoryTask.from_dict({
                "schema_version": TASK_SCHEMA_VERSION,
                "farmer_id": identity,
                "session_id": session_id,
                "source": self.task_sources[surface],
                "turns": [{
                    "trace_id": trace_id,
                    "timestamp": timestamp,
                    "query": query,
                    "answer": answer,
                }],
            })
        except TaskValidationError as exc:
            return None, exc.code, surface
        return task, None, surface


def produce_once(
    source: LangfuseEventSource,
    publisher: TaskPublisher | None,
    config: ProducerConfig,
    *,
    now: datetime | None = None,
    adapter: CurrentMainTelemetryAdapter | None = None,
) -> ScanResult:
    now = (now or _utc_now()).astimezone(timezone.utc)
    adapter = adapter or CurrentMainTelemetryAdapter()
    settled_before = now - timedelta(seconds=config.settle_seconds)
    not_before = config.event_not_before or (
        settled_before - timedelta(seconds=config.discovery_lookback_seconds)
    )
    stored_watermark = publisher.watermark() if publisher is not None else None
    cursor = max(not_before, stored_watermark or not_before)
    scan_start = max(
        not_before,
        cursor - timedelta(seconds=config.discovery_lookback_seconds),
    )
    recent = (
        source.recent(scan_start, settled_before)
        if settled_before >= cursor else []
    )
    skipped: Counter[str] = Counter()
    by_surface: dict[str, Counter[str]] = {
        surface: Counter() for surface in adapter.task_sources
    }
    eligible = restricted = would_publish = published = duplicates = 0
    seen_task_keys: set[str] = set()
    for row in sorted(
        recent,
        key=lambda value: (
            _timestamp(value.get("timestamp"))
            if isinstance(value, Mapping) else None
        ) or datetime.min.replace(tzinfo=timezone.utc),
    ):
        if not isinstance(row, Mapping):
            skipped["invalid_trace"] += 1
            continue
        observed_at = _timestamp(row.get("timestamp"))
        if observed_at is None:
            skipped["invalid_trace"] += 1
            continue
        if observed_at < not_before or observed_at > settled_before:
            skipped["outside_event_window"] += 1
            continue
        task, reason, surface = adapter.event(row)
        if task is None:
            skipped[reason or "invalid_event"] += 1
            if surface is not None:
                by_surface[surface][reason or "invalid_event"] += 1
            continue
        if task.task_key in seen_task_keys:
            duplicates += 1
            by_surface[surface]["duplicates"] += 1
            continue
        seen_task_keys.add(task.task_key)
        eligible += 1
        by_surface[surface]["eligible"] += 1
        if (
            config.allowlist_enabled
            and task.farmer_id not in config.farmer_allowlist
        ):
            restricted += 1
            by_surface[surface]["restricted"] += 1
            continue
        would_publish += 1
        by_surface[surface]["would_publish"] += 1
        if config.dry_run:
            continue
        if publisher is None:
            raise RuntimeError("publisher is required when dry-run is disabled")
        if publisher.publish(task):
            published += 1
            by_surface[surface]["published"] += 1
        else:
            duplicates += 1
            by_surface[surface]["duplicates"] += 1

    if not config.dry_run and settled_before >= cursor:
        if publisher is None:
            raise RuntimeError("publisher is required when dry-run is disabled")
        publisher.advance_watermark(settled_before)

    return ScanResult(
        events_seen=len(recent),
        eligible=eligible,
        restricted=restricted,
        would_publish=would_publish,
        published=published,
        duplicates=duplicates,
        skipped=skipped,
        by_surface=by_surface,
    )


def _run_once(config: ProducerConfig) -> ScanResult:
    source = LangfuseEventSource()
    publisher = None if config.dry_run else RedisStreamTaskQueue()
    return produce_once(source, publisher, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find settled chat and voice traces and enqueue normalized memory tasks."
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
