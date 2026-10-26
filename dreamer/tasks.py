"""Telemetry-independent event contract for the memory worker.

Source adapters may change as telemetry changes.  The worker accepts only this
normalized task shape, so source field names cannot leak into extraction, retry,
or storage code.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


TASK_SCHEMA_VERSION = 1
MAX_TASK_BYTES = 256 * 1024
MAX_TURNS = 1
MAX_SESSION_ID_CHARS = 200
MAX_TRACE_ID_CHARS = 200
MAX_SOURCE_CHARS = 64
MAX_QUERY_CHARS = 8_000
MAX_ANSWER_CHARS = 16_000
MAX_CONTENT_CHARS = 128_000
_TASK_FIELDS = {"schema_version", "farmer_id", "session_id", "source", "turns"}
_TURN_FIELDS = {"trace_id", "timestamp", "query", "answer"}
_FARMER_ID = re.compile(r"^[0-9]{10}$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class TaskValidationError(ValueError):
    """A task does not satisfy the normalized worker contract."""

    def __init__(self, code: str):
        # Codes are deliberately bounded and contain no task data.  They are safe
        # to use as metric labels and dead-letter reasons.
        super().__init__(code)
        self.code = code


def _bounded_string(value: Any, code: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskValidationError(code)
    if len(value) > limit:
        raise TaskValidationError(code + "_too_long")
    return value


def _identifier(value: Any, code: str, limit: int) -> str:
    value = _bounded_string(value, code, limit)
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise TaskValidationError(code)
    return value


def _aware_timestamp(value: Any) -> tuple[str, datetime]:
    timestamp = _bounded_string(value, "invalid_turn_timestamp", 64)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TaskValidationError("invalid_turn_timestamp") from exc
    if parsed.tzinfo is None:
        raise TaskValidationError("turn_timestamp_needs_timezone")
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z"), parsed


@dataclass(frozen=True)
class SourceTurn:
    trace_id: str
    timestamp: str
    query: str
    answer: str

    @classmethod
    def from_dict(cls, payload: Any) -> "SourceTurn":
        if not isinstance(payload, Mapping):
            raise TaskValidationError("turn_is_not_an_object")
        if set(payload) != _TURN_FIELDS:
            raise TaskValidationError("invalid_turn_fields")
        timestamp, _ = _aware_timestamp(payload.get("timestamp"))
        return cls(
            trace_id=_identifier(payload.get("trace_id"), "invalid_trace_id", MAX_TRACE_ID_CHARS),
            timestamp=timestamp,
            query=_bounded_string(payload.get("query"), "invalid_turn_query", MAX_QUERY_CHARS),
            answer=_bounded_string(payload.get("answer"), "invalid_turn_answer", MAX_ANSWER_CHARS),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class MemoryTask:
    schema_version: int
    farmer_id: str
    session_id: str
    source: str
    turns: tuple[SourceTurn, ...]

    @classmethod
    def from_dict(cls, payload: Any) -> "MemoryTask":
        if not isinstance(payload, Mapping):
            raise TaskValidationError("task_is_not_an_object")
        try:
            encoded_size = len(json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise TaskValidationError("task_is_not_json_serializable") from exc
        if encoded_size > MAX_TASK_BYTES:
            raise TaskValidationError("task_too_large")
        if set(payload) != _TASK_FIELDS:
            raise TaskValidationError("invalid_task_fields")
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != TASK_SCHEMA_VERSION:
            raise TaskValidationError("unsupported_schema_version")

        raw_turns = payload.get("turns")
        if not isinstance(raw_turns, list) or len(raw_turns) != MAX_TURNS:
            raise TaskValidationError("invalid_turn_count")
        turns = tuple(SourceTurn.from_dict(turn) for turn in raw_turns)
        if sum(len(turn.query) + len(turn.answer) for turn in turns) > MAX_CONTENT_CHARS:
            raise TaskValidationError("task_content_too_large")

        trace_ids = [turn.trace_id for turn in turns]
        if len(trace_ids) != len(set(trace_ids)):
            raise TaskValidationError("duplicate_trace_id")
        parsed_times = [_aware_timestamp(turn.timestamp)[1] for turn in turns]
        if parsed_times != sorted(parsed_times):
            raise TaskValidationError("turns_not_chronological")

        return cls(
            schema_version=TASK_SCHEMA_VERSION,
            farmer_id=cls._farmer_id(payload.get("farmer_id")),
            session_id=_identifier(payload.get("session_id"), "invalid_session_id", MAX_SESSION_ID_CHARS),
            source=_identifier(payload.get("source"), "invalid_source", MAX_SOURCE_CHARS),
            turns=turns,
        )

    @staticmethod
    def _farmer_id(value: Any) -> str:
        if not isinstance(value, str) or not _FARMER_ID.fullmatch(value):
            raise TaskValidationError("invalid_farmer_id")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "farmer_id": self.farmer_id,
            "session_id": self.session_id,
            "source": self.source,
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @property
    def task_key(self) -> str:
        """Opaque idempotency key for this exact normalized task revision."""
        canonical = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def subject_key(self) -> str:
        """Opaque stable key for one immutable source event.

        ``session_id`` is deliberately excluded. Sessions can be reopened weeks
        later; the trace is the ingestion and processing boundary.
        """
        subject = (
            f"{self.farmer_id}\0{self.source}\0{self.turns[0].trace_id}"
        ).encode("utf-8")
        return hashlib.sha256(subject).hexdigest()

    def redacted_summary(self) -> dict[str, Any]:
        """Return task information safe for structured logs and metrics."""
        return {
            "schema_version": self.schema_version,
            "task_key": self.task_key,
            "subject_key": self.subject_key,
            "source_key": hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
            "turn_count": len(self.turns),
        }
