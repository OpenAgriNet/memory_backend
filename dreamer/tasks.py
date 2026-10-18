"""Telemetry-independent input contract for the memory worker.

Source adapters may change as telemetry changes.  The worker accepts only this
normalized task shape, so source field names cannot leak into extraction, retry,
or storage code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


TASK_SCHEMA_VERSION = 1
_TASK_FIELDS = {"schema_version", "farmer_id", "session_id", "source", "turns"}
_TURN_FIELDS = {"trace_id", "timestamp", "query", "answer"}


class TaskValidationError(ValueError):
    """A task does not satisfy the normalized worker contract."""

    def __init__(self, code: str):
        # Codes are deliberately bounded and contain no task data.  They are safe
        # to use as metric labels and dead-letter reasons.
        super().__init__(code)
        self.code = code


def _non_blank_string(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskValidationError(code)
    return value


def _aware_timestamp(value: Any) -> tuple[str, datetime]:
    timestamp = _non_blank_string(value, "invalid_turn_timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TaskValidationError("invalid_turn_timestamp") from exc
    if parsed.tzinfo is None:
        raise TaskValidationError("turn_timestamp_needs_timezone")
    return timestamp, parsed


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
            trace_id=_non_blank_string(payload.get("trace_id"), "invalid_trace_id"),
            timestamp=timestamp,
            query=_non_blank_string(payload.get("query"), "invalid_turn_query"),
            answer=_non_blank_string(payload.get("answer"), "invalid_turn_answer"),
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
        if set(payload) != _TASK_FIELDS:
            raise TaskValidationError("invalid_task_fields")
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != TASK_SCHEMA_VERSION:
            raise TaskValidationError("unsupported_schema_version")

        raw_turns = payload.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise TaskValidationError("invalid_turn_count")
        turns = tuple(SourceTurn.from_dict(turn) for turn in raw_turns)

        trace_ids = [turn.trace_id for turn in turns]
        if len(trace_ids) != len(set(trace_ids)):
            raise TaskValidationError("duplicate_trace_id")
        parsed_times = [_aware_timestamp(turn.timestamp)[1] for turn in turns]
        if parsed_times != sorted(parsed_times):
            raise TaskValidationError("turns_not_chronological")

        return cls(
            schema_version=TASK_SCHEMA_VERSION,
            farmer_id=_non_blank_string(payload.get("farmer_id"), "invalid_farmer_id"),
            session_id=_non_blank_string(payload.get("session_id"), "invalid_session_id"),
            source=_non_blank_string(payload.get("source"), "invalid_source"),
            turns=turns,
        )

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
        """Opaque stable key for one farmer/session, without exposing either."""
        subject = f"{self.farmer_id}\0{self.session_id}".encode("utf-8")
        return hashlib.sha256(subject).hexdigest()

    def redacted_summary(self) -> dict[str, Any]:
        """Return task information safe for structured logs and metrics."""
        return {
            "schema_version": self.schema_version,
            "task_key": self.task_key,
            "subject_key": self.subject_key,
            "source": self.source,
            "turn_count": len(self.turns),
        }
