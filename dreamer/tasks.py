"""Telemetry-independent conversation-episode contract for the memory worker.

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


TASK_SCHEMA_VERSION = 2
MAX_TASK_BYTES = 256 * 1024
MAX_TURNS = 100
MAX_SESSION_ID_CHARS = 200
MAX_TRACE_ID_CHARS = 200
MAX_SOURCE_CHARS = 64
MAX_QUERY_CHARS = 8_000
MAX_ANSWER_CHARS = 16_000
MAX_CONTENT_CHARS = 128_000
MAX_SEQUENCE_NUMBER = 2_147_483_647
_TASK_FIELDS = {
    "schema_version",
    "farmer_id",
    "surface",
    "session_id",
    "source",
    "episode_number",
    "episode_revision",
    "episode_started_at",
    "episode_ended_at",
    "part_number",
    "part_count",
    "pending_trace_ids",
    "turns",
}
_TURN_FIELDS = {"trace_id", "timestamp", "query", "answer"}
_FARMER_ID = re.compile(r"^[0-9]{10}$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SURFACES = {"chat", "voice"}


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


def _positive_int(value: Any, code: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_SEQUENCE_NUMBER:
        raise TaskValidationError(code)
    return value


def _aware_timestamp(
    value: Any,
    code: str = "invalid_turn_timestamp",
) -> tuple[str, datetime]:
    timestamp = _bounded_string(value, code, 64)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TaskValidationError(code) from exc
    if parsed.tzinfo is None:
        timezone_code = (
            "turn_timestamp_needs_timezone"
            if code == "invalid_turn_timestamp"
            else f"{code}_needs_timezone"
        )
        raise TaskValidationError(timezone_code)
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
    surface: str
    session_id: str
    source: str
    episode_number: int
    episode_revision: int
    episode_started_at: str
    episode_ended_at: str
    part_number: int
    part_count: int
    pending_trace_ids: tuple[str, ...]
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

        surface = payload.get("surface")
        if surface not in _SURFACES:
            raise TaskValidationError("invalid_surface")
        episode_number = _positive_int(
            payload.get("episode_number"), "invalid_episode_number"
        )
        episode_revision = _positive_int(
            payload.get("episode_revision"), "invalid_episode_revision"
        )
        part_number = _positive_int(payload.get("part_number"), "invalid_part_number")
        part_count = _positive_int(payload.get("part_count"), "invalid_part_count")
        if part_number > part_count:
            raise TaskValidationError("invalid_part_range")

        episode_started_at, episode_start = _aware_timestamp(
            payload.get("episode_started_at"), "invalid_episode_started_at"
        )
        episode_ended_at, episode_end = _aware_timestamp(
            payload.get("episode_ended_at"), "invalid_episode_ended_at"
        )
        if episode_start > episode_end:
            raise TaskValidationError("invalid_episode_range")

        raw_turns = payload.get("turns")
        if not isinstance(raw_turns, list) or not 1 <= len(raw_turns) <= MAX_TURNS:
            raise TaskValidationError("invalid_turn_count")
        turns = tuple(SourceTurn.from_dict(turn) for turn in raw_turns)
        if sum(len(turn.query) + len(turn.answer) for turn in turns) > MAX_CONTENT_CHARS:
            raise TaskValidationError("task_content_too_large")

        trace_ids = [turn.trace_id for turn in turns]
        if len(trace_ids) != len(set(trace_ids)):
            raise TaskValidationError("duplicate_trace_id")
        turn_order = [(_aware_timestamp(turn.timestamp)[1], turn.trace_id) for turn in turns]
        if turn_order != sorted(turn_order):
            raise TaskValidationError("turns_not_chronological")
        if any(not episode_start <= timestamp <= episode_end for timestamp, _ in turn_order):
            raise TaskValidationError("turn_outside_episode")

        raw_pending = payload.get("pending_trace_ids")
        if not isinstance(raw_pending, list) or not raw_pending:
            raise TaskValidationError("invalid_pending_trace_ids")
        if any(not isinstance(trace_id, str) for trace_id in raw_pending):
            raise TaskValidationError("invalid_pending_trace_ids")
        if len(raw_pending) != len(set(raw_pending)):
            raise TaskValidationError("duplicate_pending_trace_id")
        pending_set = set(raw_pending)
        if not pending_set.issubset(trace_ids):
            raise TaskValidationError("unknown_pending_trace_id")
        pending_trace_ids = tuple(
            turn.trace_id for turn in turns if turn.trace_id in pending_set
        )

        return cls(
            schema_version=TASK_SCHEMA_VERSION,
            farmer_id=cls._farmer_id(payload.get("farmer_id")),
            surface=surface,
            session_id=_identifier(payload.get("session_id"), "invalid_session_id", MAX_SESSION_ID_CHARS),
            source=_identifier(payload.get("source"), "invalid_source", MAX_SOURCE_CHARS),
            episode_number=episode_number,
            episode_revision=episode_revision,
            episode_started_at=episode_started_at,
            episode_ended_at=episode_ended_at,
            part_number=part_number,
            part_count=part_count,
            pending_trace_ids=pending_trace_ids,
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
            "surface": self.surface,
            "session_id": self.session_id,
            "source": self.source,
            "episode_number": self.episode_number,
            "episode_revision": self.episode_revision,
            "episode_started_at": self.episode_started_at,
            "episode_ended_at": self.episode_ended_at,
            "part_number": self.part_number,
            "part_count": self.part_count,
            "pending_trace_ids": list(self.pending_trace_ids),
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
        """Opaque stable key for one numbered episode across revisions and parts."""
        subject = (
            f"{self.farmer_id}\0{self.surface}\0{self.session_id}\0{self.episode_number}"
        ).encode("utf-8")
        return hashlib.sha256(subject).hexdigest()

    def redacted_summary(self) -> dict[str, Any]:
        """Return task information safe for structured logs and metrics."""
        return {
            "schema_version": self.schema_version,
            "task_key": self.task_key,
            "subject_key": self.subject_key,
            "source_key": hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
            "surface": self.surface,
            "episode_number": self.episode_number,
            "episode_revision": self.episode_revision,
            "part_number": self.part_number,
            "part_count": self.part_count,
            "turn_count": len(self.turns),
            "pending_turn_count": len(self.pending_trace_ids),
        }
