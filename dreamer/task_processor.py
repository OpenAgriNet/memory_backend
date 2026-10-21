"""Process one normalized session task without depending on its telemetry source."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

import requests

if __package__:
    from .queue_worker import PermanentTaskError, RetryableTaskError
    from .tasks import MemoryTask, SourceTurn
else:
    from queue_worker import PermanentTaskError, RetryableTaskError
    from tasks import MemoryTask, SourceTurn

StateOutcome = Literal["started", "completed", "deferred", "needs_review"]
CompletionOutcome = Literal["completed", "no_op"]
FailureDisposition = Literal["retryable", "needs_review"]
BeforeMutation = Callable[[], None]
Runner = Callable[[MemoryTask, Sequence[list[dict[str, str]]], BeforeMutation], int]


@dataclass(frozen=True)
class ProcessedTurn:
    trace_key: str
    fingerprint: str


@dataclass(frozen=True)
class BeginResult:
    outcome: StateOutcome
    processed_turns: tuple[ProcessedTurn, ...] = ()


@dataclass(frozen=True)
class ProcessResult:
    outcome: Literal["completed", "no_op", "already_completed"]
    processed_chunks: int = 0
    writes: int = 0


class TaskStateStore(Protocol):
    def begin(self, task: MemoryTask) -> BeginResult: ...
    def mark_writing(self, task: MemoryTask) -> None: ...
    def complete(
        self, task: MemoryTask, outcome: CompletionOutcome,
        processed_turns: Sequence[ProcessedTurn],
    ) -> None: ...
    def fail(self, task: MemoryTask, disposition: FailureDisposition) -> None: ...


def trace_key(trace_id: str) -> str:
    return hashlib.sha256(trace_id.encode("utf-8")).hexdigest()


def turn_fingerprint(turn: SourceTurn) -> str:
    raw = json.dumps(
        turn.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def processed_turn(turn: SourceTurn) -> ProcessedTurn:
    return ProcessedTurn(trace_key(turn.trace_id), turn_fingerprint(turn))


def _worker_turn(task: MemoryTask, turn: SourceTurn) -> dict[str, str]:
    return {
        "trace_id": turn.trace_id,
        "ts": turn.timestamp,
        "session_id": task.session_id,
        "query": turn.query,
        "answer": turn.answer,
    }


def _memory_enabled(task: MemoryTask) -> bool:
    """Check the authoritative service before any model or memory tool call."""
    if __package__:
        from . import dreamer
    else:
        import dreamer
    response = requests.get(
        f"{dreamer.MEMORY_API}/memory/{task.farmer_id}/settings", timeout=30
    )
    response.raise_for_status()
    body = response.json()
    if type(body.get("memory_enabled")) is not bool:
        raise RuntimeError("invalid_memory_setting")
    return body["memory_enabled"]


def _default_runner(
    task: MemoryTask,
    chunks: Sequence[list[dict[str, str]]],
    before_mutation: BeforeMutation,
) -> int:
    if __package__:
        from . import agent, dreamer
    else:
        import agent
        import dreamer

    writes = 0
    uncertain_mutation = False
    try:
        known_keys = dreamer._known_keys(strict=True)
        for chunk in chunks:
            ctx = agent.dream(
                dreamer.MEMORY_API,
                task.farmer_id,
                chunk,
                source_ts=chunk[-1]["ts"],
                session_id=task.session_id,
                known_keys=known_keys,
                verbose=False,
                before_mutation=before_mutation,
            )
            landed = [
                row for row in ctx.writes
                if row.get("action") in ("write", "update", "fold")
            ]
            uncertain_mutation |= ctx.uncertain_mutation or bool(landed)
            if ctx.error_code:
                raise RuntimeError(ctx.error_code)
            writes += len(landed)
            if landed:
                known_keys = dreamer._known_keys(strict=True)
    except Exception as exc:
        error = RuntimeError("dreamer_runner_failed")
        error.uncertain_mutation = uncertain_mutation  # type: ignore[attr-defined]
        raise error from exc
    return writes


def _session_chunks(turns: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    cap = int(os.environ.get("DREAMER_CHUNK_TURNS", "10"))
    if cap <= 0:
        raise RuntimeError("DREAMER_CHUNK_TURNS must be positive")
    return [turns[index:index + cap] for index in range(0, len(turns), cap)]


def _record_failure(
    state_store: TaskStateStore, task: MemoryTask, disposition: FailureDisposition
) -> None:
    try:
        state_store.fail(task, disposition)
    except Exception:
        pass


def process_task(
    task: MemoryTask,
    state_store: TaskStateStore,
    *,
    runner: Runner | None = None,
    settings_check: Callable[[MemoryTask], bool] | None = None,
) -> ProcessResult:
    """Process only new immutable turns and never repeat an uncertain mutation."""
    try:
        begin = state_store.begin(task)
    except Exception as exc:
        raise RetryableTaskError("task_state_unavailable") from exc

    if begin.outcome == "completed":
        return ProcessResult("already_completed")
    if begin.outcome == "deferred":
        raise RetryableTaskError("task_lease_active")
    if begin.outcome == "needs_review":
        raise PermanentTaskError("task_requires_review")

    current = [processed_turn(turn) for turn in task.turns]
    previous = {item.trace_key: item.fingerprint for item in begin.processed_turns}
    for item in current:
        if item.trace_key in previous and previous[item.trace_key] != item.fingerprint:
            _record_failure(state_store, task, "needs_review")
            raise PermanentTaskError("source_turn_changed")

    all_processed_by_key = {
        item.trace_key: item for item in begin.processed_turns
    }
    all_processed_by_key.update({item.trace_key: item for item in current})
    all_processed = tuple(
        all_processed_by_key[key] for key in sorted(all_processed_by_key)
    )

    pending_keys = {
        item.trace_key for item in current if previous.get(item.trace_key) != item.fingerprint
    }
    pending_turns = [
        _worker_turn(task, turn) for turn in task.turns
        if trace_key(turn.trace_id) in pending_keys
    ]

    check = settings_check or _memory_enabled
    try:
        enabled = check(task)
    except Exception as exc:
        _record_failure(state_store, task, "retryable")
        raise RetryableTaskError("memory_setting_unavailable") from exc
    if not enabled:
        try:
            state_store.complete(task, "no_op", all_processed)
        except Exception as exc:
            _record_failure(state_store, task, "retryable")
            raise RetryableTaskError("task_completion_uncertain") from exc
        return ProcessResult("no_op")

    chunks = _session_chunks(pending_turns)
    if not chunks:
        try:
            state_store.complete(task, "no_op", all_processed)
        except Exception as exc:
            _record_failure(state_store, task, "retryable")
            raise RetryableTaskError("task_completion_uncertain") from exc
        return ProcessResult("no_op")

    run = runner or _default_runner
    try:
        writes = run(task, chunks, lambda: state_store.mark_writing(task))
    except Exception as exc:
        uncertain = bool(getattr(exc, "uncertain_mutation", False))
        disposition: FailureDisposition = "needs_review" if uncertain else "retryable"
        _record_failure(state_store, task, disposition)
        error_type = PermanentTaskError if uncertain else RetryableTaskError
        raise error_type("task_processing_failed") from exc

    outcome: CompletionOutcome = "completed" if writes else "no_op"
    try:
        state_store.complete(task, outcome, all_processed)
    except Exception as exc:
        disposition: FailureDisposition = "needs_review" if writes else "retryable"
        _record_failure(state_store, task, disposition)
        error_type = PermanentTaskError if writes else RetryableTaskError
        raise error_type("task_completion_uncertain") from exc
    return ProcessResult(outcome, processed_chunks=len(chunks), writes=writes)
