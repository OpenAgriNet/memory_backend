"""Process one normalized session task without depending on its telemetry source."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

if __package__:
    from .queue_worker import PermanentTaskError, RetryableTaskError
    from .tasks import MemoryTask, SourceTurn
else:
    from queue_worker import PermanentTaskError, RetryableTaskError
    from tasks import MemoryTask, SourceTurn

StateOutcome = Literal["started", "completed", "processing", "needs_review"]
CompletionOutcome = Literal["completed", "no_op"]
FailureDisposition = Literal["retryable", "needs_review"]


@dataclass(frozen=True)
class BeginResult:
    outcome: StateOutcome
    processed_turns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessResult:
    outcome: Literal["completed", "no_op", "already_completed"]
    processed_chunks: int = 0
    writes: int = 0


class TaskStateStore(Protocol):
    def begin(self, task: MemoryTask) -> BeginResult: ...
    def complete(
        self, task: MemoryTask, outcome: CompletionOutcome, processed_turns: Sequence[str]
    ) -> None: ...
    def fail(self, task: MemoryTask, disposition: FailureDisposition) -> None: ...


def turn_fingerprint(turn: SourceTurn) -> str:
    """Hash a complete normalized turn so edits are processed like late turns."""
    raw = json.dumps(
        turn.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _worker_turn(task: MemoryTask, turn: SourceTurn) -> dict[str, str]:
    return {
        "trace_id": turn.trace_id,
        "ts": turn.timestamp,
        "session_id": task.session_id,
        "query": turn.query,
        "answer": turn.answer,
    }


def _default_runner(task: MemoryTask, chunks: Sequence[list[dict[str, str]]]) -> tuple[int, bool]:
    if __package__:
        from . import agent, dreamer
    else:
        import agent
        import dreamer

    writes = 0
    mutation_attempted = False
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
            )
            mutation_attempted |= ctx.mutation_attempted
            if ctx.error_code:
                raise RuntimeError(ctx.error_code)
            landed = [
                row for row in ctx.writes
                if row.get("action") in ("write", "update", "fold")
            ]
            writes += len(landed)
            if landed:
                known_keys = dreamer._known_keys(strict=True)
    except Exception as exc:
        error = RuntimeError("dreamer_runner_failed")
        error.mutation_attempted = mutation_attempted  # type: ignore[attr-defined]
        raise error from exc
    return writes, mutation_attempted


def _session_chunks(turns: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """The existing chunking rule, simplified for one normalized session task."""
    cap = int(os.environ.get("DREAMER_CHUNK_TURNS", "10"))
    if cap <= 0:
        raise RuntimeError("DREAMER_CHUNK_TURNS must be positive")
    return [turns[index:index + cap] for index in range(0, len(turns), cap)]


def _record_failure(state_store: TaskStateStore, task: MemoryTask, disposition: FailureDisposition) -> None:
    try:
        state_store.fail(task, disposition)
    except Exception:
        # Queue routing must still be conservative if recording the failure is down.
        pass


def process_task(
    task: MemoryTask,
    state_store: TaskStateStore,
    *,
    runner: Callable[[MemoryTask, Sequence[list[dict[str, str]]]], tuple[int, bool]] | None = None,
) -> ProcessResult:
    """Process one session revision; never automatically repeat an uncertain write."""
    try:
        begin = state_store.begin(task)
    except Exception as exc:
        raise RetryableTaskError("task_state_unavailable") from exc

    if begin.outcome == "completed":
        return ProcessResult("already_completed")
    if begin.outcome in ("processing", "needs_review"):
        if begin.outcome == "processing":
            _record_failure(state_store, task, "needs_review")
        raise PermanentTaskError("task_requires_review")

    fingerprints = [turn_fingerprint(turn) for turn in task.turns]
    pending = set(fingerprints) - set(begin.processed_turns)
    turns = [_worker_turn(task, turn) for turn in task.turns]
    fingerprint_by_trace = dict(zip((turn.trace_id for turn in task.turns), fingerprints))
    chunks = [
        chunk for chunk in _session_chunks(turns)
        if any(fingerprint_by_trace[turn["trace_id"]] in pending for turn in chunk)
    ]

    if not chunks:
        try:
            state_store.complete(task, "no_op", fingerprints)
        except Exception as exc:
            _record_failure(state_store, task, "retryable")
            raise RetryableTaskError("task_completion_uncertain") from exc
        return ProcessResult("no_op")

    run = runner or _default_runner
    mutation_attempted = False
    try:
        writes, mutation_attempted = run(task, chunks)
    except Exception as exc:
        mutation_attempted = mutation_attempted or bool(getattr(exc, "mutation_attempted", False))
        disposition: FailureDisposition = "needs_review" if mutation_attempted else "retryable"
        _record_failure(state_store, task, disposition)
        error_type = PermanentTaskError if mutation_attempted else RetryableTaskError
        raise error_type("task_processing_failed") from exc

    outcome: CompletionOutcome = "completed" if writes else "no_op"
    try:
        state_store.complete(task, outcome, fingerprints)
    except Exception as exc:
        disposition: FailureDisposition = "needs_review" if mutation_attempted or writes else "retryable"
        _record_failure(state_store, task, disposition)
        error_type = PermanentTaskError if mutation_attempted else RetryableTaskError
        raise error_type("task_completion_uncertain") from exc
    return ProcessResult(outcome, processed_chunks=len(chunks), writes=writes)
