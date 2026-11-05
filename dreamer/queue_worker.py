"""Broker-neutral, one-message consumer for normalized memory tasks.

A later transport can implement ``TaskQueue`` without changing task validation or
memory processing.  Acknowledgement happens only after processing succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

if __package__:  # Package invocation: python -m dreamer.queue_worker
    from .tasks import MemoryTask, TaskValidationError
else:  # Existing repository scripts also import dreamer modules flat.
    from tasks import MemoryTask, TaskValidationError


@dataclass(frozen=True)
class QueueMessage:
    receipt: str
    body: Mapping[str, Any]


class TaskQueue(Protocol):
    def receive(self) -> QueueMessage | None: ...
    def acknowledge(self, receipt: str) -> None: ...
    def retry(self, receipt: str, reason: str) -> None: ...
    def dead_letter(self, receipt: str, reason: str) -> None: ...


class RetryableTaskError(RuntimeError):
    """Processing failed temporarily; the queue should retry the task."""


class PermanentTaskError(RuntimeError):
    """The same task cannot succeed and should not be retried."""


@dataclass(frozen=True)
class ConsumeResult:
    outcome: str
    task_key: str | None = None
    reason: str | None = None


def _failure_reason(prefix: str, error: BaseException) -> str:
    # Exception text can contain conversation data, URLs, or credentials.  The
    # class name is sufficient for retry/dead-letter routing and safe metrics.
    return f"{prefix}:{type(error).__name__}"


def consume_one(
    queue: TaskQueue,
    processor: Callable[[MemoryTask], Any],
) -> ConsumeResult:
    """Receive and handle at most one task.

    Delivery is at-least-once: acknowledgement can be lost after writes succeed.
    The eventual processor must therefore use the episode ``subject_key``, exact
    ``task_key`` revision, and pending source trace IDs to make writes idempotent.
    """
    message = queue.receive()
    if message is None:
        return ConsumeResult("idle")

    try:
        task = MemoryTask.from_dict(message.body)
    except TaskValidationError as error:
        queue.dead_letter(message.receipt, f"invalid_task:{error.code}")
        return ConsumeResult("dead_lettered", reason=error.code)

    try:
        processor(task)
    except PermanentTaskError as error:
        reason = _failure_reason("permanent", error)
        queue.dead_letter(message.receipt, reason)
        return ConsumeResult("dead_lettered", task.task_key, reason)
    except Exception as error:
        # Unknown processor failures are retried by default.  Retry delay and max
        # attempts belong to the concrete queue adapter added in a later PR.
        reason = _failure_reason("retryable", error)
        queue.retry(message.receipt, reason)
        return ConsumeResult("retry_scheduled", task.task_key, reason)

    queue.acknowledge(message.receipt)
    return ConsumeResult("completed", task.task_key)
