"""Supported one-task worker composition; concrete broker adapters plug in later."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from .qdrant_task_state import QdrantTaskStateStore
    from .queue_worker import QueueMessage, TaskQueue, consume_one
    from .task_processor import TaskStateStore, process_task
else:
    from qdrant_task_state import QdrantTaskStateStore
    from queue_worker import QueueMessage, TaskQueue, consume_one
    from task_processor import TaskStateStore, process_task


def run_one(queue: TaskQueue, state_store: TaskStateStore):
    """Compose receive -> validate -> process -> state -> ack/retry/dead-letter."""
    return consume_one(queue, lambda task: process_task(task, state_store))


@dataclass
class _FileQueue:
    body: Mapping[str, Any]
    delivered: bool = False

    def receive(self):
        if self.delivered:
            return None
        self.delivered = True
        return QueueMessage("local-task", self.body)

    def acknowledge(self, receipt: str) -> None:
        self.action = "acknowledged"

    def retry(self, receipt: str, reason: str) -> None:
        self.action = "retry"

    def dead_letter(self, receipt: str, reason: str) -> None:
        self.action = "dead_letter"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process one normalized memory task from a JSON file (or stdin)."
    )
    parser.add_argument("task_file", help="JSON file path, or - for stdin")
    args = parser.parse_args(argv)
    raw = sys.stdin.read() if args.task_file == "-" else Path(args.task_file).read_text("utf-8")
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise SystemExit("task JSON must be an object")
    queue = _FileQueue(body)
    result = run_one(queue, QdrantTaskStateStore())
    print(json.dumps({"outcome": result.outcome, "task_key": result.task_key,
                      "reason": result.reason}, sort_keys=True))
    return 0 if result.outcome in ("completed", "idle") else 1


if __name__ == "__main__":
    raise SystemExit(main())
