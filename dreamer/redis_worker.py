"""Single-process Redis worker for normalized memory tasks."""
from __future__ import annotations

import argparse
import json
import time

if __package__:
    from .qdrant_task_state import QdrantTaskStateStore
    from .redis_queue import RedisStreamTaskQueue
    from .worker import run_one
else:
    from qdrant_task_state import QdrantTaskStateStore
    from redis_queue import RedisStreamTaskQueue
    from worker import run_one


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process normalized memory tasks from Redis one at a time."
    )
    parser.add_argument(
        "--once", action="store_true", help="handle at most one message and exit"
    )
    args = parser.parse_args(argv)
    try:
        queue = RedisStreamTaskQueue()
        state = QdrantTaskStateStore()
    except Exception as exc:
        print(json.dumps({"outcome": "configuration_error", "error": type(exc).__name__}))
        return 2

    while True:
        try:
            result = run_one(queue, state)
            if result.outcome != "idle":
                print(json.dumps({
                    "outcome": result.outcome,
                    "task_key": result.task_key,
                    "reason": result.reason,
                }, sort_keys=True), flush=True)
        except Exception as exc:
            print(json.dumps(
                {"outcome": "worker_failed", "error": type(exc).__name__}, sort_keys=True
            ), flush=True)
            if args.once:
                return 1
            time.sleep(5)
            continue
        if args.once:
            return 0 if result.outcome in ("completed", "idle") else 1


if __name__ == "__main__":
    raise SystemExit(main())
