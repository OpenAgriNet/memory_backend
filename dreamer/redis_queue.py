"""Redis Streams transport for normalized memory tasks.

The transport owns delivery attempts, visibility recovery and the dead-letter
stream.  Task validation and processing remain broker-neutral.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__:
    from .queue_worker import QueueMessage
    from .tasks import MemoryTask
else:
    from queue_worker import QueueMessage
    from tasks import MemoryTask


_SAFE_KEY = re.compile(r"^[A-Za-z0-9:._-]+$")


def _positive_env(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def _key(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    if not _SAFE_KEY.fullmatch(value):
        raise RuntimeError(f"{name} contains unsupported characters")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RedisStreamTaskQueue:
    """At-least-once Redis Stream queue with a separate dead-letter stream."""

    def __init__(self, client: Any | None = None):
        load_dotenv(
            os.environ.get(
                "AMUL_MEMORY_ENV_FILE", Path(__file__).resolve().parents[1] / ".env"
            ),
            override=False,
        )
        self.stream = _key("MEMORY_QUEUE_STREAM", "amul:memory:tasks:v1")
        self.dead_letter_stream = _key(
            "MEMORY_QUEUE_DLQ_STREAM", "amul:memory:tasks:dead:v1"
        )
        self.group = _key("MEMORY_QUEUE_CONSUMER_GROUP", "memory-workers-v1")
        self.consumer = _key(
            "MEMORY_QUEUE_CONSUMER_NAME",
            f"worker-{os.getpid()}",
        )
        self.dedupe_prefix = _key(
            "MEMORY_QUEUE_DEDUPE_PREFIX", "amul:memory:task-published:v1"
        )
        self.revision_prefix = _key(
            "MEMORY_QUEUE_REVISION_PREFIX", "amul:memory:session-revision:v1"
        )
        self.retry_set = _key(
            "MEMORY_QUEUE_RETRY_SET", "amul:memory:tasks:retry:v1"
        )
        self.visibility_ms = _positive_env(
            "MEMORY_QUEUE_VISIBILITY_SECONDS", 900, minimum=30
        ) * 1000
        self.max_attempts = _positive_env("MEMORY_QUEUE_MAX_ATTEMPTS", 5)
        self.retry_base_seconds = _positive_env(
            "MEMORY_QUEUE_RETRY_BASE_SECONDS", 60
        )
        self.retry_max_seconds = _positive_env(
            "MEMORY_QUEUE_RETRY_MAX_SECONDS", 900
        )
        if self.retry_max_seconds < self.retry_base_seconds:
            raise RuntimeError(
                "MEMORY_QUEUE_RETRY_MAX_SECONDS must not be smaller than the base delay"
            )
        self.block_ms = _positive_env("MEMORY_QUEUE_BLOCK_MS", 5000, minimum=100)
        self.max_length = _positive_env("MEMORY_QUEUE_MAX_LENGTH", 100000)
        self.dedupe_seconds = _positive_env(
            "MEMORY_QUEUE_DEDUPE_SECONDS", 30 * 24 * 60 * 60,
            minimum=3600,
        )
        if client is None:
            redis_url = os.environ.get("MEMORY_REDIS_URL", "").strip()
            if not redis_url:
                raise RuntimeError("MEMORY_REDIS_URL is required")
            try:
                import redis
            except ImportError as exc:
                raise RuntimeError("redis package is required for Redis Streams") from exc
            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
            )
        self.client = client
        self._ready = False
        self._delivered: dict[str, dict[str, str]] = {}

    def _ensure_group(self) -> None:
        if self._ready:
            return
        try:
            self.client.xgroup_create(
                self.stream, self.group, id="0-0", mkstream=True
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._ready = True

    @staticmethod
    def revision_key(trace_id: str) -> str:
        return hashlib.sha256(trace_id.encode("utf-8")).hexdigest()

    def revision_published(self, subject_key: str, revision_key: str) -> bool:
        value = self.client.get(f"{self.revision_prefix}:{subject_key}")
        return value == revision_key

    def publish(self, task: MemoryTask) -> bool:
        """Publish one task once per task key.

        The Lua operation keeps the dedupe marker and XADD atomic.  False means
        this exact task revision was already published.
        """
        body = json.dumps(
            task.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        script = """
        local marker = KEYS[1]
        if redis.call('EXISTS', marker) == 1 then
          return 0
        end
        redis.call('XADD', KEYS[2], 'MAXLEN', '~', ARGV[1], '*',
          'body', ARGV[2], 'attempts', '0', 'task_key', ARGV[3])
        redis.call('SET', marker, '1', 'EX', ARGV[4])
        return 1
        """
        marker = f"{self.dedupe_prefix}:{task.task_key}"
        return bool(self.client.eval(
            script, 2, marker, self.stream,
            self.max_length, body, task.task_key, self.dedupe_seconds,
        ))

    def mark_revision_published(self, subject_key: str, revision_key: str) -> None:
        self.client.set(
            f"{self.revision_prefix}:{subject_key}",
            revision_key,
            ex=self.dedupe_seconds,
        )

    def _promote_due_retries(self) -> None:
        script = """
        local due = redis.call(
          'ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 100
        )
        for _, member in ipairs(due) do
          local item = cjson.decode(member)
          redis.call('XADD', KEYS[2], 'MAXLEN', '~', ARGV[2], '*',
            'body', item.body, 'attempts', tostring(item.attempts),
            'task_key', item.task_key, 'retry_reason', item.retry_reason)
          redis.call('ZREM', KEYS[1], member)
        end
        return #due
        """
        self.client.eval(
            script, 2, self.retry_set, self.stream,
            time.time(), self.max_length,
        )

    @staticmethod
    def _message(rows: Any) -> tuple[str, dict[str, str]] | None:
        if not rows:
            return None
        # XREADGROUP: [[stream, [[id, fields]]]]
        # XAUTOCLAIM: [next-id, [[id, fields]], deleted-ids]
        messages = rows[1] if isinstance(rows, (list, tuple)) and len(rows) >= 2 else []
        if messages and isinstance(messages[0], (list, tuple)) and len(messages[0]) == 2:
            receipt, fields = messages[0]
            return str(receipt), dict(fields)
        return None

    def _reclaim(self) -> tuple[str, dict[str, str]] | None:
        try:
            rows = self.client.xautoclaim(
                self.stream,
                self.group,
                self.consumer,
                min_idle_time=self.visibility_ms,
                start_id="0-0",
                count=1,
            )
        except Exception as exc:
            # Redis older than 6.2 cannot provide safe visibility recovery.
            if "unknown command" in str(exc).lower():
                raise RuntimeError("Redis 6.2 or newer is required") from exc
            raise
        return self._message(rows)

    def receive(self) -> QueueMessage | None:
        self._ensure_group()
        self._promote_due_retries()
        item = self._reclaim()
        if item is None:
            rows = self.client.xreadgroup(
                self.group,
                self.consumer,
                {self.stream: ">"},
                count=1,
                block=self.block_ms,
            )
            if not rows:
                return None
            receipt, fields = rows[0][1][0]
            item = str(receipt), dict(fields)
        receipt, fields = item
        self._delivered[receipt] = fields
        pending = self.client.xpending_range(
            self.stream, self.group, min=receipt, max=receipt, count=1
        )
        if pending:
            deliveries = int(pending[0].get("times_delivered", 1))
            if deliveries > self.max_attempts:
                self.dead_letter(receipt, "delivery_limit")
                return None
        try:
            body = json.loads(fields.get("body", ""))
        except (TypeError, ValueError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        return QueueMessage(receipt, body)

    def _fields(self, receipt: str) -> dict[str, str]:
        fields = self._delivered.get(receipt)
        if fields is None:
            rows = self.client.xrange(self.stream, min=receipt, max=receipt, count=1)
            if not rows:
                raise RuntimeError("queue receipt is no longer available")
            fields = dict(rows[0][1])
        return fields

    def acknowledge(self, receipt: str) -> None:
        pipe = self.client.pipeline(transaction=True)
        pipe.xack(self.stream, self.group, receipt)
        pipe.xdel(self.stream, receipt)
        pipe.execute()
        self._delivered.pop(receipt, None)

    def retry(self, receipt: str, reason: str) -> None:
        fields = self._fields(receipt)
        attempts = int(fields.get("attempts", "0")) + 1
        if attempts >= self.max_attempts:
            self.dead_letter(receipt, "retry_limit")
            return
        delay = min(
            self.retry_base_seconds * (2 ** max(0, attempts - 1)),
            self.retry_max_seconds,
        )
        delayed = json.dumps({
            "body": fields.get("body", ""),
            "attempts": attempts,
            "task_key": fields.get("task_key", ""),
            "retry_reason": reason,
            "nonce": uuid.uuid4().hex,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        pipe = self.client.pipeline(transaction=True)
        # Schedule first, then acknowledge. A lost response can duplicate the
        # task, which the task/state idempotency boundary tolerates.
        pipe.zadd(self.retry_set, {delayed: time.time() + delay})
        pipe.xack(self.stream, self.group, receipt)
        pipe.xdel(self.stream, receipt)
        pipe.execute()
        self._delivered.pop(receipt, None)

    def dead_letter(self, receipt: str, reason: str) -> None:
        fields = self._fields(receipt)
        pipe = self.client.pipeline(transaction=True)
        pipe.xadd(
            self.dead_letter_stream,
            {
                "body": fields.get("body", ""),
                "attempts": fields.get("attempts", "0"),
                "task_key": fields.get("task_key", ""),
                "reason": reason,
                "failed_at": _utc_now(),
            },
            maxlen=self.max_length,
            approximate=True,
        )
        pipe.xack(self.stream, self.group, receipt)
        pipe.xdel(self.stream, receipt)
        pipe.execute()
        self._delivered.pop(receipt, None)
