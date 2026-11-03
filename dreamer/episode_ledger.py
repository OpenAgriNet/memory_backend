"""Text-free Redis state for numbered conversation episodes."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__:
    from .tasks import MemoryTask
else:
    from tasks import MemoryTask


def _epoch(value: datetime) -> float:
    return value.astimezone(timezone.utc).timestamp()


@dataclass
class TraceRecord:
    observed_at: str
    fingerprint: str
    eligible: bool
    published_fingerprint: str | None = None


@dataclass
class EpisodeRecord:
    number: int
    started_at: str
    ended_at: str
    last_activity_at: str
    revision: int = 0
    traces: dict[str, TraceRecord] = field(default_factory=dict)


@dataclass
class SessionRecord:
    surface: str
    session_id: str
    episodes: list[EpisodeRecord] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {"schema_version": 2, **asdict(self)},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "SessionRecord":
        value = json.loads(raw)
        if value.get("schema_version") != 2:
            raise RuntimeError("unsupported episode ledger schema")
        episodes = []
        for item in value.get("episodes", []):
            traces = {
                key: TraceRecord(**trace) for key, trace in item.get("traces", {}).items()
            }
            episodes.append(EpisodeRecord(
                number=item["number"],
                started_at=item["started_at"],
                ended_at=item["ended_at"],
                last_activity_at=item["last_activity_at"],
                revision=item.get("revision", 0),
                traces=traces,
            ))
        return cls(value["surface"], value["session_id"], episodes)


class MemoryEpisodeLedger:
    """Small non-persistent implementation used by dry-run and focused tests."""

    def __init__(self):
        self.records: dict[str, SessionRecord] = {}
        self.due_at: dict[str, datetime] = {}
        self.reconcile_at: dict[str, datetime] = {}

    def load(self, key: str) -> SessionRecord | None:
        raw = self.records.get(key)
        return SessionRecord.from_json(raw.to_json()) if raw else None

    def due(self, now: datetime, limit: int = 100) -> list[str]:
        return sorted(
            (key for key, due in self.due_at.items() if due <= now),
            key=lambda key: self.due_at[key],
        )[:limit]

    def reconcile_due(self, now: datetime, limit: int = 25) -> list[str]:
        return sorted(
            (key for key, due in self.reconcile_at.items() if due <= now),
            key=lambda key: self.reconcile_at[key],
        )[:limit]

    def save(
        self,
        key: str,
        farmer_hash: str,
        record: SessionRecord,
        due_at: datetime | None,
        reconcile_at: datetime,
    ) -> None:
        self.records[key] = SessionRecord.from_json(record.to_json())
        if due_at is None:
            self.due_at.pop(key, None)
        else:
            self.due_at[key] = due_at
        self.reconcile_at[key] = reconcile_at

    def commit(
        self,
        key: str,
        farmer_hash: str,
        record: SessionRecord,
        tasks: list[MemoryTask],
        publisher: Any,
        due_at: datetime | None,
        reconcile_at: datetime,
    ) -> tuple[int, int]:
        published = duplicates = 0
        for task in tasks:
            if publisher.publish(task):
                published += 1
            else:
                duplicates += 1
        self.save(key, farmer_hash, record, due_at, reconcile_at)
        return published, duplicates


class RedisEpisodeLedger:
    """Redis ledger with atomic task publication and revision checkpointing."""

    def __init__(self, client: Any):
        load_dotenv(
            os.environ.get(
                "AMUL_MEMORY_ENV_FILE", Path(__file__).resolve().parents[1] / ".env"
            ),
            override=False,
        )
        self.client = client
        self.prefix = os.environ.get("MEMORY_EPISODE_LEDGER_PREFIX", "amul:memory:episodes:v2")
        self.due_set = os.environ.get("MEMORY_EPISODE_DUE_SET", "amul:memory:episodes:due:v2")
        self.reconcile_set = os.environ.get(
            "MEMORY_EPISODE_RECONCILE_SET", "amul:memory:episodes:reconcile:v2"
        )
        self.farmer_index_prefix = os.environ.get(
            "MEMORY_EPISODE_FARMER_INDEX_PREFIX", "amul:memory:episodes:farmer:v2"
        )

    def redis_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def load(self, key: str) -> SessionRecord | None:
        raw = self.client.get(self.redis_key(key))
        return SessionRecord.from_json(raw) if raw else None

    def due(self, now: datetime, limit: int = 100) -> list[str]:
        return list(self.client.zrangebyscore(
            self.due_set, "-inf", _epoch(now), start=0, num=limit
        ))

    def reconcile_due(self, now: datetime, limit: int = 25) -> list[str]:
        return list(self.client.zrangebyscore(
            self.reconcile_set, "-inf", _epoch(now), start=0, num=limit
        ))

    def save(
        self,
        key: str,
        farmer_hash: str,
        record: SessionRecord,
        due_at: datetime | None,
        reconcile_at: datetime,
    ) -> None:
        pipe = self.client.pipeline(transaction=True)
        pipe.set(self.redis_key(key), record.to_json())
        if due_at is None:
            pipe.zrem(self.due_set, key)
        else:
            pipe.zadd(self.due_set, {key: _epoch(due_at)})
        pipe.zadd(self.reconcile_set, {key: _epoch(reconcile_at)})
        pipe.sadd(f"{self.farmer_index_prefix}:{farmer_hash}", self.redis_key(key))
        pipe.execute()

    def commit(
        self,
        key: str,
        farmer_hash: str,
        record: SessionRecord,
        tasks: list[MemoryTask],
        publisher: Any,
        due_at: datetime | None,
        reconcile_at: datetime,
    ) -> tuple[int, int]:
        payloads = []
        for task in tasks:
            payloads.extend((
                task.task_key,
                json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ))
        script = """
        local published = 0
        for index = 1, tonumber(ARGV[1]) do
          local offset = 9 + ((index - 1) * 2)
          local task_key = ARGV[offset]
          local body = ARGV[offset + 1]
          local marker = KEYS[6] .. ':' .. ARGV[8] .. ':' .. task_key
          if redis.call('EXISTS', marker, KEYS[6] .. ':' .. task_key) == 0 then
            redis.call('XADD', KEYS[5], 'MAXLEN', '~', ARGV[2], '*',
              'body', body, 'attempts', '0', 'task_key', task_key)
            redis.call('SET', marker, '1', 'EX', ARGV[3])
            published = published + 1
          end
        end
        redis.call('SET', KEYS[1], ARGV[4])
        if tonumber(ARGV[7]) > 0 then
          redis.call('ZADD', KEYS[2], ARGV[7], ARGV[5])
        else
          redis.call('ZREM', KEYS[2], ARGV[5])
        end
        redis.call('ZADD', KEYS[3], ARGV[6], ARGV[5])
        redis.call('SADD', KEYS[4], KEYS[1])
        return published
        """
        published = int(self.client.eval(
            script,
            6,
            self.redis_key(key),
            self.due_set,
            self.reconcile_set,
            f"{self.farmer_index_prefix}:{farmer_hash}",
            publisher.stream,
            publisher.dedupe_prefix,
            len(tasks),
            publisher.max_length,
            publisher.dedupe_seconds,
            record.to_json(),
            key,
            _epoch(reconcile_at),
            _epoch(due_at) if due_at else 0,
            farmer_hash,
            *payloads,
        ))
        return published, len(tasks) - published
