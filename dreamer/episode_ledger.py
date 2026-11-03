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
    revision: int = 0
    traces: dict[str, TraceRecord] = field(default_factory=dict)


@dataclass
class SessionRecord:
    surface: str
    session_id: str
    episodes: list[EpisodeRecord] = field(default_factory=list)

    def to_json(self) -> str:
        value = asdict(self)
        # Keep the v2 wire field for readers on older branches.
        for episode in value["episodes"]:
            episode["last_activity_at"] = episode["ended_at"]
        return json.dumps(
            {"schema_version": 2, **value},
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

    def snapshot(self, key: str) -> str | None:
        return record.to_json() if (record := self.records.get(key)) else None

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
        *, expected: str | None, dead_letters: list[tuple[str, str]] | None = None,
    ) -> tuple[int, int] | None:
        if self.snapshot(key) != expected:
            return None
        published = duplicates = 0
        for task in tasks:
            if publisher.publish(task):
                published += 1
            else:
                duplicates += 1
        for reason, subject in dead_letters or ():
            publisher.publish_producer_dead_letter(reason, subject)
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

    def snapshot(self, key: str) -> str | None:
        return self.client.get(self.redis_key(key))

    def due(self, now: datetime, limit: int = 100) -> list[str]:
        return list(self.client.zrangebyscore(
            self.due_set, "-inf", _epoch(now), start=0, num=limit
        ))

    def reconcile_due(self, now: datetime, limit: int = 25) -> list[str]:
        return list(self.client.zrangebyscore(
            self.reconcile_set, "-inf", _epoch(now), start=0, num=limit
        ))

    def commit(
        self,
        key: str,
        farmer_hash: str,
        record: SessionRecord,
        tasks: list[MemoryTask],
        publisher: Any,
        due_at: datetime | None,
        reconcile_at: datetime,
        *, expected: str | None, dead_letters: list[tuple[str, str]] | None = None,
    ) -> tuple[int, int] | None:
        payloads = []
        for task in tasks:
            payloads.extend((
                task.task_key,
                json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ))
        script = """
        if (redis.call('GET', KEYS[1]) or '') ~= ARGV[2] then
          return -1
        end
        local published = 0
        for index = 1, tonumber(ARGV[1]) do
          local offset = 8 + ((index - 1) * 2)
          local task_key = ARGV[offset]
          local body = ARGV[offset + 1]
          local marker = KEYS[6] .. ':' .. task_key
          if redis.call('EXISTS', marker) == 0 then
            redis.call('XADD', KEYS[5], '*',
              'body', body, 'attempts', '0', 'task_key', task_key)
            redis.call('SET', marker, '1', 'EX', ARGV[3])
            published = published + 1
          end
        end
        local dead_start = 8 + 2 * tonumber(ARGV[1])
        for index = dead_start + 2, #ARGV, 2 do
          redis.call('XADD', KEYS[7], 'MAXLEN', '~', ARGV[dead_start], '*',
            'body', '', 'attempts', '0', 'task_key', ARGV[index + 1],
            'reason', ARGV[index], 'failed_at', ARGV[dead_start + 1])
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
            7,
            self.redis_key(key),
            self.due_set,
            self.reconcile_set,
            f"{self.farmer_index_prefix}:{farmer_hash}",
            publisher.stream,
            publisher.dedupe_prefix,
            publisher.dead_letter_stream,
            len(tasks),
            expected or "",
            publisher.dedupe_seconds,
            record.to_json(),
            key,
            _epoch(reconcile_at),
            _epoch(due_at) if due_at else 0,
            *payloads,
            publisher.max_length,
            datetime.now(timezone.utc).isoformat(),
            *(value for pair in dead_letters or () for value in pair),
        ))
        if published < 0:
            return None
        return published, len(tasks) - published
