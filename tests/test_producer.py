"""Focused tests for numbered conversation-episode production."""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

os.environ.setdefault("MEMORY_ID_HASH_KEY", "0123456789abcdef0123456789abcdef")
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "dreamer"))

from episode_ledger import MemoryEpisodeLedger  # noqa: E402
from producer import ProducerConfig, produce_once  # noqa: E402
from redis_queue import RedisStreamTaskQueue  # noqa: E402


UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
FARMER = "9876543210"


def chat_event(
    trace_id,
    observed_at,
    *,
    session_id="long-session",
    farmer=FARMER,
    successful=True,
    query=None,
    answer="answer",
):
    return {
        "id": trace_id,
        "name": "chat.translation",
        "sessionId": session_id,
        "userId": farmer,
        "timestamp": observed_at.isoformat().replace("+00:00", "Z"),
        "metadata": {"amul.schema_version": "chat.turn.v1", "user_id": farmer, "persona": "farmer"},
        "input": {"persona": "farmer", "query": query or f"question {trace_id}"},
        "output": answer,
        "scores": [{"name": "turn_outcome", "value": "success" if successful else "error"}],
    }


def voice_event(trace_id, observed_at, *, session_id="voice-session"):
    return {
        "id": trace_id,
        "name": "turn",
        "sessionId": session_id,
        "userId": FARMER,
        "timestamp": observed_at.isoformat().replace("+00:00", "Z"),
        "metadata": {
            "amul.schema_version": "turn.v1",
            "amul.pipeline": "telephony",
            "amul.service": "amul-oan-api",
            "amul.persona": "farmer",
            "amul.signed_in": "true",
            "amul.farmer_id": FARMER,
            "amul.outcome": "success",
        },
        "input": {"text": "voice question"},
        "output": {"text": "voice answer"},
    }


class Source:
    def __init__(self, rows):
        self.rows = list(rows)

    @staticmethod
    def _time(row):
        return dt.datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))

    def recent(self, start, stop):
        return [row for row in self.rows if start <= self._time(row) <= stop]

    def session(self, session_id, start, stop):
        return [
            row for row in self.rows
            if row.get("sessionId") == session_id and start <= self._time(row) <= stop
        ]


class Publisher:
    def __init__(self):
        self.watermark_value = None
        self.keys = set()
        self.tasks = []
        self.dead = []

    def watermark(self):
        return self.watermark_value

    def publish(self, task):
        if task.task_key in self.keys:
            return False
        self.keys.add(task.task_key)
        self.tasks.append(task)
        return True

    def advance_watermark(self, value):
        self.watermark_value = value

    def publish_producer_dead_letter(self, reason, subject_key):
        self.dead.append((reason, subject_key))


def config(*, dry_run=False):
    return ProducerConfig(
        inactivity_seconds=40 * 60,
        scan_seconds=30 * 60,
        discovery_lookback_seconds=30 * 24 * 60 * 60,
        reconcile_seconds=24 * 60 * 60,
        reconcile_horizon_seconds=30 * 24 * 60 * 60,
        event_not_before=NOW - dt.timedelta(days=30),
        dry_run=dry_run,
        allowlist_enabled=False,
        farmer_allowlist=frozenset(),
    )


class TestEpisodeProducer(unittest.TestCase):
    def setUp(self):
        self.ledger = MemoryEpisodeLedger()
        self.publisher = Publisher()

    def produce(self, source, now=NOW, *, dry_run=False):
        return produce_once(
            source,
            None if dry_run else self.publisher,
            config(dry_run=dry_run),
            ledger=self.ledger,
            now=now,
        )

    def test_turns_within_40_minutes_form_one_ordered_episode(self):
        source = Source([
            chat_event("trace-2", NOW - dt.timedelta(minutes=60)),
            chat_event("trace-1", NOW - dt.timedelta(minutes=70)),
        ])
        result = self.produce(source)
        self.assertEqual(result.episodes_closed, 1)
        self.assertEqual(len(self.publisher.tasks), 1)
        task = self.publisher.tasks[0]
        self.assertEqual(task.episode_number, 1)
        self.assertEqual([turn.trace_id for turn in task.turns], ["trace-1", "trace-2"])

    def test_recent_activity_prevents_session_closure(self):
        source = Source([
            chat_event("old", NOW - dt.timedelta(minutes=41)),
            chat_event("recent", NOW - dt.timedelta(minutes=10)),
        ])
        result = self.produce(source)
        self.assertEqual(result.published, 0)
        self.assertEqual(self.publisher.tasks, [])
        self.assertTrue(self.ledger.due(NOW + dt.timedelta(minutes=31)))

    def test_same_session_five_days_later_becomes_episode_two(self):
        first_now = NOW - dt.timedelta(days=5)
        source = Source([chat_event("old", first_now - dt.timedelta(hours=1))])
        self.produce(source, first_now)
        source.rows.append(chat_event("new", NOW - dt.timedelta(hours=1)))
        self.produce(source, NOW)
        self.assertEqual([task.episode_number for task in self.publisher.tasks], [1, 2])
        self.assertEqual([task.turns[0].trace_id for task in self.publisher.tasks], ["old", "new"])

    def test_same_session_28_days_later_becomes_episode_two(self):
        first_now = NOW - dt.timedelta(days=28)
        source = Source([chat_event("old", first_now - dt.timedelta(hours=1))])
        self.produce(source, first_now)
        source.rows.append(chat_event("new", NOW - dt.timedelta(hours=1)))
        self.produce(source, NOW)
        self.assertEqual([task.episode_number for task in self.publisher.tasks], [1, 2])

    def test_new_source_session_starts_its_own_episode_one(self):
        source = Source([
            chat_event("one", NOW - dt.timedelta(hours=1), session_id="session-a"),
            chat_event("two", NOW - dt.timedelta(hours=1), session_id="session-b"),
        ])
        self.produce(source)
        self.assertEqual([task.episode_number for task in self.publisher.tasks], [1, 1])
        self.assertEqual({task.session_id for task in self.publisher.tasks}, {"session-a", "session-b"})

    def test_late_old_timestamp_revises_closed_episode(self):
        source = Source([
            chat_event("one", NOW - dt.timedelta(hours=3)),
            chat_event("three", NOW - dt.timedelta(hours=2, minutes=40)),
        ])
        self.produce(source)
        source.rows.append(chat_event("two", NOW - dt.timedelta(hours=2, minutes=50)))
        self.produce(source, NOW + dt.timedelta(days=1))
        self.assertEqual([task.episode_revision for task in self.publisher.tasks], [1, 2])
        self.assertEqual(self.publisher.tasks[-1].pending_trace_ids, ("two",))
        self.assertEqual([turn.trace_id for turn in self.publisher.tasks[-1].turns], ["one", "two", "three"])

    def test_late_trace_tie_uses_earlier_episode(self):
        source = Source([
            chat_event("early", NOW - dt.timedelta(hours=3)),
            chat_event("late", NOW - dt.timedelta(hours=2)),
        ])
        self.produce(source)
        source.rows.append(chat_event("middle", NOW - dt.timedelta(hours=2, minutes=30)))
        self.produce(source, NOW + dt.timedelta(days=1))
        self.assertEqual(self.publisher.tasks[-1].episode_number, 1)
        self.assertEqual(self.publisher.tasks[-1].episode_revision, 2)

    def test_changed_trace_id_is_published_for_processor_review(self):
        source = Source([chat_event("same", NOW - dt.timedelta(hours=1), answer="original")])
        self.produce(source)
        source.rows[0] = chat_event("same", NOW - dt.timedelta(hours=1), answer="changed")
        self.produce(source, NOW + dt.timedelta(days=1))
        self.assertEqual(self.publisher.tasks[-1].episode_revision, 2)
        self.assertEqual(self.publisher.tasks[-1].pending_trace_ids, ("same",))

    def test_trusted_failed_turn_extends_activity_but_is_not_evidence(self):
        source = Source([
            chat_event("eligible", NOW - dt.timedelta(minutes=41)),
            chat_event("failed", NOW - dt.timedelta(minutes=10), successful=False),
        ])
        self.produce(source)
        self.assertEqual(self.publisher.tasks, [])

    def test_chat_and_voice_do_not_mix(self):
        source = Source([
            chat_event("chat", NOW - dt.timedelta(hours=1), session_id="shared"),
            voice_event("voice", NOW - dt.timedelta(hours=1), session_id="shared"),
        ])
        self.produce(source)
        self.assertEqual({task.surface for task in self.publisher.tasks}, {"chat", "voice"})

    def test_mixed_farmer_session_fails_closed(self):
        source = Source([
            chat_event("one", NOW - dt.timedelta(hours=1)),
            chat_event("two", NOW - dt.timedelta(hours=1), farmer="9123456789"),
        ])
        result = self.produce(source)
        self.assertEqual(result.published, 0)
        self.assertGreaterEqual(result.skipped["mixed_or_missing_identity"], 1)

    def test_duplicate_scan_is_idempotent(self):
        source = Source([chat_event("one", NOW - dt.timedelta(hours=1))])
        self.produce(source)
        self.produce(source, NOW + dt.timedelta(minutes=30))
        self.assertEqual(len(self.publisher.tasks), 1)

    def test_large_episode_splits_without_new_episode_number(self):
        source = Source([
            chat_event(f"trace-{index:03}", NOW - dt.timedelta(hours=3) + dt.timedelta(seconds=index), answer="a" * 1000)
            for index in range(101)
        ])
        result = self.produce(source)
        self.assertEqual(result.split_episodes, 1)
        self.assertEqual(len(self.publisher.tasks), 2)
        self.assertEqual({task.episode_number for task in self.publisher.tasks}, {1})
        self.assertEqual([task.part_number for task in self.publisher.tasks], [1, 2])

    def test_single_oversized_turn_is_dead_lettered(self):
        source = Source([chat_event("large", NOW - dt.timedelta(hours=1), query="q" * 8001)])
        result = self.produce(source)
        self.assertEqual(result.published, 0)
        self.assertEqual(len(self.publisher.dead), 1)

    def test_dry_run_does_not_write_queue_or_ledger(self):
        source = Source([chat_event("one", NOW - dt.timedelta(hours=1))])
        result = self.produce(source, dry_run=True)
        self.assertEqual(result.would_publish, 1)
        self.assertEqual(self.ledger.records, {})
        self.assertEqual(self.publisher.tasks, [])

    def test_ledger_contains_no_farmer_or_conversation_text(self):
        source = Source([chat_event("one", NOW - dt.timedelta(hours=1), answer="private answer")])
        self.produce(source)
        serialized = next(iter(self.ledger.records.values())).to_json()
        self.assertNotIn(FARMER, serialized)
        self.assertNotIn("question one", serialized)
        self.assertNotIn("private answer", serialized)


class WatermarkRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def eval(self, _script, key_count, key, value):
        current = self.values.get(key)
        if current is not None and int(current) >= int(value):
            return 0
        self.values[key] = str(value)
        return 1


class TestQueueDefaults(unittest.TestCase):
    def test_v2_watermark_only_moves_forward(self):
        queue = RedisStreamTaskQueue(client=WatermarkRedis())
        first, latest = NOW - dt.timedelta(hours=2), NOW - dt.timedelta(hours=1)
        queue.advance_watermark(first)
        queue.advance_watermark(latest)
        queue.advance_watermark(first)
        self.assertEqual(queue.watermark(), latest)
        self.assertTrue(queue.watermark_key.endswith(":v2"))


if __name__ == "__main__":
    unittest.main()
