"""Offline tests for settled, trace-scoped memory ingestion."""
from __future__ import annotations

import datetime as dt
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "dreamer"))

from producer import ProducerConfig, produce_once  # noqa: E402
from redis_queue import RedisStreamTaskQueue  # noqa: E402


UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
FARMER = "9876543210"


def chat_event(trace_id, observed_at, *, session_id="long-session", answer="answer"):
    return {
        "id": trace_id,
        "name": "chat.translation",
        "sessionId": session_id,
        "userId": FARMER,
        "timestamp": observed_at.isoformat().replace("+00:00", "Z"),
        "metadata": {
            "amul.schema_version": "chat.turn.v1",
            "user_id": FARMER,
            "persona": "farmer",
        },
        "input": {"persona": "farmer", "query": f"question {trace_id}"},
        "output": answer,
        "scores": [{"name": "turn_outcome", "value": "success"}],
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
        self.rows = rows
        self.calls = []

    def recent(self, start, stop):
        self.calls.append((start, stop))
        return [
            row for row in self.rows
            if start <= dt.datetime.fromisoformat(
                row["timestamp"].replace("Z", "+00:00")
            ) <= stop
        ]


class Publisher:
    def __init__(self, watermark=None, fail_after=None):
        self.watermark_value = watermark
        self.fail_after = fail_after
        self.keys = set()
        self.tasks = []
        self.advances = []

    def watermark(self):
        return self.watermark_value

    def publish(self, task):
        if self.fail_after is not None and len(self.tasks) >= self.fail_after:
            raise RuntimeError("publish failed")
        self.tasks.append(task)
        if task.task_key in self.keys:
            return False
        self.keys.add(task.task_key)
        return True

    def advance_watermark(self, value):
        self.advances.append(value)
        self.watermark_value = value


class WatermarkRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def eval(self, _script, key_count, key, value):
        assert key_count == 1
        current = self.values.get(key)
        if current is not None and int(current) >= int(value):
            return 0
        self.values[key] = str(value)
        return 1


def config(*, not_before=None, dry_run=False):
    return ProducerConfig(
        settle_seconds=40 * 60,
        scan_seconds=30 * 60,
        discovery_lookback_seconds=24 * 60 * 60,
        event_not_before=not_before or NOW - dt.timedelta(days=30),
        dry_run=dry_run,
        allowlist_enabled=False,
        farmer_allowlist=frozenset(),
    )


class TestEventProducer(unittest.TestCase):
    def test_reopened_28_day_session_ingests_only_the_new_trace(self):
        old = chat_event("trace-old", NOW - dt.timedelta(days=28))
        new = chat_event("trace-new", NOW - dt.timedelta(hours=1))
        source = Source([old, new])
        publisher = Publisher(watermark=NOW - dt.timedelta(hours=2))

        result = produce_once(source, publisher, config(), now=NOW)

        self.assertEqual(result.events_seen, 1)
        self.assertEqual(result.published, 1)
        self.assertEqual([task.turns[0].trace_id for task in publisher.tasks], ["trace-new"])
        self.assertEqual(publisher.tasks[0].session_id, "long-session")

    def test_settlement_is_per_trace_not_per_session(self):
        settled = chat_event("trace-settled", NOW - dt.timedelta(minutes=41))
        recent = chat_event("trace-recent", NOW - dt.timedelta(minutes=20))
        publisher = Publisher(watermark=NOW - dt.timedelta(hours=2))

        result = produce_once(Source([settled, recent]), publisher, config(), now=NOW)

        self.assertEqual(result.published, 1)
        self.assertEqual(publisher.tasks[0].turns[0].trace_id, "trace-settled")

    def test_chat_and_voice_events_keep_independent_sources(self):
        observed = NOW - dt.timedelta(hours=1)
        publisher = Publisher(watermark=NOW - dt.timedelta(hours=2))

        result = produce_once(
            Source([chat_event("chat-trace", observed), voice_event("voice-trace", observed)]),
            publisher,
            config(),
            now=NOW,
        )

        self.assertEqual(result.published, 2)
        self.assertEqual(
            {task.source for task in publisher.tasks},
            {"langfuse_chat_turn_v1", "langfuse_voice_turn_v1"},
        )

    def test_overlap_scan_deduplicates_the_same_event(self):
        row = chat_event("trace-one", NOW - dt.timedelta(hours=1))
        source = Source([row])
        publisher = Publisher(watermark=NOW - dt.timedelta(hours=2))

        first = produce_once(source, publisher, config(), now=NOW)
        second = produce_once(source, publisher, config(), now=NOW + dt.timedelta(minutes=30))

        self.assertEqual(first.published, 1)
        self.assertEqual(second.published, 0)
        self.assertEqual(second.duplicates, 1)

    def test_publish_failure_does_not_advance_watermark(self):
        rows = [
            chat_event("trace-one", NOW - dt.timedelta(hours=2)),
            chat_event("trace-two", NOW - dt.timedelta(hours=1)),
        ]
        initial = NOW - dt.timedelta(hours=3)
        publisher = Publisher(watermark=initial, fail_after=1)

        with self.assertRaises(RuntimeError):
            produce_once(Source(rows), publisher, config(), now=NOW)

        self.assertEqual(publisher.watermark_value, initial)
        self.assertEqual(publisher.advances, [])

    def test_redis_watermark_only_moves_forward(self):
        queue = RedisStreamTaskQueue(client=WatermarkRedis())
        first = NOW - dt.timedelta(hours=2)
        latest = NOW - dt.timedelta(hours=1)

        queue.advance_watermark(first)
        queue.advance_watermark(latest)
        queue.advance_watermark(first)

        self.assertEqual(queue.watermark(), latest)

    def test_rollout_cutoff_prevents_historical_backfill(self):
        source = Source([chat_event("trace-old", NOW - dt.timedelta(hours=1))])
        publisher = Publisher()

        result = produce_once(
            source, publisher, config(not_before=NOW), now=NOW
        )

        self.assertEqual(result.events_seen, 0)
        self.assertEqual(publisher.tasks, [])
        self.assertEqual(publisher.advances, [])


if __name__ == "__main__":
    unittest.main()
