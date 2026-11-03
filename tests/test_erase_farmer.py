"""Focused contracts for the dry-run-first farmer erasure command."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import env_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dreamer"), str(ROOT / "service")]

import erase_farmer as erase
from tasks import MemoryTask

PHONE = "9876543210"
PRIVATE_TEXT = "private farmer question"


def task_body(phone=PHONE):
    timestamp = "2026-10-19T08:00:00Z"
    return {
        "schema_version": 2,
        "surface": "chat",
        "episode_number": 1,
        "episode_revision": 2,
        "episode_started_at": timestamp,
        "episode_ended_at": timestamp,
        "part_number": 1,
        "part_count": 2,
        "pending_trace_ids": ["trace-one"],
        "farmer_id": phone,
        "session_id": "session-one",
        "source": "chat.translation",
        "turns": [{
            "trace_id": "trace-one",
            "timestamp": "2026-10-19T08:00:00Z",
            "query": PRIVATE_TEXT,
            "answer": "private assistant answer",
        }],
    }


class FakeQdrant:
    def __init__(self, actions=None):
        task = MemoryTask.from_dict(task_body())
        self.actions = actions if actions is not None else []
        self.memory_points = 4
        self.state_points = 1
        self.disabled = False
        self.task_keys = (task.task_key,)
        self.fail_stage = None

    def inventory(self, _phone, _farmer_key, _subjects):
        return erase.QdrantInventory(
            self.memory_points, self.state_points, self.task_keys
        )

    def disable(self, _phone):
        self.actions.append("disable")
        self.disabled = True

    def delete_state(self, _key, _subjects):
        self.actions.append("state")
        if self.fail_stage == "state":
            raise RuntimeError(f"failure containing {PHONE} and {PRIVATE_TEXT}")
        self.state_points = 0
        self.task_keys = ()

    def delete_memory(self, _phone):
        self.actions.append("memory")
        self.memory_points = 0

    def settings_disabled(self, _phone):
        return self.disabled


class FakeQueue:
    def __init__(self, actions=None):
        self.actions = actions if actions is not None else []
        self.stream = self.dead = self.retries = 1
        self.markers = True

    def inventory(self, _phone, **_keys):
        return erase.RedisInventory(
            self.stream, self.dead, self.retries,
            marker_keys=("dedupe:hash",) if self.markers else (),
        )

    def purge(self, _inventory):
        self.actions.append("queue")
        self.stream = self.dead = self.retries = 0
        self.markers = False

    def episode_subjects(self, _phone):
        return ()

    def finish(self, _inventory):
        pass

    def markers_exist(self, _keys):
        return self.markers


class FakeRedisClient:
    def __init__(self):
        task = MemoryTask.from_dict(task_body())
        body = json.dumps(task.to_dict())
        self.streams = {
            "amul:memory:tasks:v2": [("1-0", {"body": body, "task_key": task.task_key})],
            "amul:memory:tasks:dead:v2": [("2-0", {"body": body, "task_key": task.task_key})],
        }
        retry = json.dumps({"body": body, "task_key": task.task_key})
        self.retries = {retry}
        identity = erase.farmer_key(PHONE)
        self.ledger = "amul:memory:episodes:v2:session"
        self.index = f"amul:memory:episodes:farmer:v2:{identity}"
        self.sorted_sets = {
            "amul:memory:episodes:due:v2": {"session": 1, "other-session": 2},
            "amul:memory:episodes:reconcile:v2": {"session": 3, "other-session": 4},
        }
        self.keys = {
            self.ledger, self.index,
            f"amul:memory:task-published:v2:{identity}:{'0' * 64}",
            f"amul:memory:task-published:v2:{task.task_key}",

        }
        other = MemoryTask.from_dict(task_body("0123456789"))
        self.streams["amul:memory:tasks:v2"].append(("3-0", {"body": json.dumps(other.to_dict())}))
        self.pending = {"1-0", "3-0"}

    def smembers(self, key):
        return {self.ledger} if key == self.index and key in self.keys else set()

    def scan_iter(self, match):
        return iter(key for key in self.keys if key.startswith(match.removesuffix("*")))

    def zscore(self, key, member):
        return self.sorted_sets[key].get(member)

    def xrange(self, stream, min="-", max="+", count=100):
        return [] if str(min).startswith("(") else list(self.streams.get(stream, []))

    def zscan_iter(self, _key, count=100):
        return iter((member, 0.0) for member in list(self.retries))

    def pipeline(self, transaction=True):
        return self

    def xack(self, _stream, _group, receipt):
        self.pending.discard(receipt)
        return self

    def xdel(self, stream, receipt):
        self.streams[stream] = [row for row in self.streams.get(stream, []) if row[0] != receipt]
        return self

    def zrem(self, key, *members):
        if key in self.sorted_sets:
            for member in members:
                self.sorted_sets[key].pop(member, None)
        else:
            self.retries.difference_update(members)
        return self

    def delete(self, *keys):
        self.keys.difference_update(keys)
        return self

    def execute(self):
        return []

    def exists(self, key):
        return key in self.keys


class EraseContract(unittest.TestCase):
    def test_dry_run_is_safe_and_non_mutating(self):
        qdrant, queue = FakeQdrant(), FakeQueue()
        result = erase.erase_farmer(PHONE, qdrant=qdrant, redis=queue)
        self.assertEqual(result["outcome"], "dry_run")
        self.assertEqual(result["memory_points"], 4)
        self.assertEqual(qdrant.actions + queue.actions, [])
        rendered = json.dumps(result)
        self.assertNotIn(PHONE, rendered)
        self.assertNotIn(PRIVATE_TEXT, rendered)

    def test_execute_requires_exact_phone_and_pause_confirmation(self):
        for kwargs in (
            {"confirm_phone": "0000000000", "workers_paused": True},
            {"confirm_phone": PHONE, "workers_paused": False},
        ):
            with self.assertRaises(erase.EraseUsageError):
                erase.erase_farmer(PHONE, execute=True, **kwargs)
        with self.assertRaises(ValueError):
            erase.erase_farmer("not-a-phone")

    def test_execute_orders_cleanup_and_is_idempotent(self):
        actions = []
        qdrant, queue = FakeQdrant(actions), FakeQueue(actions)
        first = erase.erase_farmer(
            PHONE, execute=True, confirm_phone=PHONE, workers_paused=True,
            qdrant=qdrant, redis=queue,
        )
        second = erase.erase_farmer(
            PHONE, execute=True, confirm_phone=PHONE, workers_paused=True,
            qdrant=qdrant, redis=queue,
        )
        self.assertEqual(first["outcome"], second["outcome"])
        self.assertEqual(actions[:4], ["disable", "queue", "memory", "state"])
        self.assertTrue(qdrant.disabled)
        filter_store = object.__new__(erase.QdrantEraseStore)
        self.assertEqual(
            filter_store._memory_filter(PHONE)["must_not"],
            [{"has_id": [filter_store._settings_point_id(PHONE)]}],
        )

    def test_redis_removes_stream_retry_dead_letter_and_markers(self):
        client = FakeRedisClient()
        store = erase.RedisEraseStore(client)
        plan = store.inventory(PHONE)
        self.assertEqual((plan.stream_entries, plan.dead_letters, plan.retries), (1, 1, 1))
        store.purge(plan)
        store.finish(plan)
        remaining = store.inventory(PHONE)
        self.assertEqual(
            (remaining.stream_entries, remaining.dead_letters, remaining.retries),
            (0, 0, 0),
        )
        self.assertFalse(store.markers_exist(plan.marker_keys))
        self.assertEqual(remaining.ledger_keys, ())
        self.assertEqual(client.pending, {"3-0"})
        self.assertEqual(store.inventory("0123456789").stream_entries, 1)
        for members in client.sorted_sets.values():
            self.assertEqual(set(members), {"other-session"})

    def test_partial_failure_reports_only_fixed_safe_fields(self):
        qdrant, queue = FakeQdrant(), FakeQueue()
        qdrant.fail_stage = "state"
        with patch.object(erase, "QdrantEraseStore", return_value=qdrant), patch.object(
            erase, "RedisEraseStore", return_value=queue
        ), patch("sys.stdout", new_callable=io.StringIO) as output:
            code = erase.main([
                "--farmer-id", PHONE, "--execute", "--confirm-phone", PHONE,
                "--workers-paused",
            ])
        self.assertEqual(code, 1)
        rendered = output.getvalue()
        self.assertIn('"stage": "delete_task_state"', rendered)
        self.assertNotIn(PHONE, rendered)
        self.assertNotIn(PRIVATE_TEXT, rendered)
        qdrant.fail_stage = None
        recovered = erase.erase_farmer(
            PHONE, execute=True, confirm_phone=PHONE, workers_paused=True,
            qdrant=qdrant, redis=queue,
        )
        self.assertEqual(recovered["outcome"], "erased")


if __name__ == "__main__":
    unittest.main()
