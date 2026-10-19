"""Focused safety tests for one-task processing and lease recovery."""
from __future__ import annotations

import copy
import datetime as dt
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import env_fixture

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path[:0] = [os.path.join(ROOT, "dreamer"), os.path.join(ROOT, "service"), ROOT]

import agent  # noqa: E402
from qdrant_task_state import QdrantTaskStateStore  # noqa: E402
from queue_worker import PermanentTaskError, QueueMessage, RetryableTaskError  # noqa: E402
from task_processor import (  # noqa: E402
    BeginResult, ProcessedTurn, process_task, processed_turn,
)
from tasks import MemoryTask  # noqa: E402
from worker import run_one  # noqa: E402


def task(turns=1):
    return MemoryTask.from_dict({
        "schema_version": 1, "farmer_id": "9876543210",
        "session_id": "session-1", "source": "chat-main",
        "turns": [{
            "trace_id": f"trace-{index}", "timestamp": f"2026-10-17T08:0{index}:00Z",
            "query": f"question {index}", "answer": f"answer {index}",
        } for index in range(turns)],
    })


class FakeState:
    def __init__(self, begin=None, complete_error=None):
        self.begin_result = begin or BeginResult("started")
        self.complete_error = complete_error
        self.actions = []

    def begin(self, value):
        self.actions.append("begin")
        return self.begin_result

    def mark_writing(self, value):
        self.actions.append("writing")

    def complete(self, value, outcome, processed):
        self.actions.append(("complete", outcome, tuple(processed)))
        if self.complete_error:
            raise self.complete_error

    def fail(self, value, disposition):
        self.actions.append(("fail", disposition))


class MemoryQdrant(QdrantTaskStateStore):
    def __init__(self, backing, clock):
        super().__init__("http://unused", "task_state", lease_seconds=30, now=lambda: clock[0])
        self.backing = backing

    def _ensure_ready(self):
        return None

    def _read(self, subject_key):
        return copy.deepcopy(self.backing.get(subject_key))

    def _write(self, payload, *, previous):
        current = self.backing.get(payload["subject_key"])
        if previous is None:
            if current is not None:
                return False
        elif current != previous:
            return False
        self.backing[payload["subject_key"]] = copy.deepcopy(payload)
        return True


class TestProcessor(unittest.TestCase):
    def test_opted_out_and_settings_failure_never_run_model(self):
        for check, error in ((lambda _: False, None),
                             (lambda _: (_ for _ in ()).throw(RuntimeError("down")), RetryableTaskError)):
            state = FakeState()
            runner = lambda *_: self.fail("model must not run")
            if error:
                with self.assertRaises(error):
                    process_task(task(), state, runner=runner, settings_check=check)
                self.assertIn(("fail", "retryable"), state.actions)
            else:
                self.assertEqual(process_task(task(), state, runner=runner,
                                              settings_check=check).outcome, "no_op")
                self.assertNotIn("writing", state.actions)

    def test_late_turn_in_same_chunk_sends_only_pending_evidence(self):
        full = task(3)
        previous = tuple(processed_turn(turn) for turn in full.turns[:2])
        state = FakeState(BeginResult("started", previous))
        seen = []
        def runner(_task, chunks, _before):
            seen.extend(chunks)
            return 0
        process_task(full, state, runner=runner, settings_check=lambda _: True)
        self.assertEqual([[row["trace_id"] for row in chunk] for chunk in seen], [["trace-2"]])

    def test_changed_content_under_same_trace_requires_review(self):
        original = task()
        body = original.to_dict()
        body["turns"][0]["answer"] = "edited answer"
        changed = MemoryTask.from_dict(body)
        state = FakeState(BeginResult("started", (processed_turn(original.turns[0]),)))
        with self.assertRaises(PermanentTaskError):
            process_task(changed, state, runner=lambda *_: 0, settings_check=lambda _: True)
        self.assertIn(("fail", "needs_review"), state.actions)

    def test_completion_failure_after_confirmed_write_requires_review(self):
        state = FakeState(complete_error=RuntimeError("down"))
        def runner(_task, _chunks, before):
            before()
            return 1
        with self.assertRaises(PermanentTaskError):
            process_task(task(), state, runner=runner, settings_check=lambda _: True)
        self.assertIn(("fail", "needs_review"), state.actions)

    def test_read_failure_is_retryable_but_uncertain_write_is_review(self):
        for uncertain, expected in ((False, RetryableTaskError), (True, PermanentTaskError)):
            state = FakeState()
            def runner(_task, _chunks, before, uncertain=uncertain):
                if uncertain:
                    before()
                error = RuntimeError("tool failed")
                error.uncertain_mutation = uncertain
                raise error
            with self.assertRaises(expected):
                process_task(task(), state, runner=runner, settings_check=lambda _: True)


class TestAgentFailures(unittest.TestCase):
    def test_write_failure_then_nothing_cannot_be_checkpointed_as_no_op(self):
        ctx = agent.tools.Ctx("http://unused", "9876543210")
        calls = [
            {"name": "write_memory", "args": {}},
            {"name": "nothing_to_remember", "args": {"why": "nothing"}},
        ]
        with patch.dict(agent.tools.IMPL, {"write_memory": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("down"))}):
            results = agent._run_tools(ctx, calls)
        self.assertTrue(ctx.uncertain_mutation)
        self.assertTrue(ctx.tool_failures)
        self.assertFalse(results[1]["acknowledged"])
        self.assertFalse(any(row.get("action") == "nothing" for row in ctx.writes))


class TestLeases(unittest.TestCase):
    def setUp(self):
        self.clock = [dt.datetime(2026, 10, 19, tzinfo=dt.timezone.utc)]
        self.backing = {}

    def store(self):
        return MemoryQdrant(self.backing, self.clock)

    def test_active_lease_defers_and_only_one_claim_wins(self):
        first, second = self.store(), self.store()
        self.assertEqual(first.begin(task()).outcome, "started")
        self.assertEqual(second.begin(task()).outcome, "deferred")

    def test_expired_claim_before_write_is_taken_over_after_restart(self):
        self.assertEqual(self.store().begin(task()).outcome, "started")
        self.clock[0] += dt.timedelta(seconds=31)
        self.assertEqual(self.store().begin(task()).outcome, "started")

    def test_expired_claim_after_write_started_becomes_needs_review(self):
        first = self.store()
        self.assertEqual(first.begin(task()).outcome, "started")
        first.mark_writing(task())
        self.clock[0] += dt.timedelta(seconds=31)
        self.assertEqual(self.store().begin(task()).outcome, "needs_review")
        self.assertEqual(self.backing[task().subject_key]["status"], "needs_review")

    def test_administrative_recovery_requires_explicit_confirmation(self):
        first = self.store()
        first.begin(task())
        first.mark_writing(task())
        self.clock[0] += dt.timedelta(seconds=31)
        recovery = self.store()
        recovery.begin(task())
        with self.assertRaises(RuntimeError):
            recovery.administrative_recover(task())
        recovery.administrative_recover(task(), replay_confirmed_safe=True)
        self.assertEqual(recovery.begin(task()).outcome, "started")

    def test_state_contains_hashes_not_farmer_or_conversation(self):
        self.store().begin(task())
        stored = repr(self.backing)
        payload = self.backing[task().subject_key]
        self.assertRegex(payload["farmer_key"], r"^[0-9a-f]{64}$")
        for private in ("9876543210", "question 0", "answer 0", "trace-0"):
            self.assertNotIn(private, stored)


class Queue:
    def __init__(self, body):
        self.body, self.actions = body, []
    def receive(self): return QueueMessage("receipt", self.body)
    def acknowledge(self, receipt): self.actions.append("ack")
    def retry(self, receipt, reason): self.actions.append("retry")
    def dead_letter(self, receipt, reason): self.actions.append("dead_letter")


class TestComposition(unittest.TestCase):
    def test_consumer_to_processor_composition(self):
        queue, state = Queue(task().to_dict()), FakeState()
        with patch("task_processor._memory_enabled", return_value=True), patch(
                "task_processor._default_runner", return_value=0):
            result = run_one(queue, state)
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(queue.actions, ["ack"])

    def test_package_imports_from_repository_root(self):
        env = dict(os.environ, QDRANT_URL="http://127.0.0.1:1",
                   MARQO_URL="http://127.0.0.1:1", DREAMER_BASE_URL="http://127.0.0.1:1",
                   MEMORY_API_URL="http://127.0.0.1:1")
        result = subprocess.run(
            [sys.executable, "-c", (
                "import dreamer.agent, dreamer.dreamer, dreamer.erase_farmer, "
                "dreamer.worker"
            )],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(os.environ.get("AMUL_TASK_STATE_TEST_QDRANT"), "disposable Qdrant not configured")
class TestDisposableQdrant(unittest.TestCase):
    def test_claim_restart_and_completion(self):
        endpoint = os.environ["AMUL_TASK_STATE_TEST_QDRANT"]
        collection = "task_state_" + os.urandom(6).hex()
        store = QdrantTaskStateStore(endpoint, collection, api_key="", lease_seconds=30)
        value = task()
        self.assertEqual(store.begin(value).outcome, "started")
        store.complete(value, "no_op", [processed_turn(value.turns[0])])
        restarted = QdrantTaskStateStore(endpoint, collection, api_key="", lease_seconds=30)
        self.assertEqual(restarted.begin(value).outcome, "completed")


if __name__ == "__main__":
    unittest.main()
