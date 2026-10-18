"""Focused contracts for the simple, one-task-at-a-time memory processor."""
import json
import os
import sys
import unittest
import uuid

import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [os.path.join(ROOT, "dreamer"), os.path.join(ROOT, "service")]

from qdrant_task_state import QdrantTaskStateStore  # noqa: E402
from queue_worker import PermanentTaskError, RetryableTaskError  # noqa: E402
from task_processor import BeginResult, process_task, turn_fingerprint  # noqa: E402
from tasks import MemoryTask  # noqa: E402


def make_task(count=1):
    return MemoryTask.from_dict({
        "schema_version": 1,
        "farmer_id": "9876543210",
        "session_id": "session-private",
        "source": "normalized-chat",
        "turns": [{
            "trace_id": f"trace-{index}",
            "timestamp": f"2026-10-17T08:{index:02d}:00Z",
            "query": f"private question {index}",
            "answer": f"private answer {index}",
        } for index in range(count)],
    })


class FakeState:
    def __init__(self, begin=None):
        self.result = begin or BeginResult("started")
        self.actions = []

    def begin(self, task):
        self.actions.append(("begin", task.task_key))
        return self.result

    def complete(self, task, outcome, processed_turns):
        self.actions.append(("complete", outcome, tuple(processed_turns)))

    def fail(self, task, disposition):
        self.actions.append(("fail", disposition))


class TestTaskProcessor(unittest.TestCase):
    def test_new_task_runs_and_completes(self):
        task, state = make_task(), FakeState()
        result = process_task(task, state, runner=lambda _task, chunks: (2, True))
        self.assertEqual(
            (result.outcome, result.processed_chunks, result.writes), ("completed", 1, 2)
        )
        self.assertEqual(state.actions[-1][0:2], ("complete", "completed"))

    def test_completed_redelivery_skips_runner(self):
        state = FakeState(BeginResult("completed", ("old",)))
        result = process_task(make_task(), state, runner=lambda *_: self.fail("must not run"))
        self.assertEqual(result.outcome, "already_completed")

    def test_late_turn_processes_only_affected_chunk(self):
        task = make_task(11)
        processed = tuple(turn_fingerprint(turn) for turn in task.turns[:10])
        state, seen = FakeState(BeginResult("started", processed)), []

        def runner(_task, chunks):
            seen.extend(chunks)
            return 1, True

        process_task(task, state, runner=runner)
        self.assertEqual(
            [[turn["trace_id"] for turn in chunk] for chunk in seen], [["trace-10"]]
        )

    def test_nothing_written_is_a_successful_no_op(self):
        state = FakeState()
        result = process_task(make_task(), state, runner=lambda *_: (0, False))
        self.assertEqual(result.outcome, "no_op")
        self.assertEqual(state.actions[-1][0:2], ("complete", "no_op"))

    def test_failure_before_write_is_retryable(self):
        state = FakeState()

        def fail(*_):
            raise RuntimeError("private failure text")

        with self.assertRaises(RetryableTaskError):
            process_task(make_task(), state, runner=fail)
        self.assertEqual(state.actions[-1], ("fail", "retryable"))

    def test_uncertain_write_requires_review(self):
        state = FakeState()

        def fail(*_):
            error = RuntimeError("private failure text")
            error.mutation_attempted = True
            raise error

        with self.assertRaises(PermanentTaskError):
            process_task(make_task(), state, runner=fail)
        self.assertEqual(state.actions[-1], ("fail", "needs_review"))


@unittest.skipUnless(
    os.getenv("AMUL_TASK_STATE_TEST_QDRANT"), "disposable Qdrant is not running"
)
class TestQdrantTaskState(unittest.TestCase):
    def test_state_is_separate_and_contains_hashes_only(self):
        endpoint = os.environ["AMUL_TASK_STATE_TEST_QDRANT"].rstrip("/")
        collection = "task_state_" + uuid.uuid4().hex
        task = make_task()
        store = QdrantTaskStateStore(endpoint, collection, "")
        try:
            self.assertEqual(store.begin(task).outcome, "started")
            store.complete(task, "completed", [turn_fingerprint(task.turns[0])])
            self.assertEqual(
                QdrantTaskStateStore(endpoint, collection, "").begin(task).outcome,
                "completed",
            )
            body = requests.post(
                f"{endpoint}/collections/{collection}/points/scroll",
                json={"limit": 10, "with_payload": True, "with_vector": False}, timeout=10,
            ).json()
            encoded = json.dumps(body)
            self.assertIn(task.task_key, encoded)
            for private in (task.farmer_id, task.session_id, task.turns[0].trace_id,
                            task.turns[0].query, task.turns[0].answer):
                self.assertNotIn(private, encoded)
        finally:
            requests.delete(f"{endpoint}/collections/{collection}", timeout=10)


if __name__ == "__main__":
    unittest.main()
