"""Offline tests for the telemetry-independent queue boundary."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamer"))

from queue_worker import PermanentTaskError, QueueMessage, RetryableTaskError, consume_one  # noqa: E402
from tasks import (  # noqa: E402
    MAX_ANSWER_CHARS, MAX_QUERY_CHARS, MAX_SESSION_ID_CHARS, MAX_SOURCE_CHARS,
    MAX_TRACE_ID_CHARS, MAX_TURNS, MemoryTask, TaskValidationError,
)


def task_body():
    return {
        "schema_version": 2,
        "farmer_id": "9876543210",
        "surface": "chat",
        "session_id": "session-1",
        "source": "current-chat-adapter",
        "episode_number": 1,
        "episode_revision": 1,
        "episode_started_at": "2026-10-17T08:00:00Z",
        "episode_ended_at": "2026-10-17T08:00:00Z",
        "part_number": 1,
        "part_count": 1,
        "pending_trace_ids": ["trace-1"],
        "turns": [
            {
                "trace_id": "trace-1",
                "timestamp": "2026-10-17T08:00:00Z",
                "query": "My cow has stopped eating",
                "answer": "Please check her temperature.",
            }
        ],
    }


class FakeQueue:
    def __init__(self, body=None):
        self.message = None if body is None else QueueMessage("receipt-1", body)
        self.actions = []

    def receive(self):
        return self.message

    def acknowledge(self, receipt):
        self.actions.append(("ack", receipt))

    def retry(self, receipt, reason):
        self.actions.append(("retry", receipt, reason))

    def dead_letter(self, receipt, reason):
        self.actions.append(("dead_letter", receipt, reason))


class TestMemoryTask(unittest.TestCase):
    def test_round_trip_and_keys_are_stable(self):
        first = MemoryTask.from_dict(task_body())
        second = MemoryTask.from_dict(first.to_dict())
        self.assertEqual(first, second)
        self.assertEqual(first.task_key, second.task_key)
        self.assertEqual(len(first.task_key), 64)
        self.assertEqual(len(first.subject_key), 64)
        summary = str(first.redacted_summary())
        for private in (first.farmer_id, first.session_id, first.turns[0].query, first.turns[0].answer):
            self.assertNotIn(private, summary)

    def test_changed_event_changes_task_key_but_not_subject_key(self):
        first = MemoryTask.from_dict(task_body())
        changed = task_body()
        changed["turns"][0]["answer"] = "Check whether she is chewing cud."
        second = MemoryTask.from_dict(changed)
        self.assertNotEqual(first.task_key, second.task_key)
        self.assertEqual(first.subject_key, second.subject_key)

    def test_episode_identity_includes_session_surface_and_number(self):
        first = MemoryTask.from_dict(task_body())
        for field, value in (
            ("session_id", "session-2"),
            ("surface", "voice"),
            ("episode_number", 2),
        ):
            changed = task_body()
            changed[field] = value
            with self.subTest(field=field):
                self.assertNotEqual(
                    first.subject_key, MemoryTask.from_dict(changed).subject_key
                )

    def test_revision_part_and_turns_change_task_not_episode_subject(self):
        first = MemoryTask.from_dict(task_body())
        variants = []
        revised = task_body()
        revised["episode_revision"] = 2
        variants.append(revised)
        part = task_body()
        part.update(part_number=2, part_count=2)
        variants.append(part)
        changed_trace = task_body()
        changed_trace["turns"][0]["trace_id"] = "trace-2"
        changed_trace["pending_trace_ids"] = ["trace-2"]
        variants.append(changed_trace)
        for body in variants:
            task = MemoryTask.from_dict(body)
            self.assertNotEqual(first.task_key, task.task_key)
            self.assertEqual(first.subject_key, task.subject_key)

    def test_timestamps_are_canonical_before_hashing(self):
        variants = []
        for timestamp in ("2026-10-17T08:00:00Z", "2026-10-17T08:00:00+00:00",
                          "2026-10-17T13:30:00+05:30"):
            body = task_body()
            body["turns"][0]["timestamp"] = timestamp
            variants.append(MemoryTask.from_dict(body))
        self.assertEqual({task.task_key for task in variants}, {variants[0].task_key})
        self.assertTrue(all(task.turns[0].timestamp == "2026-10-17T08:00:00Z" for task in variants))

    def test_identity_source_and_field_limits(self):
        body = task_body()
        body.update(session_id="s" * MAX_SESSION_ID_CHARS, source="s" * MAX_SOURCE_CHARS)
        body["turns"][0].update(trace_id="t" * MAX_TRACE_ID_CHARS,
                                query="q" * MAX_QUERY_CHARS,
                                answer="a" * MAX_ANSWER_CHARS)
        body["pending_trace_ids"] = ["t" * MAX_TRACE_ID_CHARS]
        MemoryTask.from_dict(body)
        invalid = (
            ("farmer_id", " 9876543210"), ("farmer_id", "anonymous"),
            ("farmer_id", "../9876543210"), ("farmer_id", "98765432101"),
            ("session_id", "s" * (MAX_SESSION_ID_CHARS + 1)),
            ("source", "https://user:secret@example.test"),
            ("source", "ok\ncredential"), ("source", "s" * (MAX_SOURCE_CHARS + 1)),
        )
        for field, value in invalid:
            changed = task_body()
            changed[field] = value
            with self.subTest(field=field, value=value[:20]), self.assertRaises(TaskValidationError):
                MemoryTask.from_dict(changed)

    def test_accepts_ordered_multi_turn_episode(self):
        body = task_body()
        body["episode_ended_at"] = "2026-10-17T08:20:00Z"
        body["turns"].append({
            "trace_id": "trace-2",
            "timestamp": "2026-10-17T08:20:00Z",
            "query": "She also has a fever",
            "answer": "Please measure it and contact the veterinarian.",
        })
        body["pending_trace_ids"] = ["trace-1", "trace-2"]
        task = MemoryTask.from_dict(body)
        self.assertEqual(len(task.turns), 2)
        self.assertEqual(task.pending_trace_ids, ("trace-1", "trace-2"))

    def test_turn_count_and_pending_evidence_limits(self):
        body = task_body()
        MemoryTask.from_dict(body)
        for turns in ([], [copy.deepcopy(body["turns"][0])] * (MAX_TURNS + 1)):
            changed = task_body()
            changed["turns"] = turns
            with self.assertRaisesRegex(TaskValidationError, "invalid_turn_count"):
                MemoryTask.from_dict(changed)
        for pending, code in (
            ([], "invalid_pending_trace_ids"),
            (["trace-1", "trace-1"], "duplicate_pending_trace_id"),
            (["trace-unknown"], "unknown_pending_trace_id"),
        ):
            changed = task_body()
            changed["pending_trace_ids"] = pending
            with self.subTest(code=code), self.assertRaises(TaskValidationError) as caught:
                MemoryTask.from_dict(changed)
            self.assertEqual(caught.exception.code, code)

    def test_rejects_malformed_tasks(self):
        def extra(body):
            body["telemetry_metadata"] = {}

        mutations = (
            ("unsupported_schema_version", lambda body: body.__setitem__("schema_version", 1)),
            ("invalid_task_fields", extra),
            ("invalid_surface", lambda body: body.__setitem__("surface", "email")),
            ("invalid_episode_number", lambda body: body.__setitem__("episode_number", 0)),
            ("invalid_episode_revision", lambda body: body.__setitem__("episode_revision", 0)),
            ("invalid_part_range", lambda body: body.update(part_number=2, part_count=1)),
            ("invalid_episode_range", lambda body: body.update(
                episode_started_at="2026-10-17T09:00:00Z",
                episode_ended_at="2026-10-17T08:00:00Z",
            )),
            ("invalid_turn_timestamp", lambda body: body["turns"][0].__setitem__("timestamp", "bad")),
            ("turn_timestamp_needs_timezone",
             lambda body: body["turns"][0].__setitem__("timestamp", "2026-10-17T08:00:00")),
            ("invalid_farmer_id", lambda body: body.__setitem__("farmer_id", " ")),
            ("invalid_session_id", lambda body: body.__setitem__("session_id", "")),
            ("invalid_source", lambda body: body.__setitem__("source", " ")),
            ("invalid_turn_query", lambda body: body["turns"][0].__setitem__("query", "")),
            ("invalid_turn_answer", lambda body: body["turns"][0].__setitem__("answer", " ")),
        )
        for code, mutate in mutations:
            body = task_body()
            mutate(body)
            with self.subTest(code=code), self.assertRaises(TaskValidationError) as caught:
                MemoryTask.from_dict(body)
            self.assertEqual(caught.exception.code, code)

    def test_redacted_summary_never_emits_source(self):
        body = task_body()
        body["source"] = "credential-like-source"
        summary = str(MemoryTask.from_dict(body).redacted_summary())
        self.assertNotIn(body["source"], summary)
        self.assertIn("source_key", summary)


class TestConsumer(unittest.TestCase):
    def test_acknowledges_once_and_only_after_success(self):
        queue = FakeQueue(task_body())
        seen = []

        def process(task):
            self.assertEqual(queue.actions, [])
            seen.append(task)

        result = consume_one(queue, process)
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(len(seen), 1)
        self.assertEqual(queue.actions, [("ack", "receipt-1")])

    def test_invalid_task_is_dead_lettered_without_processing(self):
        body = task_body()
        body["farmer_id"] = ""
        queue = FakeQueue(body)
        result = consume_one(queue, lambda task: self.fail("must not process"))
        self.assertEqual(result.outcome, "dead_lettered")
        self.assertEqual(queue.actions, [
            ("dead_letter", "receipt-1", "invalid_task:invalid_farmer_id")
        ])

    def test_processing_errors_retry_without_leaking_details(self):
        for error in (
            RetryableTaskError("private farmer text"),
            RuntimeError("https://secret.invalid/?token=private"),
        ):
            queue = FakeQueue(task_body())

            def fail(task, error=error):
                raise error

            result = consume_one(queue, fail)
            self.assertEqual(result.outcome, "retry_scheduled")
            self.assertEqual(queue.actions, [
                ("retry", "receipt-1", f"retryable:{type(error).__name__}")
            ])
            self.assertNotIn(str(error), str(queue.actions))

    def test_permanent_error_is_dead_lettered(self):
        queue = FakeQueue(task_body())

        def fail(task):
            raise PermanentTaskError("unsafe detail")

        result = consume_one(queue, fail)
        self.assertEqual(result.outcome, "dead_lettered")
        self.assertEqual(queue.actions, [
            ("dead_letter", "receipt-1", "permanent:PermanentTaskError")
        ])

    def test_empty_queue_is_idle(self):
        queue = FakeQueue()
        result = consume_one(queue, lambda task: self.fail("must not process"))
        self.assertEqual(result.outcome, "idle")
        self.assertEqual(queue.actions, [])


if __name__ == "__main__":
    unittest.main()
