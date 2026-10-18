"""Offline tests for the telemetry-independent queue boundary."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamer"))

from queue_worker import PermanentTaskError, QueueMessage, RetryableTaskError, consume_one  # noqa: E402
from tasks import (  # noqa: E402
    MAX_ANSWER_CHARS, MAX_CONTENT_CHARS, MAX_QUERY_CHARS, MAX_SESSION_ID_CHARS, MAX_SOURCE_CHARS,
    MAX_TRACE_ID_CHARS, MAX_TURNS, MemoryTask, TaskValidationError,
)


def task_body():
    return {
        "schema_version": 1,
        "farmer_id": "9876543210",
        "session_id": "session-1",
        "source": "current-chat-adapter",
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

    def test_late_turn_changes_task_key_but_not_subject_key(self):
        first = MemoryTask.from_dict(task_body())
        changed = task_body()
        changed["turns"].append({
            "trace_id": "trace-2",
            "timestamp": "2026-10-17T08:10:00Z",
            "query": "Her temperature is normal",
            "answer": "Check whether she is chewing cud.",
        })
        second = MemoryTask.from_dict(changed)
        self.assertNotEqual(first.task_key, second.task_key)
        self.assertEqual(first.subject_key, second.subject_key)

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
        for field, value in (
            ("trace_id", "t" * (MAX_TRACE_ID_CHARS + 1)),
            ("query", "q" * (MAX_QUERY_CHARS + 1)),
            ("answer", "a" * (MAX_ANSWER_CHARS + 1)),
        ):
            changed = task_body()
            changed["turns"][0][field] = value
            with self.subTest(field=field), self.assertRaises(TaskValidationError):
                MemoryTask.from_dict(changed)

    def test_turn_count_and_payload_limits(self):
        body = task_body()
        template = body["turns"][0]
        body["turns"] = [dict(template, trace_id=f"t-{index}",
                              timestamp=f"2026-10-17T08:{index // 60:02d}:{index % 60:02d}Z",
                              query="q", answer="a") for index in range(MAX_TURNS)]
        MemoryTask.from_dict(body)
        body["turns"].append(dict(body["turns"][-1], trace_id="too-many"))
        with self.assertRaisesRegex(TaskValidationError, "invalid_turn_count"):
            MemoryTask.from_dict(body)

        aggregate = task_body()
        template = aggregate["turns"][0]
        per_answer = (MAX_CONTENT_CHARS // 8) - 1
        aggregate["turns"] = [dict(template, trace_id=f"aggregate-{index}",
                                         timestamp=f"2026-10-17T08:0{index}:00Z",
                                         query="q", answer="a" * per_answer)
                              for index in range(8)]
        MemoryTask.from_dict(aggregate)
        aggregate["turns"][-1]["answer"] += "a"
        with self.assertRaisesRegex(TaskValidationError, "task_content_too_large"):
            MemoryTask.from_dict(aggregate)

        oversized = task_body()
        template = oversized["turns"][0]
        oversized["turns"] = [dict(template, trace_id=f"large-{index}",
                                          timestamp=f"2026-10-17T08:0{index}:00Z",
                                          query="q", answer="🐄" * MAX_ANSWER_CHARS)
                              for index in range(8)]
        with self.assertRaisesRegex(TaskValidationError, "task_too_large"):
            MemoryTask.from_dict(oversized)

    def test_rejects_malformed_tasks(self):
        def extra(body):
            body["telemetry_metadata"] = {}

        def duplicate(body):
            body["turns"].append(copy.deepcopy(body["turns"][0]))

        def unordered(body):
            later = copy.deepcopy(body["turns"][0])
            later.update(trace_id="later", timestamp="2026-10-17T09:00:00Z")
            body["turns"].insert(0, later)

        mutations = (
            ("unsupported_schema_version", lambda body: body.__setitem__("schema_version", 2)),
            ("invalid_task_fields", extra),
            ("duplicate_trace_id", duplicate),
            ("turns_not_chronological", unordered),
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
