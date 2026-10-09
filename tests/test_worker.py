"""Focused contracts for the scheduled worker boundary."""
import datetime as dt
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import env_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "dreamer"), str(ROOT / "service")]

import lf
import tools
import worker

NOW = dt.datetime(2026, 10, 9, 12, tzinfo=dt.timezone.utc)


def trace(trace_id="trace-1", **values):
    row = {
        "id": trace_id, "name": "chat.translation", "environment": "chat-production",
        "timestamp": "2026-10-09T11:00:00Z", "updatedAt": "2026-10-09T11:01:00Z",
        "sessionId": "session-1", "metadata": {"persona": "farmer", "memory_farmer_id": "9876543210"},
        "input": {"query": "My cow is unwell"}, "output": "Please contact the veterinarian.",
    }
    row.update(values)
    return row


class Reader:
    def __init__(self, rows):
        self.rows = rows

    def trace_window(self, *_args, **_kwargs):
        return self.rows

    def session_traces(self, session_id, *_args):
        return [row for row in self.rows if row.get("sessionId") == session_id]


class CheckpointStub:
    def __init__(self):
        self.calls = []
        self.version = 1

    def claim(self, kind, subject, farmer, candidate_fingerprint, last_activity, run_id):
        self.calls.append(("claim", kind, subject, candidate_fingerprint))
        return {"claimed": True, "lease_token": "lease", "checkpoint_version": self.version,
                "processed_traces": []}

    def transition(self, action, lease, **values):
        self.calls.append((action, lease.kind, lease.subject, values))
        if action == "renew":
            self.version += 1
            return {"renewed": True, "checkpoint_version": self.version}
        return {"checkpoint": {"status": "retryable"}}

    def due(self):
        raise AssertionError("dry-run must not read or mutate checkpoints")

    def capabilities(self):
        return {"atomic_mutations": True, "qdrant_version": "1.19.1"}


class WorkerContract(unittest.TestCase):
    def test_discovery_filters_identity_pipeline_and_inactivity(self):
        invalid = [
            trace("doctor", metadata={"persona": "doctor", "memory_farmer_id": "9876543210"}),
            trace("default", name="chat.default"),
            trace("wrong-env", environment="default"),
            trace("generic-id", metadata={"persona": "farmer"}, userId="9876543210"),
            trace("bad-mobile", metadata={"persona": "farmer", "memory_farmer_id": "uuid-value"}),
            trace("unfinished", output=None),
        ]
        active = trace("active", sessionId="active-session", timestamp="2026-10-09T11:40:00Z")
        found = worker.discover(
            Reader([trace(), active, *invalid]), NOW - dt.timedelta(hours=2), NOW,
            inactive_before=NOW - dt.timedelta(minutes=30),
        )
        self.assertEqual(list(found), [("9876543210", "session-1")])

    def test_langfuse_paginates_and_splits_only_oversized_windows(self):
        reader = lf.LangfuseRO.__new__(lf.LangfuseRO)
        calls = []

        def get(_path, params):
            calls.append((params["fromTimestamp"], params["toTimestamp"], params["page"]))
            if params["toTimestamp"] == "2026-10-09T12:00:00Z" and params["fromTimestamp"] == "2026-10-09T08:00:00Z":
                raise TimeoutError
            return {"data": [{"id": str(len(calls))}] if params["page"] == 1 else [],
                    "meta": {"totalPages": 2}}

        reader._get = get
        rows = reader.trace_window(NOW - dt.timedelta(hours=4), NOW, name="chat.translation")
        self.assertEqual(len(rows), 2)
        self.assertTrue(any(page == 2 for _, _, page in calls))

    def test_resumed_context_cannot_be_provenance(self):
        turns = [worker.trace_turn(trace(f"t-{n}", timestamp=f"2026-10-09T0{n}:00:00Z")) for n in range(1, 5)]
        previous = [worker.processed_trace(turn) for turn in turns[:3]]
        chunk = next(worker.chunks_with_context(turns, previous))
        self.assertEqual([turn["trace_id"] for turn in chunk], ["t-2", "t-3", "t-4"])
        self.assertEqual([turn["context_only"] for turn in chunk], [True, True, False])
        ctx = tools.Ctx("http://memory", "9876543210", turns=chunk)
        with self.assertRaisesRegex(ValueError, "context-only"):
            ctx.for_sources([1])
        self.assertEqual(ctx.for_sources([3]).trace_ids, ["t-4"])

    def test_lifecycle_renews_and_completes_with_one_fingerprint(self):
        client, rows = CheckpointStub(), [trace()]
        candidate = {"turns": [worker.trace_turn(rows[0])], "last_activity": rows[0]["timestamp"]}

        with patch.object(worker, "run_dreamer", side_effect=RuntimeError("partial failure")):
            failed, first_value = worker.process_session(
                Reader(rows), client, "run-1", "9876543210", "session-1", candidate, NOW, False,
            )
        self.assertEqual(failed["failed"], 1)
        self.assertEqual(client.calls[-1][0], "fail")

        def dream(*_args, **_kwargs):
            time.sleep(0.03)
            return "completed"

        with patch.object(worker, "LEASE_RENEW_SECONDS", 0.01), \
                patch.object(worker, "run_dreamer", side_effect=dream):
            result, value = worker.process_session(
                Reader(rows), client, "run", "9876543210", "session-1",
                candidate, NOW, False,
            )
        self.assertEqual(result["completed"], 1)
        self.assertEqual(first_value, value)
        self.assertEqual(value, worker.fingerprint([worker.trace_turn(rows[0])]))
        self.assertIn("renew", [call[0] for call in client.calls])
        self.assertEqual(client.calls[-1][0], "complete")

    def test_dry_run_and_preflight_are_read_only(self):
        reader, client = Reader([trace()]), CheckpointStub()
        with patch.object(worker, "run_dreamer", return_value="no_op"), \
                patch.object(worker, "process_maintenance", return_value="no_op"):
            self.assertEqual(worker.run("incremental", True, reader=reader, client=client, now=NOW), 0)
        self.assertEqual(client.calls, [])
        with patch.object(worker.llm.client.models, "list", return_value=[SimpleNamespace(id=worker.llm.MODEL)]):
            self.assertTrue(worker.preflight(reader, client)["atomic_mutations"])

        with patch.object(worker, "process_session") as session, \
                patch.object(worker, "process_maintenance", return_value="completed") as maintenance:
            worker.process_farmer(
                reader, client, "run", "9876543210", {},
                {"farmer_profile": "due-fingerprint"}, NOW, False,
            )
        session.assert_not_called()
        maintenance.assert_called_once()

        command = subprocess.run(
            [sys.executable, "-m", "dreamer.worker", "--help"], cwd=ROOT,
            env=os.environ, capture_output=True, text=True,
        )
        self.assertEqual(command.returncode, 0, command.stderr)


if __name__ == "__main__":
    unittest.main()
