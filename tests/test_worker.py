"""Which Langfuse turns the worker hands to the dreamer."""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import env_fixture  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))
import worker  # noqa: E402


def row(**md):
    meta = {"amul.schema_version": "turn.v1", "amul.service": "amul-oan-api", "amul.persona": "farmer",
            "amul.outcome": "success", "amul.pipeline": "webapp", "amul.farmer_id": "9876543210", **md}
    return {"id": "t1", "timestamp": "2026-11-04T10:00:00Z", "sessionId": "s1", "userId": "9876543210",
            "input": "my cow has fever", "output": "give it water", "metadata": meta}


class EligibleTest(unittest.TestCase):
    def test_farmer_turn_is_kept(self):
        self.assertEqual(worker.eligible(row())[0], "9876543210")

    def test_rejected_turns(self):
        for md in ({"amul.persona": "doctor"}, {"amul.outcome": "moderation_rejected"},
                   {"amul.pipeline": "telephony"}, {"amul.farmer_id": None}, {"amul.farmer_id": "1111111111"}):
            self.assertIsNone(worker.eligible(row(**md)), md)
        other_user = row()
        other_user["userId"] = "9123456780"
        self.assertIsNone(worker.eligible(other_user))

    def test_int_farmer_id_as_langfuse_stores_it(self):
        self.assertEqual(worker.eligible(row(**{"amul.farmer_id": 9876543210}))[0], "9876543210")

    def test_signed_in_voice_is_kept(self):
        self.assertIsNotNone(worker.eligible(row(**{"amul.pipeline": "telephony", "amul.signed_in": True})))


class AllowlistTest(unittest.TestCase):
    def run_cycle(self, allowlist, rows):
        dream = mock.Mock(return_value=0)
        with mock.patch.object(worker, "ALLOWLIST", allowlist), \
                mock.patch.object(worker, "read_mark", return_value=None), \
                mock.patch.object(worker, "write_mark"), \
                mock.patch.object(worker.lf, "LangfuseRO") as ro, \
                mock.patch.object(worker.dreamer, "dream", dream), \
                contextlib.redirect_stdout(io.StringIO()):
            ro.return_value.traces.return_value = rows
            worker.cycle()
        return ro.return_value.traces, dream

    def test_empty_list_reads_and_writes_nothing(self):
        traces, dream = self.run_cycle(set(), [row()])
        traces.assert_not_called()
        dream.assert_not_called()

    def test_list_is_read_per_farmer(self):
        traces, dream = self.run_cycle({"9876543210"}, [row()])
        self.assertEqual(traces.call_args.kwargs["userId"], "9876543210")
        self.assertEqual(list(dream.call_args.args[0]), ["9876543210"])

    def test_row_outside_the_list_is_dropped(self):
        _, dream = self.run_cycle({"9123456780"}, [row()])
        dream.assert_not_called()

    def test_star_is_everyone(self):
        traces, dream = self.run_cycle({"*"}, [row()])
        self.assertNotIn("userId", traces.call_args.kwargs)
        self.assertEqual(list(dream.call_args.args[0]), ["9876543210"])


class QuietLogTest(unittest.TestCase):
    def run_cycle(self, log_content, dream):
        out = io.StringIO()
        with mock.patch.object(worker, "LOG_CONTENT", log_content), \
                mock.patch.object(worker, "ALLOWLIST", {"*"}), \
                mock.patch.object(worker, "read_mark", return_value=None), \
                mock.patch.object(worker, "write_mark"), \
                mock.patch.object(worker.lf, "LangfuseRO") as ro, \
                mock.patch.object(worker.dreamer, "dream", side_effect=dream), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            ro.return_value.traces.return_value = [row()]
            worker.cycle()
        return out.getvalue()

    def test_quiet_log_keeps_counts_only(self):
        def dream(queues, args):
            print("headline: cow Gauri got Ceftriaxone")
            print("note: farmer worried", file=sys.stderr)
            return 1
        log = self.run_cycle(False, dream)
        self.assertNotIn("Gauri", log)
        self.assertNotIn("worried", log)
        self.assertIn("1 turns -> 1 written", log)

    def test_quiet_log_hides_error_text(self):
        def dream(queues, args):
            raise ValueError("LLM said: cow Gauri got Ceftriaxone")
        log = self.run_cycle(False, dream)
        self.assertNotIn("Gauri", log)
        self.assertIn("failed: ValueError", log)

    def test_content_log_when_enabled(self):
        self.assertIn("Gauri", self.run_cycle(True, lambda q, a: print("cow Gauri") or 1))


if __name__ == "__main__":
    unittest.main()
