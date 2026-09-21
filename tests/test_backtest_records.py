"""The backtest must never replay a request that already includes its answer."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import replay_records


def generation(timestamp, messages):
    tools = [{"name": name, "parameters": {"type": "object", "properties": {}}}
             for name in ("search_documents", "get_union_scheme_data", "create_health_call")]
    return {"startTime": timestamp, "input": {"messages": messages, "tools": tools}}


class TestReplayRecords(unittest.TestCase):
    def test_first_agent_request_excludes_later_generated_answer(self):
        before = [{"role": "system", "content": "You are Sarlaben."},
                  {"role": "user", "parts": [{"type": "text", "content": "Where is the receipt?"}]}]
        after = before + [{"role": "assistant", "parts": [{"type": "text", "content": "LEAKED ANSWER"}]}]
        trace = {"observations": [generation("2026-09-01T12:01:00Z", after),
                                  generation("2026-09-01T12:00:00Z", before)]}
        replay = replay_records.prepare(trace, "Where is the receipt?")
        self.assertEqual(replay["messages"][-1]["role"], "user")
        self.assertNotIn("LEAKED ANSWER", str(replay["messages"]))
        self.assertEqual(replay["tools"][0]["function"]["name"], "search_documents")

    def test_no_initial_request_cannot_silently_become_a_replay(self):
        trace = {"observations": [generation("now", [{"role": "system", "content": "Sarlaben"},
                 {"role": "assistant", "content": "Already answered"}])]}
        with self.assertRaises(ValueError):
            replay_records.prepare(trace, "Question")
