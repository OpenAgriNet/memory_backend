"""Which Langfuse turns the worker hands to the dreamer."""
import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
