"""Unit tests: no network, no model, no running service.

Everything here was previously "verified" by a throwaway script I ran once and
deleted. That is not verification — it is a memory of verification. These are the
behaviours that can regress silently, which is the dangerous kind: break the
document/query embedding convention and search still WORKS, it just stops meaning
anything.

Stdlib unittest on purpose — no pytest in either image, and a test suite that needs
an install before it runs is a test suite nobody runs.

    python3 -m unittest discover -s tests -v      (from the repo root)
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch
import env_fixture

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

import dates          # noqa: E402
import llm            # noqa: E402
import prompts        # noqa: E402
import writer         # noqa: E402


class TestDateFormat(unittest.TestCase):
    """Date formatting used in the background job's own prompts."""

    def test_iso_forms(self):
        for raw in ("2026-09-01T06:00:00Z", "2026-09-01", "2026-09-01T06:00:00+00:00"):
            self.assertEqual(dates.fmt(raw), "01-September-2026", raw)

    def test_day_is_zero_padded(self):
        self.assertEqual(dates.fmt("2026-09-08T10:16:33.125Z"), "08-September-2026")

    def test_unknown_is_explicit_never_blank(self):
        # A blank date reads as a missing field; an explicit marker does not.
        for raw in (None, "", "garbage", 0):
            self.assertEqual(dates.fmt(raw), dates.UNKNOWN, repr(raw))

    def test_span_collapses_same_day(self):
        self.assertEqual(
            dates.span("2026-09-01T06:00:00Z", "2026-09-01T22:00:00Z"), "01-September-2026")

    def test_span_shows_range(self):
        self.assertEqual(dates.span("2026-08-30T06:00:00Z", "2026-09-04T06:00:00Z"),
                         "30-August-2026 to 04-September-2026")

    def test_span_with_one_end_missing(self):
        self.assertEqual(dates.span(None, "2026-09-04T06:00:00Z"), "04-September-2026")

class TestStructuredOutputShape(unittest.TestCase):
    """Only the local vLLM enforces the schema. Everywhere else `json_object`
    guarantees valid JSON and says nothing about its keys — a model given a prompt
    with headings will mirror those headings instead. Valid-and-useless is the worst
    failure shape, because nothing raises."""

    SCHEMA = {"type": "object",
              "properties": {"expanded": {"type": "string"}, "n": {"type": "integer"}},
              "required": ["expanded"]}

    def test_accepts_right_shape(self):
        ok, why = llm._shape_ok({"expanded": "an account"}, self.SCHEMA)
        self.assertTrue(ok, why)

    def test_rejects_invented_keys(self):
        # The real observed failure: {"situation","status","history","account"}.
        ok, why = llm._shape_ok({"situation": "x", "account": "y"}, self.SCHEMA)
        self.assertFalse(ok)
        self.assertIn("expanded", why)

    def test_rejects_right_key_wrong_type(self):
        # Also real: {"expanded": 0.0} passed a presence check and blew up on
        # .strip() several frames from the cause.
        for bad in (0.0, None, ["a"], {"a": 1}, 7):
            ok, why = llm._shape_ok({"expanded": bad}, self.SCHEMA)
            self.assertFalse(ok, repr(bad))
            self.assertIn("must be a string", why)

    def test_rejects_non_object_top_level(self):
        for bad in ([1, 2], "text", 5):
            ok, _ = llm._shape_ok(bad, self.SCHEMA)
            self.assertFalse(ok, repr(bad))

    def test_boolean_is_not_a_number(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}},
                  "required": ["n"]}
        ok, why = llm._shape_ok({"n": True}, schema)
        self.assertFalse(ok)
        self.assertIn("boolean", why)

    def test_required_keys_are_stated_in_the_prompt(self):
        """Prompts must name their keys rather than rely on enforcement one provider
        applies and others ignore."""
        hint = llm._required_hint(self.SCHEMA)
        self.assertIn('"expanded"', hint)
        self.assertIn("verbatim", hint)

    def test_every_real_schema_has_required_keys(self):
        for name in dir(prompts):
            if not name.endswith("_SCHEMA"):
                continue
            schema = getattr(prompts, name)
            self.assertTrue(schema.get("required"),
                            f"{name} declares no required keys, so nothing can "
                            "validate what the model returns")


class TestIdentifierStripping(unittest.TestCase):
    """An ear tag may be STORED (it makes 'the same animal as last time' certain) but
    must never reach prose, because prose is read back to the farmer. Deterministic,
    and it runs after the judges so it cannot fail open."""

    def test_removes_known_tag_from_prose(self):
        out, moved = writer.strip_identifiers(
            {"headline": "Her cow 101520248441 is unwell.", "expanded": "",
             "metadata": {"animal_id": "101520248441"}})
        self.assertNotIn("101520248441", out["headline"])
        self.assertEqual(out["headline"], "Her cow is unwell.")
        self.assertTrue(moved)

    def test_handles_list_valued_animal_id(self):
        """One situation can cover two animals. str() on a list never matches the
        prose, so this silently did nothing until it was fixed."""
        out, moved = writer.strip_identifiers(
            {"headline": "Dispute over AI dates for animals 101520248441 and 105760548663.",
             "expanded": "", "metadata": {"animal_id": ["101520248441", "105760548663"]}})
        self.assertNotIn("101520248441", out["headline"])
        self.assertNotIn("105760548663", out["headline"])
        self.assertTrue(moved)

    def test_strips_unregistered_identifiers(self):
        """A tag the model wrote into prose WITHOUT registering it in metadata was
        missed entirely by the known-tag check."""
        out, moved = writer.strip_identifiers(
            {"headline": "He asked about animal 999888777666.", "expanded": "",
             "metadata": {}})
        self.assertNotIn("999888777666", out["headline"])
        self.assertTrue(moved)

    def test_keeps_amounts_and_dates(self):
        """Amounts are four or five digits and dates are punctuated, so nothing
        legitimate may be caught."""
        out, moved = writer.strip_identifiers(
            {"headline": "Paid 12000 rupees on 2026-08-17 for feed.", "expanded": "",
             "metadata": {}})
        self.assertEqual(out["headline"], "Paid 12000 rupees on 2026-08-17 for feed.")
        self.assertFalse(moved)

    def test_metadata_keeps_the_tag(self):
        out, _ = writer.strip_identifiers(
            {"headline": "Her cow 101520248441 is unwell.", "expanded": "",
             "metadata": {"animal_id": "101520248441"}})
        self.assertEqual(out["metadata"]["animal_id"], "101520248441")

    def test_no_dangling_connectors_left(self):
        out, _ = writer.strip_identifiers(
            {"headline": "Animals 101520248441 and 105760548663 are affected.",
             "expanded": "", "metadata": {"animal_id": ["101520248441", "105760548663"]}})
        self.assertNotIn(" and .", out["headline"])
        self.assertNotIn("  ", out["headline"])


class TestFarmerFilter(unittest.TestCase):
    """No date is ever compared to 'now' to decide visibility. Three separate bugs
    came from that axis, so this is pinned."""

    def setUp(self):
        os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
        import memory_api
        self.api = memory_api

    def _conds(self, flt):
        import json
        return json.dumps(flt)

    def test_farmer_id_is_always_applied(self):
        flt = self.api._farmer_filter("F1")
        self.assertIn({"key": "farmer_id", "match": {"value": "F1"}}, flt["must"])

    def test_no_date_comparison_anywhere(self):
        text = self._conds(self.api._farmer_filter("F1", only_current=True))
        self.assertNotIn("expires_at", text)
        self.assertNotIn("gte", text)
        self.assertNotIn("lte", text)

    def test_superseded_is_a_state_check_not_a_date(self):
        flt = self.api._farmer_filter("F1", only_current=True)
        self.assertIn({"is_empty": {"key": "superseded_at"}}, flt["must"])

    def test_only_current_false_drops_the_state_check(self):
        flt = self.api._farmer_filter("F1", only_current=False)
        self.assertNotIn({"is_empty": {"key": "superseded_at"}}, flt["must"])

    def test_internal_types_are_always_excluded(self):
        flt = self.api._farmer_filter("F1")
        excluded = {c["match"]["value"] for c in flt["must_not"]}
        self.assertIn(self.api.SETTINGS_TYPE, excluded)
        self.assertIn(self.api.PROFILE_TYPE, excluded)
        self.assertIn(self.api.WORKER_CHECKPOINT_TYPE, excluded)

    def test_embedding_model_detection_excludes_checkpoints(self):
        with patch.object(self.api, "_qdrant", new_callable=AsyncMock,
                          return_value={"result": {"points": []}}) as qdrant:
            self.assertIsNone(asyncio.run(self.api._collection_embed_model()))
        query = qdrant.call_args.args[2]
        excluded = {row["match"]["value"] for row in query["filter"]["must_not"]}
        self.assertIn(self.api.WORKER_CHECKPOINT_TYPE, excluded)

    def test_status_accepts_a_list(self):
        flt = self.api._farmer_filter("F1", status=["open", "pending"])
        self.assertIn({"key": "status", "match": {"any": ["open", "pending"]}}, flt["must"])

    def test_metadata_filter_is_applied(self):
        """This was accepted by callers and silently ignored for a while, so a
        'filtered' listing returned everything."""
        flt = self.api._farmer_filter("F1", metadata_key="severity", metadata_value="high")
        self.assertIn({"key": "metadata.severity", "match": {"value": "high"}}, flt["must"])

    def test_profile_type_is_not_the_profile_entry_type(self):
        """'profile' is a real remembered fact; the Layer-1 singleton is a different
        type. Conflating them hid legitimate memories from search."""
        self.assertNotEqual(self.api.PROFILE_TYPE, "profile")

    def test_agent_hidden_keys_cover_identifiers(self):
        for k in ("animal_id", "account_number", "phone"):
            self.assertIn(k, self.api.AGENT_HIDDEN_KEYS)


if __name__ == "__main__":
    unittest.main()
