"""Regression checks using synthetic memories; no services or model calls."""
import asyncio
import hashlib
import os
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch
import env_fixture

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path[:0] = [os.path.join(ROOT, "dreamer"), os.path.join(ROOT, "service")]

import dreamer
import memory_api as api
import tools
import writer


def entry(eid="old", farmer="test-farmer", **values):
    return {"id": eid, "farmer_id": farmer, "type": "tracker", "status": "open",
            "headline": "A remembered situation", "expanded": "Earlier useful detail",
            "source_ts": "2026-09-01T00:00:00Z", "source_sessions": ["s1"],
            "source_traces": ["t1"], "metadata": {"scheme": "fodder"}, **values}


class TestSourceWindow(unittest.TestCase):
    def test_frozen_replay_rejects_future_turns_before_model_or_log_calls(self):
        import json
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as source:
            json.dump([{"ts": "2026-09-03T00:00:00Z", "query": "future question"}], source)
            source.flush()
            args = ["dreamer", "--farmer", "synthetic", "--turns-file", source.name,
                    "--until", "2026-09-02T00:00:00Z"]
            with patch.object(sys, "argv", args), patch.object(dreamer.lf, "LangfuseRO", side_effect=AssertionError("no network")), patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "after the cutoff"):
                    dreamer.main()


    def test_timeout_splits_window_and_deduplicates_boundary_traces(self):
        import datetime as dt
        import lf
        client = object.__new__(lf.LangfuseRO)
        calls = []
        def get(path, params):
            calls.append(dict(params))
            start = dt.datetime.fromisoformat(params["fromTimestamp"].replace("Z", "+00:00"))
            end = dt.datetime.fromisoformat(params["toTimestamp"].replace("Z", "+00:00"))
            if end - start > dt.timedelta(hours=12):
                raise TimeoutError("narrow the window")
            return {"data": [{"id": "boundary", "timestamp": "2026-09-01T12:00:00Z",
                              "sessionId": "s1", "input": {"query": "earlier question"}, "output": "advice"}],
                    "meta": {"totalPages": 1}}
        with patch.object(client, "_get", side_effect=get):
            turns = client.farmer_turns("one", days_back=1, until="2026-09-02T00:00:00Z")
        self.assertEqual(len(calls), 3)
        self.assertEqual([t["trace_id"] for t in turns], ["boundary"])
        self.assertEqual(calls[0]["fromTimestamp"], "2026-09-01T00:00:00Z")


class TestWriterReview(unittest.TestCase):
    def test_headline_only_update_preserves_detail_and_metadata(self):
        ctx = tools.Ctx("http://unused", "test-farmer", dry_run=True)
        old = entry()
        with patch.object(writer, "_fetch", return_value=old), patch.object(
                writer.judges, "review", side_effect=lambda e, **kw: ("allow", e, [])):
            result = tools.update_memory(ctx, "old", "A clearer headline")
        self.assertTrue(result["updated"])
        saved = ctx.writes[0]["entry"]
        self.assertEqual(saved["expanded"], old["expanded"])
        self.assertEqual(saved["metadata"], old["metadata"])

    def test_new_level2_and_full_detail_are_judged_before_writing(self):
        ctx = tools.Ctx("http://unused", "test-farmer", dry_run=True)
        judged = []
        def review(proposal, **kwargs):
            judged.append(proposal)
            return "allow", proposal, []
        with patch.object(writer, "_fetch", return_value=entry()), patch.object(
                writer.judges, "review", side_effect=review):
            tools.update_memory(ctx, "old", "Updated", expanded="Full account remains intact")
        self.assertEqual(judged[0]["expanded"], "Full account remains intact")
        self.assertTrue(judged[0]["contents"])
        self.assertEqual(ctx.writes[0]["entry"]["expanded"], judged[0]["expanded"])

    def test_update_judge_sees_new_distinct_session_evidence(self):
        ctx = tools.Ctx("unused", "test-farmer", dry_run=True,
                        turns=[{"query": "again", "session_id": "s2", "trace_id": "t2",
                                "ts": "2026-09-03T00:00:00Z"}])
        evidence = []
        def review(value, **kwargs):
            evidence.append(kwargs["record"])
            return "allow", value, []
        with patch.object(writer, "_fetch", return_value=entry(times_raised=1)), patch.object(
                writer.judges, "review", side_effect=review):
            tools.update_memory(ctx, "old", "Raised again", source_turns=[1])
        self.assertEqual(evidence[0]["times_raised"], 2)
        self.assertEqual(evidence[0]["source_ts"], "2026-09-03T00:00:00Z")

    def test_failed_write_can_be_retried_in_same_run(self):
        ctx = tools.Ctx("http://unused", "test-farmer")
        response = Mock()
        response.json.return_value = {"id": "new"}
        with patch.object(writer.judges, "review", side_effect=lambda e, **kw: ("allow", e, [])), patch.object(
                writer.requests, "post", side_effect=[RuntimeError("temporary outage"), response]) as post:
            first = tools.write_memory(ctx, "An unresolved situation")
            second = tools.write_memory(ctx, "An unresolved situation")
        self.assertFalse(first["written"])
        self.assertTrue(second["written"])
        self.assertEqual(post.call_count, 2)

    def test_merge_receives_late_detail_from_every_fragment(self):
        import prompts
        late = "The farmer reported that no AI receipt was supplied."
        prompt = prompts.build_merge_prompt([entry(expanded="Earlier detail. " * 200 + late)], "Registration")
        self.assertIn(late, prompt)
        self.assertIn("Earlier detail. " * 200, prompt)
        prompt = prompts.build_merge_prompt([entry(), entry("second")], "One situation")
        self.assertIn("DISTINCT SUPPORTING CONVERSATIONS: 1", prompt)

    def test_consolidation_uses_latest_source_date(self):
        older, newer = entry(), entry("newer", source_ts="2026-09-02T00:00:00Z")
        outputs = [{"groups": [{"members": [1, 2], "confidence": "high", "situation": "One problem"}]},
                   {"headline": "Combined account", "expanded": "Both accounts", "status": "open"}]
        contexts = []
        def apply(ctx, change):
            contexts.append(ctx)
            return {"ok": True, "entry_id": "combined"}
        with patch.object(dreamer, "_current_entries", return_value=[older, newer]), patch.object(
                dreamer.llm, "ask_json", side_effect=outputs), patch.object(dreamer.writer, "apply", side_effect=apply):
            result = dreamer.consolidate("test-farmer")
        self.assertEqual(len(result), 1)
        self.assertEqual(contexts[0].source_ts, newer["source_ts"])


class TestFoldReview(unittest.IsolatedAsyncioTestCase):
    async def test_missing_or_other_farmers_target_cannot_retire_memory(self):
        source = {"id": "old", "payload": entry()}
        for target in ([], [{"id": "target", "payload": entry("target", farmer="someone-else")}],
                       [{"id": "target", "payload": entry("target", superseded_at="2026-09-02")} ]):
            with self.subTest(target=target), patch.object(api, "_qdrant", new_callable=AsyncMock) as q:
                q.side_effect = [{"result": [source]}, {"result": target}, {}]
                with self.assertRaises(api.HTTPException):
                    await api.fold_entry("test-farmer", "old", api.FoldRequest(merged_into="target"))
                self.assertFalse(any("payload?" in call.args[1] for call in q.call_args_list))


class TestSourceEvidence(unittest.TestCase):
    def test_one_situation_does_not_acquire_unrelated_sources(self):
        turns = [
            {"query": "first problem", "trace_id": "t1", "session_id": "s1", "ts": "2026-08-01"},
            {"query": "unrelated problem", "trace_id": "t2", "session_id": "s2", "ts": "2026-09-01"},
            {"query": "first problem again", "trace_id": "t3", "session_id": "s3", "ts": "2026-09-02"},
        ]
        ctx = tools.Ctx("unused", "farmer", turns=turns, dry_run=True)
        selected = ctx.for_sources([3, 1, 1])
        provenance = writer._provenance(selected)
        self.assertEqual(provenance["source_traces"], ["t1", "t3"])
        self.assertEqual(provenance["source_sessions"], ["s1", "s3"])
        self.assertEqual(provenance["first_source_ts"], "2026-08-01")
        self.assertEqual(provenance["source_ts"], "2026-09-02")
        self.assertIs(ctx.writes, selected.writes)
        for invalid in (None, [], [0], [4], [True], ["1"]):
            with self.assertRaises(ValueError):
                ctx.for_sources(invalid)

    def test_interleaved_sessions_remain_chronological(self):
        turns = [{"ts": f"2026-09-0{n}", "session_id": sid, "query": "hello"}
                 for n, sid in [(1, "a"), (2, "b"), (3, "a")]]
        self.assertEqual([t for c in dreamer.chunk_turns(turns, cap=2) for t in c], turns)


class TestChecks(unittest.TestCase):
    def test_unavailable_or_malformed_check_retries_and_blocks(self):
        for results in ([RuntimeError("down"), RuntimeError("down")],
                        [{"verdict": "probably"}, {"verdict": "probably"}]):
            with patch.object(writer.judges, "_pii", side_effect=results) as pii, patch.object(
                    writer.judges, "_keys", return_value={"decisions": []}), patch.object(
                    writer.judges, "_honesty", return_value={"verdict": "allow"}):
                decision, _, notes = writer.judges.review(entry())
            self.assertEqual(decision, "block")
            self.assertEqual(pii.call_count, 2)
            self.assertIn("retry needed", " ".join(notes))

    def test_transient_check_can_succeed_on_retry(self):
        with patch.object(writer.judges, "_pii", side_effect=[RuntimeError("down"), {"verdict": "allow"}]), patch.object(
                writer.judges, "_keys", return_value={"decisions": []}), patch.object(
                writer.judges, "_honesty", return_value={"verdict": "allow"}):
            self.assertEqual(writer.judges.review(entry())[0], "allow")

    def test_program_owned_profile_bounds_and_unknown_fields(self):
        self.assertEqual(api.profile_schema.validate({"explicit_requests": "Keep answers short"}),
                         {"explicit_requests": "Keep answers short"})
        self.assertEqual(api.profile_schema.validate({"invented": "new slot"}), {})
        for fields in ({"explicit_requests": "x" * 1000},):
            with self.assertRaises(ValueError):
                api.profile_schema.validate(fields)


class TestReadPolicy(unittest.IsolatedAsyncioTestCase):
    async def test_default_on_and_explicit_off_are_separate_from_metadata(self):
        for payload, expected in [(None, True), ({"memory_enabled": True}, True),
                                  ({"memory_enabled": False}, False), ({"memory_enabled": "false"}, False)]:
            with patch.object(api, "_qdrant", new_callable=AsyncMock) as q:
                q.return_value = {"result": [] if payload is None else [{"payload": payload}]}
                self.assertEqual(await api._memory_enabled("farmer"), expected)
                self.assertEqual(q.call_args.args[2]["ids"], [api._settings_point_id("farmer")])
        with patch.object(api, "_qdrant", new_callable=AsyncMock, side_effect=api.HTTPException(503)):
            self.assertFalse(await api._memory_enabled("farmer"))

    async def test_disabled_reply_reads_and_background_reads_do_not_change_settings(self):
        import httpx
        ref = "00000000-0000-0000-0000-000000000001"
        stored = {"id": ref, "payload": entry(ref, farmer="farmer"), "score": 0.8}
        async def qdrant(method, path, payload):
            # All calls here must be read operations, never a write or setting update.
            self.assertEqual(method, "POST")
            if path.endswith("/scroll"):
                return {"result": {"points": [stored], "next_page_offset": None}}
            if path.endswith("/query"):
                return {"result": {"points": [stored]}}
            self.assertEqual(payload["ids"], [ref])
            return {"result": [stored]}
        with patch.object(api, "_memory_enabled", new_callable=AsyncMock, return_value=False), patch.object(
                api, "_qdrant", side_effect=qdrant), patch.object(api, "_embed", new_callable=AsyncMock, return_value=[0.0]):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                for suffix in ("entries", "entries/" + ref, "profile", "keys", "entries/" + ref + "/history"):
                    response = await client.get("/memory/farmer/" + suffix)
                    self.assertEqual(response.status_code, 200)
                    self.assertFalse(response.json()["memory_enabled"])
                for suffix in ("search", "search_expanded"):
                    response = await client.post("/memory/farmer/" + suffix, json={"query": "problem"})
                    self.assertFalse(response.json()["memory_enabled"])
                    response = await client.post("/internal/memory/farmer/" + suffix, json={"query": "problem"})
                    self.assertEqual(response.json()["hits"][0]["id"], ref)
                response = await client.get("/internal/memory/farmer/entries/" + ref)
                self.assertEqual(response.json()["entry"]["id"], ref)
                self.assertNotIn("type", response.json()["entry"])
                response = await client.get("/internal/memory/farmer/entries")
                self.assertEqual(len(response.json()["entries"]), 1)

    async def test_direct_read_rejects_another_farmer_or_internal_record(self):
        for stored in (entry(farmer="someone-else"), entry(type=api.SETTINGS_TYPE)):
            with patch.object(api, "_qdrant", new_callable=AsyncMock, return_value={"result": [{"id": "old", "payload": stored}]}):
                with self.assertRaises(api.HTTPException) as error:
                    await api._read_entry("test-farmer", "old")
                self.assertEqual(error.exception.status_code, 404)

    def test_filters_keep_scalar_types_and_exclude_internal_records(self):
        flt = api._farmer_filter("farmer", filters={"scheme": "fodder", "attempt": 2, "confirmed": False, "dose": 1.5})
        self.assertIn({"key": "metadata.confirmed", "match": {"value": False}}, flt["must"])
        self.assertIn({"key": "metadata.attempt", "match": {"value": 2}}, flt["must"])
        self.assertIn({"key": "metadata.dose", "range": {"gte": 1.5, "lte": 1.5}}, flt["must"])
        for kind in (api.SETTINGS_TYPE, api.PROFILE_TYPE, api.KEY_DOC_TYPE):
            self.assertIn({"key": "type", "match": {"value": kind}}, flt["must_not"])


class TestWriteOptOut(unittest.IsolatedAsyncioTestCase):
    async def test_every_farmer_write_path_stops_before_mutation_when_disabled(self):
        calls = (
            lambda: api._store_entry("9876543210", api.NewEntry(headline="memory")),
            lambda: api.index_existing_episode("9876543210", "entry"),
            lambda: api.update_entry("9876543210", "entry", api.UpdateEntry(headline="changed")),
            lambda: api.fold_entry("9876543210", "entry", api.FoldRequest(merged_into="other")),
            lambda: api.put_profile("9876543210", api.ProfileRecord(fields={})),
        )
        with patch.object(api, "_memory_enabled_strict", new_callable=AsyncMock, return_value=False), patch.object(
                api, "_qdrant", new_callable=AsyncMock) as qdrant:
            for call in calls:
                with self.subTest(call=call), self.assertRaises(api.HTTPException) as error:
                    await call()
                self.assertEqual(error.exception.status_code, 403)
            qdrant.assert_not_called()

    async def test_write_and_settings_endpoint_fail_closed_when_lookup_is_unavailable(self):
        failure = api.HTTPException(503, "qdrant unavailable")
        with patch.object(api, "_memory_enabled_strict", new_callable=AsyncMock,
                          side_effect=failure), patch.object(
                api, "_qdrant", new_callable=AsyncMock) as qdrant:
            for call in (
                lambda: api._store_entry("9876543210", api.NewEntry(headline="memory")),
                lambda: api.get_settings("9876543210"),
            ):
                with self.subTest(call=call), self.assertRaises(api.HTTPException) as error:
                    await call()
                self.assertEqual(error.exception.status_code, 503)
            qdrant.assert_not_called()


class TestEpisodeSummaries(unittest.IsolatedAsyncioTestCase):
    def request(self, **changes):
        values = {
            "surface": "chat", "session_id": "session-1",
            "episode_number": 1, "episode_revision": 1,
            "episode_started_at": "2026-10-17T08:00:00Z",
            "episode_ended_at": "2026-10-17T08:20:00Z",
            "part_number": 1, "part_count": 2,
            "headline": "Feeding discussion", "expanded": "The first part was discussed.",
            "source_traces": ["trace-1"], "pending_trace_ids": ["trace-1"],
        }
        values.update(changes)
        return api.EpisodeSummaryRequest(**values)

    def subject(self):
        return hashlib.sha256(
            f"9876543210\0chat\0session-1\0{1}".encode("utf-8")
        ).hexdigest()

    async def test_part_advances_one_summary_and_unions_provenance(self):
        old = entry(
            record_kind="episode", episode_subject_key=self.subject(),
            episode_revision=1, episode_part_number=1, episode_part_count=2,
            first_source_ts="2026-10-17T08:00:00Z", source_traces=["trace-1"],
        )
        request = self.request(
            part_number=2, headline="Complete feeding discussion",
            expanded="Both parts were discussed.", source_traces=["trace-2"],
            pending_trace_ids=["trace-2"],
        )
        with patch.object(api, "_require_memory_enabled", new_callable=AsyncMock), patch.object(
                api, "_current_episode_summary", new_callable=AsyncMock,
                return_value={"id": "old", "payload": old}), patch.object(
                api, "_store_entry", new_callable=AsyncMock,
                return_value={"id": "new"}) as store, patch.object(
                api, "_retire", new_callable=AsyncMock) as retire:
            result = await api.put_episode_summary("9876543210", self.subject(), request)
        self.assertTrue(result["applied"])
        stored_entry = store.await_args.args[1]
        self.assertEqual(stored_entry.source_traces, ["trace-1", "trace-2"])
        self.assertEqual(store.await_args.kwargs["record_kind"], "episode")
        self.assertEqual(store.await_args.kwargs["system_fields"]["episode_part_number"], 2)
        retire.assert_awaited_once()

    async def test_retry_is_unchanged_and_identity_mismatch_is_rejected(self):
        request = self.request()
        old = entry(
            record_kind="episode", episode_subject_key=self.subject(),
            episode_revision=1, episode_part_number=1, episode_part_count=2,
            source_traces=["trace-1"],
        )
        with patch.object(api, "_require_memory_enabled", new_callable=AsyncMock), patch.object(
                api, "_current_episode_summary", new_callable=AsyncMock,
                return_value={"id": "old", "payload": old}), patch.object(
                api, "_store_entry", new_callable=AsyncMock) as store:
            result = await api.put_episode_summary("9876543210", self.subject(), request)
            self.assertTrue(result["unchanged"])
            store.assert_not_awaited()
            with self.assertRaises(api.HTTPException) as error:
                await api.put_episode_summary("9876543210", "0" * 64, request)
        self.assertEqual(error.exception.status_code, 409)

class TestPublicBoundaries(unittest.IsolatedAsyncioTestCase):
    async def test_public_metadata_cannot_expose_private_alias(self):
        import httpx
        point = {"id": "old", "score": 0.8, "payload": entry(metadata={
            "animal_id": "123456789012", "phone_number": "9876543210", "scheme": "fodder"})}
        async def qdrant(method, path, payload):
            if path.endswith("/query"):
                return {"result": {"points": [point]}}
            if path.endswith("/scroll"):
                return {"result": {"points": [point], "next_page_offset": None}}
            return {"result": [point]}
        with patch.object(api, "_memory_enabled", new_callable=AsyncMock, return_value=True), patch.object(
                api, "_qdrant", side_effect=qdrant), patch.object(api, "_embed", new_callable=AsyncMock, return_value=[0.0]):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                for path, key in (("entries?include_private=true", "entries"), ("entries/old?include_private=true", "entry")):
                    response = await client.get("/memory/test-farmer/" + path)
                    value = response.json()[key]
                    rows = value if isinstance(value, list) else [value]
                    self.assertEqual(rows[0]["metadata"], {"scheme": "fodder"})
                    self.assertNotIn("123456789012", response.text)
                response = await client.post("/memory/test-farmer/search_expanded", json={"query": "scheme"})
                self.assertEqual(response.json()["hits"][0]["metadata"], {"scheme": "fodder"})
                response = await client.get("/internal/memory/test-farmer/entries/old")
                self.assertEqual(response.json()["entry"]["metadata"]["animal_id"], "123456789012")

    async def test_post_cannot_invent_history_links(self):
        import httpx
        with patch.object(api, "_qdrant", new_callable=AsyncMock) as q:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                response = await client.post("/memory/test-farmer/entries", json={
                    "headline": "A new memory", "previous_version": "another-farmers-entry"})
                self.assertEqual(response.status_code, 422)
        q.assert_not_called()


if __name__ == "__main__":
    unittest.main()
