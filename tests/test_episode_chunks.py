"""Lossless chunk tests and optional real-Qdrant contracts with fixture embeddings."""
import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import env_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "service"), str(ROOT / "dreamer")]
import episode
import memory_api as api
import httpx


class TestLayout(unittest.TestCase):
    def test_embedding_migration_uses_actual_chunk_and_level2_text(self):
        import reembed
        parent = {"id": "parent", "payload": {"farmer_id": "one", "type": "memory",
                  "headline": "Treatment history", "expanded": "first" * 20 + "mineral" * 10,
                  "chunk_count": 2, "chunks": episode.split_text("first" * 20 + "mineral" * 10, 100),
                  "contents": [{"start_chunk": 1, "end_chunk": 2, "text": "Mineral treatment outcome"}]}}
        child = {"id": "child", "payload": {"farmer_id": "one", "type": "memory_chunk",
                 "episode_id": "parent", "chunk_number": 2}}
        self.assertEqual(reembed.vector_texts(child, {"parent": parent}), {"expanded": "mineral" * 10})
        self.assertIn("Mineral treatment outcome", reembed.vector_texts(parent, {})["headline"])
        self.assertNotIn("expanded", reembed.vector_texts(parent, {}))
        with self.assertRaises(ValueError):
            reembed.vector_texts(child, {})

    def test_chunks_preserve_every_character_and_respect_budget(self):
        for text in ("A sentence.\n\nAnother paragraph. " * 50, "દવા પછી સુધારો નહોતો. " * 70, "x" * 2041):
            chunks = episode.split_text(text, 100)
            self.assertEqual("".join(text[c["start"]:c["end"]] for c in chunks), text)
            self.assertTrue(all(0 < c["end"] - c["start"] <= 100 for c in chunks))
            self.assertEqual([c["number"] for c in chunks], list(range(1, len(chunks) + 1)))

    def test_five_rows_cover_any_number_of_chunks(self):
        text = "feed, advice and later developments. " * 200
        chunks = episode.split_text(text, 100)
        rows = episode.extractive_contents(text, chunks)
        self.assertLessEqual(len(rows), 5)
        self.assertEqual(episode.validate_contents(rows, len(chunks)), rows)
        for invalid in ([rows[0]], [dict(rows[0], start_chunk=2)] + rows[1:],
                        [dict(rows[0], text="x" * 1000)] + rows[1:]):
            with self.assertRaises(ValueError):
                episode.validate_contents(invalid, len(chunks))

    def test_whole_chunk_pages_preserve_full_text_and_next_numbers(self):
        text = "x" * 500
        payload = {"expanded": text, "chunks": episode.split_text(text, 100)}
        rows, next_ids = episode.whole_chunk_page(payload)
        self.assertEqual([r["number"] for r in rows], [1, 2, 3])
        self.assertEqual(next_ids, [4, 5])
        self.assertTrue(all(r["text"] == "x" * 100 for r in rows))
        more, next_ids = episode.whole_chunk_page(payload, next_ids)
        self.assertFalse(next_ids)
        self.assertEqual("".join(r["text"] for r in rows + more), text)
        selected, _ = episode.whole_chunk_page(payload, [2, 5])
        self.assertEqual([r["number"] for r in selected], [2, 5])

    def test_reading_budget_and_continuation_never_lose_text(self):
        text = "Alpha paragraph.\n\n" * 100
        payload = {"expanded": text, "chunks": episode.split_text(text, 100)}
        continuation, got = {}, ""
        while True:
            rows, continuation = episode.read_chunks(payload, max_chars=73, **continuation)
            self.assertLessEqual(sum(len(r["text"]) for r in rows), 73)
            got += "".join(r["text"] for r in rows)
            if continuation is None:
                break
        self.assertEqual(got, text)

    def test_nonadjacent_selection_does_not_read_intervening_chunks(self):
        text = "x" * 400
        payload = {"expanded": text, "chunks": episode.split_text(text, 100)}
        rows, next_ = episode.read_chunks(payload, [2, 4], max_chars=150)
        self.assertEqual([r["number"] for r in rows], [2, 4])
        self.assertEqual(next_, {"chunk_ids": [4], "offset": 50})
        rows, next_ = episode.read_chunks(payload, next_["chunk_ids"], offset=next_["offset"], max_chars=150)
        self.assertEqual(rows[0]["text"], "x" * 50)
        self.assertIsNone(next_)

    def test_long_accounts_are_not_condensed_before_checks(self):
        import writer
        import tools
        text = "Treatment was tried; outcome remains uncertain.\n\n" * 100
        chunks = episode.split_text(text)
        outline = {"headline": "Treatment tried; outcome uncertain.",
                   "contents": episode.extractive_contents(text, chunks)}
        ctx = tools.Ctx("http://unused", "synthetic", dry_run=True)
        with patch.object(writer.llm, "ask_json", return_value={"headline": outline["headline"], "descriptions": [r["text"] for r in outline["contents"]]}), patch.object(
                writer.judges, "review", side_effect=lambda value, **kw: ("allow", value, [])):
            result = tools.write_memory(ctx, "Treatment discussion", expanded=text)
        self.assertTrue(result["written"])
        self.assertEqual(ctx.writes[0]["entry"]["expanded"], text)
        self.assertEqual(ctx.writes[0]["entry"]["contents"], outline["contents"])


@unittest.skipUnless(os.getenv("AMUL_CHUNK_TEST_QDRANT"), "run with tests/run_isolated.py")
class TestQdrantChunks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.collection = "chunk_test_" + uuid.uuid4().hex
        self.patches = [patch.object(api, "QDRANT_URL", os.environ["AMUL_CHUNK_TEST_QDRANT"]),
                        patch.object(api, "COLLECTION", self.collection), patch.object(api, "EMBED_DIM", 3),
                        patch.object(api, "_embed_many", side_effect=self.embed)]
        for p in self.patches:
            p.start()
        await api._ensure_collection()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        await api._qdrant("DELETE", "/collections/" + self.collection)
        for p in reversed(self.patches):
            p.stop()

    @staticmethod
    async def embed(texts, **kwargs):
        # Fixture vectors test real storage, filtering and ranking plumbing—not model quality.
        return [[float("mineral" in t.lower()), float("subsidy" in t.lower()), 0.01] for t in texts]

    async def write(self, farmer="one", **extra):
        body = {"headline": "A continuing feeding episode", "expanded": "General feeding detail. " * 100 +
                "\n\nThe mineral mixture did not help.\n\n" + "Later observations. " * 100,
                "metadata": {"scheme": "feed", "animal_id": "123456789012"}, "status": "open", **extra}
        response = await self.client.post(f"/memory/{farmer}/entries", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_public_whole_chunk_read_ignores_character_budget(self):
        result = await self.write()
        ref = result["id"]
        response = await self.client.post(f"/memory/one/entries/{ref}/read", json={
            "whole_chunks": True, "chunk_ids": [1], "max_chars": 1})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertGreater(len(body["chunks"][0]["text"]), 100)
        self.assertNotIn("offset", body["chunks"][0])
        self.assertEqual(body["continuation"]["chunk_ids"][0], 2)

    async def test_write_search_and_direct_read_reach_a_buried_detail(self):
        stored = await self.write()
        ref, payload = stored["id"], stored["payload"]
        self.assertGreater(payload["chunk_count"], 2)
        listing = (await self.client.get("/memory/one/entries")).json()
        self.assertEqual(len(listing["entries"]), 1)  # chunks must never appear as episodes
        found = (await self.client.post("/memory/one/search_expanded", json={"query": "mineral", "limit": 1})).json()
        hit = found["hits"][0]
        self.assertEqual(hit["id"], ref)
        self.assertIn("mineral mixture", hit["passage"])
        self.assertNotIn("animal_id", hit["metadata"])
        part = (await self.client.post(f"/memory/one/entries/{ref}/read", json={
            "chunk_ids": [hit["chunk_number"]], "max_chars": 70})).json()
        self.assertLessEqual(part["returned_chars"], 70)
        self.assertIsNotNone(part["continuation"])
        self.assertEqual(part["chunks"][0]["number"], hit["chunk_number"])
        level2 = (await self.client.post("/memory/one/search", json={"query": "mineral"})).json()["hits"][0]
        self.assertTrue(level2["contents"])
        self.assertNotIn("expanded", level2)

    async def test_settings_filters_and_episode_scope_apply_to_chunks(self):
        first = await self.write()
        await self.write("two", headline="Other farmer's episode")
        for body in ({"query": "mineral", "filters": {"scheme": "other"}},
                     {"query": "mineral", "memory_ref": str(uuid.uuid4())}):
            result = (await self.client.post("/memory/one/search_expanded", json=body)).json()
            self.assertEqual(result["hits"], [])
        await self.client.put("/memory/one/settings", json={"memory_enabled": False})
        result = (await self.client.post("/memory/one/search_expanded", json={"query": "mineral"})).json()
        self.assertFalse(result["memory_enabled"])
        result = (await self.client.post("/internal/memory/one/search_expanded", json={"query": "mineral"})).json()
        self.assertTrue(result["hits"])
        part = await self.client.post(f"/memory/two/entries/{first['id']}/read", json={})
        self.assertEqual(part.status_code, 404)

    async def test_updates_retire_old_chunks_but_old_references_stay_readable(self):
        old = await self.write(source_traces=["t1"], source_session_id="s1")
        response = await self.client.patch(f"/memory/one/entries/{old['id']}", json={
            "headline": "Updated episode", "expanded": "Subsidy was discussed instead.",
            "source_traces": ["t2"], "source_session_id": "s2"})
        self.assertEqual(response.status_code, 200, response.text)
        new = response.json()["new"]["id"]
        hits = (await self.client.post("/memory/one/search_expanded", json={"query": "mineral"})).json()["hits"]
        self.assertEqual({h["id"] for h in hits}, {new})
        old_read = (await self.client.post(f"/memory/one/entries/{old['id']}/read", json={"chunk_ids": [1]})).json()
        self.assertFalse(old_read["current"])
        history = (await self.client.get(f"/memory/one/entries/{new}/history")).json()
        self.assertEqual(len(history["history"]), 2)

    async def test_orphan_chunks_cannot_be_returned(self):
        stored = await self.write()
        await api._qdrant("POST", f"/collections/{self.collection}/points/delete?wait=true", {"points": [stored["id"]]})
        result = (await self.client.post("/memory/one/search_expanded", json={"query": "mineral"})).json()
        self.assertEqual(result["hits"], [])

    async def test_legacy_reindex_preserves_evidence_and_does_not_enable_farmer(self):
        ref = str(uuid.uuid4())
        payload = {"type": "tracker", "farmer_id": "one", "headline": "Legacy episode", "expanded": "mineral detail " * 200,
                   "status": "open", "source_traces": ["old-trace"], "source_ts": "2026-08-01T00:00:00Z"}
        await api._qdrant("PUT", f"/collections/{self.collection}/points?wait=true", {"points": [
            {"id": ref, "payload": payload, "vector": {"headline": [1., 0., .01], "expanded": [1., 0., .01]}}]})
        await self.client.put("/memory/one/settings", json={"memory_enabled": False})
        indexed = await self.client.post(f"/internal/memory/one/entries/{ref}/index")
        self.assertEqual(indexed.status_code, 200, indexed.text)
        self.assertEqual(indexed.json()["payload"]["expanded"], payload["expanded"])
        self.assertEqual(indexed.json()["payload"]["source_traces"], ["old-trace"])
        self.assertFalse((await self.client.get("/memory/one/settings")).json()["memory_enabled"])


if __name__ == "__main__":
    unittest.main()
