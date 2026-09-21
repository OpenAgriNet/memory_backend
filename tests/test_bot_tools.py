"""Live memory-tool contracts, exercised with synthetic HTTP responses only."""
import asyncio
import json
import os
import sys
import types
import unittest
import urllib.request
import uuid

BOT_ROOT = os.environ.get("AMUL_OAN_API_PATH", "/amulpfsdata/gautam/amul-oan-api")

API = os.environ.get("MEMORY_API_URL", "http://localhost:8100").rstrip("/")
QDRANT = os.environ.get("QDRANT_HTTP", "http://localhost:6350").rstrip("/")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "amul_memory")
PREFIX = "zz-tooltest-"

# --- stub what would otherwise drag in the whole app -----------------------
if "helpers.utils" not in sys.modules:
    helpers = types.ModuleType("helpers")
    utils = types.ModuleType("helpers.utils")
    utils.get_logger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        error=lambda *a, **k: None, debug=lambda *a, **k: None)
    helpers.utils = utils
    sys.modules["helpers"] = helpers
    sys.modules["helpers.utils"] = utils

if "agents.deps" not in sys.modules:
    agents = sys.modules.get("agents") or types.ModuleType("agents")
    deps = types.ModuleType("agents.deps")
    class FarmerContext:  # noqa: D101 - stand-in for the real dataclass
        pass
    deps.FarmerContext = FarmerContext
    agents.deps = deps
    sys.modules["agents"] = agents
    sys.modules["agents.deps"] = deps

if "pydantic_ai" not in sys.modules:
    pai = types.ModuleType("pydantic_ai")
    class RunContext:  # noqa: D101
        pass
    pai.RunContext = RunContext
    tools_mod = types.ModuleType("pydantic_ai.tools")
    class ToolDefinition:  # noqa: D101
        pass
    tools_mod.ToolDefinition = ToolDefinition
    pai.tools = tools_mod
    sys.modules["pydantic_ai"] = pai
    sys.modules["pydantic_ai.tools"] = tools_mod

os.environ["MEMORY_ENABLED"] = "true"
os.environ.setdefault("MEMORY_API_URL", API)
os.environ.setdefault("QDRANT_URL", "http://qdrant.test")
os.environ.setdefault("QDRANT_COLLECTION", COLLECTION)
os.environ.setdefault("MARQO_ENDPOINT_URL", "http://marqo.test")
os.environ.setdefault("MARQO_INDEX_NAME", "test-index")

def _load(name, relpath):
    """Load one module from the bot repo by path, without importing the app."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(BOT_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BOT_PRESENT = os.path.isdir(BOT_ROOT)
if _BOT_PRESENT:
    # memory.py is imported by both tools, so it has to exist under its real name.
    _load("app", "app/__init__.py") if os.path.exists(
        os.path.join(BOT_ROOT, "app/__init__.py")) else None
    _load("app.services.memory_store", "app/services/memory_store.py")
    memory_svc = _load("app.services.memory", "app/services/memory.py")
    memory_tools = _load("agents.tools.memory_tools", "agents/tools/memory_tools.py")


import httpx
from unittest.mock import patch

_REAL_CLIENT = httpx.AsyncClient
REF = "11111111-1111-4111-8111-111111111111"


def record(**overrides):
    return {"id": REF, "headline": "Calf subsidy application discussed.",
            "expanded": "The farmer was advised to contact the society; submission was not confirmed.",
            "status": "pending", "source_ts": "2026-09-01T00:00:00Z", **overrides}


def read_body(**overrides):
    row = record()
    return {"entry": row, "current": True, "chunks": [
        {"number": 1, "offset": 0, "text": row["expanded"]}], "continuation": None, **overrides}


class FixtureStore:
    """Expose the existing response fixtures as a reader, only inside unit tests.

    Synthetic HTTP here tests tool formatting, errors and timeouts. Production
    Qdrant contracts are covered in the bot's tests/test_memory_store.py.
    Replay imports this module but never installs this fake.
    """
    def __init__(self, client, farmer):
        self.client = client

    async def request(self, method, path, **kwargs):
        response = await self.client.request(method, "http://fixture.test" + path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def search(self, query, **kwargs):
        return await self.request("POST", "/search", json={"query": query, **kwargs})

    async def list_entries(self, status=None, filters=None, cursor=None, limit=8):
        params = {"limit": limit}
        if status:
            params["status"] = status
        if filters:
            params["filters"] = json.dumps(filters)
        if cursor:
            params["cursor"] = cursor
        return await self.request("GET", "/entries", params=params)

    async def read(self, memory_ref, chunk_ids=None):
        return await self.request("POST", f"/entries/{memory_ref}/read", json={"chunk_ids": chunk_ids, "whole_chunks": True})

    async def history(self, memory_ref, limit=8):
        return await self.request("GET", f"/entries/{memory_ref}/history")


@unittest.skipUnless(_BOT_PRESENT, "bot checkout missing")
class TestMemoryTools(unittest.TestCase):
    def setUp(self):
        self.ctx = types.SimpleNamespace(deps=types.SimpleNamespace(mobile="test-farmer", memory_tool_calls=0))
        self.requests = []
        patcher = patch.object(memory_tools, "MemoryStore", FixtureStore)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_tool(self, fn, *args, body=None, status=200, raw=None, **kwargs):
        def handler(request):
            self.requests.append(request)
            return (httpx.Response(status, content=raw) if raw is not None
                    else httpx.Response(status, json=body))
        def client(**options):
            return _REAL_CLIENT(transport=httpx.MockTransport(handler), **options)
        with patch.object(memory_tools.httpx, "AsyncClient", side_effect=client):
            return asyncio.run(fn(self.ctx, *args, **kwargs))

    def test_parallel_calls_share_one_turn_output_budget(self):
        async def handler(request):
            await asyncio.sleep(0)
            return httpx.Response(200, json={"hits": [record(passage="evidence " * 1000, chunk_number=1)]})
        def client(**options):
            return _REAL_CLIENT(transport=httpx.MockTransport(handler), **options)
        async def calls():
            return await asyncio.gather(*(memory_tools.search_memories(self.ctx, "feed") for _ in range(6)))
        with patch.object(memory_tools.httpx, "AsyncClient", side_effect=client):
            outputs = asyncio.run(calls())
        self.assertLessEqual(self.ctx.deps.memory_tool_chars, memory_tools.TURN_MAX_CHARS)
        self.assertTrue(any("budget reached" in output.lower() for output in outputs))

    def test_search_never_shortens_a_matching_chunk(self):
        passage = "A recorded detail. " * 60 + "The last sentence matters."
        out = self.run_tool(memory_tools.search_memories, "detail", body={"hits": [record(passage=passage, chunk_number=1)]})
        self.assertIn(passage, out)
        self.assertNotIn("read_memory for the rest", out)

    def test_read_next_page_uses_only_chunk_numbers(self):
        out = self.run_tool(memory_tools.read_memory, REF, body=read_body(continuation={"chunk_ids": [4, 5, 6]}))
        self.assertIn("chunk_ids=[4, 5, 6]", out)
        self.assertNotIn("offset", out)
        sent = json.loads(self.requests[0].content)
        self.assertEqual(sent, {"chunk_ids": None, "whole_chunks": True})

    def test_search_needs_no_keys_or_categories(self):
        out = self.run_tool(memory_tools.search_memories, "that subsidy", body={"hits": [record()]})
        sent = json.loads(self.requests[0].content)
        self.assertNotIn("filters", sent)
        self.assertNotIn("type", sent)
        self.assertIn(REF, out)
        self.assertNotIn("submission was not confirmed", out)  # summary only

    def test_listing_supports_several_typed_farmer_keys(self):
        filters = {"scheme": "calf_rearing", "attempts": 2, "submitted": False}
        self.run_tool(memory_tools.list_memories, filters=filters, body={"entries": []})
        self.assertEqual(json.loads(self.requests[0].url.params["filters"]), filters)

    def test_listing_without_filters_includes_all_states(self):
        self.run_tool(memory_tools.list_memories, body={"entries": [record()]})
        self.assertNotIn("status", self.requests[0].url.params)
        self.assertNotIn("type", self.requests[0].url.params)

    def test_unresolved_is_open_or_pending(self):
        self.run_tool(memory_tools.list_memories, state="unresolved", body={"entries": []})
        self.assertEqual(self.requests[0].url.params.get_list("status"), ["open", "pending"])

    def test_list_preserves_keys_values_and_pagination(self):
        filters = {"feed_supplement": "mineral mix", "quantity": 2}
        out = self.run_tool(memory_tools.list_memories, filters=filters, cursor=REF,
                            body={"entries": [record()], "next_cursor": REF})
        self.assertEqual(json.loads(self.requests[0].url.params["filters"]), filters)
        self.assertEqual(self.requests[0].url.params["cursor"], REF)
        self.assertIn("More records exist", out)
        self.assertNotIn("list is complete", out.lower())

    def test_unknown_state_does_not_make_a_request(self):
        out = self.run_tool(memory_tools.list_memories, state="probably")
        self.assertIn("Unknown state", out)
        self.assertFalse(self.requests)

    def test_disabled_memory_is_not_an_empty_result(self):
        for fn, args, empty in [(memory_tools.search_memories, ["subsidy"], {"hits": []}),
                                (memory_tools.list_memories, [], {"entries": []}),
                                (memory_tools.read_memory, [REF], {"entry": None})]:
            with self.subTest(fn=fn.__name__):
                out = self.run_tool(fn, *args, body={**empty, "memory_enabled": False})
                self.assertIn("disabled", out)
                self.assertNotIn("No matching", out)

    def test_failure_and_malformed_json_do_not_claim_absence(self):
        for response in ({"status": 503, "body": {}}, {"raw": b"bad json"}, {"body": {"entries": "invalid"}}):
            with self.subTest(response=response):
                out = self.run_tool(memory_tools.list_memories, **response)
                self.assertEqual(out, memory_tools.UNAVAILABLE)

    def test_read_opens_selected_result_without_searching(self):
        out = self.run_tool(memory_tools.read_memory, REF, body=read_body())
        self.assertTrue(self.requests[0].url.path.endswith("/entries/" + REF + "/read"))
        self.assertIn("submission was not confirmed", out)

    def test_invalid_reference_cannot_become_a_path(self):
        out = self.run_tool(memory_tools.read_memory, "../another-farmer")
        self.assertIn("invalid", out)
        self.assertFalse(self.requests)

    def test_superseded_result_is_identified_as_history(self):
        out = self.run_tool(memory_tools.read_memory, REF, body=read_body(current=False))
        self.assertIn("older version", out)

    def test_history_is_requested_for_the_same_entry(self):
        def handler(request):
            self.requests.append(request)
            if request.url.path.endswith("/history"):
                return httpx.Response(200, json={"history": [record(id="older", expanded="Earlier account")], "truncated": True})
            return httpx.Response(200, json={"entry": record(), "current": True})
        with patch.object(memory_tools.httpx, "AsyncClient", side_effect=lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw)):
            out = asyncio.run(memory_tools.read_memory(self.ctx, REF, include_history=True))
        self.assertIn("Earlier account", out)
        self.assertIn("part of the history", out)
        self.assertEqual(len(self.requests), 1)

    def test_remaining_turn_budget_never_causes_partial_chunk_output(self):
        self.ctx.deps.memory_tool_chars = memory_tools.TURN_MAX_CHARS - 700
        text = "A complete sentence. " * 60
        out = self.run_tool(memory_tools.read_memory, REF, body=read_body(chunks=[{"number": 1, "text": text}]))
        self.assertEqual(out, memory_tools.BUDGET_REACHED)
        self.assertNotIn("A complete sentence", out)

    def test_read_returns_the_entire_selected_chunk(self):
        text = "An actual detail. " * 60 + "A final important detail."
        out = self.run_tool(memory_tools.read_memory, REF, chunk_ids=[3],
                            body=read_body(chunks=[{"number": 3, "text": text}]))
        self.assertIn(text, out)
        self.assertNotIn("offset", out)

    def test_search_returns_a_passage_with_a_stable_chunk_reference(self):
        out = self.run_tool(memory_tools.search_memories, "mineral", memory_ref=REF, body={"hits": [
            record(chunk_number=4, passage="The mineral mixture did not help.", chunk_indexed=True)]})
        self.assertIn("chunk 4", out)
        self.assertIn("mixture did not help", out)
        self.assertEqual(json.loads(self.requests[0].content)["memory_ref"], REF)

    def test_turn_text_budget_prevents_further_requests(self):
        self.ctx.deps.memory_tool_chars = memory_tools.TURN_MAX_CHARS
        out = self.run_tool(memory_tools.search_memories, "mineral")
        self.assertIn("budget", out)
        self.assertFalse(self.requests)

    def test_global_off_blocks_direct_calls(self):
        with patch.object(memory_tools, "MEMORY_ENABLED", False):
            out = self.run_tool(memory_tools.search_memories, "subsidy")
        self.assertIn("unavailable", out)
        self.assertFalse(self.requests)

    def test_lookup_budget_stops_further_requests(self):
        self.ctx.deps.memory_tool_calls = memory_tools.MAX_CALLS
        out = self.run_tool(memory_tools.search_memories, "subsidy")
        self.assertIn("budget reached", out)
        self.assertFalse(self.requests)

    def test_tool_hidden_without_identity(self):
        self.ctx.deps.mobile = ""
        self.assertIsNone(asyncio.run(memory_tools.prepare_memory_tool(self.ctx, object())))

    def test_read_timeout_is_bounded(self):
        async def handler(request):
            await asyncio.sleep(1)
            return httpx.Response(200, json={"entries": []})
        with patch.object(memory_tools, "TIMEOUT", 0.01), patch.object(
                memory_tools.httpx, "AsyncClient", side_effect=lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw)):
            out = asyncio.run(memory_tools.list_memories(self.ctx))
        self.assertEqual(out, memory_tools.UNAVAILABLE)


@unittest.skipUnless(_BOT_PRESENT, "bot checkout missing")
class TestAutomaticContext(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from unittest.mock import AsyncMock
        patcher = patch.object(memory_svc.MemoryStore, "enabled", new_callable=AsyncMock, return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_keys_alone_do_not_create_memory_context(self):
        from unittest.mock import AsyncMock
        with patch.object(memory_svc, "_standing_summary", new_callable=AsyncMock, return_value=""), patch.object(
                memory_svc, "_relevant_entries", new_callable=AsyncMock, return_value=[]), patch.object(
                memory_svc, "_available_filters", new_callable=AsyncMock, return_value=["scheme: fodder"]):
            self.assertEqual(await memory_svc.fetch_memory_context("farmer", "hello"), "")

    async def test_context_uses_farmer_keys_when_memory_is_relevant(self):
        from unittest.mock import AsyncMock
        with patch.object(memory_svc, "_standing_summary", new_callable=AsyncMock, return_value=""), patch.object(
                memory_svc, "_relevant_entries", new_callable=AsyncMock, return_value=["A subsidy request"]), patch.object(
                memory_svc, "_available_filters", new_callable=AsyncMock, return_value=["scheme: fodder"]):
            self.assertIn("scheme: fodder", await memory_svc.fetch_memory_context("farmer", "subsidy"))


if __name__ == "__main__":
    unittest.main()
