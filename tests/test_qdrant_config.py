"""Authenticated REST contracts; no network or real credentials."""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import env_fixture
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
import httpx
import memory_api as api
from runtime_config import qdrant_headers


class TestQdrantConnection(unittest.TestCase):
    def test_key_is_not_forwarded_to_another_service_or_http(self):
        with patch.dict(os.environ, QDRANT_URL="https://qdrant.example", QDRANT_API_KEY="test-key"):
            self.assertEqual(qdrant_headers("https://qdrant.example/collections"), {"api-key": "test-key"})
            self.assertEqual(qdrant_headers("https://embedding.example"), {})
            self.assertEqual(qdrant_headers("http://qdrant.example"), {})
        with patch.dict(os.environ, QDRANT_API_KEY=""):
            self.assertEqual(qdrant_headers(), {})

    def exercise_startup(self, lookup_status):
        calls = []
        def respond(request):
            calls.append((request.method, request.headers.get("api-key")))
            status = lookup_status if request.method == "GET" else 200
            return httpx.Response(status, json={"result": {}})
        real_client = httpx.AsyncClient
        with patch.dict(os.environ, QDRANT_URL="https://qdrant.example", QDRANT_API_KEY="test-key"), \
             patch.object(api, "QDRANT_URL", "https://qdrant.example"), \
             patch.object(api.httpx, "AsyncClient", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw)):
            if lookup_status in (401, 403, 500):
                with self.assertRaisesRegex(RuntimeError, str(lookup_status)):
                    asyncio.run(api._ensure_collection())
            else:
                asyncio.run(api._ensure_collection())
        return calls

    def test_failed_auth_or_server_error_does_not_create_collection(self):
        for status in (401, 403, 500):
            with self.subTest(status=status):
                self.assertEqual(self.exercise_startup(status), [("GET", "test-key")])

    def test_missing_collection_authenticates_create_and_indexes(self):
        calls = self.exercise_startup(404)
        self.assertEqual(calls[0], ("GET", "test-key"))
        self.assertTrue(any(method == "PUT" for method, _ in calls))
        self.assertTrue(all(key == "test-key" for _, key in calls))

    def test_existing_collection_is_not_mutated(self):
        self.assertEqual(self.exercise_startup(200), [("GET", "test-key")])
