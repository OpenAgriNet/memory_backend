"""Offline authentication contracts for the private memory API."""
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import env_fixture
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "service"), str(ROOT / "dreamer")]

import memory_api as api
import memory_client


def concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "test-value", path)


class AuthContract(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_is_the_only_unauthenticated_route(self):
        with patch.object(api, "_qdrant", new_callable=AsyncMock,
                          return_value={"result": {"points_count": 0}}), patch.object(
                              api, "_collection_embed_model", new_callable=AsyncMock, return_value=None):
            health = await self.client.get("/health")
        self.assertEqual(health.status_code, 200)

        for route in api.app.routes:
            methods = getattr(route, "methods", None) or set()
            for method in methods - {"HEAD", "OPTIONS"}:
                if route.path == "/health" and method == "GET":
                    continue
                path = concrete(route.path)
                missing = await self.client.request(method, path)
                self.assertEqual((route.path, method, missing.status_code), (route.path, method, 401))
                self.assertEqual(missing.headers.get("www-authenticate"), "Bearer")
                invalid = await self.client.request(method, path, headers={"Authorization": "Bearer wrong"})
                self.assertEqual((route.path, method, invalid.status_code), (route.path, method, 401))

    async def test_valid_token_reaches_route_validation(self):
        response = await self.client.post(
            "/memory/test/entries",
            headers={"Authorization": f"Bearer {os.environ['MEMORY_INTERNAL_TOKEN']}"},
            json={},
        )
        self.assertEqual(response.status_code, 422)


class StartupContract(unittest.TestCase):
    def test_missing_token_fails_import(self):
        env = dict(os.environ)
        env.pop("MEMORY_INTERNAL_TOKEN", None)
        env.update({
            "QDRANT_URL": "http://127.0.0.1:1",
            "MARQO_URL": "http://127.0.0.1:1",
            "AMUL_MEMORY_ENV_FILE": str(ROOT / "tests" / "does-not-exist.env"),
            "PYTHONPATH": str(ROOT / "service"),
        })
        result = subprocess.run(
            [sys.executable, "-c", "import memory_api"],
            cwd=ROOT / "service", env=env, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set MEMORY_INTERNAL_TOKEN", result.stderr)


class ClientContract(unittest.TestCase):
    def test_shared_client_attaches_bearer_token(self):
        with patch.object(memory_client.requests, "request") as request:
            memory_client.post("http://memory/keys/doc", json={"key": "feed"})
        self.assertEqual(
            request.call_args.kwargs["headers"]["Authorization"],
            f"Bearer {os.environ['MEMORY_INTERNAL_TOKEN']}",
        )


if __name__ == "__main__":
    unittest.main()
