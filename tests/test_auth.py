"""Focused regressions for private Memory API bearer authentication."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import env_fixture
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "service"), str(ROOT / "dreamer")]

import memory_api as api
import memory_client
import runtime_config


class AuthContract(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_public_get_and_head_health(self):
        with patch.object(
            api, "_qdrant", new_callable=AsyncMock,
            return_value={"result": {"points_count": 0}},
        ), patch.object(
            api, "_collection_embed_model", new_callable=AsyncMock, return_value=None,
        ):
            get_response = await self.client.get("/health")
            head_response = await self.client.head("/health")
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(head_response.status_code, 200)
        self.assertEqual(head_response.content, b"")

    async def test_protected_routes_return_safe_401_and_accept_valid_token(self):
        for authorization in (None, "Basic abc", "Bearer wrong", "Bearer two words"):
            headers = {} if authorization is None else {"Authorization": authorization}
            response = await self.client.get("/keys", headers=headers)
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json(), {"detail": "Unauthorized"})
            self.assertEqual(response.headers.get("www-authenticate"), "Bearer")

        unicode_response = await self.client.get(
            "/keys", headers=[(b"authorization", b"Bearer \xff")]
        )
        self.assertEqual(unicode_response.status_code, 401)
        self.assertNotIn("\ufffd", unicode_response.text)

        valid = await self.client.post(
            "/memory/test/entries",
            headers={"Authorization": f"Bearer {api.MEMORY_INTERNAL_TOKEN}"},
            json={},
        )
        self.assertEqual(valid.status_code, 422)


class ConfigurationAndClientContract(unittest.TestCase):
    def test_token_validation_rejects_unsafe_and_example_values(self):
        good = "a-secure-internal-token-with-32-chars"
        with patch.dict(os.environ, {"MEMORY_INTERNAL_TOKEN": good}):
            self.assertEqual(runtime_config.memory_internal_token(), good)

        invalid = ("", "short", "contains whitespace and is definitely long", "x" * 31,
                   "contains:unsafe:header:characters:here",
                   "t\N{LATIN SMALL LETTER E WITH ACUTE}" + "x" * 40,
                   "replace-with-a-long-random-internal-token")
        for value in invalid:
            with self.subTest(value=value), patch.dict(
                os.environ, {"MEMORY_INTERNAL_TOKEN": value}
            ):
                with self.assertRaises(RuntimeError):
                    runtime_config.memory_internal_token()

    def test_client_scopes_and_owns_authorization_header(self):
        target = os.environ["MEMORY_API_URL"].rstrip("/") + "/keys"
        with patch.object(memory_client.requests, "request") as request:
            memory_client.get(target, headers={"X-Test": "yes"}, timeout=5)
        headers = request.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], f"Bearer {api.MEMORY_INTERNAL_TOKEN}")
        self.assertEqual(headers["X-Test"], "yes")

        with self.assertRaises(ValueError):
            memory_client.get(target, headers={"authorization": "Bearer caller-value"})
        with self.assertRaises(RuntimeError):
            memory_client.get("https://other.example/keys")


if __name__ == "__main__":
    unittest.main()
