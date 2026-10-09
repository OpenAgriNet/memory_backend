"""Checkpoint contracts against the disposable Qdrant/service pair."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import unittest
import urllib.error
import urllib.request
import uuid

API = os.environ.get("MEMORY_API_URL", "http://127.0.0.1:1").rstrip("/")
QDRANT = os.environ.get("QDRANT_HTTP", "http://127.0.0.1:1").rstrip("/")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "amul_memory")
TOKEN = os.environ.get("MEMORY_INTERNAL_TOKEN", "")


def request(method, url, body=None, *, authenticated=True):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None,
        headers=headers, method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def service_up():
    return os.getenv("AMUL_MEMORY_INTEGRATION") == "1" and request("GET", API + "/health", authenticated=False)[0] == 200


@unittest.skipUnless(service_up(), "isolated memory service is not running")
class CheckpointIntegration(unittest.TestCase):
    def setUp(self):
        self.farmer = "zz-checkpoint-" + uuid.uuid4().hex[:10]
        self.subject = "session-" + uuid.uuid4().hex

    def tearDown(self):
        request("POST", f"{QDRANT}/collections/{COLLECTION}/points/delete?wait=true", {
            "filter": {"must": [{"key": "farmer_id", "match": {"value": self.farmer}}]},
        }, authenticated=False)

    def claim(self, *, kind="session_ingestion", subject=None, fingerprint="fp-1", run_id=None):
        return request("POST", API + "/internal/checkpoints/claim", {
            "kind": kind,
            "subject_id": subject or self.subject,
            "farmer_id": self.farmer,
            "candidate_fingerprint": fingerprint,
            "last_activity_at": "2026-10-09T00:00:00Z",
            "run_id": run_id or uuid.uuid4().hex,
        })

    def mutate_payload(self, point_id, payload):
        status, body = request("POST", f"{QDRANT}/collections/{COLLECTION}/points/payload?wait=true", {
            "payload": payload, "points": [point_id],
        }, authenticated=False)
        self.assertEqual(status, 200, body)

    def lookup(self, kind="session_ingestion", subject=None):
        status, body = request("POST", API + "/internal/checkpoints/lookup", {
            "subjects": [{"kind": kind, "subject_id": subject or self.subject}],
        })
        self.assertEqual(status, 200, body)
        return body["checkpoints"][0]

    def test_concurrent_first_claim_has_one_owner(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.claim(), range(8)))
        self.assertTrue(all(status == 200 for status, _ in results), results)
        winners = [body for _, body in results if body["claimed"]]
        self.assertEqual(len(winners), 1, results)
        self.assertEqual(self.lookup()["checkpoint_version"], 1)

    def test_concurrent_expired_reclaim_has_one_owner(self):
        status, first = self.claim()
        self.assertEqual(status, 200, first)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-checkpoint:session_ingestion:{self.subject}"))
        self.mutate_payload(point_id, {"checkpoint_lease_expires_at": "2020-01-01T00:00:00Z"})
        status, _ = request("POST", API + "/internal/checkpoints/complete", {
            "kind": "session_ingestion", "subject_id": self.subject,
            "lease_token": first["lease_token"], "checkpoint_version": first["checkpoint_version"],
            "status": "completed",
        })
        self.assertEqual(status, 409)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.claim(), range(8)))
        winners = [body for status, body in results if status == 200 and body["claimed"]]
        self.assertEqual(len(winners), 1, results)
        self.assertEqual(winners[0]["checkpoint_version"], 2)

    def test_stale_lease_and_version_cannot_transition(self):
        _, claim = self.claim()
        base = {"kind": "session_ingestion", "subject_id": self.subject,
                "lease_token": "stale", "checkpoint_version": claim["checkpoint_version"]}
        for endpoint, extra in (
            ("renew", {}),
            ("complete", {"status": "completed"}),
            ("fail", {"error_code": "boom", "error_summary": "failed"}),
        ):
            status, _ = request("POST", API + "/internal/checkpoints/" + endpoint, {**base, **extra})
            self.assertEqual(status, 409)

        valid = {**base, "lease_token": claim["lease_token"], "checkpoint_version": 999}
        status, _ = request("POST", API + "/internal/checkpoints/complete", {**valid, "status": "completed"})
        self.assertEqual(status, 409)

    def test_completed_and_no_op_are_skipped_and_changed_work_preserves_traces(self):
        for completion in ("completed", "no_op"):
            subject = self.subject + completion
            _, claim = self.claim(subject=subject)
            status, done = request("POST", API + "/internal/checkpoints/complete", {
                "kind": "session_ingestion", "subject_id": subject,
                "lease_token": claim["lease_token"], "checkpoint_version": claim["checkpoint_version"],
                "status": completion,
                "processed_traces": [{"trace_id": "trace-1", "source_version": "v1"}],
            })
            self.assertEqual(status, 200, done)
            status, skipped = self.claim(subject=subject)
            self.assertEqual((status, skipped["claimed"], skipped["reason"]), (200, False, "unchanged_complete"))
            status, changed = self.claim(subject=subject, fingerprint="fp-2")
            self.assertEqual(status, 200, changed)
            self.assertTrue(changed["claimed"])
            self.assertEqual(changed["checkpoint_version"], 3)
            self.assertEqual(changed["processed_traces"][0]["trace_id"], "trace-1")
            checkpoint = self.lookup(subject=subject)
            self.assertEqual(checkpoint["attempt_count"], 1)
            request("POST", API + "/internal/checkpoints/complete", {
                "kind": "session_ingestion", "subject_id": subject,
                "lease_token": changed["lease_token"], "checkpoint_version": changed["checkpoint_version"],
                "status": "completed",
            })

    def test_retry_schedule_fifth_failure_and_redaction(self):
        expected = [900, 3600, 21600, 86400]
        claim_status, claim = self.claim()
        self.assertEqual(claim_status, 200)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-checkpoint:session_ingestion:{self.subject}"))
        for index in range(5):
            status, failed = request("POST", API + "/internal/checkpoints/fail", {
                "kind": "session_ingestion", "subject_id": self.subject,
                "lease_token": claim["lease_token"], "checkpoint_version": claim["checkpoint_version"],
                "error_code": "worker failure!", "error_summary": "Bearer abc123 token=secret farmer 9999999999 a@b.com",
            })
            self.assertEqual(status, 200, failed)
            checkpoint = failed["checkpoint"]
            self.assertNotIn("abc123", checkpoint["last_error_summary"])
            self.assertNotIn("9999999999", checkpoint["last_error_summary"])
            self.assertLessEqual(len(checkpoint["last_error_summary"]), 500)
            if index == 4:
                self.assertEqual(checkpoint["status"], "needs_review")
                self.assertIsNone(checkpoint["next_retry_at"])
                break
            self.assertEqual(checkpoint["status"], "retryable")
            retry = datetime.fromisoformat(checkpoint["next_retry_at"])
            updated = datetime.fromisoformat(checkpoint["updated_at"])
            self.assertAlmostEqual((retry - updated).total_seconds(), expected[index], delta=1)
            self.mutate_payload(point_id, {"checkpoint_next_retry_at": "2020-01-01T00:00:00Z"})
            _, claim = self.claim()
            self.assertTrue(claim["claimed"])

    def test_kinds_are_independent_and_due_lists_only_ready_work(self):
        claims = {}
        for kind in ("session_ingestion", "farmer_consolidation", "farmer_profile"):
            status, claims[kind] = self.claim(kind=kind)
            self.assertEqual(status, 200)
        self.assertEqual({value["checkpoint_version"] for value in claims.values()}, {1})

        profile_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-checkpoint:farmer_profile:{self.subject}"))
        self.mutate_payload(profile_id, {"checkpoint_lease_expires_at": "2020-01-01T00:00:00Z"})
        status, due = request("POST", API + "/internal/checkpoints/due", {
            "kinds": ["farmer_profile"], "limit": 10,
        })
        self.assertEqual(status, 200, due)
        self.assertEqual([row["kind"] for row in due["checkpoints"]], ["farmer_profile"])

    def test_checkpoint_is_hidden_from_agent_facing_collections(self):
        _, claim = self.claim()
        checkpoint = self.lookup()
        status, listing = request("GET", API + f"/memory/{self.farmer}/entries")
        self.assertEqual(status, 200, listing)
        self.assertEqual(listing["entries"], [])
        status, search = request("POST", API + f"/memory/{self.farmer}/search", {
            "query": "checkpoint subject", "limit": 10,
        })
        self.assertEqual(status, 200, search)
        self.assertEqual(search["hits"], [])
        status, keys = request("GET", API + "/keys")
        self.assertEqual(status, 200, keys)
        self.assertNotIn(self.subject, json.dumps(keys))
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-checkpoint:session_ingestion:{self.subject}"))
        status, _ = request("GET", API + f"/memory/{self.farmer}/entries/{point_id}")
        self.assertEqual(status, 404)
        status, _ = request("GET", API + f"/memory/{self.farmer}/entries/{point_id}/history")
        self.assertEqual(status, 404)
        self.assertEqual(checkpoint["checkpoint_version"], claim["checkpoint_version"])


if __name__ == "__main__":
    unittest.main()
