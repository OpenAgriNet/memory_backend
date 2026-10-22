"""Integration tests against a running memory service.

No model is involved — these exercise storage, filtering and idempotency, which is
where the silent failures live. They need the service up (and therefore Qdrant and
Marqo); they skip cleanly if it is not, so the unit suite still runs anywhere.

Each test works on its own farmer id under a fixed prefix and deletes it afterwards.
This matters: I twice left test farmers behind in the real store, and one of them
sat next to five genuine farmers polluting measurements.

    MEMORY_API_URL=http://localhost:8100 python3 -m unittest discover -s tests -v
"""
import json
import os
import unittest
import urllib.error
import urllib.request
import uuid

API = os.environ.get("MEMORY_API_URL", "http://localhost:8100").rstrip("/")
TOKEN = os.environ.get("MEMORY_INTERNAL_TOKEN", "")
QDRANT = os.environ.get("QDRANT_HTTP", "http://localhost:6350").rstrip("/")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "amul_memory")
PREFIX = "zz-test-"


def _req(method, url, body=None):
    headers = {"Content-Type": "application/json"}
    if not (method in {"GET", "HEAD"} and url.rstrip("/").endswith("/health")):
        headers["Authorization"] = f"Bearer {TOKEN}"
    r = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None,
        headers=headers, method=method)
    return json.load(urllib.request.urlopen(r, timeout=90))


def _service_up():
    if os.getenv("AMUL_MEMORY_INTEGRATION") != "1":
        return False  # Explicit isolated runner only; never auto-write to a local demo.
    try:
        _req("GET", f"{API}/health")
        return True
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_service_up(), f"memory service not reachable at {API}")
class ServiceCase(unittest.TestCase):
    def setUp(self):
        self.farmer = PREFIX + uuid.uuid4().hex[:8]

    def tearDown(self):
        try:
            _req("POST", f"{QDRANT}/collections/{COLLECTION}/points/delete?wait=true",
                 {"filter": {"must": [{"key": "farmer_id",
                                       "match": {"value": self.farmer}}]}})
        except Exception:  # noqa: BLE001
            pass

    # -- helpers ---------------------------------------------------------
    def write(self, **kw):
        body = {"status": "open", "headline": "A thing happened.",
                "source_ts": "2026-09-01T06:00:00Z", "source_session_id": "s1",
                "source_traces": ["t1"]}
        body.update(kw)
        return _req("POST", f"{API}/memory/{self.farmer}/entries", body)

    def patch(self, entry_id, **kw):
        return _req("PATCH", f"{API}/memory/{self.farmer}/entries/{entry_id}", kw)

    def listing(self, query=""):
        return _req("GET", f"{API}/memory/{self.farmer}/entries?only_current=true" + query)

    def search(self, q, path="search", **kw):
        return _req("POST", f"{API}/memory/{self.farmer}/{path}",
                    {"query": q, "limit": 10, "only_current": True, **kw})


class TestIdempotency(ServiceCase):
    """The job is re-run: nightly retries, overlapping windows, a crash mid-farmer."""

    def test_identical_write_is_refused(self):
        self.write()
        again = self.write()
        self.assertFalse(again.get("written", True))
        self.assertIn("already", (again.get("reason") or "").lower())

    def test_checked_merge_can_combine_fragments_from_the_same_turn(self):
        first = self.write(headline="The tag is missing online.", expanded="Animal tag is absent.")
        second = self.write(headline="The AI receipt was not supplied.", expanded="No receipt was supplied.")
        merged = self.patch(first["id"], headline="Registration and receipt problems.",
                            expanded="Animal tag is absent. No receipt was supplied.",
                            source_traces=["t1"], merge_from=[second["id"]])
        self.assertIsNot(merged.get("applied"), False)
        self.assertEqual(merged["new"]["payload"]["times_raised"], 1)
        self.assertIn("No receipt was supplied", merged["new"]["payload"]["expanded"])

    def test_merge_cannot_use_a_missing_source_to_bypass_retry_guard(self):
        first = self.write()
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.patch(first["id"], headline="Unverified replacement", source_traces=["t1"],
                       merge_from=[str(uuid.uuid4())])
        self.assertEqual(error.exception.code, 422)
        self.assertEqual(self.listing()["entries"][0]["id"], first["id"])

    def test_one_turn_can_support_two_different_memories(self):
        self.write(headline="A cow's ongoing feeding issue.")
        second = self.write(headline="Also asked for shorter replies.")
        self.assertTrue(second.get("written", True))
        self.assertEqual(len(self.listing()["entries"]), 2)

    def test_update_from_already_recorded_turns_is_rejected(self):
        e = self.write()
        res = self.patch(e["id"], headline="Reworded.", source_traces=["t1"],
                         source_session_id="s1")
        self.assertIs(res.get("applied"), False)

    def test_update_from_a_new_turn_is_applied(self):
        e = self.write()
        res = self.patch(e["id"], headline="It came up again.", source_traces=["t2"],
                         source_session_id="s2", source_ts="2026-09-05T06:00:00Z")
        self.assertIsNot(res.get("applied"), False)
        self.assertEqual(res["new"]["payload"]["times_raised"], 2)


class TestTimesRaised(ServiceCase):
    """Counts occasions, not writes. An inflated count is evidence the honesty judge
    uses to allow 'raised repeatedly', so inflating it launders a false claim."""

    def test_same_session_twice_does_not_inflate(self):
        e = self.write()
        r = self.patch(e["id"], headline="Same conversation, more detail.",
                       source_traces=["t9"], source_session_id="s1")
        self.assertEqual(r["new"]["payload"]["times_raised"], 1)

    def test_caller_cannot_override_the_count(self):
        e = self.write()
        r = self.patch(e["id"], headline="Again.", source_traces=["t2"],
                       source_session_id="s2", times_raised=99)
        self.assertEqual(r["new"]["payload"]["times_raised"], 2)

    def test_distinct_sessions_accumulate(self):
        e = self.write()
        cur = e["id"]
        for i in (2, 3, 4):
            r = self.patch(cur, headline=f"Raised again, round {i}.",
                           source_traces=[f"t{i}"], source_session_id=f"s{i}")
            cur = r["new"]["id"]
        self.assertEqual(r["new"]["payload"]["times_raised"], 4)


class TestUpdateIsAllOrNothing(ServiceCase):
    """A refused update must leave the entry exactly as it was.

    This is a regression suite for real data loss. update_entry used to stamp the old
    version superseded BEFORE the idempotency check and before writing the
    replacement, so any refusal after that point left the entry retired with nothing
    replacing it — silently gone. Five real memories were destroyed that way before it
    was caught, because an update rejected as already-applied still retired the thing
    it declined to change."""

    def test_refused_by_idempotency_leaves_the_entry_alive(self):
        e = self.write(headline="An important memory that must not vanish.")
        res = self.patch(e["id"], headline="Reworded.", source_traces=["t1"],
                         source_session_id="s1")
        self.assertIs(res.get("applied"), False)
        self.assertTrue(res.get("unchanged"))
        live = self.listing()
        self.assertEqual(live["returned"], 1, "the entry was destroyed by a refusal")
        self.assertEqual(live["entries"][0]["headline"],
                         "An important memory that must not vanish.")

    def test_a_real_update_still_replaces_cleanly(self):
        e = self.write(headline="Version one.")
        self.patch(e["id"], headline="Version two.", source_traces=["t2"],
                   source_session_id="s2", source_ts="2026-09-05T06:00:00Z")
        live = self.listing()
        self.assertEqual(live["returned"], 1)
        self.assertEqual(live["entries"][0]["headline"], "Version two.")

    def test_no_entry_is_ever_retired_without_a_successor(self):
        """The invariant the bug violated, checked directly: every superseded entry
        must either have a successor pointing at it or be a folded fragment."""
        e = self.write(headline="Original.")
        self.patch(e["id"], headline="Refused.", source_traces=["t1"],
                   source_session_id="s1")
        self.patch(e["id"], headline="Accepted.", source_traces=["t2"],
                   source_session_id="s2")
        raw = _req("POST", f"{QDRANT}/collections/{COLLECTION}/points/scroll",
                   {"limit": 100, "with_payload": True,
                    "filter": {"must": [{"key": "farmer_id",
                                         "match": {"value": self.farmer}}]}})
        pts = raw["result"]["points"]
        successors = {p["payload"].get("previous_version") for p in pts
                      if p["payload"].get("previous_version")}
        for p in pts:
            pl = p["payload"]
            if pl.get("superseded_at") and not pl.get("merged_into"):
                predecessor = pl["episode_id"] if pl.get("type") == "memory_chunk" else p["id"]
                self.assertIn(predecessor, successors,
                              f"orphaned: {pl.get('headline')!r} was retired with "
                              "nothing replacing it")


class TestVersioning(ServiceCase):
    """Superseded is a STATE check, not a date comparison — which is why removing all
    date filtering did not expose old versions."""

    def test_only_the_current_version_is_returned(self):
        e = self.write(headline="VERSION ONE of the account.")
        r = self.patch(e["id"], headline="VERSION TWO of the account.",
                       source_traces=["t2"], source_session_id="s2")
        for path in ("search", "search_expanded"):
            heads = [h["headline"] for h in self.search("account", path=path)["hits"]]
            self.assertEqual(heads, ["VERSION TWO of the account."], path)
        heads = [x["headline"] for x in self.listing()["entries"]]
        self.assertEqual(heads, ["VERSION TWO of the account."])

    def test_history_is_walkable_and_ordered(self):
        e = self.write(headline="VERSION ONE.")
        r = self.patch(e["id"], headline="VERSION TWO.", source_traces=["t2"],
                       source_session_id="s2", source_ts="2026-09-05T06:00:00Z")
        cur = r["new"]["id"]
        h = _req("GET", f"{API}/memory/{self.farmer}/entries/{cur}/history")
        self.assertEqual(h["count"], 2)
        self.assertEqual([x["headline"] for x in h["history"]],
                         ["VERSION ONE.", "VERSION TWO."])

    def test_current_entry_advertises_that_history_exists(self):
        """A reader who cannot tell whether they have the whole picture assumes they
        do — so the existence of more must be stated, not inferred."""
        e = self.write()
        r = self.patch(e["id"], headline="Moved along.", source_traces=["t2"],
                       source_session_id="s2")
        hit = self.search("thing")["hits"][0]
        self.assertTrue(hit.get("earlier_versions_available"))

    def test_folded_fragments_are_excluded_from_reads(self):
        a = self.write(headline="Fragment A of one situation.")
        b = self.write(headline="Fragment B of the same situation.",
                       source_traces=["t2"], source_session_id="s2")
        _req("POST", f"{API}/memory/{self.farmer}/entries/{a['id']}/fold",
             {"merged_into": b["id"], "reason": "same situation"})
        heads = [x["headline"] for x in self.listing()["entries"]]
        self.assertEqual(heads, ["Fragment B of the same situation."])

    def test_folded_fragment_keeps_its_trail(self):
        """Nothing is deleted: a wrong merge has to stay diagnosable."""
        a = self.write(headline="Fragment A.")
        b = self.write(headline="Fragment B.", source_traces=["t2"], source_session_id="s2")
        _req("POST", f"{API}/memory/{self.farmer}/entries/{a['id']}/fold",
             {"merged_into": b["id"], "reason": "why it was merged"})
        got = _req("POST", f"{QDRANT}/collections/{COLLECTION}/points",
                   {"ids": [a["id"]], "with_payload": True})["result"][0]["payload"]
        self.assertEqual(got["merged_into"], b["id"])
        self.assertEqual(got["merge_reason"], "why it was merged")


class TestExactListing(ServiceCase):
    """The whole value of this path is that zero rows means DEFINITIVELY zero, which
    a vector search can never establish."""

    def test_metadata_filter_is_honoured(self):
        """Silently ignored for a while: a 'filtered' listing returned everything."""
        self.write(headline="High severity issue.", metadata={"severity": "high"})
        self.write(headline="Mild issue.", metadata={"severity": "mild"},
                   source_traces=["t2"], source_session_id="s2")
        got = self.listing("&metadata_key=severity&metadata_value=high")
        self.assertEqual(got["returned"], 1)
        self.assertIn("High severity", got["entries"][0]["headline"])

    def test_lapsed_but_open_entries_are_still_listed(self):
        """An open booking whose deadline passed is MORE urgent to surface, not less.
        status is the authority on resolution; a date is not."""
        self.write(headline="Requested; the scheme window has closed.",
                   expires_at="2026-09-01T00:00:00Z")
        self.write(headline="Requested; no deadline.",
                   source_traces=["t2"], source_session_id="s2")
        got = self.listing("&status=open")
        self.assertEqual(got["returned"], 2, "a passed end date must not hide an open item")

    def test_several_statuses_in_one_call(self):
        self.write(headline="An open one.", status="open")
        self.write(headline="A pending one.", status="pending",
                   source_traces=["t2"], source_session_id="s2")
        self.write(headline="A resolved one.", status="resolved",
                   source_traces=["t3"], source_session_id="s3")
        got = self.listing("&status=open&status=pending")
        self.assertEqual(got["returned"], 2)

    def test_truncation_is_reported(self):
        """'Zero means zero' survives truncation; 'this is all of them' does not."""
        for i in range(4):
            self.write(headline=f"Distinct thing number {i}.",
                       source_traces=[f"t{i}"], source_session_id=f"s{i}")
        got = self.listing("&limit=2")
        self.assertEqual(got["returned"], 2)
        self.assertTrue(got["truncated"])
        self.assertIn("not the full list", got.get("note", ""))

    def test_empty_result_is_not_truncated(self):
        got = self.listing("&status=resolved")
        self.assertEqual(got["returned"], 0)
        self.assertFalse(got["truncated"])


class TestStandingRecord(ServiceCase):
    """Read on every single turn, and re-derived on every run."""

    def put(self, fields):
        return _req("PUT", f"{API}/memory/{self.farmer}/profile", {"fields": fields})

    def test_unchanged_record_is_not_rewritten(self):
        """Without this, every run superseded the previous version with an identical
        copy — unbounded churn in the one record read on every turn."""
        f = {"name_used": "Rameshbhai", "language_register": "Gujarati"}
        self.put(f)
        again = self.put(f)
        self.assertTrue(again.get("unchanged"))

    def test_changed_record_is_written(self):
        self.put({"language_register": "Gujarati"})
        res = self.put({"language_register": "Gujarati, short answers"})
        self.assertFalse(res.get("unchanged"))

    def test_unknown_fields_are_ignored_and_reported(self):
        res = self.put({"language_register": "Gujarati", "invented_field": "x"})
        self.assertIn("invented_field", res["ignored_unknown_fields"])
        self.assertNotIn("invented_field", res["stored_fields"])

    def test_empty_fields_are_not_stored(self):
        res = self.put({"language_register": "Gujarati", "name_used": ""})
        self.assertEqual(res["stored_fields"], ["language_register"])

    def test_standing_record_is_excluded_from_search_and_listing(self):
        """It is injected wholesale every turn; a semantic hit on it would only
        duplicate what the agent already has."""
        self.put({"language_register": "Gujarati"})
        self.write(headline="A real remembered thing.")
        heads = [x["headline"] for x in self.listing()["entries"]]
        self.assertEqual(heads, ["A real remembered thing."])
        for h in self.search("Gujarati language")["hits"]:
            self.assertNotEqual(h.get("type"), "profile_record")


class TestAgentSafety(ServiceCase):
    """Things that must hold regardless of what any model does."""

    def test_identifier_keys_are_hidden_from_the_agent_key_list(self):
        """The tag is stored on purpose, for certain matching — but the key list is
        injected into conversations, so it must never carry one."""
        self.write(metadata={"animal_id": "101520248441", "severity": "high"})
        keys = _req("GET", f"{API}/memory/{self.farmer}/keys")["keys"]
        names = {k["filter_path"].removeprefix("metadata.") for k in keys}
        self.assertNotIn("animal_id", names)
        self.assertIn("severity", names)

    def test_one_farmers_memory_never_reaches_another(self):
        self.write(headline="Farmer one's private business.")
        other = PREFIX + uuid.uuid4().hex[:8]
        try:
            self.assertEqual(_req("GET", f"{API}/memory/{other}/entries")["returned"], 0)
            hits = _req("POST", f"{API}/memory/{other}/search",
                        {"query": "private business", "limit": 5})["hits"]
            self.assertEqual(hits, [])
        finally:
            _req("POST", f"{QDRANT}/collections/{COLLECTION}/points/delete?wait=true",
                 {"filter": {"must": [{"key": "farmer_id", "match": {"value": other}}]}})

    def test_memory_off_returns_nothing(self):
        self.write(headline="Something remembered.")
        _req("PUT", f"{API}/memory/{self.farmer}/settings", {"memory_enabled": False})
        try:
            self.assertEqual(self.listing()["entries"], [])
            self.assertEqual(self.search("something")["hits"], [])
            self.assertEqual(_req("GET", f"{API}/memory/{self.farmer}/keys")["keys"], [])
        finally:
            _req("PUT", f"{API}/memory/{self.farmer}/settings", {"memory_enabled": True})

    def test_embedding_convention_is_recorded_on_every_entry(self):
        """Break this and search still WORKS — it just stops meaning anything. The
        stamp is what makes that detectable instead of silent."""
        e = self.write()
        self.assertIn("doc-query", e["payload"]["embed_model"])


if __name__ == "__main__":
    unittest.main()
