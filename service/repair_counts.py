"""One-off repair: recompute times_raised from distinct conversations.

Needed because times_raised was originally incremented per WRITE rather than derived
from the conversations behind an entry. Two chunks of one conversation produced two
entries, a merge summed them, and a later update added one more — so an entry claimed
a farmer had raised something three times when they had said it once. That count is
evidence the honesty judge uses to allow "raised repeatedly" phrasing, so an inflated
count launders a false claim.

Sessions are reconstructed from what is actually on record: an entry's own
source_session_id, plus the sessions of every fragment folded into it. Where a merge
trail was broken by later updates this UNDERCOUNTS, which is the safe direction — a
claim of repetition should need evidence.

  python repair_counts.py            # report
  python repair_counts.py --apply
"""
import argparse
import json
import os
import urllib.request
from runtime_config import required, qdrant_headers

QDRANT = required("QDRANT_URL").rstrip("/")
COLLECTION = os.getenv("QDRANT_COLLECTION", "amul_memory")
SKIP_TYPES = {"settings", "key_doc", "profile_record"}


def _call(method, path, body=None):
    r = urllib.request.Request(
        f"{QDRANT}{path}", data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **qdrant_headers(QDRANT)}, method=method)
    return json.load(urllib.request.urlopen(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    points, nxt = [], None
    while True:
        body = {"limit": 256, "with_payload": True}
        if nxt:
            body["offset"] = nxt
        r = _call("POST", f"/collections/{COLLECTION}/points/scroll", body)["result"]
        points.extend(r["points"])
        nxt = r.get("next_page_offset")
        if not nxt:
            break

    # Fragments record which entry they were folded into, so the sessions they
    # contributed can be attributed back to it.
    folded_into: dict[str, set] = {}
    for p in points:
        pl = p["payload"]
        target = pl.get("merged_into")
        if not target:
            continue
        got = set(pl.get("source_sessions") or [])
        if pl.get("source_session_id"):
            got.add(pl["source_session_id"])
        folded_into.setdefault(target, set()).update(got)

    changes = []
    for p in points:
        pl = p["payload"]
        if pl.get("superseded_at") or pl.get("type") in SKIP_TYPES:
            continue
        sessions = set(pl.get("source_sessions") or [])
        if pl.get("source_session_id"):
            sessions.add(pl["source_session_id"])
        sessions |= folded_into.get(p["id"], set())
        correct = max(1, len(sessions))
        if correct != (pl.get("times_raised") or 1) or not pl.get("source_sessions"):
            changes.append((p["id"], pl, sorted(sessions), correct))

    print(f"{COLLECTION}: {len(points)} points, {len(changes)} current entries to correct")
    for pid, pl, sessions, correct in changes:
        was = pl.get("times_raised") or 1
        flag = "  <-- was overstating repetition" if correct < was else ""
        print(f"  {str(pl.get('farmer_id'))[-6:]}  {was}x -> {correct}x  "
              f"({len(sessions)} conversation(s)){flag}\n     {(pl.get('headline') or '')[:88]}")
    if not args.apply:
        print("\ndry run — pass --apply to write")
        return

    for pid, pl, sessions, correct in changes:
        _call("POST", f"/collections/{COLLECTION}/points/payload?wait=true",
              {"payload": {"times_raised": correct, "source_sessions": sessions},
               "points": [pid]})
    print(f"\ncorrected {len(changes)} entries")


if __name__ == "__main__":
    main()
