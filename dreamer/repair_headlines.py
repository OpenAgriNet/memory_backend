"""One-off repair: strip repetition claims the record does not support.

Companion to service/repair_counts.py. Correcting times_raised (which used to count
writes rather than conversations) left some entries whose TEXT still says "has
repeatedly raised" while the record now shows a single conversation. Left alone, the
assistant would tell a farmer they had raised something repeatedly when they said it
once — the exact false-confidence failure this system exists to prevent.

Only the unsupported claim is removed. Everything else in the entry is left as it is.

  python repair_headlines.py            # report
  python repair_headlines.py --apply
"""
import argparse
import json
import os
import re
import sys
import urllib.request

import llm
from runtime_config import required, qdrant_headers

QDRANT = required("QDRANT_URL").rstrip("/")
MEMORY_API = required("MEMORY_API_URL").rstrip("/")
COLLECTION = os.getenv("QDRANT_COLLECTION", "amul_memory")
SKIP_TYPES = {"settings", "key_doc", "profile_record"}

CLAIM = re.compile(
    r"repeatedly|again|several times|keeps asking|multiple times|"
    r"since (early |late )?(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)", re.I)

SYSTEM = """You are correcting one memory entry whose wording claims something the record does not support.

The entry says the farmer raised this repeatedly, or has been raising it since some date. The record shows it came up in only ONE conversation. The claim is therefore false and must go.

Rewrite the headline so it says the same thing WITHOUT any claim about repetition or duration. Change nothing else: keep the situation, the status, the specifics, the honesty about what is unconfirmed. Do not add anything new. Keep it to 1-2 sentences.

Return strict JSON only."""

SCHEMA = {"type": "object", "properties": {"headline": {"type": "string"}},
          "required": ["headline"]}


def _scroll():
    out, nxt = [], None
    while True:
        body = {"limit": 256, "with_payload": True}
        if nxt:
            body["offset"] = nxt
        r = urllib.request.Request(
            f"{QDRANT}/collections/{COLLECTION}/points/scroll",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **qdrant_headers(QDRANT)})
        res = json.load(urllib.request.urlopen(r))["result"]
        out.extend(res["points"])
        nxt = res.get("next_page_offset")
        if not nxt:
            return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    targets = []
    for p in _scroll():
        pl = p["payload"]
        if pl.get("superseded_at") or pl.get("type") in SKIP_TYPES:
            continue
        if (pl.get("times_raised") or 1) <= 1 and CLAIM.search(pl.get("headline") or ""):
            targets.append((p["id"], pl))

    print(f"{len(targets)} entries claim repetition the record does not support")
    for pid, pl in targets:
        try:
            out = llm.ask_json(SYSTEM, f"""ENTRY
  status:   {pl.get('status')}
  headline: {pl.get('headline')}
  detail:   {(pl.get('expanded') or '(none)')[:800]}

WHAT THE RECORD SHOWS: this came up in 1 conversation only.
""", SCHEMA, max_tokens=300)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not rewrite ({exc}); left alone", file=sys.stderr)
            continue
        new = (out.get("headline") or "").strip()
        if not (15 < len(new) < 400):
            print("  ! rewrite looked wrong; left alone", file=sys.stderr)
            continue
        if CLAIM.search(new):
            print(f"  ! rewrite still claims repetition; left alone: {new[:70]}", file=sys.stderr)
            continue
        print(f"\n  farmer ...{str(pl.get('farmer_id'))[-4:]}")
        print(f"    was: {pl.get('headline')[:110]}")
        print(f"    now: {new[:110]}")
        if not args.apply:
            continue
        # Straight payload edit, not a PATCH: this is a correction to a mistake, not a
        # new version of a fact that changed, and it must not create a history entry
        # implying the farmer said something new.
        req = urllib.request.Request(
            f"{QDRANT}/collections/{COLLECTION}/points/payload?wait=true",
            data=json.dumps({"payload": {"headline": new}, "points": [pid]}).encode(),
            headers={"Content-Type": "application/json", **qdrant_headers(QDRANT)})
        json.load(urllib.request.urlopen(req))
    if not args.apply:
        print("\ndry run — pass --apply to write")


if __name__ == "__main__":
    main()
