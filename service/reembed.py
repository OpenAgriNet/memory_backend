"""Re-embed an existing collection after an embedding-convention change.

Needed once (2026-09-08) because memories were originally stored encoded as though
they were questions, before the document/query split. Old vectors are not comparable
with new queries, so they have to be rebuilt — there is no way to "convert" them.

Run against a running memory service's Qdrant. Payloads are never touched, only
vectors and the embed_model stamp. Idempotent: points already carrying the current
stamp are skipped, so an interrupted run can just be re-run.

  python reembed.py                 # report what would change
  python reembed.py --apply
"""
import argparse
import json
import os
import urllib.request

import episode
from runtime_config import required, qdrant_headers

QDRANT = required("QDRANT_URL").rstrip("/")
COLLECTION = os.getenv("QDRANT_COLLECTION", "amul_memory")
MARQO = required("MARQO_URL").rstrip("/")
MARQO_INDEX = os.getenv("MARQO_INDEX", "amul-veterinary-index")
TARGET_STAMP = os.getenv("EMBED_MODEL_ID", f"marqo:{MARQO_INDEX}/doc-query")
# Types whose vectors are placeholders, never searched — no point embedding them.
PLACEHOLDER_TYPES = {"settings", "key_doc", "profile_record"}


def _post(url, body, method="POST"):
    r = urllib.request.Request(url, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", **qdrant_headers(url)}, method=method)
    return json.load(urllib.request.urlopen(r))


def _embed(texts, kind):
    return _post(f"{MARQO}/indexes/{MARQO_INDEX}/embed",
                 {"content": [t if (t or "").strip() else " " for t in texts],
                  "content_type": kind})["embeddings"]


def vector_texts(point, by_id):
    """Keep episode and chunk vectors aligned with their actual searchable text."""
    payload = point["payload"]
    if payload.get("type") in PLACEHOLDER_TYPES:
        return {}
    if payload.get("type") == "memory_chunk":
        parent = by_id.get(payload.get("episode_id"))
        if parent is None or parent["payload"].get("farmer_id") != payload.get("farmer_id"):
            raise ValueError("Chunk has no matching parent; repair the index before re-embedding")
        text, chunks, _ = episode.layout(parent["payload"])
        chunk = chunks[payload["chunk_number"] - 1]
        return {"expanded": text[chunk["start"]:chunk["end"]]}
    if payload.get("chunk_count"):
        return {"headline": episode.level2_text(payload.get("headline") or "", payload.get("contents") or [])}
    return {"headline": payload.get("headline") or " ",
            "expanded": payload.get("expanded") or payload.get("headline") or " "}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    points, nxt = [], None
    while True:
        body = {"limit": 256, "with_payload": True}
        if nxt:
            body["offset"] = nxt
        r = _post(f"{QDRANT}/collections/{COLLECTION}/points/scroll", body)["result"]
        points.extend(r["points"])
        nxt = r.get("next_page_offset")
        if not nxt:
            break

    todo = [p for p in points
            if p["payload"].get("type") not in PLACEHOLDER_TYPES
            and p["payload"].get("embed_model") != TARGET_STAMP]
    skip = len(points) - len(todo)
    print(f"{COLLECTION}: {len(points)} points, {skip} skipped "
          f"(placeholder vectors or already at {TARGET_STAMP}), {len(todo)} to re-embed")
    if not args.apply:
        print("dry run — pass --apply to write")
        return

    by_id = {p["id"]: p for p in points}
    plans = {p["id"]: vector_texts(p, by_id) for p in todo}  # Validate before any writes.
    done = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        inputs = [(p["id"], name, text) for p in chunk for name, text in plans[p["id"]].items()]
        embedded = _embed([text for _, _, text in inputs], "document")
        if len(embedded) != len(inputs):
            raise ValueError("Embedding service returned an incomplete batch; nothing in this batch was written")
        vectors = {p["id"]: {} for p in chunk}
        for (point_id, name, _), vector in zip(inputs, embedded):
            vectors[point_id][name] = vector
        # PUT, not POST: POST /points is Qdrant's fetch-by-id endpoint.
        _post(f"{QDRANT}/collections/{COLLECTION}/points?wait=true", {"points": [
            {"id": p["id"],
             "vector": vectors[p["id"]],
             "payload": {**p["payload"], "embed_model": TARGET_STAMP}}
            for p in chunk]}, method="PUT")
        done += len(chunk)
        print(f"  re-embedded {done}/{len(todo)}")
    print("done")


if __name__ == "__main__":
    main()
