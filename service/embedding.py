"""The one place text becomes vectors: multilingual-e5-large via FastEmbed, in-process.

e5 is trained asymmetrically: stored text is encoded as "passage: ...", search text as
"query: ...". Dropping the prefix still returns 1024 floats, just worse ones, so the
prefix is applied here and nowhere else.

These are the same vectors Marqo's /embed returned for content_type document/query
(checked 2026-11-04 on the H100: cosine 1.0, identical norms), so collections written
through Marqo need no re-embed.
"""
import os
from functools import lru_cache

MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
PREFIX = {"document": "passage: ", "query": "query: "}


@lru_cache(maxsize=1)
def _model():
    # Imported here so offline tests (which stub embedding) don't need fastembed.
    # Cache location follows FASTEMBED_CACHE_PATH; the image pre-downloads the model.
    from fastembed import TextEmbedding
    return TextEmbedding(MODEL)


def embed(texts, kind):
    if kind not in PREFIX:
        raise ValueError(f"bad embed kind {kind!r}")
    return [v.tolist() for v in _model().embed([PREFIX[kind] + t for t in texts])]
