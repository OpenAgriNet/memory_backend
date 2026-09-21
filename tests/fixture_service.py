"""Isolated HTTP test service with deterministic fixture vectors, no model calls."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
import memory_api

async def embed(texts, **kwargs):
    return [[float("mineral" in t.lower()), float("subsidy" in t.lower()), 0.01] for t in texts]

memory_api._embed_many = embed
memory_api.EMBED_DIM = 3
app = memory_api.app
