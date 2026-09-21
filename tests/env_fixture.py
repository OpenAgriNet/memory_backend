"""Non-routable defaults let offline tests run without deployment credentials."""
import os

for _key in ("QDRANT_URL", "MARQO_URL", "DREAMER_BASE_URL", "MEMORY_API_URL"):
    os.environ.setdefault(_key, "http://127.0.0.1:1")
os.environ.setdefault("QDRANT_API_KEY", "")
