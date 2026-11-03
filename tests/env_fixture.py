"""Non-routable defaults let offline tests run without deployment credentials."""
import os

for _key in ("QDRANT_URL", "MARQO_URL", "DREAMER_BASE_URL", "MEMORY_API_URL"):
    os.environ.setdefault(_key, "http://127.0.0.1:1")
os.environ.setdefault("QDRANT_API_KEY", "")
os.environ.setdefault("MEMORY_INTERNAL_TOKEN", "offline-test-token-32-characters-minimum")
os.environ.setdefault("MEMORY_ID_HASH_KEY", "offline-id-hash-key-32-characters-minimum")
