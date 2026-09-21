"""Environment configuration shared by the service and maintenance commands."""
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

_root_env = Path(__file__).resolve().parents[1] / ".env"
ENV_FILE = Path(os.environ.get("AMUL_MEMORY_ENV_FILE", str(_root_env)))
load_dotenv(ENV_FILE, override=False)


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in the environment or AMUL_MEMORY_ENV_FILE")
    return value


def qdrant_headers(url=None):
    """Send credentials only to the configured Qdrant origin, never embeddings."""
    key = os.environ.get("QDRANT_API_KEY", "").strip()
    if not key:
        return {}
    configured = urlsplit(required("QDRANT_URL"))
    target = urlsplit(url or required("QDRANT_URL"))
    if (target.scheme, target.netloc) != (configured.scheme, configured.netloc):
        return {}
    return {"api-key": key}
