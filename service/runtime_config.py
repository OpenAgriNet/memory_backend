"""Environment configuration shared by the service and maintenance commands."""
import os
import re
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


_INVALID_MEMORY_TOKENS = {
    "replace-with-a-long-random-internal-token",
    "change-me",
    "changeme",
}


def bearer_token_is_safe(token):
    """Accept the token68 characters used by an HTTP Bearer credential."""
    return bool(re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", token))


def memory_internal_token():
    """Return a bearer-safe internal token, failing closed on unsafe values."""
    token = os.environ.get("MEMORY_INTERNAL_TOKEN", "")
    valid = (
        len(token) >= 32
        and bearer_token_is_safe(token)
        and token.lower() not in _INVALID_MEMORY_TOKENS
    )
    if not valid:
        raise RuntimeError(
            "Set MEMORY_INTERNAL_TOKEN to at least 32 printable ASCII characters "
            "with no whitespace or example value"
        )
    return token


def memory_api_headers(url=None):
    """Attach the service token only to the configured Memory API origin."""
    configured = urlsplit(required("MEMORY_API_URL"))
    target = urlsplit(url or required("MEMORY_API_URL"))
    configured_origin = (configured.scheme.lower(), configured.netloc.lower())
    target_origin = (target.scheme.lower(), target.netloc.lower())
    if configured_origin != target_origin:
        raise RuntimeError("Refusing to send Memory API credentials to another origin")
    return {"Authorization": f"Bearer {memory_internal_token()}"}
