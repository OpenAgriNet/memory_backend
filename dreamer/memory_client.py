"""Authenticated HTTP client for the private Memory API."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import requests

try:
    from service.runtime_config import memory_api_headers, required
except ImportError:  # Direct script execution adds service/ to sys.path.
    from runtime_config import memory_api_headers, required


def _keep_bearer(request):
    """Explicit auth keeps Requests from applying machine-local credentials."""
    return request


def request(method: str, url: str, **kwargs):
    """Send a request without allowing callers to omit or replace authentication."""
    headers = dict(kwargs.pop("headers", {}) or {})
    if "auth" in kwargs or any(name.lower() == "authorization" for name in headers):
        raise ValueError("Authorization is managed by the Memory API client")
    headers.update(memory_api_headers(url))
    kwargs.pop("allow_redirects", None)
    response = requests.request(
        method, url, headers=headers, auth=_keep_bearer, allow_redirects=False, **kwargs
    )
    if 300 <= response.status_code < 400:
        raise requests.HTTPError("Unexpected redirect from Memory API", response=response)
    return response


def get(url: str, **kwargs):
    return request("GET", url, **kwargs)


def post(url: str, **kwargs):
    return request("POST", url, **kwargs)


def put(url: str, **kwargs):
    return request("PUT", url, **kwargs)


def patch(url: str, **kwargs):
    return request("PATCH", url, **kwargs)


@dataclass(frozen=True)
class MemoryPolicy:
    enabled: bool
    erase_cutoff: datetime | None
    erase_in_progress: bool


def memory_policy(farmer_id: str) -> MemoryPolicy:
    api = required("MEMORY_API_URL").rstrip("/")
    response = get(f"{api}/memory/{farmer_id}/settings", timeout=30)
    response.raise_for_status()
    data = response.json()
    enabled = data["memory_enabled"]
    erasing = data.get("erase_in_progress", False)
    cutoff = data.get("erase_cutoff")
    if type(enabled) is not bool or type(erasing) is not bool:
        raise ValueError("invalid memory policy")
    if cutoff is not None:
        cutoff = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
        if cutoff.utcoffset() is None:
            raise ValueError("invalid erase cutoff")
    return MemoryPolicy(enabled, cutoff, erasing)
