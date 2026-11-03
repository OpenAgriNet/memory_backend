"""Authenticated HTTP client for the private Memory API."""
from __future__ import annotations

import requests

try:
    from service.runtime_config import memory_api_headers, required
except ImportError:  # Direct script execution adds service/ to sys.path.
    from runtime_config import memory_api_headers, required


def request(method: str, url: str, **kwargs):
    """Send a request without allowing callers to omit or replace authentication."""
    headers = dict(kwargs.pop("headers", {}) or {})
    if any(name.lower() == "authorization" for name in headers):
        raise ValueError("Authorization is managed by the Memory API client")
    headers.update(memory_api_headers(url))
    return requests.request(method, url, headers=headers, **kwargs)


def get(url: str, **kwargs):
    return request("GET", url, **kwargs)


def post(url: str, **kwargs):
    return request("POST", url, **kwargs)


def put(url: str, **kwargs):
    return request("PUT", url, **kwargs)


def patch(url: str, **kwargs):
    return request("PATCH", url, **kwargs)


def delete(url: str, **kwargs):
    return request("DELETE", url, **kwargs)


def memory_enabled(farmer_id: str) -> bool:
    response = get(f"{required('MEMORY_API_URL').rstrip('/')}/memory/{farmer_id}/settings", timeout=30)
    response.raise_for_status()
    enabled = response.json().get("memory_enabled")
    if type(enabled) is not bool:
        raise RuntimeError("invalid_memory_setting")
    return enabled
