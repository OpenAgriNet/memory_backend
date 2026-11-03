"""Authenticated HTTP client for the private Memory API."""
from __future__ import annotations

import requests

try:
    from service.runtime_config import memory_api_headers
except ImportError:  # Direct script execution adds service/ to sys.path.
    from runtime_config import memory_api_headers


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
