"""Authenticated HTTP client for the private memory service.

All Dreamer reads and writes go through this module so adding a new call cannot
accidentally omit the service credential.  The token is attached as a header and
is never included in exceptions or logs by this wrapper.
"""
from __future__ import annotations

import requests

from runtime_config import memory_api_headers


def request(method: str, url: str, **kwargs):
    headers = dict(kwargs.pop("headers", {}) or {})
    headers.update(memory_api_headers())
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
