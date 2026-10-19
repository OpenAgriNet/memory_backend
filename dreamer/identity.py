"""Stable, non-reversible identifiers for internal memory bookkeeping."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(
    os.environ.get("AMUL_MEMORY_ENV_FILE", Path(__file__).resolve().parents[1] / ".env"),
    override=False,
)

_PHONE = re.compile(r"^[0-9]{10}$")
_KEY = re.compile(r"^[A-Za-z0-9._~+/-]+=*$")


def validate_farmer_id(value: str) -> str:
    if not isinstance(value, str) or not _PHONE.fullmatch(value):
        raise ValueError("farmer_id must be exactly 10 digits")
    return value


def memory_id_hash_key() -> bytes:
    value = os.environ.get("MEMORY_ID_HASH_KEY", "")
    if len(value) < 32 or not _KEY.fullmatch(value):
        raise RuntimeError(
            "Set MEMORY_ID_HASH_KEY to at least 32 bearer-safe ASCII characters"
        )
    return value.encode("ascii")


def farmer_key(farmer_id: str) -> str:
    identity = validate_farmer_id(farmer_id)
    return hmac.new(
        memory_id_hash_key(), identity.encode("ascii"), hashlib.sha256
    ).hexdigest()
