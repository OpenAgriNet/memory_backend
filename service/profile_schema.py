"""Program-owned standing-record schema, shared by storage and the dreamer."""
import json
import os
from pathlib import Path

SCHEMA_PATH = Path(os.environ.get("MEMORY_PROFILE_SCHEMA", Path(__file__).with_name("profile_fields.json")))
SCHEMA = json.loads(SCHEMA_PATH.read_text())
FIELDS = SCHEMA["fields"]
MAX_TOTAL_CHARS = SCHEMA["max_total_chars"]

if not FIELDS or not isinstance(MAX_TOTAL_CHARS, int) or MAX_TOTAL_CHARS <= 0:
    raise ValueError("Standing schema needs fields and a positive max_total_chars")
for name, spec in FIELDS.items():
    if not name.isidentifier() or not spec.get("description") or not isinstance(spec.get("max_chars"), int) or spec["max_chars"] <= 0:
        raise ValueError(f"Invalid standing-record field: {name}")


def validate(fields):
    """Keep only approved, populated fields; reject oversized or non-text values."""
    kept = {}
    for name, value in fields.items():
        if name not in FIELDS or value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ValueError(f"{name} must contain text")
        value = value.strip()
        if len(value) > FIELDS[name]["max_chars"]:
            raise ValueError(f"{name} exceeds its {FIELDS[name]['max_chars']}-character limit")
        if value:
            kept[name] = value
    if sum(len(value) for value in kept.values()) > MAX_TOTAL_CHARS:
        raise ValueError(f"Standing record exceeds its {MAX_TOTAL_CHARS}-character total")
    return kept
