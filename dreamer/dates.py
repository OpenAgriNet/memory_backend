"""Date display for the background job's own prompts.

Stored timestamps remain ISO strings. The Amul bot displays those stored values
directly and does not depend on this formatter.
"""
from datetime import datetime

UNKNOWN = "date not recorded"


def fmt(ts) -> str:
    """An ISO timestamp (or anything) as dd-Month-yyyy. Never raises."""
    if not ts:
        return UNKNOWN
    if isinstance(ts, datetime):
        return f"{ts.day:02d}-{ts.strftime('%B')}-{ts.year}"
    text = str(ts).strip()
    if not text:
        return UNKNOWN
    try:
        d = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            d = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            # Better to show something honest than a half-parsed date.
            return UNKNOWN
    return f"{d.day:02d}-{d.strftime('%B')}-{d.year}"


def span(first, last) -> str:
    """A date range, collapsed when both ends are the same day."""
    a, b = fmt(first), fmt(last)
    if a == UNKNOWN and b == UNKNOWN:
        return UNKNOWN
    if a == UNKNOWN:
        return b
    if b == UNKNOWN or a == b:
        return a
    return f"{a} to {b}"
