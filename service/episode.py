"""Lossless episode chunks and a bounded Level 2 contents page.

Character positions refer to the exact stored expanded text. Numbering belongs to
one immutable episode version. Changing a reading budget never changes the layout.
"""
import os
import re

CHUNK_MAX_CHARS = int(os.getenv("MEMORY_CHUNK_MAX_CHARS", "1200"))
SUMMARY_MAX_CHARS = int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "350"))
CONTENTS_ROW_MAX_CHARS = int(os.getenv("MEMORY_CONTENTS_ROW_MAX_CHARS", "120"))
MAX_CONTENTS_ROWS = 5
if CHUNK_MAX_CHARS < 100 or SUMMARY_MAX_CHARS < 40 or CONTENTS_ROW_MAX_CHARS < 30:
    raise ValueError("Memory budgets are too small: chunk >=100, summary >=40, contents row >=30")


def split_text(text, max_chars=CHUNK_MAX_CHARS):
    """Partition every character once, preferring paragraph/sentence boundaries."""
    if max_chars < 1:
        raise ValueError("chunk size must be positive")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            # Avoid tiny fragments while keeping natural boundaries when possible.
            for pattern in (r"\n\s*\n", r"[.!?।]\s+", r"\s+"):
                boundaries = [m.end() for m in re.finditer(pattern, window) if m.end() >= max_chars // 2]
                if boundaries:
                    end = start + boundaries[-1]
                    break
        chunks.append({"number": len(chunks) + 1, "start": start, "end": end})
        start = end
    return chunks


def validate_contents(rows, chunk_count):
    """At most five meaningful rows; contiguous ranges cover the entire account."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= min(MAX_CONTENTS_ROWS, chunk_count):
        raise ValueError("contents must have 1–5 rows covering all chunks")
    cleaned, next_chunk = [], 1
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"start_chunk", "end_chunk", "text"}:
            raise ValueError("Each contents row needs start_chunk, end_chunk, text")
        first, last, text = row["start_chunk"], row["end_chunk"], row["text"]
        if type(first) is not int or type(last) is not int or first != next_chunk or not first <= last <= chunk_count:
            raise ValueError("contents ranges must cover all chunks in order without gaps or overlaps")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= CONTENTS_ROW_MAX_CHARS:
            raise ValueError(f"contents text must be 1–{CONTENTS_ROW_MAX_CHARS} characters")
        cleaned.append({"start_chunk": first, "end_chunk": last, "text": text.strip()})
        next_chunk = last + 1
    if next_chunk != chunk_count + 1:
        raise ValueError("contents must cover the last chunk")
    return cleaned


def contents_groups(count):
    """At most five adjacent ranges; numbering and coverage are code-owned."""
    groups = min(MAX_CONTENTS_ROWS, count)
    return [(i * count // groups + 1, (i + 1) * count // groups) for i in range(groups)]


def extractive_contents(text, chunks):
    """No-model fallback for legacy reads/direct API writes; excerpts, not invented labels."""
    count = len(chunks)
    rows = []
    for i in range(min(MAX_CONTENTS_ROWS, count)):
        groups = min(MAX_CONTENTS_ROWS, count)
        first, last = i * count // groups, (i + 1) * count // groups
        excerpt = " ".join(text[chunks[first]["start"]:chunks[last - 1]["end"]].split())
        if len(excerpt) > CONTENTS_ROW_MAX_CHARS:
            excerpt = excerpt[:CONTENTS_ROW_MAX_CHARS - 1].rstrip() + "…"
        rows.append({"start_chunk": first + 1, "end_chunk": last, "text": excerpt or "(blank passage)"})
    return rows


def layout(payload):
    """Old entries remain readable without altering their stored data."""
    text = payload.get("expanded") or payload.get("headline") or ""
    chunks = payload.get("chunks") or split_text(text)
    rows = payload.get("contents") or extractive_contents(text, chunks)
    return text, chunks, rows


def level2_text(headline, contents):
    return headline + "\n" + "\n".join(
        f"Chunks {r['start_chunk']}-{r['end_chunk']}: {r['text']}" for r in contents)


def read_chunks(payload, numbers=None, *, offset=0, max_chars=2000, start_chunk=1):
    """Read selected chunks with a shared text budget and an exact continuation."""
    text, chunks, _ = layout(payload)
    selected = sorted(set(numbers)) if numbers is not None else list(range(start_chunk, len(chunks) + 1))
    if not selected or any(type(n) is not int or not 1 <= n <= len(chunks) for n in selected):
        raise ValueError("Select chunk numbers shown for this episode")
    if type(offset) is not int or offset < 0 or max_chars < 1:
        raise ValueError("Invalid reading offset or budget")
    if offset >= chunks[selected[0] - 1]["end"] - chunks[selected[0] - 1]["start"]:
        raise ValueError("Offset must be inside the first selected chunk")
    output, remaining = [], max_chars
    for pos, number in enumerate(selected):
        chunk = chunks[number - 1]
        begin = chunk["start"] + (offset if pos == 0 else 0)
        end = min(chunk["end"], begin + remaining)
        output.append({"number": number, "offset": begin - chunk["start"],
                       "text": text[begin:end]})
        remaining -= end - begin
        if end < chunk["end"]:
            return output, ({"chunk_ids": selected[pos:], "offset": end - chunk["start"]} if numbers is not None
                            else {"start_chunk": number, "offset": end - chunk["start"]})
        if not remaining and pos + 1 < len(selected):
            return output, ({"chunk_ids": selected[pos + 1:], "offset": 0} if numbers is not None
                            else {"start_chunk": selected[pos + 1], "offset": 0})
    return output, None


READ_CHUNKS = max(1, min(10, int(os.getenv("MEMORY_READ_CHUNKS", "3"))))


def whole_chunk_page(payload, numbers=None):
    """Return whole chunks only. The program controls page size, never the model."""
    text, chunks, _ = layout(payload)
    if numbers is None:
        numbers = list(range(1, min(READ_CHUNKS, len(chunks)) + 1))
    if not numbers or any(type(n) is not int or not 1 <= n <= len(chunks) for n in numbers):
        raise ValueError("Select chunk numbers shown for this episode")
    selected = sorted(set(numbers))
    page = selected[:READ_CHUNKS]
    rows = [{"number": n, "text": text[chunks[n - 1]["start"]:chunks[n - 1]["end"]]}
            for n in page]
    next_ids = selected[READ_CHUNKS:] or list(range(page[-1] + 1, min(len(chunks), page[-1] + READ_CHUNKS) + 1))
    return rows, next_ids
