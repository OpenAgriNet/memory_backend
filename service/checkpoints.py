"""Atomic worker checkpoints stored as internal zero-vector Qdrant points."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

CheckpointKind = Literal["session_ingestion", "farmer_consolidation", "farmer_profile"]
CheckpointStatus = Literal["processing", "completed", "no_op", "retryable", "needs_review"]
CompletionStatus = Literal["completed", "no_op"]
CHECKPOINT_TYPE = "worker_checkpoint"
RETRY_DELAYS = (15 * 60, 60 * 60, 6 * 60 * 60, 24 * 60 * 60)
MAX_PROCESSED_TRACES = 500
QdrantCall = Callable[[str, str, dict[str, Any] | None], Awaitable[dict[str, Any]]]


class CheckpointRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: CheckpointKind
    subject_id: str = Field(min_length=1, max_length=300)


class CheckpointLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subjects: list[CheckpointRef] = Field(min_length=1, max_length=100)


class ProcessedTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: str = Field(min_length=1, max_length=200)
    source_version: str | None = Field(default=None, max_length=200)
    source_timestamp: datetime | None = None

    @model_validator(mode="after")
    def _has_source_marker(self):
        if self.source_version is None and self.source_timestamp is None:
            raise ValueError("source_version or source_timestamp is required")
        return self


class CheckpointClaimRequest(CheckpointRef):
    farmer_id: str = Field(min_length=1, max_length=200)
    candidate_fingerprint: str = Field(min_length=1, max_length=200)
    last_activity_at: datetime | None = None
    run_id: str = Field(min_length=1, max_length=200)


class CheckpointLeaseRequest(CheckpointRef):
    lease_token: str = Field(min_length=1, max_length=200)
    checkpoint_version: int = Field(ge=1)


class CheckpointRenewRequest(CheckpointLeaseRequest):
    pass


class CheckpointCompleteRequest(CheckpointLeaseRequest):
    status: CompletionStatus
    processed_traces: list[ProcessedTrace] = Field(default_factory=list, max_length=MAX_PROCESSED_TRACES)


class CheckpointFailRequest(CheckpointLeaseRequest):
    error_code: str = Field(min_length=1, max_length=80)
    error_summary: str = Field(min_length=1, max_length=2000)


class CheckpointDueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before: datetime | None = None
    kinds: list[CheckpointKind] | None = Field(default=None, max_length=3)
    limit: int = Field(default=100, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=200)


def checkpoint_id(kind: str, subject_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-checkpoint:{kind}:{subject_id}"))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def public_checkpoint(point: dict[str, Any]) -> dict[str, Any]:
    payload = point.get("payload", point)
    return {
        "kind": payload.get("checkpoint_kind"),
        "subject_id": payload.get("checkpoint_subject_id"),
        "farmer_id": payload.get("farmer_id"),
        "candidate_fingerprint": payload.get("checkpoint_candidate_fingerprint"),
        "last_activity_at": payload.get("checkpoint_last_activity_at"),
        "status": payload.get("checkpoint_status"),
        "processed_traces": payload.get("checkpoint_processed_traces") or [],
        "attempt_count": payload.get("checkpoint_attempt_count", 0),
        "run_id": payload.get("checkpoint_run_id"),
        "lease_expires_at": payload.get("checkpoint_lease_expires_at"),
        "next_retry_at": payload.get("checkpoint_next_retry_at"),
        "completed_at": payload.get("checkpoint_completed_at"),
        "updated_at": payload.get("checkpoint_updated_at"),
        "checkpoint_version": payload.get("checkpoint_version"),
        "last_error_code": payload.get("checkpoint_last_error_code"),
        "last_error_summary": payload.get("checkpoint_last_error_summary"),
    }


async def read_checkpoint(call: QdrantCall, collection: str, kind: str, subject_id: str) -> dict[str, Any] | None:
    result = await call("POST", f"/collections/{collection}/points", {
        "ids": [checkpoint_id(kind, subject_id)], "with_payload": True, "with_vector": False,
    })
    points = result.get("result") or []
    return points[0] if points else None


def _zero_vectors(embed_dim: int) -> dict[str, list[float]]:
    zero = [0.0] * embed_dim
    return {"headline": zero, "expanded": zero}


def _version_filter(
    payload: dict[str, Any], *, lease_token: str | None = None,
    lease_valid_after: datetime | None = None,
) -> dict[str, Any]:
    must = [
        {"key": "type", "match": {"value": CHECKPOINT_TYPE}},
        {"key": "checkpoint_kind", "match": {"value": payload["checkpoint_kind"]}},
        {"key": "checkpoint_subject_id", "match": {"value": payload["checkpoint_subject_id"]}},
        {"key": "checkpoint_version", "match": {"value": payload["checkpoint_version"]}},
    ]
    if lease_token is not None:
        must.extend([
            {"key": "checkpoint_status", "match": {"value": "processing"}},
            {"key": "checkpoint_lease_token", "match": {"value": lease_token}},
        ])
    if lease_valid_after is not None:
        must.append({
            "key": "checkpoint_lease_expires_at",
            "range": {"gt": iso(lease_valid_after)},
        })
    return {"must": must}


async def _atomic_write(
    call: QdrantCall,
    collection: str,
    embed_dim: int,
    payload: dict[str, Any],
    *,
    previous: dict[str, Any] | None,
    owner_token: str,
) -> bool:
    body: dict[str, Any] = {
        "points": [{
            "id": checkpoint_id(payload["checkpoint_kind"], payload["checkpoint_subject_id"]),
            "vector": _zero_vectors(embed_dim),
            "payload": payload,
        }],
        "update_mode": "insert_only" if previous is None else "update_only",
    }
    if previous is not None:
        body["update_filter"] = _version_filter(previous)
    await call("PUT", f"/collections/{collection}/points?wait=true&ordering=strong", body)
    stored = await read_checkpoint(call, collection, payload["checkpoint_kind"], payload["checkpoint_subject_id"])
    stored_payload = (stored or {}).get("payload", {})
    return (
        stored_payload.get("checkpoint_version") == payload["checkpoint_version"]
        and stored_payload.get("checkpoint_lease_token") == owner_token
        and stored_payload.get("checkpoint_status") == payload["checkpoint_status"]
    )


async def _atomic_lease_transition(
    call: QdrantCall,
    collection: str,
    embed_dim: int,
    previous: dict[str, Any],
    payload: dict[str, Any],
    lease_token: str,
) -> bool:
    body = {
        "points": [{
            "id": checkpoint_id(payload["checkpoint_kind"], payload["checkpoint_subject_id"]),
            "vector": _zero_vectors(embed_dim),
            "payload": payload,
        }],
        "update_mode": "update_only",
        "update_filter": _version_filter(
            previous, lease_token=lease_token, lease_valid_after=utc_now(),
        ),
    }
    await call("PUT", f"/collections/{collection}/points?wait=true&ordering=strong", body)
    stored = await read_checkpoint(call, collection, payload["checkpoint_kind"], payload["checkpoint_subject_id"])
    stored_payload = (stored or {}).get("payload", {})
    return (
        stored_payload.get("checkpoint_version") == payload["checkpoint_version"]
        and stored_payload.get("checkpoint_status") == payload["checkpoint_status"]
        and stored_payload.get("checkpoint_lease_token") == payload.get("checkpoint_lease_token")
    )


def _base_payload(req: CheckpointClaimRequest, now: datetime, lease_seconds: int, previous: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    old = (previous or {}).get("payload", {})
    changed = bool(old) and old.get("checkpoint_candidate_fingerprint") != req.candidate_fingerprint
    attempt = 1 if changed or not old else int(old.get("checkpoint_attempt_count") or 0) + 1
    lease_token = uuid.uuid4().hex
    return {
        "farmer_id": req.farmer_id,
        "type": CHECKPOINT_TYPE,
        "status": "n/a",
        "recorded_at": old.get("recorded_at") or iso(now),
        "superseded_at": None,
        "checkpoint_kind": req.kind,
        "checkpoint_subject_id": req.subject_id,
        "checkpoint_candidate_fingerprint": req.candidate_fingerprint,
        "checkpoint_last_activity_at": iso(req.last_activity_at),
        "checkpoint_status": "processing",
        "checkpoint_processed_traces": old.get("checkpoint_processed_traces") or [],
        "checkpoint_attempt_count": attempt,
        "checkpoint_run_id": req.run_id,
        "checkpoint_lease_token": lease_token,
        "checkpoint_lease_expires_at": iso(now + timedelta(seconds=lease_seconds)),
        "checkpoint_next_retry_at": None,
        "checkpoint_completed_at": None,
        "checkpoint_updated_at": iso(now),
        "checkpoint_version": int(old.get("checkpoint_version") or 0) + 1,
        "checkpoint_last_error_code": None,
        "checkpoint_last_error_summary": None,
    }, lease_token


async def claim_checkpoint(call: QdrantCall, collection: str, embed_dim: int, lease_seconds: int, req: CheckpointClaimRequest) -> dict[str, Any]:
    now = utc_now()
    current = await read_checkpoint(call, collection, req.kind, req.subject_id)
    old = (current or {}).get("payload", {})
    if old and old.get("farmer_id") != req.farmer_id:
        raise HTTPException(409, "checkpoint subject already belongs to another farmer")
    unchanged = bool(old) and old.get("checkpoint_candidate_fingerprint") == req.candidate_fingerprint
    if unchanged:
        status = old.get("checkpoint_status")
        if status in ("completed", "no_op"):
            return {"claimed": False, "reason": "unchanged_complete", "checkpoint": public_checkpoint(current)}
        if status == "needs_review":
            return {"claimed": False, "reason": "needs_review", "checkpoint": public_checkpoint(current)}
        if status == "processing" and (parse_time(old.get("checkpoint_lease_expires_at")) or now) > now:
            return {"claimed": False, "reason": "active_lease", "checkpoint": public_checkpoint(current)}
        if status == "retryable" and (parse_time(old.get("checkpoint_next_retry_at")) or now) > now:
            return {"claimed": False, "reason": "retry_not_due", "checkpoint": public_checkpoint(current)}

    payload, lease_token = _base_payload(req, now, lease_seconds, current)
    won = await _atomic_write(call, collection, embed_dim, payload, previous=old or None, owner_token=lease_token)
    if not won:
        latest = await read_checkpoint(call, collection, req.kind, req.subject_id)
        return {"claimed": False, "reason": "lost_race", "checkpoint": public_checkpoint(latest or {})}
    return {
        "claimed": True,
        "lease_token": lease_token,
        "checkpoint_version": payload["checkpoint_version"],
        "processed_traces": payload["checkpoint_processed_traces"],
        "lease_expires_at": payload["checkpoint_lease_expires_at"],
    }


async def _lease_record(call: QdrantCall, collection: str, req: CheckpointLeaseRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    point = await read_checkpoint(call, collection, req.kind, req.subject_id)
    payload = (point or {}).get("payload", {})
    if (
        not payload
        or payload.get("checkpoint_status") != "processing"
        or payload.get("checkpoint_lease_token") != req.lease_token
        or payload.get("checkpoint_version") != req.checkpoint_version
    ):
        raise HTTPException(409, "stale checkpoint lease or version")
    if (parse_time(payload.get("checkpoint_lease_expires_at")) or utc_now()) <= utc_now():
        raise HTTPException(409, "checkpoint lease has expired")
    return point, payload


async def renew_checkpoint(call: QdrantCall, collection: str, embed_dim: int, lease_seconds: int, req: CheckpointRenewRequest) -> dict[str, Any]:
    _, old = await _lease_record(call, collection, req)
    now = utc_now()
    payload = dict(old)
    payload.update({
        "checkpoint_version": req.checkpoint_version + 1,
        "checkpoint_lease_expires_at": iso(now + timedelta(seconds=lease_seconds)),
        "checkpoint_updated_at": iso(now),
    })
    if not await _atomic_lease_transition(call, collection, embed_dim, old, payload, req.lease_token):
        raise HTTPException(409, "stale checkpoint lease or version")
    return {"renewed": True, "checkpoint_version": payload["checkpoint_version"], "lease_expires_at": payload["checkpoint_lease_expires_at"]}


def _merge_traces(previous: list[dict[str, Any]], incoming: list[ProcessedTrace]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {item["trace_id"]: dict(item) for item in previous if item.get("trace_id")}
    for item in incoming:
        row = item.model_dump(mode="json")
        merged[row["trace_id"]] = row
    return list(merged.values())[-MAX_PROCESSED_TRACES:]


async def complete_checkpoint(call: QdrantCall, collection: str, embed_dim: int, req: CheckpointCompleteRequest) -> dict[str, Any]:
    _, old = await _lease_record(call, collection, req)
    now = utc_now()
    payload = dict(old)
    payload.update({
        "checkpoint_status": req.status,
        "checkpoint_processed_traces": _merge_traces(old.get("checkpoint_processed_traces") or [], req.processed_traces),
        "checkpoint_version": req.checkpoint_version + 1,
        "checkpoint_lease_token": None,
        "checkpoint_lease_expires_at": None,
        "checkpoint_next_retry_at": None,
        "checkpoint_completed_at": iso(now),
        "checkpoint_updated_at": iso(now),
        "checkpoint_last_error_code": None,
        "checkpoint_last_error_summary": None,
    })
    if not await _atomic_lease_transition(call, collection, embed_dim, old, payload, req.lease_token):
        raise HTTPException(409, "stale checkpoint lease or version")
    return {"completed": True, "checkpoint": public_checkpoint(payload)}


_SECRET = re.compile(r"(?i)(authorization|bearer|token|secret|password|api[-_ ]?key)\s*[:=]?\s*\S+")
_LONG_DIGITS = re.compile(r"\b\d{8,}\b")
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\b")


def sanitize_error(code: str, summary: str) -> tuple[str, str]:
    safe_code = re.sub(r"[^A-Za-z0-9_.-]", "_", code.strip())[:80] or "worker_error"
    safe = _SECRET.sub("[redacted]", summary)
    safe = _EMAIL.sub("[redacted]", safe)
    safe = _LONG_DIGITS.sub("[redacted]", safe)
    safe = " ".join(safe.split())[:500]
    return safe_code, safe or "worker failed"


async def fail_checkpoint(call: QdrantCall, collection: str, embed_dim: int, req: CheckpointFailRequest) -> dict[str, Any]:
    _, old = await _lease_record(call, collection, req)
    now = utc_now()
    attempt = int(old.get("checkpoint_attempt_count") or 1)
    final = attempt >= 5
    code, summary = sanitize_error(req.error_code, req.error_summary)
    payload = dict(old)
    payload.update({
        "checkpoint_status": "needs_review" if final else "retryable",
        "checkpoint_version": req.checkpoint_version + 1,
        "checkpoint_lease_token": None,
        "checkpoint_lease_expires_at": None,
        "checkpoint_next_retry_at": None if final else iso(now + timedelta(seconds=RETRY_DELAYS[attempt - 1])),
        "checkpoint_completed_at": None,
        "checkpoint_updated_at": iso(now),
        "checkpoint_last_error_code": code,
        "checkpoint_last_error_summary": summary,
    })
    if not await _atomic_lease_transition(call, collection, embed_dim, old, payload, req.lease_token):
        raise HTTPException(409, "stale checkpoint lease or version")
    return {"failed": True, "checkpoint": public_checkpoint(payload)}


async def due_checkpoints(call: QdrantCall, collection: str, req: CheckpointDueRequest) -> dict[str, Any]:
    before = iso(req.before or utc_now())
    must: list[dict[str, Any]] = [{"key": "type", "match": {"value": CHECKPOINT_TYPE}}]
    if req.kinds:
        must.append({"key": "checkpoint_kind", "match": {"any": sorted(set(req.kinds))}})
    body: dict[str, Any] = {
        "filter": {"must": must, "should": [
            {"must": [
                {"key": "checkpoint_status", "match": {"value": "retryable"}},
                {"key": "checkpoint_next_retry_at", "range": {"lte": before}},
            ]},
            {"must": [
                {"key": "checkpoint_status", "match": {"value": "processing"}},
                {"key": "checkpoint_lease_expires_at", "range": {"lte": before}},
            ]},
        ]},
        "limit": req.limit,
        "with_payload": True,
        "with_vector": False,
    }
    if req.cursor:
        body["offset"] = req.cursor
    result = await call("POST", f"/collections/{collection}/points/scroll", body)
    page = result.get("result") or {}
    return {
        "checkpoints": [public_checkpoint(point) for point in page.get("points") or []],
        "next_cursor": page.get("next_page_offset"),
    }
