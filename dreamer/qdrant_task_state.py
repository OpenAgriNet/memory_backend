"""Lease-protected, hash-only episode progress outside farmer memory."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

import requests

if __package__:
    from .task_processor import (
        BeginResult, CompletionOutcome, FailureDisposition, ProcessedTurn,
    )
    from .tasks import MemoryTask
else:
    from task_processor import BeginResult, CompletionOutcome, FailureDisposition, ProcessedTurn
    from tasks import MemoryTask


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _point_id(subject_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-task-state-v2:{subject_key}"))


def _processed(payload: dict[str, Any] | None) -> tuple[ProcessedTurn, ...]:
    rows = (payload or {}).get("processed_turns") or []
    return tuple(
        ProcessedTurn(row["trace_key"], row["fingerprint"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("trace_key"), str)
        and isinstance(row.get("fingerprint"), str)
    )


class QdrantTaskStateStore:
    """Atomic task claims with conservative recovery after worker crashes."""

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        api_key: str | None = None,
        *,
        lease_seconds: int | None = None,
        now: Callable[[], datetime] = _utc_now,
    ):
        self.url = (url or os.environ.get("QDRANT_URL") or "").rstrip("/")
        if not self.url:
            raise RuntimeError("QDRANT_URL is required")
        memory_collection = os.environ.get("QDRANT_COLLECTION", "amul_memory")
        self.collection = collection or os.environ.get(
            "QDRANT_TASK_STATE_COLLECTION", f"{memory_collection}_task_state_v2"
        )
        if self.collection == memory_collection:
            raise RuntimeError("task state collection must be separate from farmer memory")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.collection):
            raise RuntimeError("invalid Qdrant task-state collection name")
        self.api_key = os.environ.get("QDRANT_API_KEY", "") if api_key is None else api_key
        self.lease_seconds = lease_seconds or int(os.environ.get("MEMORY_TASK_LEASE_SECONDS", "900"))
        if self.lease_seconds < 30:
            raise RuntimeError("MEMORY_TASK_LEASE_SECONDS must be at least 30")
        self._now = now
        self._ready = False
        self._claims: dict[str, str] = {}

    def _headers(self) -> dict[str, str]:
        return {"api-key": self.api_key} if self.api_key else {}

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        target = f"{self.url}{path}"
        if urlsplit(target).netloc != urlsplit(self.url).netloc:
            raise RuntimeError("invalid Qdrant task-state target")
        response = requests.request(method, target, json=body, headers=self._headers(), timeout=30)
        if response.status_code >= 300:
            raise RuntimeError(f"Qdrant task-state request failed with HTTP {response.status_code}")
        return response.json()

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        info = self._request("GET", "/")
        try:
            version = tuple(int(part) for part in str(info.get("version", "0.0")).split(".")[:2])
        except ValueError as exc:
            raise RuntimeError("cannot verify Qdrant task-state support") from exc
        if version < (1, 17):
            raise RuntimeError("Qdrant 1.17 or newer is required for task state")
        response = requests.get(
            f"{self.url}/collections/{self.collection}", headers=self._headers(), timeout=30
        )
        if response.status_code == 404:
            self._request("PUT", f"/collections/{self.collection}", {
                "vectors": {"size": 1, "distance": "Cosine"},
            })
        elif response.status_code >= 300:
            raise RuntimeError(
                f"Qdrant task-state collection check failed with HTTP {response.status_code}"
            )
        self._ready = True

    def _read(self, subject_key: str) -> dict[str, Any] | None:
        result = self._request("POST", f"/collections/{self.collection}/points", {
            "ids": [_point_id(subject_key)], "with_payload": True, "with_vector": False,
        })
        points = result.get("result") or []
        return points[0].get("payload") if points else None

    def _write(self, payload: dict[str, Any], *, previous: dict[str, Any] | None) -> bool:
        body: dict[str, Any] = {
            "points": [{
                "id": _point_id(payload["subject_key"]),
                "vector": [0.0],
                "payload": payload,
            }],
            "update_mode": "insert_only" if previous is None else "update_only",
        }
        if previous is not None:
            must = [
                {"key": "subject_key", "match": {"value": previous["subject_key"]}},
                {"key": "version", "match": {"value": previous["version"]}},
                {"key": "status", "match": {"value": previous["status"]}},
            ]
            if previous.get("lease_owner") is not None:
                must.append({"key": "lease_owner", "match": {"value": previous["lease_owner"]}})
            body["update_filter"] = {"must": must}
        self._request(
            "PUT", f"/collections/{self.collection}/points?wait=true&ordering=strong", body
        )
        stored = self._read(payload["subject_key"]) or {}
        return (
            stored.get("version") == payload["version"]
            and stored.get("task_key") == payload["task_key"]
            and stored.get("status") == payload["status"]
            and stored.get("lease_owner") == payload.get("lease_owner")
        )

    def _latest_result(self, task: MemoryTask) -> BeginResult:
        latest = self._read(task.subject_key) or {}
        if task.task_key in latest.get("completed_task_keys", []):
            return BeginResult("completed", _processed(latest))
        if latest.get("status") == "needs_review":
            return BeginResult("needs_review", _processed(latest))
        return BeginResult("deferred", _processed(latest))

    def begin(self, task: MemoryTask) -> BeginResult:
        self._ensure_ready()
        current = self._read(task.subject_key)
        processed = _processed(current)
        if current and task.task_key in current.get("completed_task_keys", []):
            return BeginResult("completed", processed)
        if current and current.get("status") == "needs_review":
            return BeginResult("needs_review", processed)

        now = self._now()
        if current and current.get("status") == "processing":
            expiry = _parse(current.get("lease_expires_at"))
            if expiry and expiry > now:
                return BeginResult("deferred", processed)
            if current.get("write_state") == "started":
                review = dict(current)
                review.update(
                    status="needs_review", lease_owner=None, lease_expires_at=None,
                    version=int(current["version"]) + 1, updated_at=_iso(now),
                )
                if self._write(review, previous=current):
                    return BeginResult("needs_review", processed)
                return self._latest_result(task)

        completed_revision = int((current or {}).get("completed_revision") or 0)
        completed_part = int((current or {}).get("completed_part") or 0)
        completed_part_count = int((current or {}).get("completed_part_count") or 0)
        if completed_revision == 0:
            sequence = "started" if (
                task.episode_revision == 1 and task.part_number == 1
            ) else "deferred"
        elif task.episode_revision < completed_revision:
            sequence = "needs_review"
        elif task.episode_revision == completed_revision:
            if task.part_count != completed_part_count or task.part_number <= completed_part:
                sequence = "needs_review"
            else:
                sequence = "started" if task.part_number == completed_part + 1 else "deferred"
        elif task.episode_revision == completed_revision + 1:
            sequence = "started" if (
                completed_part == completed_part_count and task.part_number == 1
            ) else "deferred"
        else:
            sequence = "deferred"

        if sequence == "needs_review":
            review = dict(current or {})
            review.update(
                subject_key=task.subject_key, task_key=task.task_key,
                status="needs_review", write_state="not_started",
                lease_owner=None, lease_expires_at=None,
                version=int((current or {}).get("version") or 0) + 1,
                updated_at=_iso(now),
            )
            if self._write(review, previous=current):
                return BeginResult("needs_review", processed)
            return self._latest_result(task)
        if sequence == "deferred":
            return BeginResult("deferred", processed)

        owner = uuid.uuid4().hex
        payload = {
            **(current or {}),
            "schema_version": 2,
            "subject_key": task.subject_key,
            "task_key": task.task_key,
            "status": "processing",
            "write_state": "not_started",
            "processed_turns": [
                {"trace_key": row.trace_key, "fingerprint": row.fingerprint}
                for row in processed
            ],
            "episode_revision": task.episode_revision,
            "part_number": task.part_number,
            "part_count": task.part_count,
            "attempt_count": int((current or {}).get("attempt_count") or 0) + 1,
            "version": int((current or {}).get("version") or 0) + 1,
            "lease_owner": owner,
            "lease_expires_at": _iso(now + timedelta(seconds=self.lease_seconds)),
            "started_at": _iso(now),
            "updated_at": _iso(now),
        }
        if not self._write(payload, previous=current):
            return self._latest_result(task)
        self._claims[task.subject_key] = owner
        return BeginResult("started", processed)

    def _owned(self, task: MemoryTask) -> dict[str, Any]:
        current = self._read(task.subject_key)
        owner = self._claims.get(task.subject_key)
        if (
            not current or not owner or current.get("lease_owner") != owner
            or current.get("task_key") != task.task_key or current.get("status") != "processing"
        ):
            raise RuntimeError("task state is not owned by this worker")
        expiry = _parse(current.get("lease_expires_at"))
        if not expiry or expiry <= self._now():
            raise RuntimeError("task lease expired")
        return current

    def mark_writing(self, task: MemoryTask) -> None:
        current = self._owned(task)
        if current.get("write_state") == "started":
            return
        payload = dict(current)
        payload.update(
            write_state="started", version=int(current["version"]) + 1,
            updated_at=_iso(self._now()),
        )
        if not self._write(payload, previous=current):
            raise RuntimeError("task state changed before mutation")

    def _transition(
        self, task: MemoryTask, status: str,
        processed_turns: Sequence[ProcessedTurn] | None = None,
    ) -> None:
        current = self._owned(task)
        payload = dict(current)
        payload.update(
            status=status, lease_owner=None, lease_expires_at=None,
            version=int(current["version"]) + 1, updated_at=_iso(self._now()),
        )
        if processed_turns is not None:
            payload["processed_turns"] = sorted(
                ({"trace_key": row.trace_key, "fingerprint": row.fingerprint}
                 for row in processed_turns), key=lambda row: row["trace_key"]
            )
            completed = list(dict.fromkeys([
                *(current.get("completed_task_keys") or []), task.task_key,
            ]))
            payload.update(
                completed_task_keys=completed,
                completed_revision=task.episode_revision,
                completed_part=task.part_number,
                completed_part_count=task.part_count,
            )
        if not self._write(payload, previous=current):
            raise RuntimeError("task state changed concurrently")
        self._claims.pop(task.subject_key, None)

    def complete(
        self, task: MemoryTask, outcome: CompletionOutcome,
        processed_turns: Sequence[ProcessedTurn],
    ) -> None:
        self._transition(task, outcome, processed_turns)

    def fail(self, task: MemoryTask, disposition: FailureDisposition) -> None:
        self._transition(task, disposition)

    def administrative_recover(
        self, task: MemoryTask, *, replay_confirmed_safe: bool = False
    ) -> None:
        """Manual recovery after inspecting memory writes for an uncertain task."""
        if not replay_confirmed_safe:
            raise RuntimeError("explicit replay confirmation is required")
        self._ensure_ready()
        current = self._read(task.subject_key)
        if not current or current.get("status") not in ("needs_review", "processing"):
            raise RuntimeError("task is not awaiting recovery")
        expiry = _parse(current.get("lease_expires_at"))
        if current.get("status") == "processing" and expiry and expiry > self._now():
            raise RuntimeError("active lease cannot be recovered")
        payload = dict(current)
        payload.update(
            status="retryable", write_state="not_started", lease_owner=None,
            lease_expires_at=None, version=int(current["version"]) + 1,
            updated_at=_iso(self._now()),
        )
        if not self._write(payload, previous=current):
            raise RuntimeError("task state changed concurrently")
