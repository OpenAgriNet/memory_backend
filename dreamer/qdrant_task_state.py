"""Simple, hash-only task progress stored outside the farmer-memory collection."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence
from urllib.parse import urlsplit

import requests

if __package__:
    from .task_processor import BeginResult, CompletionOutcome, FailureDisposition
    from .tasks import MemoryTask
else:
    from task_processor import BeginResult, CompletionOutcome, FailureDisposition
    from tasks import MemoryTask


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _point_id(subject_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-task-state:{subject_key}"))


class QdrantTaskStateStore:
    """One non-overlapping worker's durable task progress.

    Conditional Qdrant writes prevent accidental second ownership, but there is no
    lease, renewal thread, retry ladder, or automatic takeover in v0.
    """

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        api_key: str | None = None,
    ):
        self.url = (url or os.environ.get("QDRANT_URL") or "").rstrip("/")
        if not self.url:
            raise RuntimeError("QDRANT_URL is required")
        memory_collection = os.environ.get("QDRANT_COLLECTION", "amul_memory")
        self.collection = collection or os.environ.get(
            "QDRANT_TASK_STATE_COLLECTION", f"{memory_collection}_task_state"
        )
        if self.collection == memory_collection:
            raise RuntimeError("task state collection must be separate from farmer memory")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.collection):
            raise RuntimeError("invalid Qdrant task-state collection name")
        self.api_key = os.environ.get("QDRANT_API_KEY", "") if api_key is None else api_key
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

    def _write(
        self, payload: dict[str, Any], *, previous: dict[str, Any] | None
    ) -> bool:
        body: dict[str, Any] = {
            "points": [{
                "id": _point_id(payload["subject_key"]),
                "vector": [0.0],
                "payload": payload,
            }],
            "update_mode": "insert_only" if previous is None else "update_only",
        }
        if previous is not None:
            body["update_filter"] = {"must": [
                {"key": "subject_key", "match": {"value": previous["subject_key"]}},
                {"key": "version", "match": {"value": previous["version"]}},
                {"key": "status", "match": {"value": previous["status"]}},
            ]}
        self._request(
            "PUT", f"/collections/{self.collection}/points?wait=true&ordering=strong", body
        )
        stored = self._read(payload["subject_key"]) or {}
        return (
            stored.get("version") == payload["version"]
            and stored.get("task_key") == payload["task_key"]
            and stored.get("status") == payload["status"]
            and stored.get("claim_token") == payload.get("claim_token")
        )

    def begin(self, task: MemoryTask) -> BeginResult:
        self._ensure_ready()
        current = self._read(task.subject_key)
        processed = tuple((current or {}).get("processed_turns") or ())
        if current and current.get("task_key") == task.task_key and current.get("status") in (
            "completed", "no_op",
        ):
            return BeginResult("completed", processed)
        if current and current.get("status") == "needs_review":
            return BeginResult("needs_review", processed)
        if current and current.get("status") == "processing":
            return BeginResult("processing", processed)

        now = _now()
        claim_token = uuid.uuid4().hex
        payload = {
            "subject_key": task.subject_key,
            "task_key": task.task_key,
            "status": "processing",
            "processed_turns": list(processed),
            "attempt_count": int((current or {}).get("attempt_count") or 0) + 1,
            "version": int((current or {}).get("version") or 0) + 1,
            "claim_token": claim_token,
            "started_at": now,
            "updated_at": now,
        }
        if not self._write(payload, previous=current):
            latest = self._read(task.subject_key) or {}
            outcome = "completed" if latest.get("status") in ("completed", "no_op") else "processing"
            return BeginResult(outcome, tuple(latest.get("processed_turns") or ()))
        self._claims[task.subject_key] = claim_token
        return BeginResult("started", processed)

    def _transition(
        self, task: MemoryTask, status: str, processed_turns: Sequence[str] | None = None
    ) -> None:
        current = self._read(task.subject_key)
        claim_token = self._claims.get(task.subject_key)
        if (
            not current
            or not claim_token
            or current.get("claim_token") != claim_token
            or current.get("task_key") != task.task_key
            or current.get("status") != "processing"
        ):
            raise RuntimeError("task state is not owned by this task")
        payload = dict(current)
        payload.update(status=status, version=current["version"] + 1, updated_at=_now())
        if processed_turns is not None:
            payload["processed_turns"] = sorted(set(processed_turns))
        if not self._write(payload, previous=current):
            raise RuntimeError("task state changed concurrently")
        self._claims.pop(task.subject_key, None)

    def complete(
        self, task: MemoryTask, outcome: CompletionOutcome, processed_turns: Sequence[str]
    ) -> None:
        self._transition(task, outcome, processed_turns)

    def fail(self, task: MemoryTask, disposition: FailureDisposition) -> None:
        self._transition(task, disposition)
