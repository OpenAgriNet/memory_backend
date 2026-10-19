"""Dry-run-first operator command for erasing one farmer's derived memory data."""
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

if __package__:
    from . import memory_client
    from .identity import farmer_key, validate_farmer_id
    from .tasks import MemoryTask, TaskValidationError
    from service.runtime_config import qdrant_headers, required
else:
    import memory_client
    from identity import farmer_key, validate_farmer_id
    from tasks import MemoryTask, TaskValidationError
    from runtime_config import qdrant_headers, required


load_dotenv(
    os.environ.get("AMUL_MEMORY_ENV_FILE", os.path.join(os.path.dirname(__file__), "..", ".env")),
    override=False,
)

_HASH = re.compile(r"^[0-9a-f]{64}$")
_COLLECTION = re.compile(r"^[A-Za-z0-9._-]+$")


class EraseUsageError(ValueError):
    pass


class EraseFailure(RuntimeError):
    def __init__(self, stage: str, cause: Exception):
        super().__init__(stage)
        self.stage = stage
        self.cause = cause


@dataclass(frozen=True)
class QdrantInventory:
    memory_points: int
    state_points: int
    task_keys: tuple[str, ...] = ()
    subject_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class RedisInventory:
    stream_entries: int
    dead_letters: int
    retries: int
    marker_keys: tuple[str, ...] = ()
    stream_ids: tuple[str, ...] = field(default=(), repr=False)
    dead_letter_ids: tuple[str, ...] = field(default=(), repr=False)
    retry_members: tuple[str, ...] = field(default=(), repr=False)


class QdrantEraseStore:
    def __init__(self):
        self.url = required("QDRANT_URL").rstrip("/")
        self.memory_collection = os.environ.get("QDRANT_COLLECTION", "amul_memory")
        self.state_collection = os.environ.get(
            "QDRANT_TASK_STATE_COLLECTION", f"{self.memory_collection}_task_state"
        )
        if self.state_collection == self.memory_collection:
            raise RuntimeError("task state collection must be separate from farmer memory")
        if not _COLLECTION.fullmatch(self.memory_collection) or not _COLLECTION.fullmatch(
            self.state_collection
        ):
            raise RuntimeError("invalid Qdrant collection name")
        self.memory_api = required("MEMORY_API_URL").rstrip("/")

    def _request(self, method: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        target = f"{self.url}{path}"
        configured = urlsplit(self.url)
        requested = urlsplit(target)
        if (configured.scheme, configured.netloc) != (requested.scheme, requested.netloc):
            raise RuntimeError("invalid Qdrant erase target")
        response = requests.request(
            method, target, json=body, headers=qdrant_headers(target), timeout=30
        )
        if response.status_code >= 300:
            raise RuntimeError(f"Qdrant erase request failed with HTTP {response.status_code}")
        return response.json()

    def _collection_exists(self, collection: str) -> bool:
        target = f"{self.url}/collections/{collection}"
        response = requests.get(target, headers=qdrant_headers(target), timeout=30)
        if response.status_code == 404:
            return False
        if response.status_code >= 300:
            raise RuntimeError(
                f"Qdrant collection check failed with HTTP {response.status_code}"
            )
        return True

    @staticmethod
    def _farmer_filter(farmer_id: str) -> dict[str, Any]:
        return {"must": [{"key": "farmer_id", "match": {"value": farmer_id}}]}

    @staticmethod
    def _settings_point_id(farmer_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-settings:{farmer_id}"))

    def _memory_filter(self, farmer_id: str) -> dict[str, Any]:
        result = self._farmer_filter(farmer_id)
        result["must_not"] = [{"has_id": [self._settings_point_id(farmer_id)]}]
        return result

    @staticmethod
    def _state_filter(identity_key: str) -> dict[str, Any]:
        return {"must": [{"key": "farmer_key", "match": {"value": identity_key}}]}

    def _count(self, collection: str, query_filter: dict[str, Any]) -> int:
        result = self._request(
            "POST", f"/collections/{collection}/points/count",
            {"filter": query_filter, "exact": True},
        )
        return int(result["result"]["count"])

    def _state_markers(self, identity_key: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        task_keys: set[str] = set()
        subject_keys: set[str] = set()
        offset: Any | None = None
        while True:
            body: dict[str, Any] = {
                "filter": self._state_filter(identity_key),
                "limit": 100,
                "with_payload": ["task_key", "subject_key"],
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            page = self._request(
                "POST", f"/collections/{self.state_collection}/points/scroll", body
            )["result"]
            for point in page.get("points") or []:
                payload = point.get("payload") or {}
                if _HASH.fullmatch(str(payload.get("task_key", ""))):
                    task_keys.add(payload["task_key"])
                if _HASH.fullmatch(str(payload.get("subject_key", ""))):
                    subject_keys.add(payload["subject_key"])
            offset = page.get("next_page_offset")
            if offset is None:
                break
        return tuple(sorted(task_keys)), tuple(sorted(subject_keys))

    def inventory(self, farmer_id: str, identity_key: str) -> QdrantInventory:
        state_exists = self._collection_exists(self.state_collection)
        task_keys, subject_keys = (
            self._state_markers(identity_key) if state_exists else ((), ())
        )
        return QdrantInventory(
            memory_points=self._count(
                self.memory_collection, self._memory_filter(farmer_id)
            ),
            state_points=(
                self._count(self.state_collection, self._state_filter(identity_key))
                if state_exists else 0
            ),
            task_keys=task_keys,
            subject_keys=subject_keys,
        )

    def disable(self, farmer_id: str) -> None:
        response = memory_client.put(
            f"{self.memory_api}/memory/{farmer_id}/settings",
            json={"memory_enabled": False, "note": "erasure_requested"},
            timeout=30,
        )
        response.raise_for_status()

    def settings_disabled(self, farmer_id: str) -> bool:
        response = memory_client.get(
            f"{self.memory_api}/memory/{farmer_id}/settings", timeout=30
        )
        response.raise_for_status()
        return response.json().get("memory_enabled") is False

    def delete_memory(self, farmer_id: str) -> None:
        self._request(
            "POST", f"/collections/{self.memory_collection}/points/delete?wait=true",
            {"filter": self._memory_filter(farmer_id)},
        )

    def delete_state(self, identity_key: str) -> None:
        if not self._collection_exists(self.state_collection):
            return
        self._request(
            "POST", f"/collections/{self.state_collection}/points/delete?wait=true",
            {"filter": self._state_filter(identity_key)},
        )


class RedisEraseStore:
    def __init__(self, client: Any | None = None):
        if __package__:
            from .redis_queue import RedisStreamTaskQueue
        else:
            from redis_queue import RedisStreamTaskQueue
        queue = RedisStreamTaskQueue(client=client)
        self.client = queue.client
        self.stream = queue.stream
        self.dead_letter_stream = queue.dead_letter_stream
        self.group = queue.group
        self.retry_set = queue.retry_set
        self.dedupe_prefix = queue.dedupe_prefix
        self.revision_prefix = queue.revision_prefix

    @staticmethod
    def _decoded_body(raw: Any) -> dict[str, Any] | None:
        try:
            body = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            return None
        return body if isinstance(body, dict) else None

    @staticmethod
    def _matches(raw: Any, farmer_id: str) -> tuple[bool, str | None, str | None]:
        body = RedisEraseStore._decoded_body(raw)
        if body is None or body.get("farmer_id") != farmer_id:
            return False, None, None
        try:
            task = MemoryTask.from_dict(body)
        except TaskValidationError:
            return True, None, None
        return True, task.task_key, task.subject_key

    def _stream_rows(self, stream: str) -> Iterable[tuple[str, dict[str, str]]]:
        cursor = "-"
        while True:
            rows = self.client.xrange(stream, min=cursor, max="+", count=100)
            if not rows:
                return
            for receipt, fields in rows:
                yield str(receipt), dict(fields)
            cursor = f"({rows[-1][0]}"

    def inventory(
        self, farmer_id: str, *, task_keys: Iterable[str] = (),
        subject_keys: Iterable[str] = (),
    ) -> RedisInventory:
        matched_task_keys = {value for value in task_keys if _HASH.fullmatch(value)}
        matched_subject_keys = {value for value in subject_keys if _HASH.fullmatch(value)}
        stream_ids: list[str] = []
        dead_letter_ids: list[str] = []
        retry_members: list[str] = []

        for stream, target in (
            (self.stream, stream_ids), (self.dead_letter_stream, dead_letter_ids)
        ):
            for receipt, fields in self._stream_rows(stream):
                matches, task_key, subject_key = self._matches(fields.get("body"), farmer_id)
                if not matches:
                    continue
                target.append(receipt)
                outer_task_key = str(fields.get("task_key", ""))
                if _HASH.fullmatch(outer_task_key):
                    matched_task_keys.add(outer_task_key)
                if task_key:
                    matched_task_keys.add(task_key)
                if subject_key:
                    matched_subject_keys.add(subject_key)

        for member, _score in self.client.zscan_iter(self.retry_set, count=100):
            item = self._decoded_body(member)
            if item is None:
                continue
            matches, task_key, subject_key = self._matches(item.get("body"), farmer_id)
            if not matches:
                continue
            retry_members.append(member)
            outer_task_key = str(item.get("task_key", ""))
            if _HASH.fullmatch(outer_task_key):
                matched_task_keys.add(outer_task_key)
            if task_key:
                matched_task_keys.add(task_key)
            if subject_key:
                matched_subject_keys.add(subject_key)

        markers = tuple(sorted(
            [f"{self.dedupe_prefix}:{value}" for value in matched_task_keys]
            + [f"{self.revision_prefix}:{value}" for value in matched_subject_keys]
        ))
        return RedisInventory(
            stream_entries=len(stream_ids),
            dead_letters=len(dead_letter_ids),
            retries=len(retry_members),
            marker_keys=markers,
            stream_ids=tuple(stream_ids),
            dead_letter_ids=tuple(dead_letter_ids),
            retry_members=tuple(retry_members),
        )

    def purge(self, inventory: RedisInventory) -> None:
        pipe = self.client.pipeline(transaction=True)
        for receipt in inventory.stream_ids:
            pipe.xdel(self.stream, receipt)
        for receipt in inventory.dead_letter_ids:
            pipe.xdel(self.dead_letter_stream, receipt)
        if inventory.retry_members:
            pipe.zrem(self.retry_set, *inventory.retry_members)
        if inventory.marker_keys:
            pipe.delete(*inventory.marker_keys)
        pipe.execute()

    def markers_exist(self, marker_keys: Iterable[str]) -> bool:
        return any(bool(self.client.exists(key)) for key in marker_keys)


def _safe_result(
    identity_key: str, qdrant: QdrantInventory, redis: RedisInventory, outcome: str
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "farmer_key": identity_key,
        "memory_points": qdrant.memory_points,
        "task_state_points": qdrant.state_points,
        "queue_entries": redis.stream_entries,
        "dead_letters": redis.dead_letters,
        "retries": redis.retries,
        "markers": len(redis.marker_keys),
    }


def erase_farmer(
    farmer_id: str, *, execute: bool = False, confirm_phone: str | None = None,
    workers_paused: bool = False, qdrant: Any | None = None, redis: Any | None = None,
) -> dict[str, Any]:
    identity = validate_farmer_id(farmer_id)
    identity_key = farmer_key(identity)
    if execute and confirm_phone != identity:
        raise EraseUsageError("confirmation_mismatch")
    if execute and not workers_paused:
        raise EraseUsageError("workers_not_paused")

    qdrant = qdrant or QdrantEraseStore()
    redis = redis or RedisEraseStore()
    try:
        qdrant_plan = qdrant.inventory(identity, identity_key)
        redis_plan = redis.inventory(
            identity,
            task_keys=qdrant_plan.task_keys,
            subject_keys=qdrant_plan.subject_keys,
        )
    except Exception as exc:
        raise EraseFailure("inventory", exc) from exc
    if not execute:
        return _safe_result(identity_key, qdrant_plan, redis_plan, "dry_run")

    for stage, action in (
        ("disable_memory", lambda: qdrant.disable(identity)),
        ("purge_queue", lambda: redis.purge(redis_plan)),
        ("delete_task_state", lambda: qdrant.delete_state(identity_key)),
        ("delete_memory", lambda: qdrant.delete_memory(identity)),
    ):
        try:
            action()
        except Exception as exc:
            raise EraseFailure(stage, exc) from exc

    try:
        remaining_qdrant = qdrant.inventory(identity, identity_key)
        remaining_redis = redis.inventory(identity)
        verified = (
            qdrant.settings_disabled(identity)
            and remaining_qdrant.memory_points == 0
            and remaining_qdrant.state_points == 0
            and remaining_redis.stream_entries == 0
            and remaining_redis.dead_letters == 0
            and remaining_redis.retries == 0
            and not redis.markers_exist(redis_plan.marker_keys)
        )
        if not verified:
            raise RuntimeError("erase verification failed")
    except Exception as exc:
        raise EraseFailure("verification", exc) from exc
    return _safe_result(identity_key, qdrant_plan, redis_plan, "erased")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Erase one farmer's derived memory data")
    parser.add_argument("--farmer-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-phone")
    parser.add_argument("--workers-paused", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = erase_farmer(
            args.farmer_id,
            execute=args.execute,
            confirm_phone=args.confirm_phone,
            workers_paused=args.workers_paused,
        )
    except EraseUsageError as exc:
        print(json.dumps({"outcome": "refused", "reason": str(exc)}, sort_keys=True))
        return 2
    except Exception as exc:
        stage = exc.stage if isinstance(exc, EraseFailure) else "configuration"
        cause = exc.cause if isinstance(exc, EraseFailure) else exc
        print(json.dumps({
            "outcome": "failed", "stage": stage, "error": type(cause).__name__,
        }, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
