"""Dry-run-first operator command for erasing one farmer's derived memory data."""
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

if __package__:
    from . import memory_client
    from .episode_ledger import RedisEpisodeLedger
    from .identity import farmer_key, validate_farmer_id
    from .tasks import MemoryTask, TaskValidationError, episode_subject_key
    from service.runtime_config import memory_erase_point_id, qdrant_headers, required
else:
    import memory_client
    from episode_ledger import RedisEpisodeLedger
    from identity import farmer_key, validate_farmer_id
    from tasks import MemoryTask, TaskValidationError, episode_subject_key
    from runtime_config import memory_erase_point_id, qdrant_headers, required


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


@dataclass(frozen=True)
class RedisInventory:
    stream_entries: int
    dead_letters: int
    retries: int
    marker_keys: tuple[str, ...] = ()
    ledger_keys: tuple[str, ...] = ()
    ledger_index: str = ""
    stream_ids: tuple[str, ...] = field(default=(), repr=False)
    dead_letter_ids: tuple[str, ...] = field(default=(), repr=False)
    retry_members: tuple[str, ...] = field(default=(), repr=False)


class QdrantEraseStore:
    def __init__(self):
        self.url = required("QDRANT_URL").rstrip("/")
        self.memory_collection = os.environ.get("QDRANT_COLLECTION", "amul_memory")
        self.state_collection = os.environ.get(
            "QDRANT_TASK_STATE_COLLECTION", f"{self.memory_collection}_task_state_v2"
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
        result["must_not"] = [{"has_id": [self._settings_point_id(farmer_id), memory_erase_point_id(farmer_id)]}]
        return result

    @staticmethod
    def _state_filter(identity_key: str, subjects: tuple[str, ...]) -> dict[str, Any]:
        filters = [{"key": "farmer_key", "match": {"value": identity_key}}]
        if subjects:
            filters.append({"key": "subject_key", "match": {"any": list(subjects)}})
        return {"should": filters}

    def _count(self, collection: str, query_filter: dict[str, Any]) -> int:
        result = self._request(
            "POST", f"/collections/{collection}/points/count",
            {"filter": query_filter, "exact": True},
        )
        return int(result["result"]["count"])

    def _state_markers(self, query_filter: dict[str, Any]) -> tuple[str, ...]:
        task_keys: set[str] = set()
        offset: Any | None = None
        while True:
            body: dict[str, Any] = {
                "filter": query_filter,
                "limit": 100,
                "with_payload": ["task_key", "completed_task_keys"],
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
                task_keys.update(
                    key for key in payload.get("completed_task_keys", ()) if _HASH.fullmatch(key)
                )
            offset = page.get("next_page_offset")
            if offset is None:
                break
        return tuple(sorted(task_keys))

    def inventory(self, farmer_id: str, identity_key: str, subjects: tuple[str, ...]) -> QdrantInventory:
        state_exists = self._collection_exists(self.state_collection)
        state_filter = self._state_filter(identity_key, subjects)
        return QdrantInventory(
            memory_points=self._count(
                self.memory_collection, self._memory_filter(farmer_id)
            ),
            state_points=(
                self._count(self.state_collection, state_filter)
                if state_exists else 0
            ),
            task_keys=self._state_markers(state_filter) if state_exists else (),
        )

    def _erase_record(self, farmer_id: str) -> dict:
        path = f"/collections/{self.memory_collection}/points"
        points = self._request("POST", path, {
            "ids": [memory_erase_point_id(farmer_id)], "with_payload": True,
        })["result"]
        return points[0]["payload"] if points else {}

    def finish_erase(self, farmer_id: str) -> None:
        path = f"/collections/{self.memory_collection}/points/payload?wait=true"
        self._request("POST", path, {
            "points": [memory_erase_point_id(farmer_id)],
            "payload": {"erase_in_progress": False},
        })
        record = self._erase_record(farmer_id)
        if not record.get("erase_cutoff") or record.get("erase_in_progress") is not False:
            raise RuntimeError("erase cutoff was not retained")

    def disable(self, farmer_id: str) -> None:
        record = self._erase_record(farmer_id)
        if not record.get("erase_in_progress"):
            self._request("PUT", f"/collections/{self.memory_collection}/points?wait=true", {
                "points": [{
                    "id": memory_erase_point_id(farmer_id), "vector": {},
                    "payload": {"type": "memory_erase", "erase_in_progress": True,
                                "erase_cutoff": datetime.now(timezone.utc).isoformat()},
                }],
            })
        response = memory_client.put(
            f"{self.memory_api}/memory/{farmer_id}/settings",
            json={"memory_enabled": False, "note": "erasure_requested"},
            timeout=30,
        )
        response.raise_for_status()

    def settings_disabled(self, farmer_id: str) -> bool:
        points = self._request("POST", f"/collections/{self.memory_collection}/points", {
            "ids": [self._settings_point_id(farmer_id)], "with_payload": True,
        })["result"]
        return bool(points) and points[0]["payload"].get("memory_enabled") is False

    def delete_memory(self, farmer_id: str) -> None:
        self._request(
            "POST", f"/collections/{self.memory_collection}/points/delete?wait=true",
            {"filter": self._memory_filter(farmer_id)},
        )

    def delete_state(self, identity_key: str, subjects: tuple[str, ...]) -> None:
        if not self._collection_exists(self.state_collection):
            return
        self._request(
            "POST", f"/collections/{self.state_collection}/points/delete?wait=true",
            {"filter": self._state_filter(identity_key, subjects)},
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
        self.ledger = RedisEpisodeLedger(self.client)

    @staticmethod
    def _decoded_body(raw: Any) -> dict[str, Any] | None:
        try:
            body = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            return None
        return body if isinstance(body, dict) else None

    @staticmethod
    def _matches(raw: Any, farmer_id: str) -> tuple[bool, str | None]:
        body = RedisEraseStore._decoded_body(raw)
        if body is None or body.get("farmer_id") != farmer_id:
            return False, None
        try:
            task = MemoryTask.from_dict(body)
        except TaskValidationError:
            return True, None
        return True, task.task_key

    def _stream_rows(self, stream: str) -> Iterable[tuple[str, dict[str, str]]]:
        cursor = "-"
        while True:
            rows = self.client.xrange(stream, min=cursor, max="+", count=100)
            if not rows:
                return
            for receipt, fields in rows:
                yield str(receipt), dict(fields)
            cursor = f"({rows[-1][0]}"

    def episode_subjects(self, farmer_id: str) -> tuple[str, ...]:
        index = f"{self.ledger.farmer_index_prefix}:{farmer_key(farmer_id)}"
        subjects = []
        for key in self.client.smembers(index):
            record = self.ledger.load(key.removeprefix(f"{self.ledger.prefix}:"))
            if record:
                subjects.extend(
                    episode_subject_key(farmer_id, record.surface, record.session_id, episode.number)
                    for episode in record.episodes
                )
        return tuple(subjects)

    def inventory(
        self, farmer_id: str, *, task_keys: Iterable[str] = (),
    ) -> RedisInventory:
        matched_task_keys = {value for value in task_keys if _HASH.fullmatch(value)}
        identity = farmer_key(farmer_id)
        ledger_index = f"{self.ledger.farmer_index_prefix}:{identity}"
        ledger_keys = tuple(sorted(self.client.smembers(ledger_index)))
        stream_ids: list[str] = []
        dead_letter_ids: list[str] = []
        retry_members: list[str] = []

        for stream, target in (
            (self.stream, stream_ids), (self.dead_letter_stream, dead_letter_ids)
        ):
            for receipt, fields in self._stream_rows(stream):
                matches, task_key = self._matches(fields.get("body"), farmer_id)
                if not matches and fields.get("farmer_key") != identity:
                    continue
                target.append(receipt)
                outer_task_key = str(fields.get("task_key", ""))
                if _HASH.fullmatch(outer_task_key):
                    matched_task_keys.add(outer_task_key)
                if task_key:
                    matched_task_keys.add(task_key)

        for member, _score in self.client.zscan_iter(self.retry_set, count=100):
            item = self._decoded_body(member)
            if item is None:
                continue
            matches, task_key = self._matches(item.get("body"), farmer_id)
            if not matches:
                continue
            retry_members.append(member)
            outer_task_key = str(item.get("task_key", ""))
            if _HASH.fullmatch(outer_task_key):
                matched_task_keys.add(outer_task_key)
            if task_key:
                matched_task_keys.add(task_key)

        markers = set(self.client.scan_iter(match=f"{self.dedupe_prefix}:{identity}:*"))
        markers.update(f"{self.dedupe_prefix}:{value}" for value in matched_task_keys)
        return RedisInventory(
            stream_entries=len(stream_ids),
            dead_letters=len(dead_letter_ids),
            retries=len(retry_members),
            marker_keys=tuple(sorted(markers)),
            ledger_keys=ledger_keys,
            ledger_index=ledger_index,
            stream_ids=tuple(stream_ids),
            dead_letter_ids=tuple(dead_letter_ids),
            retry_members=tuple(retry_members),
        )

    def purge(self, inventory: RedisInventory) -> None:
        pipe = self.client.pipeline(transaction=True)
        for receipt in inventory.stream_ids:
            pipe.xack(self.stream, self.group, receipt)
            pipe.xdel(self.stream, receipt)
        for receipt in inventory.dead_letter_ids:
            pipe.xdel(self.dead_letter_stream, receipt)
        if inventory.retry_members:
            pipe.zrem(self.retry_set, *inventory.retry_members)
        if inventory.marker_keys:
            pipe.delete(*inventory.marker_keys)
        for key in inventory.ledger_keys:
            member = key.removeprefix(f"{self.ledger.prefix}:")
            pipe.zrem(self.ledger.due_set, member)
            pipe.zrem(self.ledger.reconcile_set, member)
        pipe.execute()
        for key in inventory.ledger_keys:
            member = key.removeprefix(f"{self.ledger.prefix}:")
            if (
                self.client.zscore(self.ledger.due_set, member) is not None
                or self.client.zscore(self.ledger.reconcile_set, member) is not None
            ):
                raise RuntimeError("episode ledger cleanup incomplete")

    def finish(self, inventory: RedisInventory) -> None:
        self.client.delete(*inventory.ledger_keys, inventory.ledger_index)
        if self.markers_exist((*inventory.ledger_keys, inventory.ledger_index)):
            raise RuntimeError("episode ledger cleanup incomplete")

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
        "episode_ledgers": len(redis.ledger_keys),
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
    if execute:
        try:
            qdrant.disable(identity)
            if not qdrant.settings_disabled(identity):
                raise RuntimeError("memory tombstone not verified")
        except Exception as exc:
            raise EraseFailure("disable_memory", exc) from exc
    try:
        subjects = redis.episode_subjects(identity)
        qdrant_plan = qdrant.inventory(identity, identity_key, subjects)
        redis_plan = redis.inventory(
            identity,
            task_keys=qdrant_plan.task_keys,
        )
    except Exception as exc:
        raise EraseFailure("inventory", exc) from exc
    if not execute:
        return _safe_result(identity_key, qdrant_plan, redis_plan, "dry_run")

    for stage, action in (
        ("purge_queue", lambda: redis.purge(redis_plan)),
        ("delete_memory", lambda: qdrant.delete_memory(identity)),
    ):
        try:
            action()
        except Exception as exc:
            raise EraseFailure(stage, exc) from exc

    try:
        remaining_queue = redis.inventory(identity)
        if (
            remaining_queue.stream_entries or remaining_queue.dead_letters or remaining_queue.retries
            or redis.markers_exist(redis_plan.marker_keys)
        ):
            raise RuntimeError("Redis cleanup incomplete")
        qdrant.delete_state(identity_key, subjects)
        if qdrant.inventory(identity, identity_key, subjects).state_points:
            raise RuntimeError("task state cleanup incomplete")
    except Exception as exc:
        raise EraseFailure("delete_task_state", exc) from exc

    try:
        redis.finish(redis_plan)
        remaining_qdrant = qdrant.inventory(identity, identity_key, subjects)
        remaining_redis = redis.inventory(identity)
        verified = (
            qdrant.settings_disabled(identity)
            and remaining_qdrant.memory_points == 0
            and remaining_qdrant.state_points == 0
            and remaining_redis.stream_entries == 0
            and remaining_redis.dead_letters == 0
            and remaining_redis.retries == 0
            and not remaining_redis.ledger_keys
            and not redis.markers_exist(remaining_redis.marker_keys)
        )
        if not verified:
            raise RuntimeError("erase verification failed")
    except Exception as exc:
        raise EraseFailure("verification", exc) from exc
    try:
        qdrant.finish_erase(identity)
    except Exception as exc:
        raise EraseFailure("finish_erase", exc) from exc
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
