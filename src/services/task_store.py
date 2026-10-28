"""In-memory thread-safe task store for tracking background ingestion jobs."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import threading
from typing import Any

from src.models.schemas import IngestTaskDetail, TaskStatus


class TaskStore:
    """Thread-safe in-memory store for async background tasks with LRU-like eviction."""

    def __init__(self, max_tasks: int = 500) -> None:
        self._max_tasks = max_tasks
        self._tasks: OrderedDict[str, IngestTaskDetail] = OrderedDict()
        self._lock = threading.Lock()

    def create_task(
        self,
        task_id: str,
        file_name: str,
        collection_name: str,
    ) -> IngestTaskDetail:
        """Register a new task with initial PENDING status."""
        now = datetime.now(timezone.utc)
        record = IngestTaskDetail(
            task_id=task_id,
            status=TaskStatus.PENDING,
            file_name=file_name,
            collection_name=collection_name,
            files_processed=None,
            chunks_created=None,
            failed_files=[],
            error=None,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            if len(self._tasks) >= self._max_tasks:
                # Evict the oldest completed/failed or oldest item
                self._tasks.popitem(last=False)
            self._tasks[task_id] = record

        return record

    def get_task(self, task_id: str) -> IngestTaskDetail | None:
        """Retrieve task details by task_id."""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                return record.model_copy()
            return None

    def update_task(
        self,
        task_id: str,
        status: TaskStatus,
        files_processed: int | None = None,
        chunks_created: int | None = None,
        failed_files: list[str] | None = None,
        error: str | None = None,
    ) -> IngestTaskDetail | None:
        """Update an existing task's status and results."""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None

            now = datetime.now(timezone.utc)
            updated = record.model_copy(
                update={
                    "status": status,
                    "files_processed": (
                        files_processed if files_processed is not None else record.files_processed
                    ),
                    "chunks_created": (
                        chunks_created if chunks_created is not None else record.chunks_created
                    ),
                    "failed_files": (
                        failed_files if failed_files is not None else record.failed_files
                    ),
                    "error": error if error is not None else record.error,
                    "updated_at": now,
                }
            )
            self._tasks[task_id] = updated
            return updated.model_copy()

    def clear(self) -> None:
        """Clear all tasks (primarily for unit tests)."""
        with self._lock:
            self._tasks.clear()


# Global singleton instance
task_store = TaskStore()

