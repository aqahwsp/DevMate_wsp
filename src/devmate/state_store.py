"""Persistent runtime state storage for Web and CLI sessions."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from devmate.config import AppConfig

LOGGER = logging.getLogger(__name__)
_ACTIVE_STATUSES = {"queued", "running"}
_INTERRUPTED_STATUSES = {"interrupted", "abandoned"}


class RunStateStore:
    """Persist run state so Web and CLI sessions can be resumed."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.root = config.state_dir
        self.web_root = self.root / "web_jobs"
        self.cli_root = self.root / "cli_runs"

    def ensure_directories(self) -> None:
        """Create all runtime state directories."""

        for path in (self.root, self.web_root, self.cli_root):
            path.mkdir(parents=True, exist_ok=True)

    def _scope_root(self, scope: str) -> Path:
        if scope == "web":
            return self.web_root
        if scope == "cli":
            return self.cli_root
        raise ValueError(f"Unsupported state scope: {scope}")

    def state_path(self, scope: str, run_id: str) -> Path:
        """Return the JSON state file path for a run."""

        return self._scope_root(scope) / f"{run_id}.json"

    def control_path(self, scope: str, run_id: str) -> Path:
        """Return the control file path for a run."""

        return self._scope_root(scope) / f"{run_id}.control.json"

    def log_path(self, scope: str, run_id: str) -> Path:
        """Return the log file path for a detached worker."""

        return self._scope_root(scope) / f"{run_id}.log"

    def write_state(
        self,
        scope: str,
        run_id: str,
        payload: dict[str, Any],
    ) -> Path:
        """Write a run-state JSON file atomically."""

        self.ensure_directories()
        path = self.state_path(scope, run_id)
        temp_path = path.with_suffix(".json.tmp")
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        temp_path.write_text(
            serialized,
            encoding="utf-8",
        )
        temp_path.replace(path)
        return path

    def read_state(self, scope: str, run_id: str) -> dict[str, Any] | None:
        """Read a persisted run-state JSON file if it exists."""

        path = self.state_path(scope, run_id)
        if not path.exists() or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Unable to read runtime state from %s: %s", path, exc)
            return None
        if not isinstance(payload, dict):
            return None
        return self._normalize_loaded_record(scope, payload)

    def list_states(self, scope: str) -> list[dict[str, Any]]:
        """Return all persisted states for a given scope."""

        root = self._scope_root(scope)
        if not root.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(root.glob("*.json")):
            if path.name.endswith(".control.json"):
                continue
            record = self.read_state(scope, path.stem)
            if record is not None:
                records.append(record)
        return records

    def latest_state(
        self,
        scope: str,
        *,
        statuses: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return the newest persisted run state for the given scope."""

        candidates = self.list_states(scope)
        if statuses is not None:
            candidates = [
                item for item in candidates if str(item.get("status", "")) in statuses
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: float(item.get("updated_at", 0.0)))

    def write_control(
        self,
        scope: str,
        run_id: str,
        payload: dict[str, Any],
    ) -> Path:
        """Write a control file for a running worker."""

        self.ensure_directories()
        path = self.control_path(scope, run_id)
        temp_path = path.with_suffix(".json.tmp")
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        temp_path.write_text(
            serialized,
            encoding="utf-8",
        )
        temp_path.replace(path)
        return path

    def read_control(self, scope: str, run_id: str) -> dict[str, Any] | None:
        """Read a persisted control file if available."""

        path = self.control_path(scope, run_id)
        if not path.exists() or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Unable to read control file from %s: %s", path, exc)
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def consume_control(self, scope: str, run_id: str) -> dict[str, Any] | None:
        """Read and clear a control file in one logical operation."""

        payload = self.read_control(scope, run_id)
        if payload is not None:
            self.clear_control(scope, run_id)
        return payload

    def clear_control(self, scope: str, run_id: str) -> None:
        """Delete a control file after it has been consumed."""

        self._delete_path(
            self.control_path(scope, run_id),
            description="control file",
        )

    def delete_state(self, scope: str, run_id: str) -> None:
        """Delete a persisted run-state JSON file if present."""

        self._delete_path(
            self.state_path(scope, run_id),
            description="state file",
        )

    def clear_scope(
        self,
        scope: str,
        *,
        preserve_run_id: str | None = None,
        include_controls: bool = True,
        include_logs: bool = False,
    ) -> None:
        """Delete persisted snapshots for a scope before a fresh run starts."""

        root = self._scope_root(scope)
        if not root.exists():
            return

        protected_names: set[str] = set()
        if preserve_run_id:
            protected_names.add(f"{preserve_run_id}.json")
            if include_controls:
                protected_names.add(f"{preserve_run_id}.control.json")
            if include_logs:
                protected_names.add(f"{preserve_run_id}.log")

        for path in sorted(root.iterdir()):
            if not path.is_file() or path.name in protected_names:
                continue
            if path.name.endswith(".control.json"):
                if include_controls:
                    self._delete_path(path, description="control file")
                continue
            if path.suffix == ".json":
                self._delete_path(path, description="state file")
                continue
            if include_logs and path.suffix == ".log":
                self._delete_path(path, description="log file")

    def _delete_path(self, path: Path, *, description: str) -> None:
        """Remove a persisted runtime artifact if it exists."""

        if not path.exists():
            return
        try:
            path.unlink()
        except OSError as exc:
            LOGGER.debug("Unable to delete %s %s: %s", description, path, exc)

    def _normalize_loaded_record(
        self,
        scope: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize persisted states after a process restart."""

        status = str(payload.get("status", ""))
        if status not in _ACTIVE_STATUSES:
            return payload

        record_id = str(payload.get("run_id", payload.get("job_id", "")))
        if not record_id:
            return payload

        if scope != "cli":
            return payload

        worker_pid = payload.get("worker_pid")
        if isinstance(worker_pid, int) and self._is_process_alive(worker_pid):
            return payload

        normalized = dict(payload)
        normalized["status"] = "interrupted"
        normalized["stage"] = "CLI 会话已恢复，但后台 worker 不再运行。"
        normalized["error"] = normalized.get("error") or (
            "上一次 CLI 会话已中断，请重新执行或查看最近产物。"
        )
        self.write_state(scope, record_id, normalized)
        return normalized

    def _is_process_alive(self, pid: int) -> bool:
        """Return whether a process is alive."""

        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def list_active_states(self, scope: str) -> list[dict[str, Any]]:
        """Return persisted active run states."""

        return [
            item
            for item in self.list_states(scope)
            if str(item.get("status", "")) in _ACTIVE_STATUSES
        ]

    def latest_active_state(self, scope: str) -> dict[str, Any] | None:
        """Return the newest active run state for a scope."""

        return self.latest_state(scope, statuses=_ACTIVE_STATUSES)

    def latest_resumable_state(self, scope: str) -> dict[str, Any] | None:
        """Return the newest state that can be resumed or inspected."""

        return self.latest_state(
            scope,
            statuses=(
                _ACTIVE_STATUSES | _INTERRUPTED_STATUSES | {"completed", "error"}
            ),
        )
