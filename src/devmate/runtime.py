"""Core runtime that assembles the DevMate agent."""

from __future__ import annotations

import asyncio
import filecmp
import inspect
import json
import logging
import mimetypes
import os
import re
import shutil
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from threading import Event, Lock
from typing import Literal
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from devmate.config import AppConfig, load_config
from devmate.llm_factory import build_chat_model
from devmate.local_python_tool import execute_local_python
from devmate.observability import configure_langsmith
from devmate.prompts import SYSTEM_PROMPT
from devmate.rag import KnowledgeBase
from devmate.skills import SkillManager
from devmate.state_store import RunStateStore
from devmate.workflow import (
    StageExecutionResult,
    VerificationReport,
    WorkflowArtifacts,
    WorkflowOptions,
    build_builder_prompt,
    build_packager_prompt,
    build_planner_prompt,
    build_researcher_prompt,
    build_verifier_pytest_prompt,
    build_verifier_text_review_prompt,
    parse_verifier_report,
)

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[dict[str, object]], Awaitable[None] | None]

_VERIFIER_MAX_TOTAL_CHARS = 120_000
_VERIFIER_MAX_FILE_CHARS = 16_000
_VERIFIER_MAX_FILES = 80
_BUILDER_REVIEW_MAX_TOTAL_CHARS = 48_000
_BUILDER_REVIEW_MAX_FILE_CHARS = 8_000
_BUILDER_REVIEW_MAX_FILES = 40
_INITIAL_FILE_MAX_TOTAL_CHARS = 48_000
_INITIAL_FILE_MAX_FILE_CHARS = 12_000
_INITIAL_FILE_MAX_FILES = 20
_INITIAL_FILE_MAX_READ_BYTES = 256_000
_READONLY_RUNTIME_MAX_CHARS = 12_000
_READONLY_RUNTIME_MAX_READ_BYTES = 256_000
_READONLY_RUNTIME_MAX_LINES = 240
_READONLY_RUNTIME_MAX_LIST_ENTRIES = 200
_LLM_RETRY_DELAY_SECONDS = 30
_LLM_MAX_RETRIES_AFTER_FAILURE = 3
_WRITE_EXTERNAL_FILE_MAX_CHARS = 256_000

_PLANNER_DECISION_PATTERN = re.compile(
    r"PLANNER_DECISION\s*:\s*(APPROVE|REJECT)",
    flags=re.IGNORECASE,
)
_INTERNAL_WORKSPACE_DIR_NAMES = {
    ".devmate_runtime",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
_TEXT_LIKE_UPLOAD_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".css",
    ".csv",
    ".env",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".log",
    ".md",
    ".mjs",
    ".py",
    ".rb",
    ".rs",
    ".scss",
    ".sh",
    ".sql",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(slots=True, frozen=True)
class WorkspaceFileFingerprint:
    """Stable file fingerprint for workspace change detection."""

    size: int
    mtime_ns: int


@dataclass(slots=True, frozen=True)
class WorkspaceChange:
    """Structured description of a single workspace file operation."""

    path: str
    action: str


@dataclass(slots=True, frozen=True)
class StageReplySample:
    """Normalized stage reply used for loop-detection history."""

    stage_name: str
    reply: str


class RunController:
    """Thread-safe run controls shared by CLI, web UI, and runtime."""

    def __init__(
        self,
        external_request_reader: Callable[[], dict[str, object] | None] | None = None,
    ) -> None:
        self._immediate_output_requested = Event()
        self._lock = Lock()
        self._reason = ""
        self._external_request_reader = external_request_reader

    def bind_external_request_reader(
        self,
        reader: Callable[[], dict[str, object] | None] | None,
    ) -> None:
        """Bind an external control reader used by detached workers."""

        with self._lock:
            self._external_request_reader = reader

    def _refresh_external_requests(self) -> None:
        """Import an immediate-output request from an external control file."""

        with self._lock:
            reader = self._external_request_reader
        if reader is None or self._immediate_output_requested.is_set():
            return
        try:
            payload = reader()
        except Exception as exc:  # pragma: no cover - defensive safeguard
            LOGGER.debug("Failed to read external run control: %s", exc)
            return
        if not payload or not bool(payload.get("immediate_output_requested", False)):
            return
        self.request_immediate_output(str(payload.get("reason", "")))

    def request_immediate_output(self, reason: str = "") -> bool:
        """Request that the workflow jump to the Packager stage."""

        with self._lock:
            was_already_requested = self._immediate_output_requested.is_set()
            if reason and not self._reason:
                self._reason = reason
            self._immediate_output_requested.set()
            return not was_already_requested

    @property
    def immediate_output_requested(self) -> bool:
        """Return whether finalization has been requested."""

        self._refresh_external_requests()
        return self._immediate_output_requested.is_set()

    @property
    def reason(self) -> str:
        """Return the first recorded finalization reason."""

        self._refresh_external_requests()
        with self._lock:
            return self._reason


class StageInterruptedError(RuntimeError):
    """Raised when a stage is interrupted to jump to final packaging."""

    def __init__(self, stage_name: str, partial_reply: str = "") -> None:
        super().__init__(f"Stage interrupted: {stage_name}")
        self.stage_name = stage_name
        self.partial_reply = partial_reply


class StageLoopDetectedError(RuntimeError):
    """Raised when the runtime detects a repeated-output loop."""

    def __init__(self, stage_name: str, partial_reply: str = "") -> None:
        super().__init__(f"Loop detected in stage: {stage_name}")
        self.stage_name = stage_name
        self.partial_reply = partial_reply


@dataclass(slots=True)
class WorkspaceTracker:
    """Live tracker for workspace file operations during a run."""

    runtime: DevMateRuntime
    snapshot: dict[str, WorkspaceFileFingerprint]
    file_operations: list[str] = field(default_factory=list)
    changed_paths: set[str] = field(default_factory=set)

    @property
    def changed_files(self) -> list[str]:
        """Return the sorted set of changed workspace paths."""

        return sorted(self.changed_paths)

    async def scan_and_emit(
        self,
        *,
        stage_name: str,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Detect new workspace changes and emit live progress updates."""

        after = self.runtime.snapshot_workspace()
        changes = self.runtime.diff_workspace_changes(self.snapshot, after)
        self.snapshot = after
        if not changes:
            return

        for change in changes:
            self.changed_paths.add(change.path)
            self.file_operations.append(
                self.runtime.format_workspace_change(change, stage_name)
            )

        await self.runtime._emit_progress(
            progress_callback,
            event_type="workspace_changes",
            message=f"检测到 {len(changes)} 项文件变更。",
            current_phase=stage_name,
            file_operations=list(self.file_operations),
            changed_files=self.changed_files,
            output_files=sorted(after),
        )


@dataclass(slots=True)
class AgentRunResult:
    """Structured result from a single DevMate execution."""

    prompt: str
    reply: str
    changed_files: list[str]
    output_files: list[str] = field(default_factory=list)
    file_operations: list[str] = field(default_factory=list)
    saved_skill: str | None = None
    delivery_zip: str | None = None
    verification_passed: bool | None = None
    optimization_rounds_used: int = 0
    deep_optimization: bool = False
    local_python_tool_enabled: bool = False
    max_deep_optimization_rounds: int = 0
    immediate_output_requested: bool = False


class DevMateRuntime:
    """Build and run the DevMate agent."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.knowledge_base = KnowledgeBase(config)
        self.skill_manager = SkillManager(config)
        self.state_store = RunStateStore(config)
        self._mcp_tools_cache: tuple[object, ...] | None = None

    @classmethod
    def from_path(cls, config_path: str | Path | None = None) -> DevMateRuntime:
        """Create a runtime from a TOML configuration file."""

        return cls(load_config(config_path=config_path))

    async def prepare(self, rebuild_kb: bool = False) -> None:
        """Ensure workspace, skills, artifacts, and vector store are ready."""

        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._sync_project_context_into_workspace()
        self.config.skills_dir.mkdir(parents=True, exist_ok=True)
        self.config.docs_dir.mkdir(parents=True, exist_ok=True)
        self.config.research_cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.workflow_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir("cli").mkdir(parents=True, exist_ok=True)
        self.upload_dir("web").mkdir(parents=True, exist_ok=True)
        self.state_store.ensure_directories()

        if rebuild_kb or not self._knowledge_base_exists():
            self.knowledge_base.ingest(rebuild=True)

    def _knowledge_base_exists(self) -> bool:
        """Check whether the Chroma persistence directory already has content."""

        if not self.config.persist_dir.exists():
            return False
        return any(self.config.persist_dir.rglob("*"))

    def uploads_root(self) -> Path:
        """Return the managed uploads root inside the workspace."""

        return self.config.resolve_workspace_output_path("data/uploads")

    def upload_dir(self, scope: str) -> Path:
        """Return the staged-upload directory for a specific scope."""

        scope_name = scope.strip().lower()
        if scope_name not in {"cli", "web"}:
            raise ValueError(f"Unsupported upload scope: {scope}")
        return (self.uploads_root() / scope_name).resolve()

    def _ensure_within_root(self, root: Path, candidate: Path) -> None:
        """Validate that a candidate path stays inside the allowed root."""

        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
        if (
            candidate_resolved != root_resolved
            and root_resolved not in candidate_resolved.parents
        ):
            raise ValueError(f"Path is outside the allowed root: {candidate}")

    def _path_is_within_root(self, root: Path, candidate: Path) -> bool:
        """Return whether a candidate path resolves under the given root."""

        try:
            self._ensure_within_root(root, candidate)
        except ValueError:
            return False
        return True

    def _read_file_bytes_with_limit(
        self,
        path: Path,
        *,
        max_bytes: int,
    ) -> tuple[bytes, bool]:
        """Read a file with a hard byte cap and report truncation."""

        with path.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
        truncated = len(payload) > max_bytes
        if truncated:
            payload = payload[:max_bytes]
        return payload, truncated

    def _deduplicate_upload_path(self, destination: Path) -> Path:
        """Avoid clobbering an existing staged upload."""

        if not destination.exists():
            return destination

        stem = destination.stem
        suffix = destination.suffix
        counter = 2
        while True:
            candidate = destination.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def copy_initial_file_to_uploads(self, source: str | Path, scope: str) -> str:
        """Copy a local file into the staged uploads directory."""

        source_path = Path(source).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Initial file does not exist: {source_path}")

        destination_dir = self.upload_dir(scope)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = self._deduplicate_upload_path(destination_dir / source_path.name)
        shutil.copy2(source_path, destination)
        return destination.relative_to(self.config.project_root).as_posix()

    def save_uploaded_file(self, filename: str, content: bytes, scope: str) -> str:
        """Persist raw uploaded bytes into the staged uploads directory."""

        safe_name = Path(filename or "upload.bin").name
        destination_dir = self.upload_dir(scope)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = self._deduplicate_upload_path(destination_dir / safe_name)
        destination.write_bytes(content)
        return destination.relative_to(self.config.project_root).as_posix()

    def list_initial_upload_files(self, scope: str) -> list[str]:
        """List staged upload files for a scope."""

        root = self.upload_dir(scope)
        root.mkdir(parents=True, exist_ok=True)
        return sorted(
            path.relative_to(self.config.project_root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )

    def describe_initial_upload_files(self, scope: str) -> list[dict[str, object]]:
        """Return structured metadata for staged upload files."""

        entries: list[dict[str, object]] = []
        for relative_path in self.list_initial_upload_files(scope):
            absolute_path = (self.config.project_root / relative_path).resolve()
            entries.append(
                {
                    "path": relative_path,
                    "name": absolute_path.name,
                    "size_bytes": absolute_path.stat().st_size,
                }
            )
        return entries

    def _normalize_runtime_root_name(self, root_name: str) -> str:
        """Normalize root aliases used by runtime file tools."""

        normalized = root_name.strip().lower()
        aliases = {
            "artifact": "workflow_artifacts",
            "artifacts": "workflow_artifacts",
            "doc": "docs",
            "upload": "uploads",
        }
        return aliases.get(normalized, normalized)

    def _runtime_read_roots(self) -> dict[str, Path]:
        """Return approved read-only roots available to the agent."""

        return {
            "project": self.config.project_root,
            "docs": self.config.docs_dir,
            "skills": self.config.skills_dir,
            "uploads": self.uploads_root(),
            "uploads_cli": self.upload_dir("cli"),
            "uploads_web": self.upload_dir("web"),
            "workflow_artifacts": self.config.workflow_artifacts_dir,
        }

    def _resolve_runtime_read_root(self, root_name: str) -> Path:
        """Resolve a named read-only root used by runtime file tools."""

        normalized = self._normalize_runtime_root_name(root_name) or "project"
        roots = self._runtime_read_roots()
        if normalized not in roots:
            supported = ", ".join(sorted(roots))
            raise ValueError(
                f"Unsupported runtime read root: {root_name}. "
                f"Supported roots: {supported}"
            )
        return roots[normalized].resolve()

    def _resolve_runtime_read_path(self, root_name: str, relative_path: str) -> Path:
        """Resolve a read-only file path under an approved runtime root."""

        root = self._resolve_runtime_read_root(root_name)
        raw_value = (relative_path or ".").strip() or "."
        candidate = Path(raw_value)
        resolved = (
            candidate.expanduser().resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )
        self._ensure_within_root(root, resolved)
        return resolved

    def _display_path_from_runtime_root(self, root_name: str, path: Path) -> str:
        """Return a stable display path for a runtime read-only file."""

        root = self._resolve_runtime_read_root(root_name)
        resolved = path.resolve()
        if resolved == root:
            return "."
        return resolved.relative_to(root).as_posix()

    def _runtime_write_roots(self) -> dict[str, Path]:
        """Return approved managed output roots inside the workspace."""

        roots = {
            "docs": self.config.docs_dir,
            "skills": self.config.skills_dir,
            "uploads": self.uploads_root(),
            "uploads_cli": self.upload_dir("cli"),
            "uploads_web": self.upload_dir("web"),
            "workflow_artifacts": self.config.workflow_artifacts_dir,
        }
        for root in roots.values():
            self._ensure_within_root(self.config.workspace_dir, root)
        return roots

    def _resolve_runtime_write_root(self, root_name: str) -> Path:
        """Resolve a named writable root used by the external write tool."""

        normalized = self._normalize_runtime_root_name(root_name)
        roots = self._runtime_write_roots()
        if normalized not in roots:
            supported = ", ".join(sorted(roots))
            raise ValueError(
                f"Unsupported runtime write root: {root_name}. "
                f"Supported roots: {supported}"
            )
        return roots[normalized].resolve()

    def _resolve_runtime_write_path(self, root_name: str, relative_path: str) -> Path:
        """Resolve a writable file path under an approved external root."""

        root = self._resolve_runtime_write_root(root_name)
        raw_value = (relative_path or "").strip()
        if not raw_value or raw_value == ".":
            raise ValueError("relative_path must point to a file path")

        candidate = Path(raw_value)
        resolved = (
            candidate.expanduser().resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )
        self._ensure_within_root(root, resolved)
        if resolved == root:
            raise ValueError("relative_path must point to a file path")
        return resolved

    def write_external_file(
        self,
        *,
        root_name: str,
        relative_path: str,
        content: str,
        mode: str = "overwrite",
    ) -> str:
        """Write UTF-8 text to an approved managed workspace output root."""

        normalized_mode = mode.strip().lower() or "overwrite"
        if normalized_mode not in {"append", "fail_if_exists", "overwrite"}:
            raise ValueError(
                "Unsupported write mode. Use overwrite, append, or fail_if_exists."
            )

        text = str(content)
        if len(text) > _WRITE_EXTERNAL_FILE_MAX_CHARS:
            raise ValueError(
                "External file content is too large for a single tool call: "
                f"{len(text)} > {_WRITE_EXTERNAL_FILE_MAX_CHARS}"
            )

        target = self._resolve_runtime_write_path(root_name, relative_path)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"External write target is a directory: {target}")

        target.parent.mkdir(parents=True, exist_ok=True)
        if normalized_mode == "append":
            with target.open("a", encoding="utf-8") as handle:
                handle.write(text)
        elif normalized_mode == "fail_if_exists":
            if target.exists():
                raise FileExistsError(f"External write target already exists: {target}")
            target.write_text(text, encoding="utf-8")
        else:
            target.write_text(text, encoding="utf-8")

        self._sync_runtime_writeback()
        return target.relative_to(self.config.project_root).as_posix()

    def _skills_dir_is_workspace_local(self) -> bool:
        """Return whether skill writes stay within the workspace boundary."""

        return self._path_is_within_root(
            self.config.workspace_dir,
            self.config.skills_dir,
        )

    def _agent_skill_paths(self) -> list[str]:
        """Return skill search paths anchored to the workspace root."""

        skill_root = self.config.skills_dir.resolve()
        if self._path_is_within_root(self.config.workspace_dir, skill_root):
            relative_path = skill_root.relative_to(self.config.workspace_dir).as_posix()
            if relative_path in {"", "."}:
                return ["."]
            return [f"./{relative_path}"]
        return [str(skill_root)]

    def list_runtime_files(
        self,
        *,
        root_name: str = "project",
        relative_path: str = ".",
        max_entries: int = _READONLY_RUNTIME_MAX_LIST_ENTRIES,
    ) -> str:
        """List files from an approved read-only runtime root."""

        target = self._resolve_runtime_read_path(root_name, relative_path)
        if not target.exists():
            raise FileNotFoundError(f"Runtime path does not exist: {relative_path}")

        safe_limit = max(1, min(int(max_entries), _READONLY_RUNTIME_MAX_LIST_ENTRIES))
        lines = [
            f"ROOT: {root_name}",
            f"TARGET: {self._display_path_from_runtime_root(root_name, target)}",
        ]

        if target.is_file():
            lines.append(
                f"[FILE] {self._display_path_from_runtime_root(root_name, target)}"
            )
            return "\n".join(lines)

        entries = sorted(
            target.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        )
        for entry in entries[:safe_limit]:
            label = "DIR" if entry.is_dir() else "FILE"
            lines.append(
                f"[{label}] {self._display_path_from_runtime_root(root_name, entry)}"
            )

        omitted = len(entries) - safe_limit
        if omitted > 0:
            lines.append(f"[... omitted {omitted} additional entries ...]")
        return "\n".join(lines)

    def read_runtime_file(
        self,
        *,
        root_name: str = "project",
        relative_path: str,
        start_line: int = 1,
        max_lines: int = _READONLY_RUNTIME_MAX_LINES,
    ) -> str:
        """Read text from an approved read-only runtime root."""

        target = self._resolve_runtime_read_path(root_name, relative_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Runtime file does not exist: {relative_path}")

        payload, truncated_by_bytes = self._read_file_bytes_with_limit(
            target,
            max_bytes=_READONLY_RUNTIME_MAX_READ_BYTES,
        )
        mime_type, _ = mimetypes.guess_type(target.name)
        if self._looks_binary_upload(target, payload, mime_type):
            display_path = self._display_path_from_runtime_root(root_name, target)
            return f"[binary file omitted from runtime read tool: {display_path}]"

        text = payload.decode("utf-8", errors="replace")
        lines = text.splitlines()
        safe_start_line = max(1, int(start_line))
        safe_max_lines = max(1, min(int(max_lines), _READONLY_RUNTIME_MAX_LINES))
        start_index = safe_start_line - 1
        end_index = min(len(lines), start_index + safe_max_lines)
        if lines and start_index < len(lines):
            selected_text = "\n".join(lines[start_index:end_index])
        elif lines:
            selected_text = "[requested line range starts after the end of the file]"
        else:
            selected_text = ""

        display_path = self._display_path_from_runtime_root(root_name, target)
        selected_text = self._trim_for_prompt(
            selected_text,
            max_chars=_READONLY_RUNTIME_MAX_CHARS,
            label=f"{display_path} runtime content",
        )
        line_end = max(safe_start_line, end_index)
        result = (
            f"ROOT: {root_name}\n"
            f"FILE: {display_path}\n"
            f"LINES: {safe_start_line}-{line_end}\n\n"
            f"{selected_text}"
        )
        if truncated_by_bytes or end_index < len(lines):
            result = (
                f"{result}\n\n[... output truncated by line or byte safety limits ...]"
            )
        return result

    def _resolve_initial_file_candidate(self, relative_path: str) -> Path:
        """Resolve a staged initial-file path under the approved upload roots."""

        candidate = (self.config.project_root / relative_path).resolve()
        allowed_roots = [self.upload_dir("cli"), self.upload_dir("web")]
        for root in allowed_roots:
            try:
                self._ensure_within_root(root, candidate)
            except ValueError:
                continue
            if not candidate.exists() or not candidate.is_file():
                raise FileNotFoundError(f"Initial file not found: {relative_path}")
            return candidate
        raise ValueError(
            f"Initial file is outside approved upload roots: {relative_path}"
        )

    def _read_initial_file_bytes(self, path: Path) -> tuple[bytes, bool]:
        """Read an initial file with a hard byte cap."""

        with path.open("rb") as handle:
            payload = handle.read(_INITIAL_FILE_MAX_READ_BYTES + 1)
        truncated = len(payload) > _INITIAL_FILE_MAX_READ_BYTES
        if truncated:
            payload = payload[:_INITIAL_FILE_MAX_READ_BYTES]
        return payload, truncated

    def _looks_binary_upload(
        self,
        path: Path,
        payload: bytes,
        mime_type: str | None,
    ) -> bool:
        """Best-effort binary detection for staged initial files."""

        if path.suffix.lower() in _TEXT_LIKE_UPLOAD_EXTENSIONS:
            return False
        if mime_type and mime_type.startswith("text/"):
            return False
        if mime_type in {
            "application/json",
            "application/javascript",
            "application/xml",
            "application/x-yaml",
            "application/toml",
        }:
            return False
        if b"\x00" in payload:
            return True
        if not payload:
            return False

        sample = payload[:4096]
        non_text = sum(
            1 for byte in sample if byte < 9 or (13 < byte < 32 and byte not in {27})
        )
        return non_text / max(1, len(sample)) > 0.12

    def _extract_text_from_pdf_bytes(self, payload: bytes) -> str | None:
        """Best-effort PDF text extraction for staged initial files."""

        try:
            from pypdf import PdfReader
        except Exception:  # pragma: no cover - optional dependency
            return None

        try:
            import io

            reader = PdfReader(io.BytesIO(payload))
            fragments = [page.extract_text() or "" for page in reader.pages]
        except Exception:  # pragma: no cover - extraction fallback
            return None

        text = "\n\n".join(fragment.strip() for fragment in fragments if fragment)
        return text or None

    def _extract_initial_file_text(self, path: Path) -> str:
        """Extract prompt-safe text from a staged initial file."""

        try:
            payload, truncated_by_bytes = self._read_initial_file_bytes(path)
        except OSError as exc:
            return f"[unable to read initial file: {exc}]"

        mime_type, _ = mimetypes.guess_type(path.name)
        suffix = path.suffix.lower()
        text: str | None = None
        if suffix == ".pdf":
            text = self._extract_text_from_pdf_bytes(payload)
            if text is None:
                return (
                    "[binary file skipped: PDF text extraction unavailable or failed]"
                )
        elif self._looks_binary_upload(path, payload, mime_type):
            return "[binary file skipped from initial file context]"
        else:
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                text = payload.decode("utf-8", errors="replace")

        text = self._trim_for_prompt(
            text,
            max_chars=_INITIAL_FILE_MAX_FILE_CHARS,
            label=f"{path.name} initial file content",
        )
        if truncated_by_bytes:
            text = (
                f"{text}\n\n"
                f"[... {path.name} was truncated to the first "
                f"{_INITIAL_FILE_MAX_READ_BYTES} bytes before extraction ...]"
            )
        return text

    def _build_initial_file_context(self, initial_file_paths: list[str]) -> str:
        """Serialize staged initial files into prompt context blocks."""

        if not initial_file_paths:
            return ""

        parts: list[str] = [
            "The following staged initial files were supplied before planning. ",
            "Treat them as additional user context and requirements when relevant.",
            "",
        ]
        remaining_budget = _INITIAL_FILE_MAX_TOTAL_CHARS
        processed_files = 0

        for relative_path in initial_file_paths:
            if processed_files >= _INITIAL_FILE_MAX_FILES:
                omitted = len(initial_file_paths) - processed_files
                parts.append(
                    f"[... omitted {omitted} additional initial files due to limit ...]"
                )
                break

            try:
                absolute_path = self._resolve_initial_file_candidate(relative_path)
            except (FileNotFoundError, ValueError) as exc:
                content = f"[initial file unavailable: {exc}]"
                display_path = relative_path
            else:
                content = self._extract_initial_file_text(absolute_path)
                display_path = absolute_path.relative_to(
                    self.config.project_root
                ).as_posix()

            block = (
                f"<<<INITIAL FILE: {display_path}>>>\n"
                f"{content}\n"
                f"<<<END INITIAL FILE: {display_path}>>>\n"
            )
            if len(block) > remaining_budget:
                if remaining_budget <= 256:
                    omitted = len(initial_file_paths) - processed_files
                    parts.append(
                        "[... omitted "
                        f"{omitted} additional initial files due to prompt "
                        "budget ...]"
                    )
                    break
                parts.append(
                    self._trim_for_prompt(
                        block,
                        max_chars=remaining_budget,
                        label="initial file context",
                    )
                )
                break

            parts.append(block)
            remaining_budget -= len(block)
            processed_files += 1

        return "\n".join(part for part in parts if part)

    def _compose_prompt_with_initial_files(
        self,
        prompt: str,
        *,
        include_initial_files: bool,
        initial_file_paths: list[str] | None,
    ) -> tuple[str, list[str], str]:
        """Merge staged initial-file context into the user request."""

        normalized_paths = [path for path in (initial_file_paths or []) if path]
        if not include_initial_files or not normalized_paths:
            return prompt, normalized_paths, ""

        context_text = self._build_initial_file_context(normalized_paths)
        if not context_text:
            return prompt, normalized_paths, ""

        merged_prompt = (
            f"{prompt}\n\n"
            "===== INITIAL FILE CONTEXT =====\n"
            f"{context_text}\n"
            "===== END INITIAL FILE CONTEXT ====="
        )
        return merged_prompt, normalized_paths, context_text

    def _planner_request_rejected(self, planner_text: str) -> bool:
        """Return whether the Planner explicitly rejected the request."""

        match = _PLANNER_DECISION_PATTERN.search(planner_text or "")
        if match is None:
            return False
        return match.group(1).strip().upper() == "REJECT"

    async def _emit_progress(
        self,
        callback: ProgressCallback | None,
        *,
        event_type: str,
        message: str = "",
        **extra: object,
    ) -> None:
        """Dispatch a progress event when a callback is available."""

        if callback is None:
            return
        payload: dict[str, object] = {"type": event_type, "message": message, **extra}
        maybe_awaitable = callback(payload)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

    async def _load_mcp_tools(self) -> list:
        """Load MCP tools from the configured Streamable HTTP endpoint."""

        if self._mcp_tools_cache is not None:
            return list(self._mcp_tools_cache)

        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                client = MultiServerMCPClient(
                    {
                        "search": {
                            "transport": "http",
                            "url": self.config.search.mcp_url,
                        }
                    }
                )
                tools = await client.get_tools()
                self._mcp_tools_cache = tuple(tools)
                LOGGER.info("Loaded %s MCP tool(s)", len(tools))
                return list(self._mcp_tools_cache)
            except Exception as exc:  # pragma: no cover - defensive logging
                last_error = exc
                LOGGER.warning(
                    "Failed to connect to MCP search server on attempt %s: %s",
                    attempt,
                    exc,
                )
                await asyncio.sleep(0.5)
        raise RuntimeError(
            "Unable to load MCP tools from the search server"
        ) from last_error

    def _build_kb_tool(self):
        """Create the local knowledge-base search tool."""

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the local knowledge base for internal guidelines and templates."""

            return self.knowledge_base.format_search_results(query)

        return search_knowledge_base

    def _build_runtime_file_list_tool(self):
        """Create the read-only runtime file listing tool."""

        @tool
        def list_runtime_files(
            root_name: Literal[
                "project",
                "docs",
                "skills",
                "uploads",
                "uploads_cli",
                "uploads_web",
                "workflow_artifacts",
            ] = "project",
            relative_path: str = ".",
            max_entries: int = _READONLY_RUNTIME_MAX_LIST_ENTRIES,
        ) -> str:
            """List approved read-only runtime files when extra context is needed."""

            return self.list_runtime_files(
                root_name=root_name,
                relative_path=relative_path,
                max_entries=max_entries,
            )

        return list_runtime_files

    def _build_runtime_file_read_tool(self):
        """Create the read-only runtime file content tool."""

        @tool
        def read_runtime_file(
            root_name: Literal[
                "project",
                "docs",
                "skills",
                "uploads",
                "uploads_cli",
                "uploads_web",
                "workflow_artifacts",
            ] = "project",
            relative_path: str = ".",
            start_line: int = 1,
            max_lines: int = _READONLY_RUNTIME_MAX_LINES,
        ) -> str:
            """Read approved runtime files with read-only access when needed."""

            return self.read_runtime_file(
                root_name=root_name,
                relative_path=relative_path,
                start_line=start_line,
                max_lines=max_lines,
            )

        return read_runtime_file

    def _build_runtime_file_write_tool(self):
        """Create the controlled external text-write tool."""

        @tool
        def write_external_file(
            root_name: Literal[
                "docs",
                "skills",
                "uploads",
                "uploads_cli",
                "uploads_web",
                "workflow_artifacts",
            ],
            relative_path: str,
            content: str,
            mode: Literal[
                "overwrite",
                "append",
                "fail_if_exists",
            ] = "overwrite",
        ) -> str:
            """Write UTF-8 text to approved managed output roots."""

            return self.write_external_file(
                root_name=root_name,
                relative_path=relative_path,
                content=content,
                mode=mode,
            )

        return write_external_file

    def _build_save_skill_tool(self):
        """Create the skill-persistence tool."""

        @tool
        def save_skill_pattern(
            name: str,
            description: str,
            instructions: str,
        ) -> str:
            """Save a reusable workflow as a Deep Agents skill for future tasks."""

            skill_dir = self.skill_manager.save_skill_pattern(
                name=name,
                description=description,
                instructions=instructions,
            )
            return str(skill_dir.relative_to(self.config.project_root))

        return save_skill_pattern

    def _build_local_python_tool(self):
        """Create the guarded local Python execution tool."""

        @tool
        async def invoke_local_python(
            execution_mode: Literal["module", "script", "code"],
            target: str,
            args: list[str] | None = None,
            working_directory: str = ".",
            timeout_seconds: int = 120,
            purpose: str = "",
        ) -> str:
            """Run Python inside the workspace and return structured JSON.

            Use module mode for commands like pytest, script mode for workspace
            files, and code mode for short inline Python snippets. Keep runs
            focused and prefer the smallest useful command.
            """

            try:
                result = await execute_local_python(
                    workspace_root=self.config.workspace_dir,
                    execution_mode=execution_mode,
                    target=target,
                    args=args,
                    working_directory=working_directory,
                    timeout_seconds=timeout_seconds,
                    purpose=purpose,
                )
            except Exception as exc:  # pragma: no cover - defensive tool guard
                return json.dumps(
                    {
                        "ok": False,
                        "execution_mode": execution_mode,
                        "purpose": purpose.strip(),
                        "target": target,
                        "working_directory": working_directory,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            return result.to_json()

        return invoke_local_python

    async def build_agent(self, *, local_python_tool_enabled: bool = False):
        """Assemble the Deep Agent with MCP, RAG, skills, and memory."""

        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._sync_project_context_into_workspace()
        configure_langsmith(self.config)
        mcp_tools = await self._load_mcp_tools()
        custom_tools = [
            self._build_kb_tool(),
            self._build_runtime_file_list_tool(),
            self._build_runtime_file_read_tool(),
            self._build_runtime_file_write_tool(),
        ]
        if self._skills_dir_is_workspace_local():
            custom_tools.append(self._build_save_skill_tool())
        if local_python_tool_enabled:
            custom_tools.append(self._build_local_python_tool())
        backend = FilesystemBackend(root_dir=str(self.config.workspace_dir))
        agent = create_deep_agent(
            name="DevMate",
            model=build_chat_model(self.config.model),
            tools=[*mcp_tools, *custom_tools],
            system_prompt=SYSTEM_PROMPT,
            backend=backend,
            skills=self._agent_skill_paths(),
            memory=["./AGENTS.md"],
        )
        return agent

    def _extract_agent_tool_names(self, agent) -> list[str]:
        """Best-effort extraction of tool names from an agent object."""

        raw_tools = getattr(agent, "tools", None) or getattr(agent, "_tools", None)
        if raw_tools is None:
            return []
        names: list[str] = []
        for tool_obj in raw_tools:
            candidate = None
            if isinstance(tool_obj, dict):
                candidate = tool_obj.get("name")
            else:
                candidate = getattr(tool_obj, "name", None) or getattr(
                    tool_obj,
                    "__name__",
                    None,
                )
            if candidate:
                names.append(str(candidate).strip().lower())
        return names

    def _agent_has_planner_tool(self, agent) -> bool:
        """Return whether the current agent exposes a planner-like tool."""

        tool_names = self._extract_agent_tool_names(agent)
        planner_aliases = {"planner", "plan", "planning", "planning_tool"}
        return any(
            name in planner_aliases or "planner" in name or "plan" == name
            for name in tool_names
        )

    def _format_verifier_feedback_summary(
        self,
        report: VerificationReport | None,
    ) -> str:
        """Serialize verifier findings for targeted Researcher retries."""

        if report is None:
            return ""
        lines: list[str] = []
        if report.summary:
            lines.append(f"Summary: {report.summary}")
        if report.research_feedback:
            lines.append(f"Research feedback: {report.research_feedback}")
        if report.builder_feedback:
            lines.append(f"Builder feedback: {report.builder_feedback}")
        for item in report.module_results:
            finding_bits = [
                f"module={item.module}",
                f"has_error={str(item.has_error).lower()}",
                f"missing_detail={str(item.missing_detail).lower()}",
            ]
            if item.details:
                finding_bits.append(f"details={item.details}")
            if item.recommendation:
                finding_bits.append(f"recommendation={item.recommendation}")
            lines.append("Finding: " + " | ".join(finding_bits))
        return "\n".join(lines)

    def _cache_research_artifact(
        self,
        *,
        run_id: str,
        round_index: int,
        prompt: str,
        researcher_text: str,
    ) -> Path | None:
        """Persist Researcher findings into the local TXT knowledge cache."""

        content = (researcher_text or "").strip()
        if not content:
            return None
        cached_path = self.knowledge_base.cache_research_knowledge(
            run_id=run_id,
            round_index=round_index,
            prompt=prompt,
            content=content,
        )
        if cached_path is not None:
            LOGGER.info("Cached Researcher knowledge at %s", cached_path)
        return cached_path

    def _stabilize_directory_permissions(self, root: Path) -> None:
        """Best-effort permission repair for Docker volume write-back."""

        if not root.exists():
            return
        try:
            root_stat = root.stat()
        except OSError as exc:
            LOGGER.debug("Unable to stat %s for permission repair: %s", root, exc)
            return

        desired_uid = root_stat.st_uid
        desired_gid = root_stat.st_gid
        for path in [root, *sorted(root.rglob("*"))]:
            try:
                if hasattr(os, "chown"):
                    os.chown(path, desired_uid, desired_gid)
                mode = 0o775 if path.is_dir() else 0o664
                path.chmod(mode)
            except PermissionError:
                LOGGER.debug(
                    "Skipping ownership repair for %s due to permissions",
                    path,
                )
            except OSError as exc:
                LOGGER.debug("Unable to repair permissions for %s: %s", path, exc)

    def _write_stage_artifact(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((text or "").strip(), encoding="utf-8")

    def _sync_runtime_writeback(self) -> None:
        """Best-effort write-back stabilization for managed workspace dirs."""

        roots = {
            self.config.workspace_dir,
            self.config.skills_dir,
            self.config.skills_source_dir,
            self.config.docs_dir,
            self.config.docs_source_dir,
            self.config.workflow_artifacts_dir,
            self.config.state_dir,
            self.config.persist_dir,
            self.config.persist_source_dir,
            self.config.research_cache_dir,
            self.uploads_root(),
            self.upload_dir("cli"),
            self.upload_dir("web"),
            self.config.workflow_artifacts_dir.parent,
            self.config.state_dir.parent,
            self.config.persist_dir.parent,
            self.config.persist_source_dir.parent,
        }
        for root in sorted(roots, key=str):
            root.mkdir(parents=True, exist_ok=True)
            self._stabilize_directory_permissions(root)

    def _verification_passed_for_mode(
        self,
        report: VerificationReport | None,
        *,
        detail_review_enabled: bool,
    ) -> bool | None:
        """Return the pass/fail decision for the active verification mode."""

        if report is None:
            return None
        if detail_review_enabled:
            return report.passed
        return not report.contains_errors

    def _hidden_workspace_roots(self) -> tuple[Path, ...]:
        """Return managed workspace roots omitted from user file tracking."""

        return (
            self.config.research_cache_dir,
            self.config.workflow_artifacts_dir,
            self.config.state_dir,
            self.config.persist_dir,
            self.uploads_root(),
        )

    def _should_track_workspace_file(self, path: Path) -> bool:
        """Return whether a workspace file should be exposed to users."""

        if path.name == ".gitkeep":
            return False
        if path.name.startswith(".coverage"):
            return False
        if any(part in _INTERNAL_WORKSPACE_DIR_NAMES for part in path.parts):
            return False
        resolved = path.resolve()
        for root in self._hidden_workspace_roots():
            if self._path_is_within_root(root, resolved):
                return False
        return path.is_file()

    def snapshot_workspace(self) -> dict[str, WorkspaceFileFingerprint]:
        """Capture workspace fingerprints for change tracking."""

        snapshot: dict[str, WorkspaceFileFingerprint] = {}
        workspace = self.config.workspace_dir
        workspace.mkdir(parents=True, exist_ok=True)
        for path in workspace.rglob("*"):
            if not self._should_track_workspace_file(path):
                continue
            stat = path.stat()
            relative_path = path.relative_to(self.config.project_root).as_posix()
            snapshot[relative_path] = WorkspaceFileFingerprint(
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        return snapshot

    def diff_workspace_changes(
        self,
        before: dict[str, WorkspaceFileFingerprint],
        after: dict[str, WorkspaceFileFingerprint],
    ) -> list[WorkspaceChange]:
        """Return structured workspace create/modify/delete operations."""

        changes: list[WorkspaceChange] = []
        after_paths = set(after)
        before_paths = set(before)

        for path in sorted(after_paths - before_paths):
            changes.append(WorkspaceChange(path=path, action="created"))
        for path in sorted(before_paths & after_paths):
            if before[path] != after[path]:
                changes.append(WorkspaceChange(path=path, action="modified"))
        for path in sorted(before_paths - after_paths):
            changes.append(WorkspaceChange(path=path, action="deleted"))
        return changes

    def _ensure_workspace_changes_present(
        self,
        changed_files: list[str],
    ) -> None:
        """Require at least one real workspace change before success."""

        if changed_files:
            return
        raise RuntimeError(
            "Agent finished without modifying any workspace files. "
            "Enforce workspace-relative writes and require at least one real "
            "file change before completion."
        )

    def format_workspace_change(self, change: WorkspaceChange, stage_name: str) -> str:
        """Render a workspace change record for UI and CLI presentation."""

        label_map = {
            "created": "创建",
            "modified": "修改",
            "deleted": "删除",
        }
        label = label_map.get(change.action, change.action)
        return f"[{stage_name}] {label} {change.path}"

    def list_workspace_files(self) -> list[str]:
        """List all user-visible files currently under the workspace directory."""

        workspace = self.config.workspace_dir
        workspace.mkdir(parents=True, exist_ok=True)
        return sorted(
            path.relative_to(self.config.project_root).as_posix()
            for path in workspace.rglob("*")
            if self._should_track_workspace_file(path)
        )

    def _flatten_content(self, content) -> str:
        """Normalize agent message content to plain text."""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, str):
                    fragments.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        fragments.append(str(text))
                else:
                    fragments.append(str(item))
            return "\n".join(fragment for fragment in fragments if fragment)
        return str(content)

    def _is_retryable_llm_error(self, exc: Exception) -> bool:
        """Best-effort classification for transient LLM/API failures."""

        if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
            return True

        message_parts: list[str] = []
        current: BaseException | None = exc
        while current is not None:
            message_parts.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        message = " ".join(message_parts).lower()
        retry_markers = {
            "api",
            "bad gateway",
            "connect",
            "connection",
            "deepseek",
            "gateway timeout",
            "http",
            "llm",
            "openai",
            "rate limit",
            "service unavailable",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "too many requests",
        }
        return any(marker in message for marker in retry_markers)

    async def _run_with_llm_retries(
        self,
        operation: Callable[[], Awaitable[object]],
        *,
        operation_name: str,
        progress_callback: ProgressCallback | None,
        current_phase: str,
    ) -> object:
        """Run an LLM-dependent operation with retry-on-failure semantics."""

        attempt = 0
        max_attempts = _LLM_MAX_RETRIES_AFTER_FAILURE + 1
        while True:
            attempt += 1
            try:
                return await operation()
            except (StageInterruptedError, StageLoopDetectedError):
                raise
            except Exception as exc:
                should_retry = attempt < max_attempts and self._is_retryable_llm_error(
                    exc
                )
                if not should_retry:
                    LOGGER.exception(
                        "LLM request failed during %s after %s attempt(s): %s",
                        operation_name,
                        attempt,
                        exc,
                    )
                    raise

                LOGGER.warning(
                    "LLM request failed during %s on attempt %s/%s: %s. "
                    "Retrying in %s seconds.",
                    operation_name,
                    attempt,
                    max_attempts,
                    exc,
                    _LLM_RETRY_DELAY_SECONDS,
                )
                await self._emit_progress(
                    progress_callback,
                    event_type="status",
                    message=(
                        f"{operation_name} 调用大模型失败，"
                        f"将在 {_LLM_RETRY_DELAY_SECONDS} 秒后重试 "
                        f"({attempt}/{_LLM_MAX_RETRIES_AFTER_FAILURE})…"
                    ),
                    current_phase=current_phase,
                )
                await asyncio.sleep(_LLM_RETRY_DELAY_SECONDS)

    def _normalize_stream_part(self, chunk) -> tuple[tuple[str, ...], str, object]:
        """Normalize LangGraph stream parts into a stable tuple."""

        if isinstance(chunk, dict):
            return (
                tuple(chunk.get("ns", ())),
                str(chunk.get("type", "")),
                chunk.get("data"),
            )
        if isinstance(chunk, tuple) and len(chunk) == 3:
            namespace, mode, data = chunk
            return tuple(namespace), str(mode), data
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, data = chunk
            return (), str(mode), data
        return (), "", chunk

    def _message_text_from_chunk(self, message) -> str:
        """Extract readable text from a streamed message chunk."""

        content = getattr(message, "content", "")
        return self._flatten_content(content)

    def _merge_stream_text(self, previous: str, incoming: str) -> str:
        """Merge a streamed token fragment with the buffered reply."""

        if not incoming:
            return previous
        if not previous:
            return incoming
        if incoming == previous:
            return previous
        if incoming.startswith(previous):
            return incoming
        if previous.endswith(incoming):
            return previous
        return previous + incoming

    def _humanize_task_name(self, task_name: str) -> str:
        """Turn low-level task names into more readable UI text."""

        aliases = {
            "__start__": "启动任务",
            "__end__": "完成任务",
            "model": "调用大模型",
            "tools": "调用工具",
        }
        if task_name in aliases:
            return aliases[task_name]
        return task_name.replace("_", " ").replace("-", " ")

    def _normalize_output_for_similarity(self, text: str) -> str:
        """Normalize model output before similarity comparison."""

        return " ".join(text.lower().split())

    def _outputs_look_like_loop(self, outputs: list[str]) -> bool:
        """Return whether the last three outputs are nearly identical."""

        if len(outputs) < 3:
            return False
        recent = outputs[-3:]
        normalized = [self._normalize_output_for_similarity(item) for item in recent]
        if any(not item for item in normalized):
            return False
        return all(
            SequenceMatcher(None, normalized[index], normalized[index + 1]).ratio()
            >= 0.95
            and SequenceMatcher(None, normalized[0], normalized[2]).ratio() >= 0.95
            for index in range(2)
        )

    async def _stream_agent_run(
        self,
        agent,
        payload: dict[str, object],
        run_config: dict[str, object],
        progress_callback: ProgressCallback | None,
        *,
        stage_name: str,
        workspace_tracker: WorkspaceTracker | None,
        controller: RunController | None,
    ):
        """Execute the agent with LangGraph streaming enabled."""

        last_state = None
        reply_buffers: dict[str, str] = {}
        visible_message_id: str | None = None
        finalized_message_id: str | None = None
        recent_outputs: deque[StageReplySample] = deque(maxlen=3)

        def finalize_visible_message() -> None:
            nonlocal finalized_message_id
            if visible_message_id is None or visible_message_id == finalized_message_id:
                return
            text = reply_buffers.get(visible_message_id, "").strip()
            finalized_message_id = visible_message_id
            if not text:
                return
            recent_outputs.append(text)
            if self._outputs_look_like_loop(list(recent_outputs)):
                raise StageLoopDetectedError(stage_name=stage_name, partial_reply=text)

        async for chunk in agent.astream(
            payload,
            config=run_config,
            stream_mode=["values", "messages", "tasks"],
            subgraphs=True,
        ):
            if controller is not None and controller.immediate_output_requested:
                partial_reply = reply_buffers.get(visible_message_id or "", "")
                raise StageInterruptedError(
                    stage_name=stage_name,
                    partial_reply=partial_reply,
                )

            namespace, mode, data = self._normalize_stream_part(chunk)

            if mode == "values" and not namespace:
                last_state = data
                continue

            if mode == "tasks" and isinstance(data, dict):
                task_name = self._humanize_task_name(str(data.get("name", "步骤")))
                if "input" in data:
                    await self._emit_progress(
                        progress_callback,
                        event_type="status",
                        message=f"正在执行：{task_name}",
                        current_phase=stage_name,
                    )
                elif data.get("error"):
                    await self._emit_progress(
                        progress_callback,
                        event_type="status",
                        message=f"步骤失败：{task_name}",
                        current_phase=stage_name,
                    )
                else:
                    await self._emit_progress(
                        progress_callback,
                        event_type="status",
                        message=f"已完成：{task_name}",
                        current_phase=stage_name,
                    )
                if workspace_tracker is not None:
                    await workspace_tracker.scan_and_emit(
                        stage_name=stage_name,
                        progress_callback=progress_callback,
                    )
                continue

            if mode != "messages" or not isinstance(data, tuple) or len(data) != 2:
                continue

            message, metadata = data
            message_type = str(getattr(message, "type", "")).lower()
            if "ai" not in message_type and "assistant" not in message_type:
                continue

            message_text = self._message_text_from_chunk(message)
            if not message_text:
                continue

            chunk_message_id = getattr(message, "id", None) or (
                f"{metadata.get('langgraph_node', 'assistant')}::"
                f"{metadata.get('langgraph_step', 0)}"
            )
            if (
                visible_message_id is not None
                and chunk_message_id != visible_message_id
            ):
                finalize_visible_message()

            reply_buffers[chunk_message_id] = self._merge_stream_text(
                reply_buffers.get(chunk_message_id, ""),
                message_text,
            )
            visible_message_id = chunk_message_id
            await self._emit_progress(
                progress_callback,
                event_type="reply",
                message=reply_buffers[visible_message_id],
                current_phase=stage_name,
            )

        finalize_visible_message()
        if last_state is not None:
            return last_state
        if visible_message_id is not None and reply_buffers.get(visible_message_id):
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": reply_buffers[visible_message_id],
                    }
                ]
            }
        return {"messages": []}

    def extract_reply(self, result) -> str:
        """Extract the final assistant message from a Deep Agent result."""

        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            return str(result)

        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, dict):
                role = message.get("role") or message.get("type")
                if role in {"assistant", "ai"}:
                    return self._flatten_content(message.get("content", ""))
                continue

            message_type = getattr(message, "type", "")
            if message_type in {"assistant", "ai"}:
                return self._flatten_content(getattr(message, "content", ""))
        return "No assistant response was captured."

    def _resolve_workflow_options(
        self,
        *,
        deep_optimization: bool | None,
        local_python_tool_enabled: bool | None,
        max_deep_optimization_rounds: int | None,
    ) -> WorkflowOptions:
        """Resolve workflow options from config defaults and request overrides."""

        workflow_cfg = self.config.workflow
        enabled = workflow_cfg.deep_optimization_default
        if deep_optimization is not None:
            enabled = deep_optimization

        python_tool_enabled = workflow_cfg.local_python_tool_default
        if local_python_tool_enabled is not None:
            python_tool_enabled = local_python_tool_enabled

        max_rounds = workflow_cfg.max_deep_optimization_rounds
        if max_deep_optimization_rounds is not None:
            max_rounds = max(0, max_deep_optimization_rounds)

        return WorkflowOptions(
            deep_optimization=enabled,
            local_python_tool_enabled=python_tool_enabled,
            max_deep_optimization_rounds=max_rounds,
            standard_builder_repair_rounds=max(
                0,
                workflow_cfg.standard_builder_repair_rounds,
            ),
        )

    def _create_workflow_artifacts(self, run_id: str) -> WorkflowArtifacts:
        """Create a per-run artifact directory."""

        root = self.config.workflow_artifacts_dir / run_id
        root.mkdir(parents=True, exist_ok=True)
        return WorkflowArtifacts(root=root)

    async def _invoke_stage(
        self,
        *,
        agent,
        stage_name: str,
        stage_prompt: str,
        run_config: dict[str, object],
        progress_callback: ProgressCallback | None,
        workspace_tracker: WorkspaceTracker | None,
        controller: RunController | None,
    ) -> StageExecutionResult:
        """Execute a single stage with the shared agent."""

        await self._emit_progress(
            progress_callback,
            event_type="status",
            message=f"正在执行阶段：{stage_name}",
            current_phase=stage_name,
        )
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": stage_prompt,
                }
            ]
        }

        try:

            async def invoke_once() -> object:
                if hasattr(agent, "astream"):
                    return await self._stream_agent_run(
                        agent,
                        payload,
                        run_config,
                        progress_callback,
                        stage_name=stage_name,
                        workspace_tracker=workspace_tracker,
                        controller=controller,
                    )
                return await agent.ainvoke(payload, config=run_config)

            graph_result = await self._run_with_llm_retries(
                invoke_once,
                operation_name=stage_name,
                progress_callback=progress_callback,
                current_phase=stage_name,
            )
        except StageInterruptedError as exc:
            if workspace_tracker is not None:
                await workspace_tracker.scan_and_emit(
                    stage_name=stage_name,
                    progress_callback=progress_callback,
                )
            self._sync_runtime_writeback()
            await self._emit_progress(
                progress_callback,
                event_type="status",
                message=f"已终止阶段：{stage_name}，准备进入最终整合与打包…",
                current_phase=stage_name,
            )
            return StageExecutionResult(
                stage_name=stage_name,
                reply=exc.partial_reply,
                interrupted=True,
            )
        except StageLoopDetectedError as exc:
            if controller is not None:
                controller.request_immediate_output(
                    reason=(
                        f"{stage_name} 阶段最近三次输出相似度达到 95%，"
                        "已触发防循环保护。"
                    )
                )
            if workspace_tracker is not None:
                await workspace_tracker.scan_and_emit(
                    stage_name=stage_name,
                    progress_callback=progress_callback,
                )
            self._sync_runtime_writeback()
            await self._emit_progress(
                progress_callback,
                event_type="status",
                message=(
                    f"{stage_name} 阶段检测到重复输出，"
                    "已触发防循环保护并进入最终整合与打包。"
                ),
                current_phase=stage_name,
            )
            return StageExecutionResult(
                stage_name=stage_name,
                reply=exc.partial_reply,
                interrupted=True,
                loop_detected=True,
            )

        if workspace_tracker is not None:
            await workspace_tracker.scan_and_emit(
                stage_name=stage_name,
                progress_callback=progress_callback,
            )
        self._sync_runtime_writeback()

        reply = self.extract_reply(graph_result)
        await self._emit_progress(
            progress_callback,
            event_type="status",
            message=f"阶段完成：{stage_name}",
            current_phase=stage_name,
        )
        return StageExecutionResult(stage_name=stage_name, reply=reply)

    def _package_workspace_archive(self, artifacts: WorkflowArtifacts) -> str | None:
        """Create a zip archive of the current workspace."""

        files = self.list_workspace_files()
        if not files:
            return None

        zip_path = artifacts.delivery_zip_path()
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
            for relative_path in files:
                candidate = self.config.project_root / relative_path
                if not candidate.exists() or not candidate.is_file():
                    continue
                archive.write(candidate, arcname=relative_path)
        return zip_path.relative_to(self.config.project_root).as_posix()

    def _select_packager_reply(
        self,
        packager_result: StageExecutionResult,
        verifier_report: VerificationReport | None,
        fallback_replies: list[str],
    ) -> str:
        """Select the most useful final reply text."""

        if packager_result.reply:
            return packager_result.reply
        if verifier_report is not None and verifier_report.summary:
            return verifier_report.summary
        for reply in reversed(fallback_replies):
            if reply:
                return reply
        return "Task execution completed."

    def _read_text_file(self, path: Path, fallback_label: str) -> str:
        """Read a UTF-8 text file for prompt construction."""

        if not path.exists() or not path.is_file():
            return f"[{fallback_label} unavailable: {path.name} does not exist]"
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"[{fallback_label} unavailable: {path.name} is not UTF-8 text]"
        except OSError as exc:
            return f"[{fallback_label} unavailable: {exc}]"

    def _trim_for_prompt(self, text: str, *, max_chars: int, label: str) -> str:
        """Trim very large prompt inputs while preserving a clear marker."""

        if len(text) <= max_chars:
            return text
        trimmed = text[:max_chars]
        return (
            f"{trimmed}\n\n"
            f"[... {label} truncated to the first {max_chars} characters ...]"
        )

    def _copy_tree_if_source_is_newer(
        self,
        *,
        source_root: Path,
        destination_root: Path,
    ) -> list[Path]:
        """Copy files into destination when source files are missing or newer."""

        source_root = source_root.resolve()
        destination_root = destination_root.resolve()
        destination_root.mkdir(parents=True, exist_ok=True)
        if not source_root.exists() or source_root == destination_root:
            return []

        copied_paths: list[Path] = []
        for source_path in sorted(source_root.rglob("*")):
            if not source_path.is_file():
                continue
            relative_path = source_path.relative_to(source_root)
            destination_path = destination_root / relative_path
            if self._copy_file_if_source_is_newer(
                source_path=source_path,
                destination_path=destination_path,
            ):
                copied_paths.append(destination_path)

        return copied_paths

    def _copy_file_if_source_is_newer(
        self,
        *,
        source_path: Path,
        destination_path: Path,
    ) -> bool:
        """Copy one file when the destination is missing or older."""

        source_path = source_path.resolve()
        destination_path = destination_path.resolve()
        if not source_path.exists() or not source_path.is_file():
            return False
        if source_path == destination_path:
            return False

        if destination_path.exists():
            try:
                files_match = filecmp.cmp(
                    source_path,
                    destination_path,
                    shallow=False,
                )
            except OSError:
                files_match = False
            if files_match:
                return False
            try:
                source_mtime = source_path.stat().st_mtime_ns
                destination_mtime = destination_path.stat().st_mtime_ns
            except OSError:
                source_mtime = 0
                destination_mtime = 0
            if destination_mtime > source_mtime:
                return False

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return True

    def _sync_project_context_into_workspace(self) -> None:
        """Mirror project context files that the agent must read in workspace."""

        mirrors = (
            (self.config.skills_source_dir, self.config.skills_dir),
            (self.config.docs_source_dir, self.config.docs_dir),
            (self.config.persist_source_dir, self.config.persist_dir),
        )
        for source_root, destination_root in mirrors:
            self._copy_tree_if_source_is_newer(
                source_root=source_root,
                destination_root=destination_root,
            )

        self._copy_file_if_source_is_newer(
            source_path=self.config.project_root / 'AGENTS.md',
            destination_path=self.config.workspace_dir / 'AGENTS.md',
        )

    def sync_workspace_skills_back_to_source(self) -> list[str]:
        """Copy new workspace Markdown skill files back to the source root."""

        copied_paths = self.skill_manager.sync_new_markdown_to_source(
            source_root=self.config.skills_source_dir,
        )
        return [
            path.relative_to(self.config.project_root).as_posix()
            for path in copied_paths
        ]

    def _build_workspace_snapshot_text(
        self,
        *,
        max_total_chars: int = _VERIFIER_MAX_TOTAL_CHARS,
        max_file_chars: int = _VERIFIER_MAX_FILE_CHARS,
        max_files: int = _VERIFIER_MAX_FILES,
    ) -> str:
        """Serialize current workspace files as plain text for prompt context."""

        files = self.list_workspace_files()
        if not files:
            return "[workspace is currently empty]"

        parts: list[str] = []
        remaining_budget = max_total_chars
        processed_files = 0

        for relative_path in files:
            if processed_files >= max_files:
                omitted = len(files) - processed_files
                parts.append(
                    "\n[... omitted "
                    f"{omitted} additional workspace files due to limit ...]"
                )
                break

            absolute_path = self.config.project_root / relative_path
            try:
                content = absolute_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = "[binary or non-UTF-8 file omitted from text verification]"
            except OSError as exc:
                content = f"[unable to read file for verification: {exc}]"

            content = self._trim_for_prompt(
                content,
                max_chars=max_file_chars,
                label=f"{relative_path} content",
            )
            block = (
                f"<<<FILE: {relative_path}>>>\n"
                f"{content}\n"
                f"<<<END FILE: {relative_path}>>>\n"
            )
            if len(block) > remaining_budget:
                if remaining_budget <= 256:
                    omitted = len(files) - processed_files
                    parts.append(
                        "\n[... omitted "
                        f"{omitted} additional workspace files due to "
                        "prompt budget ...]"
                    )
                    break
                truncated_block = self._trim_for_prompt(
                    block,
                    max_chars=remaining_budget,
                    label="workspace snapshot",
                )
                parts.append(truncated_block)
                break

            parts.append(block)
            remaining_budget -= len(block)
            processed_files += 1

        return "\n".join(parts)

    def _build_workspace_file_index_text(self) -> str:
        """Return a compact newline list of workspace files for prompts."""

        files = self.list_workspace_files()
        if not files:
            return "- [workspace is currently empty]"
        return "\n".join(f"- {path}" for path in files)

    async def _invoke_text_model(
        self,
        prompt: str,
        *,
        progress_callback: ProgressCallback | None = None,
        current_phase: str = "Verifier",
    ) -> str:
        """Invoke the configured chat model directly and return plain text."""

        model = build_chat_model(self.config.model)

        async def invoke_once() -> object:
            return await model.ainvoke(prompt)

        response = await self._run_with_llm_retries(
            invoke_once,
            operation_name=current_phase,
            progress_callback=progress_callback,
            current_phase=current_phase,
        )
        content = getattr(response, "content", response)
        return self._flatten_content(content)

    async def _invoke_text_model_with_context(
        self,
        prompt: str,
        *,
        progress_callback: ProgressCallback | None = None,
        current_phase: str = "Verifier",
    ) -> str:
        """Call the text model while remaining compatible with simple test doubles."""

        invoke = self._invoke_text_model
        try:
            return await invoke(
                prompt,
                progress_callback=progress_callback,
                current_phase=current_phase,
            )
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message and (
                "positional argument" not in message or "given" not in message
            ):
                raise
            return await invoke(prompt)

    def _build_fallback_verifier_payload(self, reason: str) -> dict[str, object]:
        """Build a safe fallback verifier payload when model output is unusable."""

        return {
            "overall_status": "needs_fix",
            "contains_errors": False,
            "contains_missing_details": True,
            "summary": reason,
            "module_results": [
                {
                    "module": "workspace",
                    "has_error": False,
                    "missing_detail": True,
                    "details": reason,
                    "recommendation": (
                        "Review the current workspace and rerun verification after "
                        "fixing the reported issue."
                    ),
                }
            ],
            "builder_feedback": reason,
            "research_feedback": "",
        }

    async def _run_verifier_stage(
        self,
        *,
        agent,
        run_config: dict[str, object],
        planner_text: str,
        researcher_text: str,
        builder_text: str,
        verifier_path: Path,
        round_index: int,
        user_prompt: str,
        progress_callback: ProgressCallback | None,
        workspace_tracker: WorkspaceTracker | None,
        detail_review_enabled: bool,
        controller: RunController | None,
        local_python_tool_enabled: bool,
    ) -> tuple[StageExecutionResult, VerificationReport, str]:
        """Run the Verifier in either text-review or pytest mode."""

        stage_name = "Verifier"
        await self._emit_progress(
            progress_callback,
            event_type="status",
            message="正在执行阶段：Verifier",
            current_phase=stage_name,
        )
        verifier_mode_message = (
            (
                "Verifier 采用 pytest + 本地 Python 工具模式，将编写 pytest 测试、"
                "本地执行并根据结果分析实现质量。"
            )
            if local_python_tool_enabled
            else (
                "Verifier 采用深度细节文本审查模式，直接检查 "
                "planner、research、builder 产物和 workspace 文件，"
                "重点判断细节是否全部完成，不执行任何脚本。"
                if detail_review_enabled
                else (
                    "Verifier 采用常规文本审查模式，直接检查 "
                    "planner、research、builder 产物和 workspace 文件，"
                    "只关注代码错误与项目级缺陷，不执行任何脚本。"
                )
            )
        )
        await self._emit_progress(
            progress_callback,
            event_type="status",
            message=verifier_mode_message,
            current_phase=stage_name,
        )
        if workspace_tracker is not None:
            await workspace_tracker.scan_and_emit(
                stage_name=stage_name,
                progress_callback=progress_callback,
            )

        planner_snapshot = self._trim_for_prompt(
            planner_text,
            max_chars=24_000,
            label="planner artifact",
        )
        researcher_snapshot = self._trim_for_prompt(
            researcher_text,
            max_chars=28_000,
            label="research artifact",
        )
        builder_snapshot = self._trim_for_prompt(
            builder_text,
            max_chars=20_000,
            label="builder artifact",
        )
        raw_reply = ""
        stage_result = StageExecutionResult(stage_name=stage_name, reply="")

        if local_python_tool_enabled:
            verifier_prompt = build_verifier_pytest_prompt(
                round_index=round_index,
                user_prompt=user_prompt,
                planner_text=planner_snapshot,
                researcher_text=researcher_snapshot,
                builder_text=builder_snapshot,
                workspace_file_index=self._build_workspace_file_index_text(),
                detail_review_enabled=detail_review_enabled,
            )
            stage_result = await self._invoke_stage(
                agent=agent,
                stage_name=stage_name,
                stage_prompt=verifier_prompt,
                run_config=run_config,
                progress_callback=progress_callback,
                workspace_tracker=workspace_tracker,
                controller=controller,
            )
            raw_reply = stage_result.reply
        else:
            workspace_snapshot_text = self._build_workspace_snapshot_text()
            verifier_prompt = build_verifier_text_review_prompt(
                round_index=round_index,
                user_prompt=user_prompt,
                planner_text=planner_snapshot,
                researcher_text=researcher_snapshot,
                builder_text=builder_snapshot,
                workspace_snapshot_text=workspace_snapshot_text,
                detail_review_enabled=detail_review_enabled,
            )
            raw_reply = await self._invoke_text_model_with_context(
                verifier_prompt,
                progress_callback=progress_callback,
                current_phase=stage_name,
            )

        raw_verifier_text = raw_reply.strip()
        if not raw_verifier_text and (
            stage_result.interrupted or stage_result.loop_detected
        ):
            fallback_payload = self._build_fallback_verifier_payload(
                "Verifier 在完成本地 pytest 校验前被中断，当前结果需要人工复核。"
            )
            raw_verifier_text = json.dumps(
                fallback_payload,
                ensure_ascii=False,
                indent=2,
            )
        self._write_stage_artifact(verifier_path, raw_verifier_text)
        self._sync_runtime_writeback()
        verifier_report = parse_verifier_report(verifier_path)

        if (
            not verifier_report.summary
            and not verifier_report.module_results
            and not stage_result.interrupted
            and not stage_result.loop_detected
        ):
            repair_prompt = (
                "Convert the following verifier response into strict JSON only. "
                "Do not add markdown fences. Use this schema exactly: "
                '{"overall_status": "pass" | "needs_fix", '
                '"contains_errors": true | false, '
                '"contains_missing_details": true | false, '
                '"summary": "string", '
                '"module_results": [{"module": "string", '
                '"has_error": true | false, '
                '"missing_detail": true | false, '
                '"details": "string", '
                '"recommendation": "string"}], '
                '"builder_feedback": "string", '
                '"research_feedback": "string"}.\n\n'
                "Original response:\n"
                f"{raw_reply}"
            )
            repaired_reply = await self._invoke_text_model_with_context(
                repair_prompt,
                progress_callback=progress_callback,
                current_phase=stage_name,
            )
            raw_verifier_text = repaired_reply.strip()
            self._write_stage_artifact(verifier_path, raw_verifier_text)
            self._sync_runtime_writeback()
            verifier_report = parse_verifier_report(verifier_path)
            raw_reply = repaired_reply

        if not verifier_report.summary and not verifier_report.module_results:
            fallback_payload = self._build_fallback_verifier_payload(
                "Verifier 未返回可解析的 JSON，当前结果需要人工复核。"
            )
            raw_verifier_text = json.dumps(
                fallback_payload,
                ensure_ascii=False,
                indent=2,
            )
            self._write_stage_artifact(verifier_path, raw_verifier_text)
            self._sync_runtime_writeback()
            verifier_report = parse_verifier_report(verifier_path)
            raw_reply = raw_verifier_text

        reply = verifier_report.summary or raw_reply
        await self._emit_progress(
            progress_callback,
            event_type="reply",
            message=reply,
            current_phase=stage_name,
        )
        await self._emit_progress(
            progress_callback,
            event_type="status",
            message="阶段完成：Verifier",
            current_phase=stage_name,
        )
        return (
            StageExecutionResult(
                stage_name=stage_name,
                reply=reply,
                interrupted=stage_result.interrupted,
                loop_detected=stage_result.loop_detected,
            ),
            verifier_report,
            raw_verifier_text,
        )

    async def _check_recent_output_loop(
        self,
        *,
        recent_outputs: deque[StageReplySample],
        latest_reply: str,
        stage_name: str,
        controller: RunController,
        progress_callback: ProgressCallback | None,
    ) -> bool:
        """Track recent stage replies and stop only on same-stage repetition."""

        if not latest_reply.strip():
            return False
        recent_outputs.append(
            StageReplySample(stage_name=stage_name, reply=latest_reply)
        )
        if len(recent_outputs) < 3:
            return False
        recent_samples = list(recent_outputs)[-3:]
        stage_names = {sample.stage_name for sample in recent_samples}
        if len(stage_names) != 1:
            return False
        if not self._outputs_look_like_loop(
            [sample.reply for sample in recent_samples]
        ):
            return False

        controller.request_immediate_output(
            reason=(f"{stage_name} 阶段最近三次输出相似度达到 95%，已触发防循环保护。")
        )
        await self._emit_progress(
            progress_callback,
            event_type="status",
            message=(
                f"检测到 {stage_name} 阶段最近三次输出高度重复，"
                "已触发防循环保护并转入最终整合与打包。"
            ),
            current_phase=stage_name,
        )
        return True

    async def run(
        self,
        prompt: str,
        progress_callback: ProgressCallback | None = None,
        *,
        deep_optimization: bool | None = None,
        local_python_tool_enabled: bool | None = None,
        max_deep_optimization_rounds: int | None = None,
        include_initial_files: bool = False,
        initial_file_paths: list[str] | None = None,
        controller: RunController | None = None,
    ) -> AgentRunResult:
        """Execute the staged DevMate workflow with a single shared agent."""

        controller = controller or RunController()
        workflow_options = self._resolve_workflow_options(
            deep_optimization=deep_optimization,
            local_python_tool_enabled=local_python_tool_enabled,
            max_deep_optimization_rounds=max_deep_optimization_rounds,
        )
        detail_review_enabled = workflow_options.deep_optimization

        await self._emit_progress(
            progress_callback,
            event_type="status",
            message="正在准备运行环境…",
            current_phase="",
            deep_optimization=workflow_options.deep_optimization,
            local_python_tool_enabled=(workflow_options.local_python_tool_enabled),
            max_deep_optimization_rounds=(
                workflow_options.max_deep_optimization_rounds
            ),
        )
        await self.prepare(rebuild_kb=False)
        workspace_tracker = WorkspaceTracker(
            runtime=self,
            snapshot=self.snapshot_workspace(),
        )
        effective_prompt, normalized_initial_paths, initial_file_context = (
            self._compose_prompt_with_initial_files(
                prompt,
                include_initial_files=include_initial_files,
                initial_file_paths=initial_file_paths,
            )
        )
        if include_initial_files:
            if normalized_initial_paths and initial_file_context:
                await self._emit_progress(
                    progress_callback,
                    event_type="status",
                    message=(
                        "已在 Planner 开始前注入 "
                        f"{len(normalized_initial_paths)} 个初始文件。"
                    ),
                    current_phase="Planner",
                )
            else:
                await self._emit_progress(
                    progress_callback,
                    event_type="status",
                    message="本次已开启初始文件注入，但未找到可注入的暂存文件。",
                    current_phase="Planner",
                )
        before_skills = set(self.skill_manager.list_skill_names())

        await self._emit_progress(
            progress_callback,
            event_type="status",
            message="正在构建单 Agent 并连接 MCP / RAG 工具…",
            current_phase="",
            deep_optimization=workflow_options.deep_optimization,
            local_python_tool_enabled=(workflow_options.local_python_tool_enabled),
            max_deep_optimization_rounds=(
                workflow_options.max_deep_optimization_rounds
            ),
        )
        agent = await self.build_agent(
            local_python_tool_enabled=workflow_options.local_python_tool_enabled,
        )
        await self._emit_progress(
            progress_callback,
            event_type="status",
            message=(
                "本次已开启本地 Python 调用工具，Builder/Verifier 可按需进行受控的 "
                "Python / pytest 本地检查。"
                if workflow_options.local_python_tool_enabled
                else (
                    "本次未开启本地 Python 调用工具，Builder/Verifier 将改用代码快照"
                    "进行静态逻辑与语法审查。"
                )
            ),
            current_phase="",
            deep_optimization=workflow_options.deep_optimization,
            local_python_tool_enabled=(workflow_options.local_python_tool_enabled),
            max_deep_optimization_rounds=(
                workflow_options.max_deep_optimization_rounds
            ),
        )
        planner_tool_available = self._agent_has_planner_tool(agent)
        if not planner_tool_available:
            await self._emit_progress(
                progress_callback,
                event_type="status",
                message=(
                    "当前未发现 Planner 工具，将启用 Planner 角色提示词回退，"
                    "直接由大模型以 Planner 身份输出计划。"
                ),
                current_phase="Planner",
                deep_optimization=workflow_options.deep_optimization,
                max_deep_optimization_rounds=(
                    workflow_options.max_deep_optimization_rounds
                ),
            )

        run_id = uuid4().hex
        run_config = {"configurable": {"thread_id": run_id}}
        artifacts = self._create_workflow_artifacts(run_id)
        planner_path = artifacts.planner_path()
        latest_researcher_path = artifacts.researcher_path(0)
        latest_builder_path = artifacts.builder_path(0)
        latest_verifier_path = artifacts.verifier_path(0)

        stage_replies: list[str] = []
        recent_outputs: deque[StageReplySample] = deque(maxlen=3)
        verification_report: VerificationReport | None = None
        optimization_rounds_used = 0
        repair_rounds_used = 0
        round_index = 0
        finalization_reason = "normal_completion"
        planner_text = ""
        researcher_text = ""
        builder_text = ""
        verifier_text = ""

        async def register_stage_result(result: StageExecutionResult) -> bool:
            if result.reply:
                stage_replies.append(result.reply)
            if result.loop_detected:
                return True
            if await self._check_recent_output_loop(
                recent_outputs=recent_outputs,
                latest_reply=result.reply,
                stage_name=result.stage_name,
                controller=controller,
                progress_callback=progress_callback,
            ):
                return True
            return result.interrupted or controller.immediate_output_requested

        async def cache_research_for_round(
            *,
            round_number: int,
            researcher_prompt: str,
            researcher_text: str,
        ) -> None:
            cached_path = self._cache_research_artifact(
                run_id=run_id,
                round_index=round_number,
                prompt=researcher_prompt,
                researcher_text=researcher_text,
            )
            if cached_path is None:
                return
            await self._emit_progress(
                progress_callback,
                event_type="status",
                message=(
                    "Researcher 阶段知识已写入本地知识库缓存："
                    f"{cached_path.relative_to(self.config.project_root).as_posix()}"
                ),
                current_phase="Researcher",
                optimization_rounds_used=optimization_rounds_used,
                deep_optimization=workflow_options.deep_optimization,
                max_deep_optimization_rounds=(
                    workflow_options.max_deep_optimization_rounds
                ),
            )

        if not controller.immediate_output_requested:
            planner_prompt = build_planner_prompt(
                user_prompt=effective_prompt,
                planner_tool_available=planner_tool_available,
            )
            planner_result = await self._invoke_stage(
                agent=agent,
                stage_name="Planner",
                stage_prompt=planner_prompt,
                run_config=run_config,
                progress_callback=progress_callback,
                workspace_tracker=workspace_tracker,
                controller=controller,
            )
            planner_text = (planner_result.reply or "").strip()
            self._write_stage_artifact(planner_path, planner_text)
            if await register_stage_result(planner_result):
                finalization_reason = controller.reason or "immediate_output_requested"

        if planner_text and self._planner_request_rejected(planner_text):
            await self._emit_progress(
                progress_callback,
                event_type="final",
                message="Planner 审核判定当前请求不应继续执行，已终止生成。",
                current_phase="Planner",
                reply=planner_text,
                changed_files=workspace_tracker.changed_files,
                output_files=self.list_workspace_files(),
                file_operations=list(workspace_tracker.file_operations),
                saved_skill=None,
                delivery_zip=None,
                verification_passed=None,
                optimization_rounds_used=optimization_rounds_used,
                deep_optimization=workflow_options.deep_optimization,
                local_python_tool_enabled=(workflow_options.local_python_tool_enabled),
                max_deep_optimization_rounds=(
                    workflow_options.max_deep_optimization_rounds
                ),
                immediate_output_requested=controller.immediate_output_requested,
            )
            return AgentRunResult(
                prompt=prompt,
                reply=planner_text,
                changed_files=workspace_tracker.changed_files,
                output_files=self.list_workspace_files(),
                file_operations=list(workspace_tracker.file_operations),
                saved_skill=None,
                delivery_zip=None,
                verification_passed=None,
                optimization_rounds_used=optimization_rounds_used,
                deep_optimization=workflow_options.deep_optimization,
                local_python_tool_enabled=(workflow_options.local_python_tool_enabled),
                max_deep_optimization_rounds=(
                    workflow_options.max_deep_optimization_rounds
                ),
                immediate_output_requested=controller.immediate_output_requested,
            )

        if not controller.immediate_output_requested:
            researcher_prompt = build_researcher_prompt(
                planner_text=self._trim_for_prompt(
                    planner_text,
                    max_chars=24_000,
                    label="planner artifact",
                ),
                verifier_feedback_text=None,
                user_prompt=effective_prompt,
                deep_optimization_round=round_index,
                detail_focus_enabled=workflow_options.deep_optimization,
            )
            researcher_result = await self._invoke_stage(
                agent=agent,
                stage_name="Researcher",
                stage_prompt=researcher_prompt,
                run_config=run_config,
                progress_callback=progress_callback,
                workspace_tracker=workspace_tracker,
                controller=controller,
            )
            researcher_text = (researcher_result.reply or "").strip()
            self._write_stage_artifact(latest_researcher_path, researcher_text)
            await cache_research_for_round(
                round_number=round_index,
                researcher_prompt=effective_prompt,
                researcher_text=researcher_text,
            )
            if await register_stage_result(researcher_result):
                finalization_reason = controller.reason or "immediate_output_requested"

        if not controller.immediate_output_requested:
            await self._emit_progress(
                progress_callback,
                event_type="status",
                message=(
                    "Builder 当前采用本地 Python 工具模式，可按需运行受控的 Python / "
                    "pytest 检查。"
                    if workflow_options.local_python_tool_enabled
                    else (
                        "Builder 当前采用静态代码审查模式，"
                        "如需检测程序将直接基于代码快照分析逻辑与语法错误，"
                        "不运行本地测试程序。"
                    )
                ),
                current_phase="Builder",
            )
            builder_prompt = build_builder_prompt(
                planner_text=self._trim_for_prompt(
                    planner_text,
                    max_chars=24_000,
                    label="planner artifact",
                ),
                researcher_text=self._trim_for_prompt(
                    researcher_text,
                    max_chars=28_000,
                    label="research artifact",
                ),
                verifier_feedback_text=None,
                workspace_snapshot_text=(
                    self._build_workspace_snapshot_text(
                        max_total_chars=_BUILDER_REVIEW_MAX_TOTAL_CHARS,
                        max_file_chars=_BUILDER_REVIEW_MAX_FILE_CHARS,
                        max_files=_BUILDER_REVIEW_MAX_FILES,
                    )
                    if not workflow_options.local_python_tool_enabled
                    else None
                ),
                local_python_tool_enabled=(workflow_options.local_python_tool_enabled),
                round_index=round_index,
                user_prompt=effective_prompt,
            )
            builder_result = await self._invoke_stage(
                agent=agent,
                stage_name="Builder",
                stage_prompt=builder_prompt,
                run_config=run_config,
                progress_callback=progress_callback,
                workspace_tracker=workspace_tracker,
                controller=controller,
            )
            builder_text = (builder_result.reply or "").strip()
            self._write_stage_artifact(latest_builder_path, builder_text)
            if await register_stage_result(builder_result):
                finalization_reason = controller.reason or "immediate_output_requested"

        if not controller.immediate_output_requested:
            (
                verifier_result,
                verification_report,
                verifier_text,
            ) = await self._run_verifier_stage(
                agent=agent,
                run_config=run_config,
                planner_text=planner_text,
                researcher_text=researcher_text,
                builder_text=builder_text,
                verifier_path=latest_verifier_path,
                round_index=round_index,
                user_prompt=effective_prompt,
                progress_callback=progress_callback,
                workspace_tracker=workspace_tracker,
                detail_review_enabled=detail_review_enabled,
                controller=controller,
                local_python_tool_enabled=(workflow_options.local_python_tool_enabled),
            )
            if await register_stage_result(verifier_result):
                finalization_reason = controller.reason or "immediate_output_requested"

        while (
            verification_report is not None
            and not self._verification_passed_for_mode(
                verification_report,
                detail_review_enabled=detail_review_enabled,
            )
            and not controller.immediate_output_requested
        ):
            if (
                workflow_options.deep_optimization
                and optimization_rounds_used
                < workflow_options.max_deep_optimization_rounds
            ):
                optimization_rounds_used += 1
                round_index += 1
                LOGGER.info(
                    "Triggering deep optimization round %s of %s",
                    optimization_rounds_used,
                    workflow_options.max_deep_optimization_rounds,
                )
                await self._emit_progress(
                    progress_callback,
                    event_type="status",
                    message=(
                        "Verifier 发现细节未完成或存在缺口，已回传到 "
                        f"Researcher，开始第 {optimization_rounds_used} 次深度优化…"
                    ),
                    current_phase="Researcher",
                    optimization_rounds_used=optimization_rounds_used,
                    deep_optimization=workflow_options.deep_optimization,
                    max_deep_optimization_rounds=(
                        workflow_options.max_deep_optimization_rounds
                    ),
                )

                latest_researcher_path = artifacts.researcher_path(round_index)
                researcher_prompt = build_researcher_prompt(
                    planner_text=self._trim_for_prompt(
                        planner_text,
                        max_chars=24_000,
                        label="planner artifact",
                    ),
                    verifier_feedback_text=self._trim_for_prompt(
                        verifier_text
                        or self._format_verifier_feedback_summary(verification_report),
                        max_chars=18_000,
                        label="verifier feedback",
                    ),
                    user_prompt=effective_prompt,
                    deep_optimization_round=round_index,
                    detail_focus_enabled=True,
                )
                researcher_result = await self._invoke_stage(
                    agent=agent,
                    stage_name="Researcher",
                    stage_prompt=researcher_prompt,
                    run_config=run_config,
                    progress_callback=progress_callback,
                    workspace_tracker=workspace_tracker,
                    controller=controller,
                )
                researcher_text = (researcher_result.reply or "").strip()
                self._write_stage_artifact(latest_researcher_path, researcher_text)
                await cache_research_for_round(
                    round_number=round_index,
                    researcher_prompt=effective_prompt,
                    researcher_text=researcher_text,
                )
                if await register_stage_result(researcher_result):
                    finalization_reason = (
                        controller.reason or "immediate_output_requested"
                    )
                    break

                latest_builder_path = artifacts.builder_path(round_index)
                await self._emit_progress(
                    progress_callback,
                    event_type="status",
                    message=(
                        "Builder 当前采用本地 Python 工具模式，"
                        "可按需运行受控的 Python / pytest 检查。"
                        if workflow_options.local_python_tool_enabled
                        else (
                            "Builder 当前采用静态代码审查模式，"
                            "如需检测程序将直接基于代码快照分析逻辑与语法错误，"
                            "不运行本地测试程序。"
                        )
                    ),
                    current_phase="Builder",
                )
                builder_prompt = build_builder_prompt(
                    planner_text=self._trim_for_prompt(
                        planner_text,
                        max_chars=24_000,
                        label="planner artifact",
                    ),
                    researcher_text=self._trim_for_prompt(
                        researcher_text,
                        max_chars=28_000,
                        label="research artifact",
                    ),
                    verifier_feedback_text=self._trim_for_prompt(
                        verifier_text,
                        max_chars=18_000,
                        label="verifier feedback",
                    ),
                    workspace_snapshot_text=(
                        self._build_workspace_snapshot_text(
                            max_total_chars=_BUILDER_REVIEW_MAX_TOTAL_CHARS,
                            max_file_chars=_BUILDER_REVIEW_MAX_FILE_CHARS,
                            max_files=_BUILDER_REVIEW_MAX_FILES,
                        )
                        if not workflow_options.local_python_tool_enabled
                        else None
                    ),
                    local_python_tool_enabled=(
                        workflow_options.local_python_tool_enabled
                    ),
                    round_index=round_index,
                    user_prompt=effective_prompt,
                )
                builder_result = await self._invoke_stage(
                    agent=agent,
                    stage_name="Builder",
                    stage_prompt=builder_prompt,
                    run_config=run_config,
                    progress_callback=progress_callback,
                    workspace_tracker=workspace_tracker,
                    controller=controller,
                )
                builder_text = (builder_result.reply or "").strip()
                self._write_stage_artifact(latest_builder_path, builder_text)
                if await register_stage_result(builder_result):
                    finalization_reason = (
                        controller.reason or "immediate_output_requested"
                    )
                    break

                latest_verifier_path = artifacts.verifier_path(round_index)
                (
                    verifier_result,
                    verification_report,
                    verifier_text,
                ) = await self._run_verifier_stage(
                    agent=agent,
                    run_config=run_config,
                    planner_text=planner_text,
                    researcher_text=researcher_text,
                    builder_text=builder_text,
                    verifier_path=latest_verifier_path,
                    round_index=round_index,
                    user_prompt=effective_prompt,
                    progress_callback=progress_callback,
                    workspace_tracker=workspace_tracker,
                    detail_review_enabled=True,
                    controller=controller,
                    local_python_tool_enabled=(
                        workflow_options.local_python_tool_enabled
                    ),
                )
                if await register_stage_result(verifier_result):
                    finalization_reason = (
                        controller.reason or "immediate_output_requested"
                    )
                    break
                continue

            if (
                not workflow_options.deep_optimization
                and verification_report.contains_errors
                and repair_rounds_used < workflow_options.standard_builder_repair_rounds
            ):
                repair_rounds_used += 1
                round_index += 1
                LOGGER.info(
                    "Triggering builder repair round %s of %s",
                    repair_rounds_used,
                    workflow_options.standard_builder_repair_rounds,
                )
                await self._emit_progress(
                    progress_callback,
                    event_type="status",
                    message="Verifier 发现代码或项目错误，正在回传到 Builder 修复…",
                    current_phase="Builder",
                    optimization_rounds_used=optimization_rounds_used,
                    deep_optimization=workflow_options.deep_optimization,
                    max_deep_optimization_rounds=(
                        workflow_options.max_deep_optimization_rounds
                    ),
                )

                latest_builder_path = artifacts.builder_path(round_index)
                await self._emit_progress(
                    progress_callback,
                    event_type="status",
                    message=(
                        "Builder 当前采用本地 Python 工具模式，"
                        "可按需运行受控的 Python / pytest 检查。"
                        if workflow_options.local_python_tool_enabled
                        else (
                            "Builder 当前采用静态代码审查模式，"
                            "如需检测程序将直接基于代码快照分析逻辑与语法错误，"
                            "不运行本地测试程序。"
                        )
                    ),
                    current_phase="Builder",
                )
                builder_prompt = build_builder_prompt(
                    planner_text=self._trim_for_prompt(
                        planner_text,
                        max_chars=24_000,
                        label="planner artifact",
                    ),
                    researcher_text=self._trim_for_prompt(
                        researcher_text,
                        max_chars=28_000,
                        label="research artifact",
                    ),
                    verifier_feedback_text=self._trim_for_prompt(
                        verifier_text,
                        max_chars=18_000,
                        label="verifier feedback",
                    ),
                    workspace_snapshot_text=(
                        self._build_workspace_snapshot_text(
                            max_total_chars=_BUILDER_REVIEW_MAX_TOTAL_CHARS,
                            max_file_chars=_BUILDER_REVIEW_MAX_FILE_CHARS,
                            max_files=_BUILDER_REVIEW_MAX_FILES,
                        )
                        if not workflow_options.local_python_tool_enabled
                        else None
                    ),
                    local_python_tool_enabled=(
                        workflow_options.local_python_tool_enabled
                    ),
                    round_index=round_index,
                    user_prompt=effective_prompt,
                )
                builder_result = await self._invoke_stage(
                    agent=agent,
                    stage_name="Builder",
                    stage_prompt=builder_prompt,
                    run_config=run_config,
                    progress_callback=progress_callback,
                    workspace_tracker=workspace_tracker,
                    controller=controller,
                )
                builder_text = (builder_result.reply or "").strip()
                self._write_stage_artifact(latest_builder_path, builder_text)
                if await register_stage_result(builder_result):
                    finalization_reason = (
                        controller.reason or "immediate_output_requested"
                    )
                    break

                latest_verifier_path = artifacts.verifier_path(round_index)
                (
                    verifier_result,
                    verification_report,
                    verifier_text,
                ) = await self._run_verifier_stage(
                    agent=agent,
                    run_config=run_config,
                    planner_text=planner_text,
                    researcher_text=researcher_text,
                    builder_text=builder_text,
                    verifier_path=latest_verifier_path,
                    round_index=round_index,
                    user_prompt=effective_prompt,
                    progress_callback=progress_callback,
                    workspace_tracker=workspace_tracker,
                    detail_review_enabled=False,
                    controller=controller,
                    local_python_tool_enabled=(
                        workflow_options.local_python_tool_enabled
                    ),
                )
                if await register_stage_result(verifier_result):
                    finalization_reason = (
                        controller.reason or "immediate_output_requested"
                    )
                    break
                continue

            break

        if controller.immediate_output_requested:
            finalization_reason = controller.reason or "immediate_output_requested"
            await self._emit_progress(
                progress_callback,
                event_type="status",
                message="已收到立即输出请求，开始最终整合与打包…",
                current_phase="Packager",
                optimization_rounds_used=optimization_rounds_used,
                deep_optimization=workflow_options.deep_optimization,
                max_deep_optimization_rounds=(
                    workflow_options.max_deep_optimization_rounds
                ),
            )
        else:
            await self._emit_progress(
                progress_callback,
                event_type="status",
                message="正在进入最终整合与打包阶段…",
                current_phase="Packager",
                optimization_rounds_used=optimization_rounds_used,
                deep_optimization=workflow_options.deep_optimization,
                max_deep_optimization_rounds=(
                    workflow_options.max_deep_optimization_rounds
                ),
            )

        output_files = self.list_workspace_files()
        delivery_zip = self._package_workspace_archive(artifacts)
        verification_passed = self._verification_passed_for_mode(
            verification_report,
            detail_review_enabled=detail_review_enabled,
        )
        packager_prompt = build_packager_prompt(
            planner_text=self._trim_for_prompt(
                planner_text,
                max_chars=24_000,
                label="planner artifact",
            ),
            researcher_text=self._trim_for_prompt(
                researcher_text,
                max_chars=28_000,
                label="research artifact",
            ),
            builder_text=self._trim_for_prompt(
                builder_text,
                max_chars=20_000,
                label="builder artifact",
            ),
            verifier_text=self._trim_for_prompt(
                verifier_text,
                max_chars=18_000,
                label="verifier artifact",
            ),
            output_files=output_files,
            deep_optimization_enabled=workflow_options.deep_optimization,
            optimization_rounds_used=optimization_rounds_used,
            verification_passed=verification_passed,
            user_prompt=effective_prompt,
            finalization_reason=finalization_reason,
        )
        packager_result = await self._invoke_stage(
            agent=agent,
            stage_name="Packager",
            stage_prompt=packager_prompt,
            run_config=run_config,
            progress_callback=progress_callback,
            workspace_tracker=workspace_tracker,
            controller=None,
        )
        self._write_stage_artifact(
            artifacts.packager_path(),
            (packager_result.reply or "").strip(),
        )
        if packager_result.reply:
            stage_replies.append(packager_result.reply)
        self._sync_runtime_writeback()

        await self._emit_progress(
            progress_callback,
            event_type="status",
            message="正在整理最终交付物…",
            current_phase="Packager",
            optimization_rounds_used=optimization_rounds_used,
            deep_optimization=workflow_options.deep_optimization,
            max_deep_optimization_rounds=(
                workflow_options.max_deep_optimization_rounds
            ),
        )
        self.skill_manager.relocate_workspace_skills(
            workspace_root=self.config.workspace_dir,
        )
        after_skills = set(self.skill_manager.list_skill_names())
        reply = self._select_packager_reply(
            packager_result,
            verification_report,
            stage_replies,
        )
        changed_files = workspace_tracker.changed_files
        self._ensure_workspace_changes_present(changed_files)
        output_files = self.list_workspace_files()
        saved_skill: str | None = None
        if self._skills_dir_is_workspace_local():
            new_skills = sorted(after_skills - before_skills)
            if new_skills:
                saved_skill = (
                    (self.config.skills_dir / new_skills[0])
                    .relative_to(self.config.project_root)
                    .as_posix()
                )
            elif self.config.skills.auto_save_on_success and changed_files:
                saved_path = self.skill_manager.save_skill_from_run(
                    prompt=prompt,
                    summary=reply,
                    changed_files=changed_files,
                )
                saved_skill = saved_path.relative_to(
                    self.config.project_root
                ).as_posix()

        self.sync_workspace_skills_back_to_source()
        self._sync_runtime_writeback()
        await self._emit_progress(
            progress_callback,
            event_type="final",
            message="任务执行完成。",
            current_phase="Packager",
            reply=reply,
            changed_files=changed_files,
            output_files=output_files,
            file_operations=list(workspace_tracker.file_operations),
            saved_skill=saved_skill,
            delivery_zip=delivery_zip,
            verification_passed=verification_passed,
            optimization_rounds_used=optimization_rounds_used,
            deep_optimization=workflow_options.deep_optimization,
            local_python_tool_enabled=(workflow_options.local_python_tool_enabled),
            max_deep_optimization_rounds=(
                workflow_options.max_deep_optimization_rounds
            ),
            immediate_output_requested=controller.immediate_output_requested,
        )
        return AgentRunResult(
            prompt=prompt,
            reply=reply,
            changed_files=changed_files,
            output_files=output_files,
            file_operations=list(workspace_tracker.file_operations),
            saved_skill=saved_skill,
            delivery_zip=delivery_zip,
            verification_passed=verification_passed,
            optimization_rounds_used=optimization_rounds_used,
            deep_optimization=workflow_options.deep_optimization,
            local_python_tool_enabled=(workflow_options.local_python_tool_enabled),
            max_deep_optimization_rounds=(
                workflow_options.max_deep_optimization_rounds
            ),
            immediate_output_requested=controller.immediate_output_requested,
        )
