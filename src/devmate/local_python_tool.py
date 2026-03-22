"""Local Python execution helper for DevMate agent stages."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Literal

ExecutionMode = Literal["module", "script", "code"]

_MAX_TIMEOUT_SECONDS = 300
_MAX_OUTPUT_CHARS = 20_000
_MAX_INLINE_CODE_CHARS = 40_000
_MAX_ARGS = 50
_MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")
_BOOTSTRAP_FILENAME = "guarded_entrypoint.py"
_LOCAL_PYTHON_BOOTSTRAP = (
    dedent(
        """
    from __future__ import annotations

    import asyncio
    import builtins
    import io
    import json
    import os
    import runpy
    import subprocess
    import sys
    from pathlib import Path

    _WORKSPACE_ROOT = Path(
        os.environ["DEVMATE_GUARD_WORKSPACE_ROOT"]
    ).resolve()
    _EXECUTION_MODE = os.environ["DEVMATE_GUARD_EXECUTION_MODE"]
    _TARGET = os.environ["DEVMATE_GUARD_TARGET"]
    _ARGS = json.loads(os.environ.get("DEVMATE_GUARD_ARGS_JSON", "[]"))

    def _resolve_candidate(path_like):
        if isinstance(path_like, int):
            return None
        raw_value = os.fspath(path_like)
        candidate = Path(raw_value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path.cwd() / candidate).resolve()

    def _ensure_write_allowed(path_like):
        candidate = _resolve_candidate(path_like)
        if candidate is None:
            return
        if (
            candidate != _WORKSPACE_ROOT
            and _WORKSPACE_ROOT not in candidate.parents
        ):
            raise PermissionError(
                "Write access is restricted to the workspace: "
                f"{candidate}"
            )

    def _mode_allows_write(mode):
        return any(marker in str(mode) for marker in ("w", "a", "x", "+"))

    def _flags_allow_write(flags):
        write_flags = (
            os.O_WRONLY
            | os.O_RDWR
            | os.O_CREAT
            | os.O_TRUNC
            | os.O_APPEND
        )
        return bool(flags & write_flags)

    _original_open = builtins.open
    _original_io_open = io.open
    _original_os_open = os.open
    _original_os_mkdir = os.mkdir
    _original_os_unlink = os.unlink
    _original_os_remove = os.remove
    _original_os_rename = os.rename
    _original_os_replace = os.replace
    _original_os_rmdir = os.rmdir
    _original_os_utime = os.utime
    _original_os_chmod = os.chmod
    _original_os_symlink = os.symlink
    _original_os_link = os.link

    def _guarded_open(file, mode="r", *args, **kwargs):
        if _mode_allows_write(mode):
            _ensure_write_allowed(file)
        return _original_open(file, mode, *args, **kwargs)

    def _guarded_io_open(file, mode="r", *args, **kwargs):
        if _mode_allows_write(mode):
            _ensure_write_allowed(file)
        return _original_io_open(file, mode, *args, **kwargs)

    def _guarded_os_open(path, flags, mode=0o777, *args, **kwargs):
        if _flags_allow_write(flags):
            _ensure_write_allowed(path)
        return _original_os_open(path, flags, mode, *args, **kwargs)

    def _guarded_mkdir(path, *args, **kwargs):
        _ensure_write_allowed(path)
        return _original_os_mkdir(path, *args, **kwargs)

    def _guarded_unlink(path, *args, **kwargs):
        _ensure_write_allowed(path)
        return _original_os_unlink(path, *args, **kwargs)

    def _guarded_remove(path, *args, **kwargs):
        _ensure_write_allowed(path)
        return _original_os_remove(path, *args, **kwargs)

    def _guarded_rename(src, dst, *args, **kwargs):
        _ensure_write_allowed(src)
        _ensure_write_allowed(dst)
        return _original_os_rename(src, dst, *args, **kwargs)

    def _guarded_replace(src, dst, *args, **kwargs):
        _ensure_write_allowed(src)
        _ensure_write_allowed(dst)
        return _original_os_replace(src, dst, *args, **kwargs)

    def _guarded_rmdir(path, *args, **kwargs):
        _ensure_write_allowed(path)
        return _original_os_rmdir(path, *args, **kwargs)

    def _guarded_utime(path, *args, **kwargs):
        _ensure_write_allowed(path)
        return _original_os_utime(path, *args, **kwargs)

    def _guarded_chmod(path, *args, **kwargs):
        _ensure_write_allowed(path)
        return _original_os_chmod(path, *args, **kwargs)

    def _guarded_symlink(src, dst, *args, **kwargs):
        _ensure_write_allowed(dst)
        return _original_os_symlink(src, dst, *args, **kwargs)

    def _guarded_link(src, dst, *args, **kwargs):
        _ensure_write_allowed(dst)
        return _original_os_link(src, dst, *args, **kwargs)

    def _disallow_subprocess(*args, **kwargs):
        raise PermissionError(
            "Spawning child processes is not allowed in invoke_local_python"
        )

    def _install_guards():
        builtins.open = _guarded_open
        io.open = _guarded_io_open
        os.open = _guarded_os_open
        os.mkdir = _guarded_mkdir
        os.unlink = _guarded_unlink
        os.remove = _guarded_remove
        os.rename = _guarded_rename
        os.replace = _guarded_replace
        os.rmdir = _guarded_rmdir
        os.utime = _guarded_utime
        os.chmod = _guarded_chmod
        os.symlink = _guarded_symlink
        os.link = _guarded_link
        os.system = _disallow_subprocess
        subprocess.Popen = _disallow_subprocess
        subprocess.run = _disallow_subprocess
        subprocess.call = _disallow_subprocess
        subprocess.check_call = _disallow_subprocess
        subprocess.check_output = _disallow_subprocess
        asyncio.create_subprocess_exec = _disallow_subprocess
        asyncio.create_subprocess_shell = _disallow_subprocess

        for name in (
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
        ):
            if hasattr(os, name):
                setattr(os, name, _disallow_subprocess)
        if hasattr(os, "startfile"):
            os.startfile = _disallow_subprocess

    def _normalize_exit_code(value):
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        print(str(value), file=sys.stderr)
        return 1

    def _run_target():
        if _EXECUTION_MODE == "module":
            sys.argv = [_TARGET, *_ARGS]
            runpy.run_module(_TARGET, run_name="__main__", alter_sys=True)
            return 0

        if _EXECUTION_MODE == "script":
            sys.argv = [_TARGET, *_ARGS]
            runpy.run_path(_TARGET, run_name="__main__")
            return 0

        if _EXECUTION_MODE == "code":
            sys.argv = ["-c", *_ARGS]
            namespace = {"__name__": "__main__", "__file__": "<inline>"}
            exec(compile(_TARGET, "<inline>", "exec"), namespace)
            return 0

        raise ValueError(f"Unsupported execution mode: {_EXECUTION_MODE}")

    _install_guards()
    try:
        _EXIT_CODE = _run_target()
    except SystemExit as exc:
        _EXIT_CODE = _normalize_exit_code(exc.code)
    sys.exit(_EXIT_CODE)
    """
    ).strip()
    + "\n"
)


class LocalPythonExecutionError(ValueError):
    """Raised when the requested local Python execution is invalid."""


@dataclass(slots=True)
class LocalPythonExecutionResult:
    """Structured result returned to the agent after local execution."""

    ok: bool
    execution_mode: ExecutionMode
    purpose: str
    command: list[str]
    working_directory: str
    target: str
    script_path: str | None
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str

    def to_json(self) -> str:
        """Serialize the execution result as human-readable JSON."""

        return json.dumps(
            {
                "ok": self.ok,
                "execution_mode": self.execution_mode,
                "purpose": self.purpose,
                "command": self.command,
                "working_directory": self.working_directory,
                "target": self.target,
                "script_path": self.script_path,
                "exit_code": self.exit_code,
                "timed_out": self.timed_out,
                "duration_seconds": round(self.duration_seconds, 3),
                "stdout": self.stdout,
                "stderr": self.stderr,
            },
            ensure_ascii=False,
            indent=2,
        )


def _ensure_within_root(root: Path, candidate: Path) -> None:
    """Ensure the resolved path stays inside the allowed workspace root."""

    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    if (
        candidate_resolved != root_resolved
        and root_resolved not in candidate_resolved.parents
    ):
        raise LocalPythonExecutionError(
            f"Path is outside the workspace root: {candidate}"
        )


def _trim_output(text: str) -> str:
    """Trim long command output before sending it back to the agent."""

    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    trimmed = text[:_MAX_OUTPUT_CHARS].rstrip()
    return (
        f"{trimmed}\n\n"
        f"[... output truncated to the first {_MAX_OUTPUT_CHARS} characters ...]"
    )


def _normalize_args(args: list[str] | None) -> list[str]:
    """Normalize CLI arguments passed to the Python subprocess."""

    normalized = [str(item) for item in (args or [])]
    if len(normalized) > _MAX_ARGS:
        raise LocalPythonExecutionError(
            f"Too many Python arguments requested: {len(normalized)} > {_MAX_ARGS}"
        )
    return normalized


def _resolve_working_directory(workspace_root: Path, working_directory: str) -> Path:
    """Resolve and validate the requested working directory."""

    raw_value = (working_directory or ".").strip() or "."
    candidate = Path(raw_value)
    resolved = (
        candidate.expanduser().resolve()
        if candidate.is_absolute()
        else (workspace_root / candidate).resolve()
    )
    _ensure_within_root(workspace_root, resolved)
    if not resolved.exists() or not resolved.is_dir():
        raise LocalPythonExecutionError(
            f"Working directory does not exist inside the workspace: {raw_value}"
        )
    return resolved


def _runtime_dir(workspace_root: Path) -> Path:
    """Return the workspace-local runtime directory for Python execution."""

    return workspace_root / ".devmate_runtime" / "local_python"


def _ensure_bootstrap_script(workspace_root: Path) -> Path:
    """Write the guarded bootstrap entrypoint inside the workspace."""

    runtime_dir = _runtime_dir(workspace_root)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_path = runtime_dir / _BOOTSTRAP_FILENAME
    if bootstrap_path.exists():
        current_text = bootstrap_path.read_text(encoding="utf-8")
        if current_text == _LOCAL_PYTHON_BOOTSTRAP:
            return bootstrap_path
    bootstrap_path.write_text(_LOCAL_PYTHON_BOOTSTRAP, encoding="utf-8")
    return bootstrap_path


async def execute_local_python(
    *,
    workspace_root: Path,
    execution_mode: ExecutionMode,
    target: str,
    args: list[str] | None = None,
    working_directory: str = ".",
    timeout_seconds: int = 120,
    purpose: str = "",
) -> LocalPythonExecutionResult:
    """Run a Python module, script, or inline code in a subprocess."""

    workspace_root = workspace_root.resolve()
    cwd = _resolve_working_directory(workspace_root, working_directory)
    normalized_args = _normalize_args(args)
    normalized_timeout = max(1, min(int(timeout_seconds), _MAX_TIMEOUT_SECONDS))
    normalized_target = str(target or "").strip()
    if not normalized_target:
        raise LocalPythonExecutionError("Python execution target must not be empty")

    bootstrap_script = _ensure_bootstrap_script(workspace_root)
    tmp_root = _runtime_dir(workspace_root) / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    home_root = _runtime_dir(workspace_root) / "home"
    home_root.mkdir(parents=True, exist_ok=True)
    cache_root = home_root / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    script_path_for_result: str | None = None
    bootstrap_target = normalized_target

    if execution_mode == "module":
        if not _MODULE_NAME_PATTERN.fullmatch(normalized_target):
            raise LocalPythonExecutionError(
                "Module mode only accepts dotted Python module names, for example "
                '"pytest" or "package.module".'
            )
        command = [sys.executable, str(bootstrap_script)]
    elif execution_mode == "script":
        requested_script = Path(normalized_target)
        script_path = (
            requested_script.expanduser().resolve()
            if requested_script.is_absolute()
            else (workspace_root / requested_script).resolve()
        )
        _ensure_within_root(workspace_root, script_path)
        if not script_path.exists() or not script_path.is_file():
            raise LocalPythonExecutionError(
                f"Script does not exist inside the workspace: {normalized_target}"
            )
        bootstrap_target = str(script_path)
        command = [sys.executable, str(bootstrap_script)]
        script_path_for_result = script_path.relative_to(workspace_root).as_posix()
    elif execution_mode == "code":
        if len(normalized_target) > _MAX_INLINE_CODE_CHARS:
            raise LocalPythonExecutionError(
                "Inline Python code is too long for the local tool. "
                f"Maximum length is {_MAX_INLINE_CODE_CHARS} characters."
            )
        command = [sys.executable, str(bootstrap_script)]
    else:  # pragma: no cover - guarded by type hints and runtime validation
        raise LocalPythonExecutionError(
            f"Unsupported local Python execution mode: {execution_mode}"
        )

    env = os.environ.copy()
    workspace_python_path = str(workspace_root)
    current_python_path = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{workspace_python_path}{os.pathsep}{current_python_path}"
        if current_python_path
        else workspace_python_path
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(tmp_root)
    env["TMP"] = str(tmp_root)
    env["TEMP"] = str(tmp_root)
    env["HOME"] = str(home_root)
    env["XDG_CACHE_HOME"] = str(cache_root)
    env["DEVMATE_GUARD_WORKSPACE_ROOT"] = str(workspace_root)
    env["DEVMATE_GUARD_EXECUTION_MODE"] = execution_mode
    env["DEVMATE_GUARD_TARGET"] = bootstrap_target
    env["DEVMATE_GUARD_ARGS_JSON"] = json.dumps(
        normalized_args,
        ensure_ascii=False,
    )

    started_at = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timed_out = False
    stdout_bytes = b""
    stderr_bytes = b""
    exit_code: int | None = None

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=normalized_timeout,
        )
        exit_code = process.returncode
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()

    duration_seconds = time.perf_counter() - started_at
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    return LocalPythonExecutionResult(
        ok=not timed_out and exit_code == 0,
        execution_mode=execution_mode,
        purpose=purpose.strip(),
        command=command,
        working_directory=cwd.relative_to(workspace_root).as_posix(),
        target=normalized_target,
        script_path=script_path_for_result,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_seconds=duration_seconds,
        stdout=_trim_output(stdout_text),
        stderr=_trim_output(stderr_text),
    )
