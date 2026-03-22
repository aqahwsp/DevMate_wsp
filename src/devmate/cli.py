"""CLI entrypoints for DevMate."""

from __future__ import annotations

import asyncio
import os
import select
import sys
import time
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
import uvicorn

from devmate.config import load_config
from devmate.logging_config import configure_logging
from devmate.mcp_server import create_mcp_app
from devmate.runtime import DevMateRuntime, RunController
from devmate.webapp import create_app

app = typer.Typer(help="DevMate - AI coding assistant")


@app.command()
def ingest(
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to config.toml"),
    ] = None,
) -> None:
    """Ingest local markdown and text files into the vector store."""

    config = load_config(config_path)
    configure_logging(config.app.log_level)
    runtime = DevMateRuntime(config)
    indexed_chunks = runtime.knowledge_base.ingest(rebuild=True)
    typer.echo(f"Indexed {indexed_chunks} chunk(s) into the knowledge base.")


def _build_terminal_progress_callback():
    """Create a progress callback suitable for terminal output."""

    last_status_line = ""
    last_operation_count = 0

    async def on_progress(event: dict[str, object]) -> None:
        nonlocal last_status_line, last_operation_count

        event_type = str(event.get("type", ""))
        message = str(event.get("message", ""))
        current_phase = str(event.get("current_phase", "") or "").strip()

        if event_type == "status":
            line = f"[{current_phase}] {message}" if current_phase else message
            if line and line != last_status_line:
                typer.echo(line)
                last_status_line = line
            return

        if event_type == "workspace_changes":
            operations = list(event.get("file_operations", []))
            new_operations = operations[last_operation_count:]
            for record in new_operations:
                typer.echo(f"[文件变更] {record}")
            last_operation_count = len(operations)

    return on_progress


async def _monitor_immediate_output(controller: RunController) -> None:
    """Watch stdin for the /finish command during an active run."""

    if not sys.stdin.isatty():
        return

    while not controller.immediate_output_requested:
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            raw_line = sys.stdin.readline()
            if not raw_line:
                return
            text = raw_line.strip().lower()
            if text in {"/finish", "finish", "/f"}:
                was_new = controller.request_immediate_output(
                    "用户在 CLI 中请求立即输出"
                )
                if was_new:
                    typer.echo("已收到立即输出请求，将尽快切换到 Packager。")
                return
            if text:
                typer.echo("运行中仅支持输入 /finish 来立即输出当前结果。")
        await asyncio.sleep(0)


def _stage_cli_initial_uploads(
    runtime: DevMateRuntime,
    local_paths: list[str] | None,
) -> list[str]:
    """Copy user-provided files into the CLI initial upload directory."""

    staged_paths: list[str] = []
    for raw_path in local_paths or []:
        candidate = Path(raw_path).expanduser()
        staged_path = runtime.copy_initial_file_to_uploads(candidate, "cli")
        staged_paths.append(staged_path)
    return staged_paths


def _result_to_payload(result: object) -> dict[str, object]:
    """Normalize a runtime result into a JSON-safe payload."""

    if hasattr(result, "model_dump"):
        return dict(result.model_dump())
    if is_dataclass(result):
        return dict(asdict(result))
    if isinstance(result, dict):
        return dict(result)
    if hasattr(result, "__dict__"):
        return dict(vars(result))
    return {"reply": str(result)}


def _build_cli_run_payload(
    runtime: DevMateRuntime,
    run_id: str,
    prompt: str,
    *,
    deep_optimization: bool | None,
    local_python_tool_enabled: bool | None,
    max_deep_optimization_rounds: int | None,
    include_initial_files: bool,
    initial_file_paths: list[str] | None,
) -> dict[str, object]:
    """Create the base payload stored for a CLI run."""

    workflow = runtime.config.workflow
    effective_deep_optimization = (
        workflow.deep_optimization_default
        if deep_optimization is None
        else deep_optimization
    )
    effective_local_python_tool = (
        workflow.local_python_tool_default
        if local_python_tool_enabled is None
        else local_python_tool_enabled
    )
    effective_max_rounds = (
        workflow.max_deep_optimization_rounds
        if max_deep_optimization_rounds is None
        else max_deep_optimization_rounds
    )

    return {
        "run_id": run_id,
        "prompt": prompt,
        "status": "queued",
        "stage": "",
        "current_phase": "",
        "reply": "",
        "changed_files": [],
        "output_files": [],
        "file_operations": [],
        "saved_skill": None,
        "error": None,
        "delivery_zip": None,
        "verification_passed": None,
        "optimization_rounds_used": 0,
        "deep_optimization": bool(effective_deep_optimization),
        "local_python_tool_enabled": bool(effective_local_python_tool),
        "max_deep_optimization_rounds": int(effective_max_rounds),
        "include_initial_files": include_initial_files,
        "initial_file_paths": list(initial_file_paths or []),
        "immediate_output_requested": False,
        "updated_at": time.time(),
        "worker_pid": os.getpid(),
    }


def _persist_cli_run_payload(
    runtime: DevMateRuntime,
    run_id: str,
    payload: dict[str, object],
) -> None:
    """Persist a CLI run snapshot."""

    payload["updated_at"] = time.time()
    runtime.state_store.write_state("cli", run_id, payload)


def _build_cli_progress_callback(
    runtime: DevMateRuntime,
    run_id: str,
    base_payload: dict[str, object],
):
    """Create a CLI progress callback for terminal output and snapshots."""

    terminal_callback = _build_terminal_progress_callback()

    async def on_progress(event: dict[str, object]) -> None:
        await terminal_callback(event)

        payload = runtime.state_store.read_state("cli", run_id)
        if payload is None:
            payload = dict(base_payload)

        event_type = str(event.get("type", "") or "")
        message = str(event.get("message", "") or "")
        current_phase = str(event.get("current_phase", "") or "")

        if payload.get("status") not in {"completed", "error"}:
            payload["status"] = "running"

        payload["error"] = None

        if current_phase:
            payload["current_phase"] = current_phase

        if event_type == "status" and message:
            payload["stage"] = message

        if event_type == "reply" and message:
            payload["reply"] = message
            payload["stage"] = "正在接收大模型最新输出…"

        if event_type == "workspace_changes" and message:
            payload["stage"] = message

        if event_type == "final":
            if message:
                payload["stage"] = message
            if event.get("reply"):
                payload["reply"] = str(event["reply"])

        if "changed_files" in event:
            payload["changed_files"] = list(event.get("changed_files", []))

        if "output_files" in event:
            payload["output_files"] = list(event.get("output_files", []))

        if "file_operations" in event:
            payload["file_operations"] = list(event.get("file_operations", []))

        if "saved_skill" in event:
            payload["saved_skill"] = event.get("saved_skill")

        if "delivery_zip" in event:
            payload["delivery_zip"] = event.get("delivery_zip")

        if "verification_passed" in event:
            payload["verification_passed"] = event.get("verification_passed")

        if "optimization_rounds_used" in event:
            payload["optimization_rounds_used"] = event.get("optimization_rounds_used")

        if "immediate_output_requested" in event:
            payload["immediate_output_requested"] = event.get(
                "immediate_output_requested"
            )

        _persist_cli_run_payload(runtime, run_id, payload)

    return on_progress


async def _run_with_cli_controls(
    runtime: DevMateRuntime,
    prompt: str,
    *,
    deep_optimization: bool | None,
    local_python_tool_enabled: bool | None,
    max_deep_optimization_rounds: int | None,
    include_initial_files: bool,
    initial_file_paths: list[str] | None,
):
    """Execute a run with CLI progress and immediate-output controls."""

    runtime.state_store.clear_scope("cli")

    run_id = uuid4().hex
    controller = RunController()
    initial_payload = _build_cli_run_payload(
        runtime,
        run_id,
        prompt,
        deep_optimization=deep_optimization,
        local_python_tool_enabled=local_python_tool_enabled,
        max_deep_optimization_rounds=max_deep_optimization_rounds,
        include_initial_files=include_initial_files,
        initial_file_paths=initial_file_paths,
    )
    initial_payload["status"] = "queued"
    initial_payload["stage"] = "任务已提交，正在启动。"
    _persist_cli_run_payload(runtime, run_id, initial_payload)

    progress_callback = _build_cli_progress_callback(
        runtime,
        run_id,
        initial_payload,
    )
    monitor_task: asyncio.Task[None] | None = None

    if sys.stdin.isatty():
        typer.echo("运行中输入 /finish 并回车，可立即输出当前结果。")
        monitor_task = asyncio.create_task(_monitor_immediate_output(controller))

    try:
        result = await runtime.run(
            prompt,
            progress_callback=progress_callback,
            deep_optimization=deep_optimization,
            local_python_tool_enabled=local_python_tool_enabled,
            max_deep_optimization_rounds=max_deep_optimization_rounds,
            include_initial_files=include_initial_files,
            initial_file_paths=initial_file_paths,
            controller=controller,
        )
    except Exception as exc:
        payload = runtime.state_store.read_state("cli", run_id)
        if payload is None:
            payload = dict(initial_payload)
        payload.update(
            {
                "status": "error",
                "stage": "任务执行失败。",
                "error": str(exc),
                "immediate_output_requested": (controller.immediate_output_requested),
            }
        )
        _persist_cli_run_payload(runtime, run_id, payload)
        raise
    else:
        payload = runtime.state_store.read_state("cli", run_id)
        if payload is None:
            payload = dict(initial_payload)
        payload.update(_result_to_payload(result))
        payload["status"] = "completed"
        payload["stage"] = "任务执行完成。"
        payload["current_phase"] = "Packager"
        payload["error"] = None
        payload["immediate_output_requested"] = controller.immediate_output_requested
        _persist_cli_run_payload(runtime, run_id, payload)
        return result
    finally:
        runtime.state_store.clear_control("cli", run_id)
        if monitor_task is not None:
            monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await monitor_task


@app.command()
def chat(
    prompt: Annotated[
        str | None,
        typer.Option(
            help=("Single prompt to execute. Omit for interactive mode."),
        ),
    ] = None,
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to config.toml"),
    ] = None,
    deep_optimization: Annotated[
        bool | None,
        typer.Option(
            "--deep-optimization/--no-deep-optimization",
            help="Enable the Verifier -> Researcher deep optimization loop.",
        ),
    ] = None,
    local_python_tool_enabled: Annotated[
        bool,
        typer.Option(
            "--local-python-tool/--no-local-python-tool",
            help=(
                "Enable the local Python tool for Builder and Verifier. "
                "Disabled by default."
            ),
        ),
    ] = False,
    max_deep_optimization_rounds: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="Maximum number of deep optimization rounds.",
        ),
    ] = None,
    upload_initial_file: Annotated[
        list[str] | None,
        typer.Option(
            "--upload-initial-file",
            help=(
                "Copy a local file into workspace/data/uploads/cli "
                "before running. "
                "Can be used multiple times."
            ),
        ),
    ] = None,
    include_initial_files: Annotated[
        bool,
        typer.Option(
            "--include-initial-files/--no-include-initial-files",
            help=(
                "Inject staged files from workspace/data/uploads/cli "
                "into the current run."
            ),
        ),
    ] = False,
) -> None:
    """Run DevMate once or in an interactive loop."""

    config = load_config(config_path)
    configure_logging(config.app.log_level)
    runtime = DevMateRuntime(config)
    asyncio.run(runtime.prepare(rebuild_kb=False))

    staged_paths = _stage_cli_initial_uploads(runtime, upload_initial_file)
    if staged_paths:
        typer.echo("已上传初始文件到 workspace/data/uploads/cli:")
        for staged_path in staged_paths:
            typer.echo(f"- {staged_path}")

    initial_file_paths = []
    if include_initial_files:
        initial_file_paths = runtime.list_initial_upload_files("cli")

    if include_initial_files:
        if initial_file_paths:
            typer.echo("本次运行将带入以下已上传初始文件:")
            for staged_path in initial_file_paths:
                typer.echo(f"- {staged_path}")
        else:
            typer.echo(
                "本次已开启初始文件注入，但 workspace/data/uploads/cli 中暂无可用文件。"
            )

    def render_result(result: object) -> None:
        typer.echo(result.reply)
        if result.changed_files:
            typer.echo("Changed files:")
            for path in result.changed_files:
                typer.echo(f"- {path}")
        if result.delivery_zip:
            typer.echo(f"Delivery zip: {result.delivery_zip}")
        if result.verification_passed is not None:
            typer.echo(
                f"Verification passed: {'yes' if result.verification_passed else 'no'}"
            )
        else:
            typer.echo("Verification passed: unknown")
        typer.echo(
            "Local Python tool: "
            f"{'enabled' if result.local_python_tool_enabled else 'disabled'}"
        )
        typer.echo(
            "Deep optimization rounds used: "
            f"{result.optimization_rounds_used}/"
            f"{result.max_deep_optimization_rounds}"
        )
        if result.immediate_output_requested:
            typer.echo("Immediate output: requested")
        if result.saved_skill:
            typer.echo(f"Saved skill: {result.saved_skill}")

    if prompt:
        result = asyncio.run(
            _run_with_cli_controls(
                runtime,
                prompt,
                deep_optimization=deep_optimization,
                local_python_tool_enabled=local_python_tool_enabled,
                max_deep_optimization_rounds=max_deep_optimization_rounds,
                include_initial_files=include_initial_files,
                initial_file_paths=initial_file_paths,
            )
        )
        render_result(result)
        return

    typer.echo("Enter a prompt. Type 'exit' to quit.")
    while True:
        user_input = typer.prompt("devmate")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        result = asyncio.run(
            _run_with_cli_controls(
                runtime,
                user_input,
                deep_optimization=deep_optimization,
                local_python_tool_enabled=local_python_tool_enabled,
                max_deep_optimization_rounds=max_deep_optimization_rounds,
                include_initial_files=include_initial_files,
                initial_file_paths=initial_file_paths,
            )
        )
        render_result(result)


@app.command(name="web")
def web_server(
    host: Annotated[str, typer.Option(help="Bind host")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port")] = 8080,
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to config.toml"),
    ] = None,
) -> None:
    """Start the FastAPI web application."""

    web_app = create_app(config_path)
    uvicorn.run(web_app, host=host, port=port)


@app.command(name="mcp-search")
def mcp_search_server(
    host: Annotated[str, typer.Option(help="Bind host")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port")] = 8001,
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to config.toml"),
    ] = None,
) -> None:
    """Start the Streamable HTTP MCP search server."""

    mcp_app = create_mcp_app(config_path)
    uvicorn.run(mcp_app, host=host, port=port)


@app.command(name="list-skills")
def list_skills(
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to config.toml"),
    ] = None,
) -> None:
    """List available skills."""

    config = load_config(config_path)
    configure_logging(config.app.log_level)
    runtime = DevMateRuntime(config)
    for name in runtime.skill_manager.list_skill_names():
        typer.echo(name)
