from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

from devmate.config import AppConfig
from devmate.runtime import DevMateRuntime
from devmate.schemas import GenerateRequest
from devmate.webapp import (
    HTML_TEMPLATE,
    _build_web_job_payload,
    _build_web_progress_callback,
    _load_web_job,
    _persist_web_job_payload,
    _reset_web_job_snapshots,
)


def _build_config(project_root: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project_root": project_root,
            "config_path": project_root / "config.toml",
        }
    )


@pytest.mark.asyncio
async def test_web_progress_callback_persists_updates_without_logging_message(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))
    runtime.state_store.ensure_directories()

    job_id = "job-1"
    request_payload = GenerateRequest(prompt="demo")
    payload = _build_web_job_payload(job_id, request_payload)
    _persist_web_job_payload(runtime, job_id, payload)
    before = float(runtime.state_store.read_state("web", job_id)["updated_at"])

    callback = _build_web_progress_callback(runtime, job_id, request_payload)
    with caplog.at_level(logging.INFO):
        await callback(
            {
                "type": "status",
                "message": "secret model text",
                "current_phase": "Planner",
            }
        )

    record = runtime.state_store.read_state("web", job_id)

    assert record is not None
    assert record["stage"] == "secret model text"
    assert record["current_phase"] == "Planner"
    assert float(record["updated_at"]) >= before
    assert "secret model text" not in caplog.text
    assert "has_message=True" in caplog.text
    assert "message_length=17" in caplog.text


@pytest.mark.asyncio
async def test_load_web_job_marks_interrupted_only_when_task_is_inactive(
    tmp_path: Path,
) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))
    runtime.state_store.ensure_directories()

    job_id = "job-1"
    request_payload = GenerateRequest(prompt="demo")
    payload = _build_web_job_payload(job_id, request_payload)
    payload["status"] = "running"
    payload["stage"] = "still running"
    _persist_web_job_payload(runtime, job_id, payload)

    blocker = asyncio.Event()

    async def wait_forever() -> None:
        await blocker.wait()

    task = asyncio.create_task(wait_forever())
    try:
        active_job = _load_web_job(runtime, job_id, {job_id: task})
        assert active_job is not None
        assert active_job.status == "running"
        assert runtime.state_store.read_state("web", job_id)["status"] == "running"
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    interrupted_job = _load_web_job(runtime, job_id, {})
    stored = runtime.state_store.read_state("web", job_id)

    assert interrupted_job is not None
    assert interrupted_job.status == "interrupted"
    assert stored is not None
    assert stored["status"] == "interrupted"
    assert stored["stage"] == "Web 服务已恢复最近状态，但原后台任务未继续执行。"


@pytest.mark.asyncio
async def test_reset_web_job_snapshots_clears_old_snapshots_and_tasks(
    tmp_path: Path,
) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))
    runtime.state_store.ensure_directories()
    runtime.state_store.write_state(
        "web",
        "old-job",
        {
            "job_id": "old-job",
            "status": "running",
            "updated_at": 1.0,
        },
    )

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(wait_forever())
    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            runtime=runtime,
            jobs={"old-job": object()},
            active_tasks={"old-job": task},
        )
    )

    _reset_web_job_snapshots(fake_app)
    with suppress(asyncio.CancelledError):
        await task

    assert fake_app.state.jobs == {}
    assert fake_app.state.active_tasks == {}
    assert runtime.state_store.list_states("web") == []


def test_html_template_keeps_polling_after_interrupted_status() -> None:
    marker = "data.status === 'interrupted'"
    start = HTML_TEMPLATE.index(marker)
    interrupted_block = HTML_TEMPLATE[start : start + 220]

    assert "stopPolling();" not in interrupted_block
    assert "shouldKeepPolling(data.status)" in HTML_TEMPLATE


def test_html_template_uses_single_flight_polling() -> None:
    assert "setInterval(" not in HTML_TEMPLATE
    assert "pollInFlight" in HTML_TEMPLATE
    assert "AbortController" in HTML_TEMPLATE
