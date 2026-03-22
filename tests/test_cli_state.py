from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from devmate.cli import _run_with_cli_controls
from devmate.config import AppConfig
from devmate.runtime import DevMateRuntime


def _build_config(project_root: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project_root": project_root,
            "config_path": project_root / "config.toml",
        }
    )


@pytest.mark.asyncio
async def test_cli_run_clears_old_snapshot_and_persists_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))
    runtime.state_store.ensure_directories()
    runtime.state_store.write_state(
        "cli",
        "old-run",
        {
            "run_id": "old-run",
            "status": "completed",
            "reply": "stale",
            "updated_at": 1.0,
        },
    )

    async def fake_run(prompt: str, **kwargs) -> SimpleNamespace:
        progress_callback = kwargs["progress_callback"]
        await progress_callback(
            {
                "type": "status",
                "message": "正在执行阶段：Planner",
                "current_phase": "Planner",
            }
        )
        await progress_callback(
            {
                "type": "workspace_changes",
                "message": "检测到 1 项文件变更。",
                "current_phase": "Builder",
                "file_operations": ["modified: demo.py"],
                "changed_files": ["demo.py"],
                "output_files": ["demo.py"],
            }
        )
        return SimpleNamespace(
            prompt=prompt,
            reply="done",
            changed_files=["demo.py"],
            output_files=["demo.py"],
            file_operations=["modified: demo.py"],
            saved_skill=None,
            delivery_zip=None,
            verification_passed=True,
            optimization_rounds_used=0,
            deep_optimization=False,
            local_python_tool_enabled=False,
            max_deep_optimization_rounds=2,
            immediate_output_requested=False,
        )

    monkeypatch.setattr(runtime, "run", fake_run)

    result = await _run_with_cli_controls(
        runtime,
        "hello",
        deep_optimization=None,
        local_python_tool_enabled=None,
        max_deep_optimization_rounds=None,
        include_initial_files=False,
        initial_file_paths=None,
    )
    records = runtime.state_store.list_states("cli")

    assert result.reply == "done"
    assert len(records) == 1
    record = records[0]
    assert record["run_id"] != "old-run"
    assert record["status"] == "completed"
    assert record["stage"] == "任务执行完成。"
    assert record["current_phase"] == "Packager"
    assert record["reply"] == "done"
    assert record["file_operations"] == ["modified: demo.py"]
    assert record["changed_files"] == ["demo.py"]
    assert record["output_files"] == ["demo.py"]
    assert record["worker_pid"] > 0
