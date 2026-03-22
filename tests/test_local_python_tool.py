from __future__ import annotations

from pathlib import Path

import pytest

from devmate.local_python_tool import execute_local_python


@pytest.mark.asyncio
async def test_local_python_allows_inside_write_and_outside_read(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    external_source = tmp_path / "external.txt"
    external_source.write_text("seed", encoding="utf-8")

    code = (
        "from pathlib import Path\n"
        f"text = Path(r'{external_source}').read_text(encoding='utf-8')\n"
        "Path('inside.txt').write_text(text + ' ok', encoding='utf-8')\n"
    )
    result = await execute_local_python(
        workspace_root=workspace_root,
        execution_mode="code",
        target=code,
        purpose="read outside and write inside",
    )

    assert result.ok is True
    assert (workspace_root / "inside.txt").read_text(encoding="utf-8") == "seed ok"
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_local_python_blocks_outside_write(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    forbidden_target = tmp_path / "outside.txt"

    code = (
        "from pathlib import Path\n"
        f"Path(r'{forbidden_target}').write_text('blocked', encoding='utf-8')\n"
    )
    result = await execute_local_python(
        workspace_root=workspace_root,
        execution_mode="code",
        target=code,
        purpose="attempt to write outside workspace",
    )

    assert result.ok is False
    assert forbidden_target.exists() is False
    assert "Write access is restricted to the workspace" in result.stderr


@pytest.mark.asyncio
async def test_local_python_disallows_spawning_child_processes(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    code = (
        "import subprocess\nsubprocess.run(['python', '-c', 'print(1)'], check=False)\n"
    )
    result = await execute_local_python(
        workspace_root=workspace_root,
        execution_mode="code",
        target=code,
        purpose="attempt to spawn a subprocess",
    )

    assert result.ok is False
    assert "Spawning child processes is not allowed" in result.stderr
