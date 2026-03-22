from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from devmate.config import AppConfig
from devmate.runtime import DevMateRuntime


def _build_config(project_root: Path, *, skills_dir: str = ".skills") -> AppConfig:
    return AppConfig.model_validate(
        {
            "project_root": project_root,
            "config_path": project_root / "config.toml",
            "skills": {
                "skills_dir": skills_dir,
                "auto_save_on_success": True,
            },
        }
    )


def test_managed_output_paths_are_forced_inside_workspace(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    workspace_root = tmp_path / "workspace"

    assert config.workspace_dir == workspace_root
    assert config.skills_dir == workspace_root / ".skills"
    assert config.docs_dir == workspace_root / "docs"
    assert config.persist_dir == workspace_root / "data" / "chroma"
    assert config.research_cache_dir == workspace_root / "docs" / "research_cache"
    assert config.state_dir == workspace_root / "data" / "runtime_state"
    assert config.workflow_artifacts_dir == workspace_root / "data" / "workflow_runs"


@pytest.mark.asyncio
async def test_build_agent_exposes_workspace_scoped_runtime_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _build_config(tmp_path)
    runtime = DevMateRuntime(config)
    monkeypatch.setattr(runtime, "_load_mcp_tools", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "devmate.runtime.build_chat_model",
        lambda model_cfg: object(),
    )

    agent = await runtime.build_agent(local_python_tool_enabled=True)
    tool_names = {
        getattr(tool_obj, "name", getattr(tool_obj, "__name__", ""))
        for tool_obj in agent.tools
    }

    assert runtime._skills_dir_is_workspace_local() is True
    assert "search_knowledge_base" in tool_names
    assert "list_runtime_files" in tool_names
    assert "read_runtime_file" in tool_names
    assert "write_external_file" in tool_names
    assert "invoke_local_python" in tool_names
    assert "save_skill_pattern" in tool_names
    assert agent.backend.root_dir == str(config.workspace_dir)


@pytest.mark.asyncio
async def test_build_agent_keeps_explicit_workspace_skills_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _build_config(tmp_path, skills_dir="workspace/.skills")
    runtime = DevMateRuntime(config)
    monkeypatch.setattr(runtime, "_load_mcp_tools", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "devmate.runtime.build_chat_model",
        lambda model_cfg: object(),
    )

    agent = await runtime.build_agent(local_python_tool_enabled=False)
    tool_names = {
        getattr(tool_obj, "name", getattr(tool_obj, "__name__", ""))
        for tool_obj in agent.tools
    }

    assert config.skills_dir == tmp_path / "workspace" / ".skills"
    assert runtime._skills_dir_is_workspace_local() is True
    assert "save_skill_pattern" in tool_names


@pytest.mark.asyncio
async def test_load_mcp_tools_reuses_cached_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))
    calls: list[str] = []

    class FakeClient:
        def __init__(self, servers) -> None:
            self.servers = servers

        async def get_tools(self):
            calls.append("get_tools")
            return ["search_web"]

    monkeypatch.setattr("devmate.runtime.MultiServerMCPClient", FakeClient)

    first_tools = await runtime._load_mcp_tools()
    second_tools = await runtime._load_mcp_tools()

    assert first_tools == ["search_web"]
    assert second_tools == ["search_web"]
    assert calls == ["get_tools"]


def test_write_external_file_writes_only_to_approved_runtime_roots(
    tmp_path: Path,
) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))

    stored_path = runtime.write_external_file(
        root_name="docs",
        relative_path="notes/research.md",
        content="# Notes\n",
        mode="overwrite",
    )

    assert stored_path == "workspace/docs/notes/research.md"
    assert (tmp_path / stored_path).read_text(encoding="utf-8") == "# Notes\n"


def test_write_external_file_blocks_path_traversal(tmp_path: Path) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))

    with pytest.raises(ValueError):
        runtime.write_external_file(
            root_name="docs",
            relative_path="../escape.txt",
            content="nope",
        )


def test_write_external_file_supports_uploads_alias(tmp_path: Path) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))

    stored_path = runtime.write_external_file(
        root_name="uploads",
        relative_path="agent/output.txt",
        content="payload",
        mode="overwrite",
    )

    assert stored_path == "workspace/data/uploads/agent/output.txt"
    assert (tmp_path / stored_path).read_text(encoding="utf-8") == "payload"


def test_runtime_read_tools_can_read_external_project_context(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("memory\n", encoding="utf-8")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    runtime = DevMateRuntime(_build_config(tmp_path))
    listing = runtime.list_runtime_files(root_name="project", relative_path=".")
    content = runtime.read_runtime_file(
        root_name="project",
        relative_path="docs/guide.md",
        start_line=2,
        max_lines=1,
    )

    assert "[FILE] AGENTS.md" in listing
    assert "[FILE] config.toml" in listing
    assert "LINES: 2-2" in content
    assert "beta" in content


def test_runtime_read_tools_block_path_traversal(tmp_path: Path) -> None:
    outside_file = tmp_path.parent / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    runtime = DevMateRuntime(_build_config(tmp_path))

    with pytest.raises(ValueError):
        runtime.read_runtime_file(
            root_name="project",
            relative_path="../outside.txt",
        )


def test_runtime_hides_internal_workspace_artifacts(tmp_path: Path) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))
    workspace_root = runtime.config.workspace_dir
    workspace_root.mkdir(parents=True, exist_ok=True)

    visible_file = workspace_root / "main.py"
    visible_file.write_text("pass\n", encoding="utf-8")

    hidden_runtime = workspace_root / ".devmate_runtime" / "local_python" / "x.py"
    hidden_runtime.parent.mkdir(parents=True, exist_ok=True)
    hidden_runtime.write_text("pass\n", encoding="utf-8")

    research_file = runtime.config.research_cache_dir / "cache.txt"
    research_file.parent.mkdir(parents=True, exist_ok=True)
    research_file.write_text("cached\n", encoding="utf-8")

    workflow_file = runtime.config.workflow_artifacts_dir / "run-1" / "planner.md"
    workflow_file.parent.mkdir(parents=True, exist_ok=True)
    workflow_file.write_text("plan\n", encoding="utf-8")

    upload_file = runtime.upload_dir("web") / "sample.txt"
    upload_file.parent.mkdir(parents=True, exist_ok=True)
    upload_file.write_text("seed\n", encoding="utf-8")

    assert runtime.list_workspace_files() == ["workspace/main.py"]


def test_skill_manager_moves_downloaded_skills_into_managed_store(
    tmp_path: Path,
) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))
    workspace_root = runtime.config.workspace_dir
    workspace_root.mkdir(parents=True, exist_ok=True)

    downloaded_skill = workspace_root / "downloaded_skill"
    downloaded_skill.mkdir()
    (downloaded_skill / "SKILL.md").write_text("# Demo Skill\n", encoding="utf-8")
    (downloaded_skill / "notes.txt").write_text("skill notes\n", encoding="utf-8")

    relocated = runtime.skill_manager.relocate_workspace_skills(
        workspace_root=workspace_root,
    )

    managed_skill = runtime.config.skills_dir / "downloaded-skill"
    assert relocated == [managed_skill]
    assert (managed_skill / "SKILL.md").read_text(encoding="utf-8") == (
        "# Demo Skill\n"
    )
    assert (managed_skill / "notes.txt").read_text(encoding="utf-8") == (
        "skill notes\n"
    )
    assert not downloaded_skill.exists()


def test_sync_project_context_copies_agents_memory_into_workspace(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("memory\n", encoding="utf-8")

    runtime = DevMateRuntime(_build_config(tmp_path))
    runtime._sync_project_context_into_workspace()

    assert (tmp_path / "workspace" / "AGENTS.md").read_text(
        encoding="utf-8"
    ) == "memory\n"


def test_runtime_requires_real_workspace_changes_before_success(
    tmp_path: Path,
) -> None:
    runtime = DevMateRuntime(_build_config(tmp_path))

    with pytest.raises(
        RuntimeError,
        match="without modifying any workspace files",
    ):
        runtime._ensure_workspace_changes_present([])

