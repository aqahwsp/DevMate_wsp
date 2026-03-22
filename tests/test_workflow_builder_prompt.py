from __future__ import annotations

from devmate.workflow import build_builder_prompt


def test_builder_prompt_appends_encouragement_for_long_context(monkeypatch) -> None:
    monkeypatch.setattr("devmate.workflow.random.choice", lambda values: "你很棒")

    prompt = build_builder_prompt(
        planner_text="P" * 6000,
        researcher_text="R" * 6000,
        verifier_feedback_text="V" * 4000,
        workspace_snapshot_text="W" * 4000,
        local_python_tool_enabled=False,
        round_index=1,
        user_prompt="U" * 1000,
    )

    assert prompt.endswith("鼓励提示：你很棒")


def test_builder_prompt_skips_encouragement_for_short_context() -> None:
    prompt = build_builder_prompt(
        planner_text="planner",
        researcher_text="research",
        verifier_feedback_text="feedback",
        workspace_snapshot_text="workspace snapshot",
        local_python_tool_enabled=False,
        round_index=0,
        user_prompt="hello",
    )

    assert "鼓励提示：" not in prompt


def test_builder_prompt_mentions_readonly_external_tools() -> None:
    prompt = build_builder_prompt(
        planner_text="planner",
        researcher_text="research",
        verifier_feedback_text=None,
        workspace_snapshot_text=None,
        local_python_tool_enabled=True,
        round_index=0,
        user_prompt="hello",
    )

    assert "list_runtime_files" in prompt
    assert "read_runtime_file" in prompt
    assert "write_external_file" in prompt
    assert "every new or modified file must stay inside the" in prompt
    assert "workspace/src/" in prompt
    assert "import src" in prompt


def test_builder_prompt_requires_real_workspace_file_changes() -> None:
    prompt = build_builder_prompt(
        planner_text="planner",
        researcher_text="research",
        verifier_feedback_text=None,
        workspace_snapshot_text=None,
        local_python_tool_enabled=False,
        round_index=0,
        user_prompt="hello",
    )

    assert "Text-only completion is not enough" in prompt
    assert "at least one real workspace file" in prompt

