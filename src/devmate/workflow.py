"""Workflow helpers for the single-agent multi-stage DevMate pipeline."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

LOGGER = logging.getLogger(__name__)
_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.DOTALL,
)
_BUILDER_ENCOURAGEMENT_THRESHOLD = 18_000
_BUILDER_ENCOURAGEMENTS = (
    "你可以的",
    "你很厉害",
    "你能完成",
    "你很棒",
)


@dataclass(slots=True)
class WorkflowOptions:
    """Runtime options for the staged workflow."""

    deep_optimization: bool = False
    local_python_tool_enabled: bool = False
    max_deep_optimization_rounds: int = 0
    standard_builder_repair_rounds: int = 1


@dataclass(slots=True)
class WorkflowArtifacts:
    """Filesystem locations used by a staged workflow run."""

    root: Path

    def __post_init__(self) -> None:
        """Ensure the workflow artifact directory exists."""
        self.root.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, filename: str) -> Path:
        """Return a path inside the workflow artifact directory."""
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / filename

    def planner_path(self) -> Path:
        """Return the planner artifact path."""
        return self._artifact_path("planner_spec.md")

    def researcher_path(self, round_index: int) -> Path:
        """Return the researcher artifact path for a round."""
        return self._artifact_path(f"researcher_evidence_round_{round_index}.md")

    def builder_path(self, round_index: int) -> Path:
        """Return the builder artifact path for a round."""
        return self._artifact_path(f"builder_notes_round_{round_index}.md")

    def verifier_path(self, round_index: int) -> Path:
        """Return the verifier report path for a round."""
        return self._artifact_path(f"verifier_report_round_{round_index}.json")

    def packager_path(self) -> Path:
        """Return the packager artifact path."""
        return self._artifact_path("packager_summary.md")

    def delivery_zip_path(self) -> Path:
        """Return the packaged archive path."""
        return self._artifact_path("delivery.zip")


@dataclass(slots=True)
class VerificationModuleResult:
    """Verification result for a single module or file group."""

    module: str
    has_error: bool = False
    missing_detail: bool = False
    details: str = ""
    recommendation: str = ""


@dataclass(slots=True)
class VerificationReport:
    """Normalized verifier report used for workflow control."""

    overall_status: str = "pass"
    contains_errors: bool = False
    contains_missing_details: bool = False
    summary: str = ""
    module_results: list[VerificationModuleResult] = field(default_factory=list)
    builder_feedback: str = ""
    research_feedback: str = ""
    source_path: str | None = None

    @property
    def passed(self) -> bool:
        """Return whether verification fully passed."""

        return not self.contains_errors and not self.contains_missing_details


@dataclass(slots=True)
class StageExecutionResult:
    """Reply captured from a single workflow stage."""

    stage_name: str
    reply: str
    interrupted: bool = False
    loop_detected: bool = False


def _normalize_path(path: Path, project_root: Path) -> str:
    """Return a project-root-relative POSIX path."""

    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _trim_block(block: str) -> str:
    """Normalize indentation for prompt templates."""

    return dedent(block).strip()


def _feedback_path(path: Path | None, project_root: Path) -> str:
    """Return a display path for optional feedback artifacts."""

    if path is None:
        return "NONE"
    return _normalize_path(path, project_root)


def _bool_or_unknown(value: bool | None) -> str:
    """Convert an optional boolean to prompt-friendly text."""

    if value is None:
        return "unknown"
    return str(value).lower()


def _text_or_none(value: str | None) -> str:
    """Convert optional text into an explicit prompt placeholder."""

    if value is None:
        return "NONE"
    normalized = value.strip()
    return normalized or "NONE"


def _builder_inputs_are_long(*parts: str | None) -> bool:
    """Return whether Builder input context is large enough for encouragement."""

    total_chars = sum(len(part or "") for part in parts)
    return total_chars >= _BUILDER_ENCOURAGEMENT_THRESHOLD


def _append_builder_encouragement(
    prompt: str,
    *,
    planner_text: str,
    researcher_text: str,
    verifier_feedback_text: str | None,
    workspace_snapshot_text: str | None,
    user_prompt: str,
) -> str:
    """Append a short encouragement line when Builder context is very long."""

    if not _builder_inputs_are_long(
        planner_text,
        researcher_text,
        verifier_feedback_text,
        workspace_snapshot_text,
        user_prompt,
    ):
        return prompt

    encouragement = random.choice(_BUILDER_ENCOURAGEMENTS)
    return f"{prompt}\n\n鼓励提示：{encouragement}"


def build_planner_prompt(
    *,
    user_prompt: str,
    planner_tool_available: bool,
    **_: object,
) -> str:
    """Build the Planner-stage instruction."""

    planner_tool_note = (
        "Planner tool detected. You may use it when helpful, but you still must "
        "produce the final planner text yourself."
        if planner_tool_available
        else (
            "Planner tool is unavailable. You MUST switch to planner-role fallback "
            "mode: act explicitly as the Planner, think in terms of requirements, "
            "scope, modules, interfaces, data flow, and acceptance criteria, then "
            "return the complete planning text yourself."
        )
    )

    return _trim_block(
        f"""
        STAGE: Planner
        WORKFLOW_MODE: single-agent-multi-stage
        PLANNER_TOOL_AVAILABLE: {str(planner_tool_available).lower()}

        USER_REQUEST:
        {user_prompt}

        Planner fallback rule:
        {planner_tool_note}

        Required actions:
        1. First audit whether the request is legal, safe, reasonable, and feasible.
        2. If the request is illegal, unsafe, clearly abusive, or fundamentally
           unreasonable, stop planning immediately and return Markdown only with:
           PLANNER_DECISION: REJECT
           # Rejection Summary
           ## Reason
           ## Safer Alternative
        3. If the request is acceptable, return Markdown only with:
           PLANNER_DECISION: APPROVE
           # Structured Specification
           ## Legality and Reasonableness Review
           ## User Goal
           ## Scope and Non-Goals
           ## Technology Stack
           ## Pages and Modules
           ## Interfaces and APIs
           ## Data Flow
           ## Directory Structure
           ## Acceptance Criteria
        4. Do not write production code in this stage.
        5. Keep the plan actionable and implementation-ready.
        6. The runtime, not the agent, will persist stage artifacts. Do not instruct
           anyone to write planner output to a file.
        7. If PLANNER_TOOL_AVAILABLE is false, state the plan as if you are the
           dedicated Planner and make the returned text detailed enough for
           downstream Researcher and Builder stages.

        Reply with the full planner markdown only.
        """
    )


def build_researcher_prompt(
    *,
    planner_text: str,
    verifier_feedback_text: str | None,
    user_prompt: str,
    deep_optimization_round: int,
    detail_focus_enabled: bool,
    **_: object,
) -> str:
    """Build the Researcher-stage instruction."""

    detail_focus_rule = (
        "Deep optimization is enabled. Prioritize targeted searches that verify "
        "whether every requested detail has been fully implemented and gather the "
        "missing requirement coverage explicitly."
        if detail_focus_enabled
        else (
            "Deep optimization is disabled. Prioritize research that helps fix code, "
            "architecture, project-structure, and integration errors first."
        )
    )

    return _trim_block(
        f"""
        STAGE: Researcher
        WORKFLOW_MODE: single-agent-multi-stage
        RESEARCH_ROUND: {deep_optimization_round}
        DETAIL_FOCUS_ENABLED: {str(detail_focus_enabled).lower()}

        USER_REQUEST:
        {user_prompt}

        Detail focus rule:
        {detail_focus_rule}

        Required actions:
        1. Read the planner text snapshot before doing any research.
        2. You MUST call search_web through MCP at least once.
        3. You MUST call search_knowledge_base at least once.
        4. Gather framework docs, templates, best practices, and example
           repositories relevant to the plan.
        5. If verifier feedback text is present, convert every finding into targeted
           search goals and explicitly explain how the research covers each finding.
        6. When DETAIL_FOCUS_ENABLED is true, search specifically for requirement
           coverage, UX/detail completion, omitted endpoints, missing fields,
           missing pages, and other incomplete implementation details.
        7. If the user explicitly wants reusable research notes, docs, or other
           managed output files, save them only through write_external_file
           with an approved root such as docs or workflow_artifacts.
        8. Return Markdown only with these exact headings:
           # Evidence Package
           ## Planner Recap
           ## Web Evidence
           ## Local Knowledge Evidence
           ## Templates and Example Repositories
           ## Recommended Decisions
           ## Risks and Caveats
           ## Verifier Feedback Coverage
        9. The runtime, not the agent, will persist the returned research text.

        ===== PLANNER TEXT SNAPSHOT =====
        {planner_text}
        ===== END PLANNER TEXT SNAPSHOT =====

        ===== VERIFIER FEEDBACK TEXT =====
        {_text_or_none(verifier_feedback_text)}
        ===== END VERIFIER FEEDBACK TEXT =====

        Reply with the full research markdown only.
        """
    )


def build_builder_prompt(
    *,
    planner_text: str,
    researcher_text: str,
    verifier_feedback_text: str | None,
    workspace_snapshot_text: str | None,
    local_python_tool_enabled: bool,
    round_index: int,
    user_prompt: str,
    **_: object,
) -> str:
    """Build the Builder-stage instruction."""

    execution_mode = (
        "local-python-tool-enabled"
        if local_python_tool_enabled
        else "text-snapshot-review-only"
    )
    execution_rule = (
        "The local Python tool is enabled. If a focused syntax check, targeted "
        "script, or narrow pytest run will materially reduce uncertainty, you may "
        "call invoke_local_python. Prefer the smallest useful command and do not "
        "rerun substantially similar failing commands more than twice."
        if local_python_tool_enabled
        else (
            "The local Python tool is disabled. You are not allowed to run local "
            "tests, Python scripts, shell commands, or any executor to inspect the "
            "program. If you need to detect logic or syntax problems, use the "
            "workspace snapshot below as the object-under-test context and reason "
            "directly from the code."
        )
    )
    workspace_snapshot_block = ""
    if workspace_snapshot_text:
        workspace_snapshot_block = _trim_block(
            f"""
            ===== CURRENT WORKSPACE SNAPSHOT =====
            {workspace_snapshot_text}
            ===== END CURRENT WORKSPACE SNAPSHOT =====
            """
        )

    prompt = _trim_block(
        f"""
        STAGE: Builder
        WORKFLOW_MODE: single-agent-multi-stage
        BUILD_ROUND: {round_index}
        BUILD_EXECUTION_MODE: {execution_mode}

        USER_REQUEST:
        {user_prompt}

        Tool execution rule:
        {execution_rule}

        Required actions:
        1. Read the planner text and research text before changing code.
        2. If verifier feedback text is present, fix every reported error and detail
           gap that is practical.
        3. Implement the project in staged, coherent edits instead of a single giant
           dump.
        4. Follow PEP 8 for Python and never use the built-in print() function.
        5. You may read approved external context through list_runtime_files and
           read_runtime_file, but every new or modified file must stay inside the
           workspace and use workspace-relative paths only. For display examples,
           a path like workspace/src/app.py is acceptable, but tool writes must
           still use src/app.py. For Python package roots, use imports such as
           import src when that matches the planned structure.
        6. If the user explicitly asks for docs, skills, uploads, or workflow
           artifact files, use write_external_file so those managed outputs stay
           under the workspace. Never try a normal direct write outside
           workspace.
        7. Do not write planner, researcher, builder, verifier, or packager stage
           artifact files yourself. The runtime will persist your returned notes.
        8. Return Markdown only with these exact headings:
           # Build Notes
           ## Files Changed
           ## Implemented Work
           ## Remaining Risks
        9. Text-only completion is not enough for implementation tasks. Create or
           modify at least one real workspace file before you finish.
        10. Keep the implementation aligned with the planner and evidence package.
        11. If BUILD_EXECUTION_MODE is text-snapshot-review-only, treat the current
           workspace snapshot as the only inspection context and manually review it
           for logic errors and obvious syntax issues instead of running anything.
        12. If BUILD_EXECUTION_MODE is local-python-tool-enabled and you do run a
            local check, keep it narrowly scoped and summarize what the command
            proved or failed to prove.

        ===== PLANNER TEXT SNAPSHOT =====
        {planner_text}
        ===== END PLANNER TEXT SNAPSHOT =====

        ===== RESEARCH TEXT SNAPSHOT =====
        {researcher_text}
        ===== END RESEARCH TEXT SNAPSHOT =====

        ===== VERIFIER FEEDBACK TEXT =====
        {_text_or_none(verifier_feedback_text)}
        ===== END VERIFIER FEEDBACK TEXT =====

        {workspace_snapshot_block}

        Reply with a concise implementation summary and mention the main files changed.
        """
    )
    return _append_builder_encouragement(
        prompt,
        planner_text=planner_text,
        researcher_text=researcher_text,
        verifier_feedback_text=verifier_feedback_text,
        workspace_snapshot_text=workspace_snapshot_text,
        user_prompt=user_prompt,
    )


_VERIFIER_JSON_SCHEMA = """{
  "overall_status": "pass" | "needs_fix",
  "contains_errors": true | false,
  "contains_missing_details": true | false,
  "summary": "string",
  "module_results": [
    {
      "module": "string",
      "has_error": true | false,
      "missing_detail": true | false,
      "details": "string",
      "recommendation": "string"
    }
  ],
  "builder_feedback": "string",
  "research_feedback": "string"
}"""


def build_verifier_pytest_prompt(
    *,
    round_index: int,
    user_prompt: str,
    planner_text: str,
    researcher_text: str,
    builder_text: str,
    workspace_file_index: str,
    detail_review_enabled: bool,
    **_: object,
) -> str:
    """Build the Verifier prompt that uses pytest with the local Python tool."""

    detail_rule = (
        "Deep optimization is enabled. You MUST judge whether requested details are "
        "fully completed. Mark missing_detail=true whenever user-requested details, "
        "requirement coverage, or implementation completeness are still missing."
        if detail_review_enabled
        else (
            "Deep optimization is disabled. Focus on code errors, broken structure, "
            "missing core files, and project-level defects. Do not fail the review "
            "for polish-only or non-critical detail gaps."
        )
    )
    pass_rule = (
        'Set overall_status to "pass" only when your targeted pytest checks '
        "succeed and every inspected module has_error=false and "
        "missing_detail=false."
        if detail_review_enabled
        else (
            'Set overall_status to "pass" only when your targeted pytest checks '
            "succeed and there are no obvious code errors, broken requirements, "
            "or project-level defects."
        )
    )
    feedback_rule = (
        "research_feedback must explain what Researcher should improve when missing "
        "design evidence, requirement coverage, or detail completeness caused the "
        "issue."
        if detail_review_enabled
        else (
            "research_feedback should usually be empty unless a core requirement or "
            "critical missing evidence is blocking implementation correctness."
        )
    )

    return _trim_block(
        f"""
        STAGE: Verifier
        WORKFLOW_MODE: single-agent-multi-stage
        VERIFY_ROUND: {round_index}
        VERIFICATION_MODE: pytest-with-local-python-tool
        DETAIL_REVIEW_ENABLED: {str(detail_review_enabled).lower()}
        LOCAL_PYTHON_TOOL_ENABLED: true

        USER_REQUEST:
        {user_prompt}

        The local Python tool is enabled for this run. Use the live workspace as the
        source of truth. You may write pytest files under tests/ and you must use
        invoke_local_python for the actual local Python execution.

        Detail review rule:
        {detail_rule}

        Required actions:
        1. Read the planner snapshot, evidence snapshot, build notes snapshot, and
           inspect the current workspace files before deciding what to test.
        2. Write focused pytest tests under tests/ that validate the highest-risk
           modules, user-requested behaviors, and recent code changes.
        3. Use invoke_local_python to run pytest locally. Prefer module mode with
           target="pytest" and precise args such as specific test files or -q.
        4. Review the returned JSON from invoke_local_python, including command,
           exit_code, timed_out, stdout, and stderr, before concluding.
        5. If pytest fails because of code issues, import errors, missing dependencies,
           or environment/setup problems, record that explicitly in module_results
           and builder_feedback.
        6. Avoid repeated identical test runs. After two substantially similar
           failures, stop rerunning and summarize the evidence you already have.
        7. Output strict JSON only. Do not add markdown fences.
        8. The JSON must follow this schema exactly:
           {_VERIFIER_JSON_SCHEMA}
        9. {pass_rule}
        10. builder_feedback must explain what Builder should fix when there are
            errors, failing tests, or incomplete details.
        11. {feedback_rule}

        ===== PLANNER TEXT SNAPSHOT =====
        {planner_text}
        ===== END PLANNER TEXT SNAPSHOT =====

        ===== RESEARCH TEXT SNAPSHOT =====
        {researcher_text}
        ===== END RESEARCH TEXT SNAPSHOT =====

        ===== BUILDER TEXT SNAPSHOT =====
        {builder_text}
        ===== END BUILDER TEXT SNAPSHOT =====

        ===== CURRENT WORKSPACE FILE INDEX =====
        {workspace_file_index}
        ===== END CURRENT WORKSPACE FILE INDEX =====
        """
    )


def build_verifier_text_review_prompt(
    *,
    round_index: int,
    user_prompt: str,
    planner_text: str,
    researcher_text: str,
    builder_text: str,
    workspace_snapshot_text: str,
    detail_review_enabled: bool,
    **_: object,
) -> str:
    """Build the text-only Verifier-stage instruction."""

    detail_rule = (
        "Deep optimization is enabled. You MUST judge whether requested details are "
        "fully completed. Mark missing_detail=true whenever user-requested details, "
        "requirement coverage, or implementation completeness are still missing."
        if detail_review_enabled
        else (
            "Deep optimization is disabled. Focus on code errors, broken structure, "
            "missing core files, and project-level defects. Do not fail the review "
            "for polish-only or non-critical detail gaps."
        )
    )
    pass_rule = (
        'Set overall_status to "pass" only when every inspected module '
        "has_error=false and missing_detail=false."
        if detail_review_enabled
        else (
            'Set overall_status to "pass" when there are no obvious code errors, '
            "project-level defects, or broken requirements."
        )
    )
    feedback_rule = (
        "research_feedback must explain what Researcher should improve when missing "
        "design evidence, requirement coverage, or detail completeness caused the "
        "issue."
        if detail_review_enabled
        else (
            "research_feedback should usually be empty unless a core requirement or "
            "critical missing evidence is blocking implementation correctness."
        )
    )

    return _trim_block(
        f"""
        STAGE: Verifier
        WORKFLOW_MODE: single-agent-multi-stage
        VERIFY_ROUND: {round_index}
        VERIFICATION_MODE: inline-text-review
        DETAIL_REVIEW_ENABLED: {str(detail_review_enabled).lower()}

        USER_REQUEST:
        {user_prompt}

        You are not allowed to run tests, shell commands, scripts,
        verification tools, or any external executors.
        Review only the plain-text snapshots that follow.
        Treat the workspace snapshot as the object-under-test context.

        Detail review rule:
        {detail_rule}

        Required actions:
        1. Read the planner snapshot, evidence snapshot, build notes snapshot,
           and workspace file snapshot as plain text.
        2. Evaluate the implementation module by module.
        3. Identify obvious code errors, broken structure, missing files,
           and user-requested details that are still incomplete.
        4. If any snapshot is missing or truncated, mention that explicitly
           in the relevant module details.
        5. Output strict JSON only. Do not add markdown fences.
        6. The JSON must follow this schema exactly:
           {_VERIFIER_JSON_SCHEMA}
        7. {pass_rule}
        8. builder_feedback must explain what Builder should fix when there are errors
           or incomplete details.
        9. {feedback_rule}

        ===== PLANNER TEXT SNAPSHOT =====
        {planner_text}
        ===== END PLANNER TEXT SNAPSHOT =====

        ===== RESEARCH TEXT SNAPSHOT =====
        {researcher_text}
        ===== END RESEARCH TEXT SNAPSHOT =====

        ===== BUILDER TEXT SNAPSHOT =====
        {builder_text}
        ===== END BUILDER TEXT SNAPSHOT =====

        ===== WORKSPACE FILE SNAPSHOT =====
        {workspace_snapshot_text}
        ===== END WORKSPACE FILE SNAPSHOT =====
        """
    )


def build_packager_prompt(
    *,
    planner_text: str,
    researcher_text: str,
    builder_text: str,
    verifier_text: str | None,
    output_files: list[str],
    deep_optimization_enabled: bool,
    optimization_rounds_used: int,
    verification_passed: bool | None,
    user_prompt: str,
    finalization_reason: str = "normal_completion",
    **_: object,
) -> str:
    """Build the Packager-stage instruction."""

    output_files_text = "\n".join(f"- {path}" for path in output_files) or "- NONE"
    return _trim_block(
        f"""
        STAGE: Packager
        WORKFLOW_MODE: single-agent-multi-stage
        DEEP_OPTIMIZATION_ENABLED: {str(deep_optimization_enabled).lower()}
        OPTIMIZATION_ROUNDS_USED: {optimization_rounds_used}
        VERIFICATION_PASSED: {_bool_or_unknown(verification_passed)}
        FINALIZATION_REASON: {finalization_reason}

        USER_REQUEST:
        {user_prompt}

        Required actions:
        1. Read the planner, research, builder, and verifier texts when available.
        2. If any stage text is missing because the user requested immediate output or
           loop protection cut the workflow short, explicitly say that it is
           unavailable and continue packaging based on the current workspace state.
        3. Summarize the final deliverables.
        4. Include entry files, output files, run commands, preview address if
           applicable, verification conclusion, and change summary.
        5. Return Markdown only with these exact headings:
           # Delivery Summary
           ## Entry Files
           ## Output Files
           ## Run Commands
           ## Preview Address
           ## Verification Conclusion
           ## Change Summary
        6. If there is no dedicated preview address, say so explicitly.
        7. The runtime, not the agent, will persist the returned markdown.

        ===== PLANNER TEXT SNAPSHOT =====
        {planner_text}
        ===== END PLANNER TEXT SNAPSHOT =====

        ===== RESEARCH TEXT SNAPSHOT =====
        {researcher_text}
        ===== END RESEARCH TEXT SNAPSHOT =====

        ===== BUILDER TEXT SNAPSHOT =====
        {builder_text}
        ===== END BUILDER TEXT SNAPSHOT =====

        ===== VERIFIER TEXT SNAPSHOT =====
        {_text_or_none(verifier_text)}
        ===== END VERIFIER TEXT SNAPSHOT =====

        ===== OUTPUT FILES =====
        {output_files_text}
        ===== END OUTPUT FILES =====

        Reply with the same delivery summary in concise markdown.
        """
    )


def parse_verifier_report_text(
    raw_text: str,
    *,
    source_path: str | None = None,
) -> VerificationReport:
    """Normalize a verifier report from raw JSON text."""

    payload = _parse_json_payload(raw_text.strip())
    if payload is None:
        LOGGER.warning("Verifier report text is not valid JSON")
        return VerificationReport(
            overall_status="needs_fix",
            contains_missing_details=True,
            summary=(
                "Verifier report was invalid JSON, so verification is unavailable. "
                "Treating the result as needing review."
            ),
            builder_feedback=(
                "The verifier response could not be parsed as JSON. Review the "
                "implementation manually or rerun verification."
            ),
            source_path=source_path,
        )

    module_results: list[VerificationModuleResult] = []
    raw_modules = payload.get("module_results", [])
    if isinstance(raw_modules, list):
        for item in raw_modules:
            if not isinstance(item, dict):
                continue
            module_results.append(
                VerificationModuleResult(
                    module=str(item.get("module", "unknown")),
                    has_error=bool(item.get("has_error", False)),
                    missing_detail=bool(item.get("missing_detail", False)),
                    details=str(item.get("details", "")),
                    recommendation=str(item.get("recommendation", "")),
                )
            )

    contains_errors = bool(payload.get("contains_errors", False)) or any(
        result.has_error for result in module_results
    )
    contains_missing_details = bool(
        payload.get("contains_missing_details", False)
    ) or any(result.missing_detail for result in module_results)
    overall_status = str(payload.get("overall_status", "")).strip() or (
        "pass" if not contains_errors and not contains_missing_details else "needs_fix"
    )

    return VerificationReport(
        overall_status=overall_status,
        contains_errors=contains_errors,
        contains_missing_details=contains_missing_details,
        summary=str(payload.get("summary", "")).strip(),
        module_results=module_results,
        builder_feedback=str(payload.get("builder_feedback", "")).strip(),
        research_feedback=str(payload.get("research_feedback", "")).strip(),
        source_path=source_path,
    )


def parse_verifier_report(path: Path) -> VerificationReport:
    """Read and normalize a verifier report from disk."""

    if not path.exists() or not path.is_file():
        LOGGER.warning("Verifier report not found at %s", path)
        return VerificationReport(
            overall_status="needs_fix",
            contains_missing_details=True,
            summary=(
                "Verifier report was not written, so verification is unavailable. "
                "Treating the result as needing review."
            ),
            builder_feedback=(
                "The verifier did not produce a report. Review the current "
                "implementation manually or rerun verification."
            ),
            source_path=str(path),
        )

    raw_text = path.read_text(encoding="utf-8").strip()
    return parse_verifier_report_text(raw_text, source_path=str(path))


def _parse_json_payload(text: str) -> dict[str, object] | None:
    """Best-effort JSON extraction for verifier reports."""

    candidates = [text]
    fence_match = _JSON_FENCE_PATTERN.search(text)
    if fence_match:
        candidates.append(fence_match.group(1))

    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index != -1 and end_index != -1 and start_index < end_index:
        candidates.append(text[start_index:end_index + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None
