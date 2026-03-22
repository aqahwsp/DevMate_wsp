"""Utilities for managing Deep Agents skill directories."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from textwrap import dedent

from devmate.config import AppConfig

DEFAULT_ALLOWED_TOOLS = [
    "search_web",
    "search_knowledge_base",
    "write_file",
    "write_external_file",
    "edit_file",
    "list_dir",
    "read_file",
]
_SKILL_MARKDOWN_NAME = "SKILL.md"


class SkillManager:
    """Create, list, and persist DevMate skills."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.skills_root = config.skills_dir
        self.skills_root.mkdir(parents=True, exist_ok=True)

    def list_skill_names(self) -> list[str]:
        """Return all skill directory names containing a SKILL.md file."""

        names: list[str] = []
        for path in self.skills_root.iterdir():
            if path.is_dir() and (path / _SKILL_MARKDOWN_NAME).exists():
                names.append(path.name)
        return sorted(names)

    def slugify(self, value: str) -> str:
        """Convert arbitrary text to a filesystem-safe slug."""

        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
        return cleaned.strip("-") or "skill"

    def _next_available_skill_dir(self, value: str) -> Path:
        """Return an unused skill directory path under the managed store."""

        slug = self.slugify(value)
        skill_dir = self.skills_root / slug
        suffix = 2
        while skill_dir.exists():
            skill_dir = self.skills_root / f"{slug}-{suffix}"
            suffix += 1
        return skill_dir

    def _skill_name_from_manifest(self, skill_manifest: Path) -> str:
        """Derive a stable destination name for an imported skill manifest."""

        parent_name = skill_manifest.parent.name.strip()
        if parent_name and parent_name != _SKILL_MARKDOWN_NAME:
            return parent_name
        try:
            lines = skill_manifest.read_text(encoding="utf-8").splitlines()
        except OSError:
            return "imported-skill"
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip() or "imported-skill"
        return "imported-skill"

    def relocate_workspace_skills(self, *, workspace_root: Path) -> list[Path]:
        """Move discovered skill bundles under workspace/.skills."""

        workspace_root = workspace_root.resolve()
        managed_root = self.skills_root.resolve()
        self.skills_root.mkdir(parents=True, exist_ok=True)
        discovered_manifests = sorted(
            workspace_root.rglob(_SKILL_MARKDOWN_NAME),
            key=lambda path: (len(path.parts), path.as_posix()),
        )
        relocated_paths: list[Path] = []

        for skill_manifest in discovered_manifests:
            if not skill_manifest.exists():
                continue
            skill_manifest = skill_manifest.resolve()
            try:
                skill_manifest.relative_to(managed_root)
                continue
            except ValueError:
                pass

            source_dir = skill_manifest.parent
            destination_dir = self._next_available_skill_dir(
                self._skill_name_from_manifest(skill_manifest)
            )
            destination_dir.parent.mkdir(parents=True, exist_ok=True)

            if source_dir == workspace_root:
                destination_dir.mkdir(parents=True, exist_ok=False)
                shutil.move(
                    str(skill_manifest),
                    str(destination_dir / _SKILL_MARKDOWN_NAME),
                )
            else:
                shutil.move(str(source_dir), str(destination_dir))
            relocated_paths.append(destination_dir)

        return relocated_paths

    def sync_new_markdown_to_source(self, *, source_root: Path) -> list[Path]:
        """Copy newly created workspace Markdown skill files back to the source."""

        managed_root = self.skills_root.resolve()
        source_root = source_root.resolve()
        source_root.mkdir(parents=True, exist_ok=True)
        if managed_root == source_root:
            return []

        copied_paths: list[Path] = []
        for source_path in sorted(managed_root.rglob("*")):
            if not source_path.is_file() or source_path.suffix.lower() != ".md":
                continue
            relative_path = source_path.relative_to(managed_root)
            destination_path = source_root / relative_path
            if destination_path.exists():
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            copied_paths.append(destination_path)

        return copied_paths


    def save_skill_pattern(
        self,
        name: str,
        description: str,
        instructions: str,
        supporting_files: dict[str, str] | None = None,
    ) -> Path:
        """Persist a Deep Agents skill as a directory with SKILL.md."""

        skill_dir = self._next_available_skill_dir(name)
        skill_dir.mkdir(parents=True, exist_ok=False)

        skill_markdown = dedent(
            f"""
            ---
            name: {self.slugify(name)}
            description: {description}
            metadata:
              author: devmate
              version: "1.0"
            allowed-tools: {DEFAULT_ALLOWED_TOOLS}
            ---

            # {name}

            {instructions.strip()}
            """
        ).strip() + "\n"
        (skill_dir / _SKILL_MARKDOWN_NAME).write_text(
            skill_markdown,
            encoding="utf-8",
        )

        for relative_path, content in (supporting_files or {}).items():
            target_path = skill_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
        return skill_dir

    def save_skill_from_run(
        self,
        prompt: str,
        summary: str,
        changed_files: list[str],
    ) -> Path:
        """Auto-save a reusable skill after a successful task."""

        task_name = prompt.splitlines()[0][:80]
        description = (
            "Use this skill when the user asks for a task similar to: "
            f"{task_name}. It captures a successful DevMate workflow for planning, "
            "consulting local knowledge and web search when needed, generating "
            "files, and summarizing the results."
        )
        file_lines = (
            "\n".join(f"- {path}" for path in changed_files)
            or "- No file list captured"
        )
        instructions = dedent(
            f"""
            ## Overview
            This skill was generated automatically from a successful DevMate run.

            ## Instructions
            1. Start by understanding the user goal.
            2. Query `search_knowledge_base` for internal conventions that may apply.
            3. Use `search_web` if external frameworks, APIs,
               or best practices are needed.
            4. Plan the target file tree before writing code.
            5. Generate or edit files using workspace-relative paths only.
            6. Use `write_external_file` only when the task explicitly requires
               saving docs, skills, uploads, or workflow artifacts into the
               managed workspace output roots.
            7. Validate the output when a lightweight verification command is
               practical.
            8. Finish with a concise summary of files and next steps.

            ## Previously changed files
            {file_lines}

            ## Notes from the successful run
            {summary.strip()[:1200]}
            """
        )
        return self.save_skill_pattern(
            name=task_name,
            description=description,
            instructions=instructions,
        )
