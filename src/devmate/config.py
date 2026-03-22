"""Configuration loading and validation for DevMate."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_NAME = "config.toml"
LOCAL_CONFIG_NAME = "config.local.toml"


class ModelSettings(BaseModel):
    """LLM and embedding model settings."""

    provider: Literal["openai", "deepseek", "auto"] = "deepseek"
    ai_base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model_name: str = "deepseek-chat"
    embedding_model_name: str = "text-embedding-3-small"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: int = 120
    embedding_provider: str = "openai"


class SearchSettings(BaseModel):
    """Web search and MCP client settings."""

    tavily_api_key: str = ""
    mcp_url: str = "http://localhost:8001/mcp"
    max_results: int = 5
    allow_mock_search: bool = False


class LangSmithSettings(BaseModel):
    """LangSmith tracing settings."""

    enabled: bool = True
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langsmith_api_key: str = ""
    project: str = "devmate"
    endpoint: str = "https://api.smith.langchain.com"


class SkillsSettings(BaseModel):
    """Agent skills settings."""

    skills_dir: str = "workspace/.skills"
    auto_save_on_success: bool = True


class RagSettings(BaseModel):
    """Local knowledge base settings."""

    docs_dir: str = "workspace/docs"
    persist_dir: str = "workspace/data/chroma"
    research_cache_dir: str = "workspace/docs/research_cache"
    collection_name: str = "devmate-knowledge"
    top_k: int = 4
    chunk_size: int = 1200
    chunk_overlap: int = 150


class AppSettings(BaseModel):
    """Application server and workspace settings."""

    workspace_dir: str = "workspace"
    state_dir: str = "workspace/data/runtime_state"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"


class MCPSettings(BaseModel):
    """MCP server settings."""

    host: str = "0.0.0.0"
    port: int = 8001
    streamable_http_path: str = "/mcp"


class WorkflowSettings(BaseModel):
    """Single-agent staged workflow settings."""

    artifacts_dir: str = "workspace/data/workflow_runs"
    deep_optimization_default: bool = False
    local_python_tool_default: bool = False
    max_deep_optimization_rounds: int = 2
    standard_builder_repair_rounds: int = 1


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: ModelSettings = Field(default_factory=ModelSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    skills: SkillsSettings = Field(default_factory=SkillsSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    project_root: Path = Field(default_factory=Path.cwd)
    config_path: Path = Field(
        default_factory=lambda: Path.cwd() / DEFAULT_CONFIG_NAME,
    )

    def resolve_path(self, path_value: str | Path) -> Path:
        """Resolve a path relative to the project root."""

        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path.resolve(strict=False)
        return (self.project_root / path).resolve(strict=False)

    def ensure_path_within_workspace(
        self,
        path: Path,
        *,
        label: str,
    ) -> Path:
        """Validate that a path stays inside the workspace."""

        workspace_root = self.workspace_root.resolve(strict=False)
        candidate = path.resolve(strict=False)

        try:
            candidate.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError(
                f"{label} must stay inside the workspace: {candidate}"
            ) from exc

        return candidate

    def resolve_relative_workspace_path(
        self,
        relative_path: str,
        *,
        label: str,
    ) -> Path:
        """Resolve a relative path under the workspace safely."""

        normalized = Path(relative_path.replace("\\", "/"))

        if normalized.is_absolute():
            raise ValueError(
                f"{label} must be a relative path: {relative_path}"
            )

        if any(part == ".." for part in normalized.parts):
            raise ValueError(
                f"{label} must not contain parent traversal: {relative_path}"
            )

        return self.ensure_path_within_workspace(
            self.workspace_root / normalized,
            label=label,
        )

    def _workspace_output_relative_path(self, path_value: str) -> Path:
        """Normalize a managed output path under the workspace root."""

        candidate = Path(path_value.replace("\\", "/"))

        if candidate.is_absolute():
            raise ValueError(
                "Managed output paths must be relative to the workspace: "
                f"{path_value}"
            )

        workspace_name = Path(self.app.workspace_dir).name or "workspace"
        normalized_parts: list[str] = []

        for part in candidate.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                raise ValueError(
                    "Managed output paths must not contain parent traversal: "
                    f"{path_value}"
                )
            normalized_parts.append(part)

        if normalized_parts and normalized_parts[0] in {
            workspace_name,
            "workspace",
        }:
            normalized_parts = normalized_parts[1:]

        if not normalized_parts:
            return Path(".")

        return Path(*normalized_parts)

    def resolve_workspace_output_path(self, path_value: str) -> Path:
        """Resolve a managed output path inside the workspace."""

        workspace_root = self.workspace_root
        relative_path = self._workspace_output_relative_path(path_value)
        return (workspace_root / relative_path).resolve(strict=False)

    def validate_workspace_boundaries(self) -> AppConfig:
        """Ensure all managed directories live under the workspace."""

        self.workspace_root.mkdir(parents=True, exist_ok=True)

        managed_paths = {
            "app.state_dir": self.state_dir,
            "skills.skills_dir": self.skills_dir,
            "rag.docs_dir": self.docs_dir,
            "rag.persist_dir": self.persist_dir,
            "rag.research_cache_dir": self.research_cache_dir,
            "workflow.artifacts_dir": self.workflow_artifacts_dir,
        }

        for label, managed_path in managed_paths.items():
            self.ensure_path_within_workspace(
                managed_path,
                label=label,
            )

        return self

    @property
    def workspace_root(self) -> Path:
        """Return the resolved workspace root directory."""

        return self.resolve_path(self.app.workspace_dir)

    @property
    def workspace_dir(self) -> Path:
        """Return the absolute workspace directory."""

        return self.workspace_root

    @property
    def state_dir(self) -> Path:
        """Return the absolute runtime state directory."""

        return self.resolve_workspace_output_path(self.app.state_dir)

    @property
    def docs_source_dir(self) -> Path:
        """Return the absolute docs source directory from config."""

        return self.resolve_path(self.rag.docs_dir)

    @property
    def docs_dir(self) -> Path:
        """Return the absolute managed docs directory in workspace."""

        return self.resolve_workspace_output_path(self.rag.docs_dir)

    @property
    def persist_source_dir(self) -> Path:
        """Return the absolute vector-store source directory."""

        return self.resolve_path(self.rag.persist_dir)

    @property
    def persist_dir(self) -> Path:
        """Return the absolute vector-store persistence directory."""

        return self.resolve_workspace_output_path(self.rag.persist_dir)

    @property
    def research_cache_dir(self) -> Path:
        """Return the absolute researcher cache directory."""

        return self.resolve_workspace_output_path(
            self.rag.research_cache_dir
        )

    @property
    def skills_source_dir(self) -> Path:
        """Return the absolute skills source directory from config."""

        return self.resolve_path(self.skills.skills_dir)

    @property
    def skills_dir(self) -> Path:
        """Return the absolute managed skills directory in workspace."""

        return self.resolve_workspace_output_path(self.skills.skills_dir)

    @property
    def workflow_artifacts_dir(self) -> Path:
        """Return the absolute workflow artifacts directory."""

        return self.resolve_workspace_output_path(
            self.workflow.artifacts_dir
        )

    @property
    def memory_file(self) -> Path:
        """Return the absolute AGENTS.md path."""

        return (self.project_root / "AGENTS.md").resolve(strict=False)


def is_config_secret_set(value: str | None) -> bool:
    """Return whether a config secret contains a usable value."""

    if not value:
        return False

    normalized = value.strip()
    placeholders = {
        "",
        "your_openai_api_key_here",
        "your_tavily_api_key_here",
        "your_langsmith_api_key_here",
    }
    return normalized not in placeholders


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge two dictionaries."""

    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
            continue

        merged[key] = value

    return merged


def read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file from disk."""

    with path.open("rb") as handle:
        return tomllib.load(handle)


def locate_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve the effective config file location."""

    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv("DEVMATE_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (Path.cwd() / DEFAULT_CONFIG_NAME).resolve()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load DevMate configuration from TOML files."""

    effective_path = locate_config_path(config_path)
    if not effective_path.exists():
        raise FileNotFoundError(
            f"DevMate config file does not exist: {effective_path}"
        )

    raw_config = read_toml(effective_path)
    local_path = effective_path.with_name(LOCAL_CONFIG_NAME)

    if local_path.exists():
        raw_config = deep_merge(raw_config, read_toml(local_path))

    raw_config["project_root"] = effective_path.parent.resolve()
    raw_config["config_path"] = effective_path

    config = AppConfig.model_validate(raw_config)
    config.validate_workspace_boundaries()
    return config
