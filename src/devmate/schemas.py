"""Pydantic schemas for API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request payload for project generation."""

    prompt: str = Field(..., min_length=1)
    deep_optimization: bool = False
    local_python_tool_enabled: bool = False
    max_deep_optimization_rounds: int = Field(0, ge=0)
    include_initial_files: bool = False
    initial_file_paths: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    """Structured generation response."""

    prompt: str
    reply: str
    changed_files: list[str] = Field(default_factory=list)
    output_files: list[str] = Field(default_factory=list)
    file_operations: list[str] = Field(default_factory=list)
    saved_skill: str | None = None
    delivery_zip: str | None = None
    verification_passed: bool | None = None
    optimization_rounds_used: int = 0
    deep_optimization: bool = False
    local_python_tool_enabled: bool = False
    max_deep_optimization_rounds: int = 0
    include_initial_files: bool = False
    initial_file_paths: list[str] = Field(default_factory=list)
    immediate_output_requested: bool = False


class GenerationJobStartResponse(BaseModel):
    """Response returned when a background generation job starts."""

    job_id: str
    status: str


class GenerationJobStatusResponse(BaseModel):
    """Current status of a generation job."""

    job_id: str
    prompt: str
    status: str
    stage: str = ""
    current_phase: str = ""
    reply: str = ""
    changed_files: list[str] = Field(default_factory=list)
    output_files: list[str] = Field(default_factory=list)
    file_operations: list[str] = Field(default_factory=list)
    saved_skill: str | None = None
    error: str | None = None
    delivery_zip: str | None = None
    verification_passed: bool | None = None
    optimization_rounds_used: int = 0
    deep_optimization: bool = False
    local_python_tool_enabled: bool = False
    max_deep_optimization_rounds: int = 0
    include_initial_files: bool = False
    initial_file_paths: list[str] = Field(default_factory=list)
    immediate_output_requested: bool = False


class UploadedInitialFile(BaseModel):
    """Metadata for an uploaded initial file."""

    path: str
    name: str
    size_bytes: int


class UploadedInitialFileList(BaseModel):
    """List response for uploaded initial files."""

    files: list[UploadedInitialFile] = Field(default_factory=list)


class WorkspaceListing(BaseModel):
    """Workspace listing response."""

    files: list[str]
