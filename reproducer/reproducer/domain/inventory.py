"""Repo inventory models (pure: no IO)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FileInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str  # posix, relative to repo root
    size: int  # bytes; 0 for symlinks (never followed)
    is_symlink: bool


class RepoInventory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str
    files: tuple[FileInfo, ...]
    readme: str | None
    requirement_files: tuple[str, ...]
    docker_files: tuple[str, ...]
    config_files: tuple[str, ...]
    notebooks: tuple[str, ...]
    python_files: tuple[str, ...]
    has_submodules: bool
