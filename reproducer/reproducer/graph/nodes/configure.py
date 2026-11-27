"""Configure node: classify a repo's file list and turn it into evidence.

The helpers are pure: file listings and file texts are produced by providers/git.py and
passed in. Only the graph node `configure` does IO (clone/reuse + reads, via providers);
it returns evidence in the state and writes nothing to the run store (persist does).
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.inventory import FileInfo, RepoInventory

MAX_CONFIG_BYTES = 1_000_000  # config candidates larger than this are data, not config

_REQUIREMENT_PATTERNS = (
    "requirements*.txt",
    "environment.yml",
    "environment.yaml",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "pipfile",
)
_DOCKER_PATTERNS = ("dockerfile*", "docker-compose*")
_CONFIG_SUFFIXES = (".yaml", ".yml", ".json", ".toml", ".cfg", ".ini")
_README_EXT_RANK = {".md": 0, ".rst": 1, ".txt": 2, "": 3}


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1].lower()


def _matches(path: str, patterns: Iterable[str]) -> bool:
    name = _basename(path)
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _readme_key(path: str) -> tuple[int, int, str]:
    name = _basename(path)
    ext = name[len("readme") :]
    return (path.count("/"), _README_EXT_RANK.get(ext, 4), path)


def classify_files(files: Iterable[FileInfo], commit: str) -> RepoInventory:
    """Bucket a repo listing into categories. Output is sorted, so it is order-independent."""
    ordered = tuple(sorted(files, key=lambda f: f.path))
    readmes = [f.path for f in ordered if _basename(f.path).startswith("readme")]
    requirements = tuple(f.path for f in ordered if _matches(f.path, _REQUIREMENT_PATTERNS))
    docker = tuple(f.path for f in ordered if _matches(f.path, _DOCKER_PATTERNS))
    excluded = set(requirements) | set(docker)
    configs = tuple(
        f.path
        for f in ordered
        if _basename(f.path).endswith(_CONFIG_SUFFIXES)
        and f.path not in excluded
        and f.size <= MAX_CONFIG_BYTES
    )
    return RepoInventory(
        commit=commit,
        files=ordered,
        readme=min(readmes, key=_readme_key) if readmes else None,
        requirement_files=requirements,
        docker_files=docker,
        config_files=configs,
        notebooks=tuple(f.path for f in ordered if _basename(f.path).endswith(".ipynb")),
        python_files=tuple(f.path for f in ordered if _basename(f.path).endswith(".py")),
        has_submodules=any(f.path == ".gitmodules" for f in ordered),
    )


def _file_content(path: str, text: str, truncated: bool) -> dict[str, Any]:
    return {
        "path": path,
        "text": text,
        "truncated": truncated,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def inventory_to_evidence(
    inv: RepoInventory, source: str, texts: Mapping[str, tuple[str, bool]]
) -> tuple[Evidence, ...]:
    """Inventory + README + requirement/docker file evidence (observed, authority=repo).

    texts maps repo-relative path -> (text, truncated). Files absent from texts (e.g. refused
    by read_text as unsafe) get no file evidence; they still appear in the inventory.
    """

    def ev(kind: Any, locator: str, content: dict[str, Any]) -> Evidence:
        return make_evidence(
            kind=kind,
            source=source,
            locator=locator,
            content=content,
            confidence="observed",
            authority="repo",
        )

    out = [ev("repo_file", "inventory", inv.model_dump(mode="json"))]
    if inv.readme is not None and inv.readme in texts:
        out.append(ev("readme", inv.readme, _file_content(inv.readme, *texts[inv.readme])))
    for path in (*inv.requirement_files, *inv.docker_files):
        if path in texts:
            out.append(ev("repo_file", path, _file_content(path, *texts[path])))
    return tuple(out)


_GITHUB_RE = re.compile(
    r"^(?:https?://(?:www\.)?github\.com/|ssh://git@github\.com/|git@github\.com:)"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?/*$"
)


def parse_github_url(url: str) -> tuple[str, str]:
    """GitHub URL -> (owner, name). Anything else -> ValueError."""
    m = _GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError(f"not a GitHub repository URL (only GitHub supported for now): {url!r}")
    return m.group("owner"), m.group("name")


LOCAL_OWNER = "local"  # source owner of a local (file:) repository snapshot


def local_repo_path(url: str) -> Path | None:
    """'file:///abs/dir' -> Path('/abs/dir'); None for any other URL."""
    u = url.strip()
    return Path(unquote(urlsplit(u).path)) if u.startswith("file://") else None


def parse_repo_url(url: str) -> tuple[str, str]:
    """GitHub URL -> (owner, name); 'file:///.../<name>' -> ('local', name)."""
    path = local_repo_path(url)
    if path is not None:
        return LOCAL_OWNER, path.name
    return parse_github_url(url)


def repo_source(url: str, sha: str) -> str:
    """'https://github.com/openai/CLIP.git' + sha -> 'repo:openai/CLIP@<sha>';
    'file:///x/toyrepo' + sha -> 'repo:local/toyrepo@<sha>'."""
    owner, name = parse_repo_url(url)
    return f"repo:{owner}/{name}@{sha}"


def configure(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Graph node: clone (or reuse <out_dir>/repo), inventory, extractors, shell scripts,
    version/metric/command scans. Returns evidence + the store's existing unknown events
    (so subtasks skip already-open unknowns). Raises ValueError / CloneError on bad input."""
    # imported here: subtasks imports this module (MAX_CONFIG_BYTES)
    from reproducer.graph.nodes.subtasks import (
        MAX_SHELL_SCRIPTS,
        collect_texts,
        command_evidence,
        extract_repo_facts,
        file_evidence,
        is_shell_script,
        metric_line_evidence,
        version_evidence,
    )
    from reproducer.providers.git import (
        CloneError,
        clone_repo,
        head_sha,
        list_files,
        read_text,
        snapshot_local_repo,
        validate_commit,
    )

    url, commit = state["repo_url"], state.get("commit")
    parse_repo_url(url)
    if commit is not None:
        validate_commit(commit)
    repo_dir = Path(state["out_dir"]) / "repo"
    if repo_dir.exists():  # reuse-clone semantics
        sha = head_sha(repo_dir)
        if commit is not None and sha != commit:
            raise CloneError(
                f"{repo_dir} is at {sha}, not the requested commit {commit}; use a fresh --out"
            )
    elif (local := local_repo_path(url)) is not None:
        if commit is not None:
            raise ValueError("--commit does not apply to a local (file:) repository snapshot")
        sha = snapshot_local_repo(local, repo_dir)
    else:
        sha = clone_repo(url, repo_dir, commit=commit)
    source = repo_source(url, sha)

    inv = classify_files(list_files(repo_dir), sha)
    texts: dict[str, tuple[str, bool]] = {}
    for path in ([inv.readme] if inv.readme else []) + [*inv.requirement_files, *inv.docker_files]:
        try:
            texts[path] = read_text(repo_dir, path)
        except (ValueError, OSError):
            continue  # unsafe/unreadable: still listed in the inventory
    evidence: dict[str, Evidence] = {}
    for ev in inventory_to_evidence(inv, source, texts):
        evidence.setdefault(ev.evidence_id, ev)
    for ev in extract_repo_facts(inv, collect_texts(repo_dir, inv), source):
        evidence.setdefault(ev.evidence_id, ev)
    scripts = [f.path for f in inv.files if not f.is_symlink and is_shell_script(f.path)]
    for path in scripts[:MAX_SHELL_SCRIPTS]:
        try:
            text, truncated = read_text(repo_dir, path)
        except (ValueError, OSError):
            continue
        ev = file_evidence(source, path, text, truncated)
        evidence.setdefault(ev.evidence_id, ev)
    scans = (*version_evidence(evidence), *metric_line_evidence(evidence))
    scans += command_evidence(evidence)
    for ev in scans:
        evidence.setdefault(ev.evidence_id, ev)
    return {
        "source": source,
        "evidence": list(evidence.values()),
        "events": list(deps["store"].load_events()),
        "status": "running",
        "configure_attempts": state.get("configure_attempts", 0) + 1,
    }
