"""Tests for reproducer.graph.nodes.configure (pure)."""

from __future__ import annotations

import hashlib

import pytest

from reproducer.domain.inventory import FileInfo
from reproducer.graph.nodes.configure import (
    MAX_CONFIG_BYTES,
    RepoInventory,
    classify_files,
    inventory_to_evidence,
    parse_github_url,
    repo_source,
)

SHA = "a" * 40


def _f(path: str, size: int = 10, link: bool = False) -> FileInfo:
    return FileInfo(path=path, size=0 if link else size, is_symlink=link)


FILES = (
    _f("docs/README.md"),
    _f("Readme.rst"),
    _f("README.md"),
    _f("requirements.txt"),
    _f("requirements-dev.txt"),
    _f("envs/environment.yml"),
    _f("setup.py"),
    _f("setup.cfg"),
    _f("pyproject.toml"),
    _f("Pipfile"),
    _f("Dockerfile"),
    _f("docker/Dockerfile.gpu"),
    _f("docker-compose.yaml"),
    _f("configs/base.yaml"),
    _f("configs/exp.yml"),
    _f("configs/meta.json"),
    _f("tox.ini"),
    _f("data/huge.json", size=MAX_CONFIG_BYTES + 1),
    _f("notebooks/demo.ipynb"),
    _f("train.py"),
    _f("pkg/model.py"),
    _f(".gitmodules"),
    _f("link.yaml", link=True),
)


def test_classify_categories() -> None:
    inv = classify_files(FILES, SHA)
    assert inv.commit == SHA
    assert inv.readme == "README.md"  # root wins; .md preferred over .rst
    assert inv.requirement_files == (
        "Pipfile",
        "envs/environment.yml",
        "pyproject.toml",
        "requirements-dev.txt",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
    )
    assert inv.docker_files == ("Dockerfile", "docker-compose.yaml", "docker/Dockerfile.gpu")
    assert inv.config_files == (
        "configs/base.yaml",
        "configs/exp.yml",
        "configs/meta.json",
        "link.yaml",
        "tox.ini",
    )
    assert "data/huge.json" not in inv.config_files
    assert inv.notebooks == ("notebooks/demo.ipynb",)
    assert inv.python_files == ("pkg/model.py", "setup.py", "train.py")
    assert inv.has_submodules is True
    assert len(inv.files) == len(FILES)


def test_classify_is_order_independent() -> None:
    assert classify_files(FILES, SHA) == classify_files(tuple(reversed(FILES)), SHA)


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["docs/README.md", "readme.txt"], "readme.txt"),
        (["a/README.md", "b/c/README"], "a/README.md"),
        (["README", "README.rst"], "README.rst"),
        (["README.MD"], "README.MD"),
        (["src/main.py"], None),
    ],
)
def test_readme_selection(paths: list[str], expected: str | None) -> None:
    assert classify_files([_f(p) for p in paths], SHA).readme == expected


def test_no_submodules() -> None:
    assert classify_files([_f("a.py")], SHA).has_submodules is False


def test_inventory_frozen_and_strict() -> None:
    inv = classify_files(FILES, SHA)
    with pytest.raises(ValueError):
        inv.commit = "x"  # type: ignore[misc]
    with pytest.raises(ValueError):
        RepoInventory.model_validate({**inv.model_dump(), "extra": 1})


def _texts(inv: RepoInventory) -> dict[str, tuple[str, bool]]:
    paths = [inv.readme, *inv.requirement_files, *inv.docker_files]
    return {p: (f"text of {p}", False) for p in paths if p}


def test_inventory_to_evidence_counts_and_shapes() -> None:
    inv = classify_files(FILES, SHA)
    source = repo_source("https://github.com/org/proj", SHA)
    evs = inventory_to_evidence(inv, source, _texts(inv))
    assert len(evs) == 1 + 1 + len(inv.requirement_files) + len(inv.docker_files)

    first, readme, *rest = evs
    assert (first.kind, first.locator) == ("repo_file", "inventory")
    assert first.content == inv.model_dump(mode="json")
    assert (readme.kind, readme.locator) == ("readme", "README.md")
    assert readme.content == {
        "path": "README.md",
        "text": "text of README.md",
        "truncated": False,
        "sha256": hashlib.sha256(b"text of README.md").hexdigest(),
    }
    assert [e.locator for e in rest] == [*inv.requirement_files, *inv.docker_files]
    assert all(e.kind == "repo_file" for e in rest)
    assert all(e.confidence == "observed" and e.authority == "repo" for e in evs)
    assert all(e.source == f"repo:org/proj@{SHA}" for e in evs)
    assert len({e.evidence_id for e in evs}) == len(evs)


def test_inventory_to_evidence_ids_stable() -> None:
    def build() -> list[str]:
        inv = classify_files(list(FILES), SHA)
        return [e.evidence_id for e in inventory_to_evidence(inv, "repo:o/n@x", _texts(inv))]

    assert build() == build()
    shuffled = classify_files(tuple(reversed(FILES)), SHA)
    assert [
        e.evidence_id for e in inventory_to_evidence(shuffled, "repo:o/n@x", _texts(shuffled))
    ] == build()


def test_inventory_to_evidence_skips_missing_texts_and_no_readme() -> None:
    inv = classify_files([_f("requirements.txt"), _f("setup.py"), _f("a.py")], SHA)
    evs = inventory_to_evidence(inv, "repo:o/n@x", {"requirements.txt": ("torch", True)})
    assert [e.locator for e in evs] == ["inventory", "requirements.txt"]
    assert evs[1].content["truncated"] is True


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/openai/CLIP",
        "https://github.com/openai/CLIP/",
        "https://github.com/openai/CLIP.git",
        "https://github.com/openai/CLIP.git/",
        "http://www.github.com/openai/CLIP",
        "git@github.com:openai/CLIP.git",
        "  https://github.com/openai/CLIP  ",
    ],
)
def test_repo_source_variants(url: str) -> None:
    assert repo_source(url, SHA) == f"repo:openai/CLIP@{SHA}"


def test_repo_source_dotted_name() -> None:
    assert parse_github_url("https://github.com/a-b/x.y_z.git") == ("a-b", "x.y_z")


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/openai/CLIP",
        "file:///tmp/repo",
        "https://github.com/openai",
        "https://github.com/openai/CLIP/tree/main",
        "",
    ],
)
def test_repo_source_non_github_raises(url: str) -> None:
    with pytest.raises(ValueError):
        repo_source(url, SHA)
