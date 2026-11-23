"""Tests for reproducer.graph.nodes.subtasks (deterministic repo-fact extractors)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from reproducer.domain.evidence import Evidence
from reproducer.domain.inventory import FileInfo
from reproducer.graph.nodes.configure import classify_files
from reproducer.graph.nodes.subtasks import (
    DIGEST_HEAD_CHARS,
    ContextSection,
    candidate_entrypoints,
    collect_texts,
    config_keys,
    digest_evidence,
    extract_repo_facts,
    facts_to_evidence,
    file_evidence,
    is_skipped_path,
    pack_context,
    parse_environment_yml,
    parse_pipfile,
    parse_pyproject,
    parse_requirements_txt,
    parse_setup_cfg,
    parse_setup_py,
)

SRC = "repo:org/proj@" + "a" * 40

# --------------------------------------------------------------------------- requirements.txt

REQS = """\
# top comment
numpy==1.21.0
torch>=1.7.1,<2.0  # inline comment
scipy ~= 1.5
requests[security,socks]>=2.0
dataclasses; python_version < "3.7"
tqdm
ftfy \\
    >=6.0
-r base.txt
--constraint constraints.txt
-cother.txt
git+https://github.com/openai/CLIP.git
pkg @ https://example.com/pkg.whl
-e .
./local_pkg
--extra-index-url https://download.pytorch.org/whl/cu113
numpy=1.2
"""


def test_requirements_parsed() -> None:
    facts = parse_requirements_txt(REQS)
    by = {r.name: r for r in facts.requirements}
    assert list(by) == [
        "numpy",
        "torch",
        "scipy",
        "requests",
        "dataclasses",
        "tqdm",
        "ftfy",
    ]
    assert by["numpy"].spec == "==1.21.0"
    assert by["torch"].spec == ">=1.7.1,<2.0"
    assert by["torch"].raw == "torch>=1.7.1,<2.0"  # comment stripped
    assert by["scipy"].spec == "~=1.5"
    assert by["requests"].extras == ("security", "socks")
    assert by["requests"].spec == ">=2.0"
    assert by["dataclasses"].spec is None
    assert by["dataclasses"].marker == 'python_version < "3.7"'
    assert by["tqdm"].spec is None
    assert by["ftfy"].spec == ">=6.0" and by["ftfy"].line_no == 8  # continuation joined
    assert [(i.kind, i.path) for i in facts.includes] == [
        ("requirement", "base.txt"),
        ("constraint", "constraints.txt"),
        ("constraint", "other.txt"),
    ]


def test_requirements_unsupported_reported_not_dropped() -> None:
    reasons = [(u.reason, u.raw) for u in parse_requirements_txt(REQS).unsupported]
    assert reasons == [
        ("url", "git+https://github.com/openai/CLIP.git"),
        ("url", "pkg @ https://example.com/pkg.whl"),
        ("editable", "-e ."),
        ("path", "./local_pkg"),
        ("option", "--extra-index-url https://download.pytorch.org/whl/cu113"),
        ("unparsable", "numpy=1.2"),
    ]


def test_requirements_hash_option_and_empty() -> None:
    facts = parse_requirements_txt("six==1.16 --hash=sha256:abc\n\n   \n# only\n")
    assert [(r.name, r.spec) for r in facts.requirements] == [("six", "==1.16")]
    assert parse_requirements_txt("").requirements == ()


# --------------------------------------------------------------------------- setup.py


def test_setup_py_literals() -> None:
    text = """
from setuptools import setup, find_packages
setup(
    name="x",
    packages=find_packages(),
    install_requires=["torch>=1.7", "ftfy"],
    python_requires=">=3.7",
    extras_require={"dev": ["pytest"], "docs": "sphinx"},
)
"""
    f = parse_setup_py(text)
    assert f.install_requires == ("torch>=1.7", "ftfy")
    assert f.python_requires == ">=3.7"
    assert f.extras_require == {"dev": ("pytest",), "docs": ("sphinx",)}
    assert f.unresolved == () and f.error is None


def test_setup_py_non_literal_unresolved() -> None:
    text = """
import setuptools
REQS = open("requirements.txt").read().splitlines()
setuptools.setup(name="x", install_requires=REQS, extras_require={"dev": ["pytest"]}, **extra)
"""
    f = parse_setup_py(text)
    assert f.install_requires is None
    assert f.extras_require == {"dev": ("pytest",)}
    assert f.python_requires is None
    assert set(f.unresolved) == {"unresolved: install_requires", "unresolved: **kwargs"}


def test_setup_py_is_never_executed(tmp_path: Path) -> None:
    sentinel = tmp_path / "SENTINEL"
    text = f"""
import os
os.system("touch {sentinel}")
open({str(sentinel)!r}, "w").write("pwned")
from setuptools import setup
setup(install_requires=[__import__("os").system("touch {sentinel}")], python_requires=">=3.8")
"""
    f = parse_setup_py(text)
    assert not sentinel.exists()
    assert f.python_requires == ">=3.8"
    assert f.unresolved == ("unresolved: install_requires",)


def test_setup_py_resolves_single_module_level_literal() -> None:
    text = """
from setuptools import setup
REQS = ["torch>=1.0", "numpy"]
PY: str = ">=3.6"
EXTRAS = {"dev": ["pytest"]}
setup(install_requires=REQS, python_requires=PY, extras_require=EXTRAS)
"""
    f = parse_setup_py(text)
    assert f.install_requires == ("torch>=1.0", "numpy")
    assert f.python_requires == ">=3.6"
    assert f.extras_require == {"dev": ("pytest",)}
    assert f.unresolved == ()


@pytest.mark.parametrize(
    "prelude",
    [
        'REQS = ["a"]\nREQS = ["b"]',  # reassigned
        'REQS = ["a"]\nREQS += ["b"]',  # augmented
        'REQS = ["a"]\nREQS.append("b")',  # mutated in place
        'if True:\n    REQS = ["a"]',  # conditional
        'try:\n    REQS = ["a"]\nexcept Exception:\n    pass',  # conditional
        'REQS = open("requirements.txt").read().splitlines()',  # non-literal
        "from deps import REQS",  # imported
        'for REQS in [["a"]]:\n    pass',  # loop target
        "",  # unbound
    ],
)
def test_setup_py_name_stays_unresolved(prelude: str) -> None:
    text = f"from setuptools import setup\n{prelude}\nsetup(install_requires=REQS)\n"
    f = parse_setup_py(text)
    assert f.install_requires is None
    assert f.unresolved == ("unresolved: install_requires",)


def test_setup_py_name_assigned_after_call_unresolved() -> None:
    f = parse_setup_py('from setuptools import setup\nsetup(install_requires=R)\nR = ["a"]\n')
    assert f.unresolved == ("unresolved: install_requires",)


def test_setup_py_errors() -> None:
    assert parse_setup_py("print 'py2'").error.startswith("SyntaxError")  # type: ignore[union-attr]
    assert parse_setup_py("x = 1").error == "no setup() call found"


# --------------------------------------------------------------------------- setup.cfg / Pipfile


def test_setup_cfg() -> None:
    text = """
[metadata]
name = x
version = %(not)s interpolated
[options]
python_requires = >=3.7
install_requires =
    torch>=1.7  # comment
    dataclasses; python_version < "3.7"

    numpy
[options.extras_require]
dev = pytest
docs = file: docs/requirements.txt
"""
    f = parse_setup_cfg(text)
    assert f.install_requires == ("torch>=1.7", 'dataclasses; python_version < "3.7"', "numpy")
    assert f.python_requires == ">=3.7"
    assert f.extras_require == {"dev": ("pytest",)}
    assert f.unresolved == ("unresolved: extras_require.docs",)


def test_setup_cfg_directives_and_errors() -> None:
    f = parse_setup_cfg("[options]\ninstall_requires = file: requirements.txt\n")
    assert f.install_requires is None
    assert f.unresolved == ("unresolved: install_requires",)
    assert parse_setup_cfg("[metadata]\nname = x\n").install_requires is None
    assert parse_setup_cfg("no section header\n").error is not None
    assert parse_setup_cfg("[a]\nx=1\n[a]\nx=2\n").error is not None


def test_pipfile() -> None:
    text = """
[[source]]
url = "https://pypi.org/simple"
[packages]
torch = "==1.7.1"
numpy = "*"
requests = {version = ">=2.0", extras = ["socks"]}
mylib = {git = "https://github.com/x/mylib.git", ref = "main"}
local = {path = ".", editable = true}
[dev-packages]
pytest = "*"
[requires]
python_version = "3.8"
"""
    f = parse_pipfile(text)
    assert f.packages == {"torch": "==1.7.1", "numpy": None, "requests": ">=2.0"}
    assert f.dev_packages == {"pytest": None}
    assert f.python_version == "3.8"
    assert [u.split(":")[0] for u in f.unsupported] == ["packages.mylib", "packages.local"]


def test_pipfile_full_version_and_invalid() -> None:
    assert parse_pipfile('[requires]\npython_full_version = "3.8.10"\n').python_version == "3.8.10"
    assert parse_pipfile("").packages == {}
    assert parse_pipfile("[packages\n").error is not None


# --------------------------------------------------------------------------- pyproject.toml


def test_pyproject() -> None:
    text = """
[project]
name = "x"
requires-python = ">=3.9"
dependencies = ["torch>=2", "numpy"]
dynamic = ["version"]
[project.optional-dependencies]
dev = ["pytest", "ruff"]
"""
    f = parse_pyproject(text)
    assert f.dependencies == ("torch>=2", "numpy")
    assert f.requires_python == ">=3.9"
    assert f.optional_dependencies == {"dev": ("pytest", "ruff")}
    assert f.notes == ("dynamic: version",)


def test_pyproject_poetry_and_invalid() -> None:
    f = parse_pyproject('[tool.poetry.dependencies]\npython = "^3.8"\n')
    assert f.dependencies is None
    assert "tool.poetry present (not parsed)" in f.notes and "no [project] table" in f.notes
    assert parse_pyproject("[project\n").error is not None


# --------------------------------------------------------------------------- environment.yml


def test_environment_yml() -> None:
    text = """
name: clip
channels: [pytorch, conda-forge]
dependencies:
  - python=3.8
  - pytorch::pytorch=1.7.1
  - cudatoolkit
  - pip
  - pip:
      - ftfy
      - regex>=2020
  - {weird: thing}
"""
    f = parse_environment_yml(text)
    assert f.name == "clip"
    assert f.channels == ("pytorch", "conda-forge")
    assert f.python == "3.8"
    assert f.conda_dependencies == ("python=3.8", "pytorch::pytorch=1.7.1", "cudatoolkit", "pip")
    assert f.pip_dependencies == ("ftfy", "regex>=2020")
    assert f.unsupported == ('{"weird": "thing"}',)


def test_environment_yml_python_range_and_errors() -> None:
    assert parse_environment_yml("dependencies: ['python>=3.7,<3.10']").python == ">=3.7,<3.10"
    assert parse_environment_yml("dependencies: [numpy]").python is None
    assert parse_environment_yml("a: [").error is not None
    assert parse_environment_yml("- a\n- b").error is not None
    # unsafe tags are rejected by safe_load, never constructed
    assert parse_environment_yml("!!python/object/apply:os.system ['true']").error is not None


# --------------------------------------------------------------------------- config keys


def test_config_keys_yaml_nested_and_lists() -> None:
    text = """
model:
  name: vit
  dims: [512, 768]
  layers:
    - {width: 64}
    - {width: 128, act: gelu}
optim: {lr: 0.001}
empty: {}
"""
    assert config_keys(text, ".yaml") == frozenset(
        {
            "model.name",
            "model.dims",
            "model.layers",
            "model.layers.0.width",
            "model.layers.1.width",
            "model.layers.1.act",
            "optim.lr",
            "empty",
        }
    )


def test_config_keys_json_toml_and_invalid() -> None:
    assert config_keys('{"a": {"b": 1}, "c": [1]}', ".json") == frozenset({"a.b", "c"})
    assert config_keys("[a]\nb = 1\n[a.c]\nd = 2\n", ".TOML") == frozenset({"a.b", "a.c.d"})
    assert config_keys("", ".yml") == frozenset()
    for text, suffix in [("a: [", ".yaml"), ("{", ".json"), ("[a", ".toml")]:
        assert isinstance(config_keys(text, suffix), str)
    assert config_keys("x=1", ".ini") == "unsupported config suffix: '.ini'"


def test_config_keys_recursive_yaml_does_not_crash() -> None:
    assert isinstance(config_keys("a: &x [*x]", ".yaml"), str)
    bomb = "a: &a [1,2,3,4,5,6,7,8,9,0]\n" + "".join(
        f"{c}: &{c} [*{p},*{p},*{p},*{p},*{p},*{p},*{p},*{p},*{p}]\n"
        for p, c in zip("abcdefgh", "bcdefghi", strict=True)
    )
    assert "more than" in config_keys(bomb, ".yaml")  # aliases explode; capped, not hung


# --------------------------------------------------------------------------- entrypoints

TRAIN = """
import argparse
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", "-l", type=float)
    p.add_argument("data")
    p.add_argument(NAME_VAR)
if __name__ == "__main__":
    main()
"""
CLICK = """
import click
@click.command()
@click.option("--epochs")
def cli(epochs): ...
"""


def test_candidate_entrypoints() -> None:
    scan = candidate_entrypoints(
        {
            "train.py": TRAIN,
            "pkg/cli.py": CLICK,
            "pkg/model.py": "class Model:\n    pass\n",
            "guard_only.py": "if '__main__' == __name__:\n    pass\n",
            "old.py": "print 'hi'\n",
        }
    )
    by = {c.path: c for c in scan.candidates}
    assert list(by) == ["guard_only.py", "pkg/cli.py", "train.py"]  # sorted; model.py absent
    t = by["train.py"]
    assert (t.has_main_guard, t.uses_argparse, t.uses_click) == (True, True, False)
    assert t.arg_names == ("--lr", "-l", "data")
    c = by["pkg/cli.py"]
    assert (c.has_main_guard, c.uses_argparse, c.uses_click) == (False, False, True)
    assert c.arg_names == ("--epochs",)
    assert by["guard_only.py"].has_main_guard
    assert list(scan.unparsable) == ["old.py"]


# --------------------------------------------------------------------------- pack_context


def test_pack_context_fits_and_priority() -> None:
    secs = [
        {"title": "low", "text": "L", "priority": 1},
        ContextSection(title="high", text="H", priority=5),
        {"title": "mid", "text": "M", "priority": 3},
        {"title": "mid2", "text": "M2", "priority": 3},
    ]
    assert pack_context(secs, 1000) == "## high\nH\n\n## mid\nM\n\n## mid2\nM2\n\n## low\nL"


def test_pack_context_truncates_with_markers_and_hard_cap() -> None:
    secs = [
        {"title": "a", "text": "x" * 100, "priority": 2},
        {"title": "b", "text": "y" * 500, "priority": 1},
        {"title": "c", "text": "z" * 50, "priority": 0},
    ]
    out = pack_context(secs, 300)
    assert len(out) <= 300
    assert out.startswith("## a\n" + "x" * 100 + "\n\n## b\n")
    assert "[truncated " in out and "chars]" in out
    assert out.endswith("[omitted 1 sections]")
    cut = int(out.split("[truncated ")[1].split(" chars]")[0])
    assert out.count("y") + cut == 500
    for n in range(0, 400, 7):
        assert len(pack_context(secs, n)) <= n
        assert pack_context(secs, n) == pack_context(list(secs), n)  # deterministic
    with pytest.raises(ValueError):
        pack_context(secs, -1)


# --------------------------------------------------------------------------- evidence


def test_facts_to_evidence_stable_ids_and_links() -> None:
    src = file_evidence(SRC, "requirements.txt", "numpy\n", False)
    facts = parse_requirements_txt("numpy\n")
    a = facts_to_evidence(
        source=SRC,
        path="requirements.txt",
        parser="requirements_txt",
        result=facts,
        source_evidence_id=src.evidence_id,
    )
    b = facts_to_evidence(
        source=SRC,
        path="requirements.txt",
        parser="requirements_txt",
        result=parse_requirements_txt("numpy\n"),
        source_evidence_id=src.evidence_id,
    )
    assert a.evidence_id == b.evidence_id
    assert (a.kind, a.locator, a.confidence, a.authority) == (
        "repo_file",
        "requirements.txt#parsed",
        "derived",
        "repo",
    )
    assert a.links == (src.evidence_id,)
    assert a.content["parser"] == "requirements_txt"
    assert a.content["result"]["requirements"][0]["name"] == "numpy"
    Evidence.model_validate_json(a.model_dump_json())  # id re-verifies
    keys = facts_to_evidence(
        source=SRC,
        path="c.yaml",
        parser="config_keys",
        result={"keys": frozenset({"b", "a"})},
        source_evidence_id=src.evidence_id,
    )
    assert keys.content["result"] == {"keys": ["a", "b"]}


def test_file_evidence_matches_configure_ids() -> None:
    from reproducer.graph.nodes.configure import inventory_to_evidence

    inv = classify_files([FileInfo(path="requirements.txt", size=6, is_symlink=False)], "a" * 40)
    ev = inventory_to_evidence(inv, SRC, {"requirements.txt": ("numpy\n", False)})
    assert file_evidence(SRC, "requirements.txt", "numpy\n", False).evidence_id in {
        e.evidence_id for e in ev
    }


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


def test_collect_and_extract(tmp_path: Path) -> None:
    from reproducer.providers.git import list_files

    repo = _repo(
        tmp_path / "r",
        {
            "requirements.txt": "numpy\n",
            "setup.py": "from setuptools import setup\nsetup(python_requires='>=3.8')\n",
            "configs/a.yaml": "lr: 1\n",
            "configs/b.ini": "[x]\n",
            "train.py": TRAIN,
            "pkg/deep/util.py": "x = 1\n",
            "pkg/run_eval.py": "if __name__ == '__main__':\n    pass\n",
            "setup.cfg": "[options]\npython_requires = >=3.8\n",
            "Pipfile": '[requires]\npython_version = "3.9"\n',
            ".github/workflows/ci.yml": "on: [push]\n",
            ".github/scripts/run_ci.py": "if __name__ == '__main__':\n    pass\n",
            "sub/.github/requirements.txt": "numpy\n",
        },
    )
    inv = classify_files(list_files(repo), "a" * 40)
    texts = collect_texts(repo, inv, max_files=2)
    assert not any(".github/" in p for p in texts)  # skipped even though name-matched
    py = [p for p in texts if p.endswith(".py") and p != "setup.py"]
    assert py == ["train.py", "pkg/run_eval.py"]  # name-matched first, capped at 2
    assert "configs/b.ini" in texts  # read; no parser for .ini

    ev = extract_repo_facts(inv, texts, SRC)
    derived = {e.locator: e for e in ev if e.confidence == "derived"}
    assert set(derived) == {
        "requirements.txt#parsed",
        "setup.py#parsed",
        "setup.cfg#parsed",
        "Pipfile#parsed",
        "configs/a.yaml#parsed",
        "train.py#parsed",
        "pkg/run_eval.py#parsed",
    }
    observed = {e.evidence_id: e for e in ev if e.confidence == "observed"}
    assert all(d.links[0] in observed for d in derived.values())
    assert derived["configs/a.yaml#parsed"].content["result"] == {"keys": ["lr"]}
    assert derived["setup.cfg#parsed"].content["result"]["python_requires"] == ">=3.8"
    assert derived["Pipfile#parsed"].content["result"]["python_version"] == "3.9"

    by_loc = {e.locator: e for e in observed.values()}
    assert set(by_loc["requirements.txt"].content) == {"path", "text", "truncated", "sha256"}
    for loc in ("configs/a.yaml", "train.py"):
        c = by_loc[loc].content
        assert set(c) == {"path", "size", "sha256", "head", "truncated"}
        assert c["size"] == (repo / loc).stat().st_size and c["truncated"] is False
        assert c["head"] == (repo / loc).read_text()
    assert extract_repo_facts(inv, texts, SRC) == ev  # deterministic

    # .github/ is filtered in the pure step too, even if a caller passes its text
    gh = {**texts, ".github/workflows/ci.yml": ("on: [push]\n", False)}
    assert extract_repo_facts(inv, gh, SRC) == ev


def test_is_skipped_path() -> None:
    assert is_skipped_path(".github/workflows/x.yml")
    assert is_skipped_path("a/.github/x.py")
    assert not is_skipped_path(".github")  # a file named .github is not "under" it
    assert not is_skipped_path("github/x.yml")


def test_digest_evidence_head_and_hash() -> None:
    big = "x" * (DIGEST_HEAD_CHARS + 5)
    e = digest_evidence(SRC, "train.py", big, False, size=123)
    assert e.locator == "train.py" and e.confidence == "observed"
    assert len(e.content["head"]) == DIGEST_HEAD_CHARS and e.content["truncated"] is True
    assert e.content["sha256"] == hashlib.sha256(big.encode()).hexdigest()
    assert e.content["size"] == 123
    assert digest_evidence(SRC, "t.py", "x", True, 1).content["truncated"] is True  # read cap
    assert digest_evidence(SRC, "t.py", "x", False, 1) == digest_evidence(
        SRC, "t.py", "x", False, 1
    )
