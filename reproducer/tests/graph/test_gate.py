"""Tests for reproducer.graph.gate (offline via FakeLLM; one llm_live test)."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from reproducer.domain.grounding import RepoPath
from reproducer.graph.gate import GateResult, gated_call, strip_fence
from reproducer.providers.llm import (
    Budget,
    BudgetExceeded,
    FakeLLM,
    LLMError,
    LLMResponse,
    MeteredLLM,
    OllamaLLM,
)
from reproducer.providers.store import RunStore

REPO = {"README.md", "requirements.txt", "setup.py", "clip/model.py"}
CTX = {"repo_files": REPO}


class KeyFiles(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    readme: RepoPath
    requirements: list[RepoPath]


GOOD = '{"readme": "README.md", "requirements": ["requirements.txt"]}'


class Recorder:
    """FakeLLM wrapper that also records the json_schema it was given."""

    def __init__(self, responses: list[str]) -> None:
        self.fake = FakeLLM(responses)
        self.schemas: list[dict | None] = []

    def complete(
        self, system: str, prompt: str, *, max_tokens: int, json_schema: dict | None = None
    ) -> LLMResponse:
        self.schemas.append(json_schema)
        return self.fake.complete(system, prompt, max_tokens=max_tokens, json_schema=json_schema)

    @property
    def prompts(self) -> list[str]:
        return [p for _, p in self.fake.prompts]


def _call(llm: Any, **kw: Any) -> GateResult[KeyFiles]:
    args: dict[str, Any] = {
        "system": "sys",
        "prompt": "Find the key files.",
        "schema": KeyFiles,
        "context": CTX,
        "max_tokens": 128,
    }
    args.update(kw)
    return gated_call(llm, **args)


def test_pass_first_attempt() -> None:
    llm = Recorder([GOOD])
    res = _call(llm)
    assert res.ok and res.attempts == 1 and res.errors == ()
    assert res.value is not None
    assert (res.value.readme, res.value.requirements) == ("README.md", ["requirements.txt"])
    assert llm.schemas == [KeyFiles.model_json_schema()]
    assert llm.prompts == ["Find the key files."]


def test_invalid_json_then_valid() -> None:
    llm = Recorder(["sure! here it is: {readme: README.md", GOOD])
    res = _call(llm)
    assert res.ok and res.attempts == 2
    assert len(res.errors) == 1 and "Invalid JSON" in res.errors[0]
    assert "Your previous answer was rejected:" in llm.prompts[1]
    assert "sure! here it is" in llm.prompts[1]


@pytest.mark.parametrize(
    "text",
    [
        f"```json\n{GOOD}\n```",
        f"```\n{GOOD}\n```",
        f"  \n```JSON\n{GOOD}```\n ",
        f"```json {GOOD}```",
    ],
)
def test_fenced_json_accepted(text: str) -> None:
    res = _call(Recorder([text]))
    assert res.ok and res.attempts == 1


def test_strip_fence_only_one_surrounding_fence() -> None:
    assert strip_fence(" {} ") == "{}"
    assert strip_fence("```json\n{}\n```") == "{}"
    inner = "text ```json\n{}\n``` more"
    assert strip_fence(inner) == inner


def test_hallucinated_repo_path_then_corrected() -> None:
    bad = '{"readme": "README.md", "requirements": ["requirements-dev.txt"]}'
    llm = Recorder([bad, GOOD])
    res = _call(llm)
    assert res.ok and res.attempts == 2
    assert res.errors[0].startswith("requirements.0: Value error, file not in repo")
    feedback = llm.prompts[1]
    assert feedback.startswith("Find the key files.\n\nYour previous answer was rejected:\n")
    assert "requirements.0:" in feedback  # error loc
    assert bad in feedback  # previous answer
    assert feedback.endswith("Return ONLY corrected JSON matching the schema.")


def test_feedback_built_from_original_prompt_not_accumulated() -> None:
    llm = Recorder(["nope", "still nope", GOOD])
    res = _call(llm)
    assert res.ok and res.attempts == 3
    assert llm.prompts[2].count("Your previous answer was rejected") == 1
    assert "still nope" in llm.prompts[2] and "\nnope\n" not in llm.prompts[2]


def test_previous_answer_truncated() -> None:
    llm = Recorder(["x" * 5000, GOOD])
    _call(llm)
    assert "x" * 2000 in llm.prompts[1] and "x" * 2001 not in llm.prompts[1]


def test_fails_after_three_attempts() -> None:
    llm = Recorder(["{}", "[]", '{"readme": "nope.md", "requirements": []}', GOOD])
    res = _call(llm)
    assert (res.ok, res.value, res.attempts) == (False, None, 3)
    assert len(res.errors) == 3
    assert "readme: Field required" in res.errors[0]
    assert "readme: Value error, file not in repo: 'nope.md'" in res.errors[2]
    assert len(llm.prompts) == 3  # 4th scripted answer never requested


def test_error_truncated_to_1000_chars() -> None:
    many = json.dumps({"readme": "README.md", "requirements": [f"bad{i}.txt" for i in range(200)]})
    res = _call(Recorder([many]), max_retries=0)
    assert not res.ok and len(res.errors[0]) == 1000


def test_max_retries_zero_single_attempt() -> None:
    llm = Recorder(["{}", GOOD])
    res = _call(llm, max_retries=0)
    assert (res.ok, res.attempts, len(res.errors)) == (False, 1, 1)
    assert len(llm.prompts) == 1


def test_negative_max_retries_rejected() -> None:
    with pytest.raises(ValueError):
        _call(Recorder([GOOD]), max_retries=-1)


def test_budget_exceeded_propagates_without_retry() -> None:
    inner = FakeLLM(["{}", GOOD])
    llm = MeteredLLM(inner, Budget(max_tokens_total=130), "test")
    with pytest.raises(BudgetExceeded):
        _call(llm, max_tokens=128)  # 1st: 128<=130, uses 6; 2nd: 6+128>130 -> raises
    assert len(inner.prompts) == 1


def test_llm_error_propagates() -> None:
    llm = Recorder(["{}"])  # second attempt -> FakeLLM exhausted -> LLMError
    with pytest.raises(LLMError):
        _call(llm)


@pytest.mark.parametrize("ctx", [{}, {"evidence_ids": set()}])
def test_missing_required_context_raises_before_llm_call(ctx: dict[str, Any]) -> None:
    llm = Recorder([GOOD])
    with pytest.raises(ValueError, match=r"missing required key\(s\) \['repo_files'\]"):
        _call(llm, context=ctx)
    assert llm.fake.prompts == []  # zero calls, zero tokens


def test_required_context_custom_keys() -> None:
    llm = Recorder([GOOD])
    with pytest.raises(ValueError, match=r"\['claims', 'evidence_ids'\]"):
        _call(llm, required_context=("repo_files", "evidence_ids", "claims"))
    assert llm.fake.prompts == []
    assert _call(Recorder([GOOD]), required_context=()).ok


@pytest.mark.parametrize("bad", [["README.md"], ("README.md",), {"README.md": 1}, "README.md"])
def test_wrong_type_required_context_raises_before_llm_call(bad: Any) -> None:
    llm = Recorder([GOOD])
    with pytest.raises(ValueError, match=r"\['repo_files'\] must be set/frozenset"):
        _call(llm, context={"repo_files": bad})
    assert llm.fake.prompts == []  # zero calls, zero tokens


def test_frozenset_context_accepted() -> None:
    assert _call(Recorder([GOOD]), context={"repo_files": frozenset(REPO)}).ok


# --------------------------------------------------------------------------- live

RUN_DIR = Path(__file__).resolve().parents[2] / "runs" / "clip"
LIVE_MODEL = os.environ.get("REPRODUCER_LIVE_MODEL", "qwen2.5-coder:7b")
LIVE_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if "://" not in LIVE_HOST:
    LIVE_HOST = f"http://{LIVE_HOST}"


def _inventory_files(run_dir: Path) -> list[str]:
    if not (run_dir / "evidence.jsonl").is_file():
        pytest.skip(f"no evidence at {run_dir}; run `python -m reproducer.cli inventory ...`")
    inventories = [
        ev
        for ev in RunStore(run_dir).load_evidence().values()
        if ev.kind == "repo_file" and ev.locator == "inventory"
    ]
    if not inventories:
        pytest.skip(f"no inventory evidence in {run_dir}")
    return [f["path"] for f in inventories[-1].content["files"]]


def _require_local_model() -> None:
    try:
        with urllib.request.urlopen(f"{LIVE_HOST}/api/tags", timeout=3) as r:
            names = {m.get("name") for m in json.loads(r.read()).get("models", [])}
    except OSError as e:
        pytest.skip(f"Ollama not reachable at {LIVE_HOST}: {e}")
    if LIVE_MODEL not in names:
        pytest.skip(f"model {LIVE_MODEL!r} not pulled (have {sorted(names)})")


@pytest.mark.llm_live
def test_live_gate_key_files_from_clip_inventory() -> None:
    files = _inventory_files(RUN_DIR)
    _require_local_model()
    metered = MeteredLLM(OllamaLLM(LIVE_HOST, LIVE_MODEL), Budget(), "gate-live")
    res = gated_call(
        metered,
        system="You are a precise assistant. Reply only with JSON matching the given schema.",
        prompt=(
            "Repository files:\n"
            + "\n".join(files)
            + "\n\nReturn the README path and the dependency/requirements files "
            "(e.g. requirements*.txt, setup.py, pyproject.toml, environment.yml). "
            "Use paths exactly as listed."
        ),
        schema=KeyFiles,
        context={"repo_files": set(files)},
        max_tokens=256,
    )
    recs = metered.records()
    print(
        f"\n[gate-live] model={LIVE_MODEL} ok={res.ok} attempts={res.attempts} "
        f"value={res.value} errors={list(res.errors)}\n"
        + "\n".join(
            f"  call {i}: in={r.input_tokens} out={r.output_tokens} latency={r.latency_s:.2f}s"
            for i, r in enumerate(recs, 1)
        )
    )
    assert res.ok, res.errors
