"""Grounded critic node (graph/nodes/critic.py) and its smoke_loop integration.

Unit tests: FakeLLM + a fake container runner that "runs" the patched per-run copy by reading
it (no Docker). Marked tests:
    docker            the real smoke path in the Cutout image (torch 1.12), FakeLLM patch
    llm_live + docker the same fixture with the real critic backend (REPRODUCER_LLM_CRITIC)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from reproducer.domain.ledger import compute_ledger, make_smoke_evidence
from reproducer.domain.traceback_parse import Frame, ParsedTraceback
from reproducer.graph.nodes.critic import (
    CriticState,
    critic_patch_evidence,
    critic_prompt,
    critic_prompt_builder,
    critic_revert_evidence,
    fold_critic_state,
    run_critic,
)
from reproducer.graph.nodes.smoke import PROBE_OK, SmokeInputs, smoke_loop
from reproducer.providers.docker_runner import RunResult
from reproducer.providers.llm import Budget, ContextTooLong, FakeLLM, MeteredLLM, llm_for_node
from reproducer.rules.classify import ClassifyContext

PROGRESS = (
    "import sys\n"
    "for i in range(1, 4):\n"
    "    sys.stderr.write('\\r%d%%|##| %d/3 [00:0%d<00:0%d, 1.00it/s]' % (i * 33, i, i, 3 - i))\n"
    "    sys.stderr.flush()\n"
    "sys.stderr.write('\\n')\n"
)
FIXTURE_TRAIN = (
    "import torch\n"
    + PROGRESS
    + "loss = torch.tensor(1.5)\n"
    + "print(loss.data[0])\n"
    + "print('test_acc: %.3f' % (loss.item()))\n"
)
BAD_LINE = FIXTURE_TRAIN.split("\n").index("print(loss.data[0])") + 1
METRIC_TEXTS = {"train.py": ["print('test_acc: %.3f' % (loss.item()))"]}
OK_PROBE = ("python", "-c", f"print({PROBE_OK!r})")
INDEX_MSG = (
    "invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in "
    "C++ to convert a 0-dim tensor to a number"
)


def _repo(tmp_path: Path, train: str = FIXTURE_TRAIN) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "train.py").write_text(train)
    return repo


def _inputs(tmp_path: Path, repo: Path, **kw: object) -> SmokeInputs:
    base: dict[str, object] = {
        "image": "img:tag",
        "clone_dir": repo,
        "example": "python train.py",
        "arguments": (),
        "third_party": (),
        "local_modules": (),
        "run_root": tmp_path / "smoke",
        "gpu": False,
        "timeout_s": 60.0,
        "probe_cmd": OK_PROBE,
        "mem": "2g",
    }
    base.update(kw)
    return SmokeInputs(**base)  # type: ignore[arg-type]


def _tb(line: int, message: str = INDEX_MSG, src: str = "print(loss.data[0])") -> ParsedTraceback:
    raw = (
        "Traceback (most recent call last):\n"
        f'  File "train.py", line {line}, in <module>\n'
        f"    {src}\n"
        f"IndexError: {message}"
    )
    frame = Frame(file="train.py", line=line, func="<module>")
    return ParsedTraceback(
        exc_type="IndexError", message=message, frames=(frame,), repo_frame=frame, raw=raw
    )


def _patch_json(line: int, original: str, replacement: str, file: str = "train.py") -> str:
    return json.dumps(
        {
            "file": file,
            "rationale": "0-dim tensor cannot be indexed; .item() reads its value",
            "edits": [
                {
                    "start_line": line,
                    "end_line": line,
                    "original": original,
                    "replacement": replacement,
                }
            ],
        }
    )


def _item_patch_for_prompt(system: str, prompt: str) -> str:
    """A grounded fake critic: patches the line the prompt points at (.data[0] -> .item())."""
    m = re.search(r"innermost repo frame is line (\d+)", prompt)
    assert m, prompt
    line = int(m.group(1))
    src = re.search(rf"^\s*>>\s+{line}\| (.*)$", prompt, re.MULTILINE)
    assert src, prompt
    return _patch_json(line, src.group(1), src.group(1).replace(".data[0]", ".item()"))


VALID = _patch_json(BAD_LINE, "print(loss.data[0])", "print(loss.item())")


def _critic(llm: FakeLLM, repo: Path, **kw: Any):  # type: ignore[no-untyped-def]
    return run_critic(
        llm,
        traceback=kw.pop("traceback", _tb(BAD_LINE)),
        command="python train.py",
        clone_dir=repo,
        repo_files=frozenset({"train.py"}),
        metric_line_texts=METRIC_TEXTS,
        **kw,
    )


# --- node ----------------------------------------------------------------------------------


def test_critic_prompt_is_grounded_and_patch_validated(tmp_path: Path) -> None:
    llm = FakeLLM([VALID])
    out = _critic(llm, _repo(tmp_path))
    assert out.status == "patched" and out.gate_attempts == 1 and out.patch is not None
    assert "-print(loss.data[0])\n+print(loss.item())\n" in out.patch.diff
    assert out.signature.startswith("IndexError|invalid index of a N-dim tensor. Use")
    assert out.signature.endswith(f"|train.py:{BAD_LINE}")
    prompt = llm.prompts[0][1]
    assert _tb(BAD_LINE).raw in prompt  # traceback verbatim
    assert "python train.py" in prompt
    assert re.search(rf">>\s+{BAD_LINE}\| print\(loss.data\[0\]\)", prompt)
    assert re.search(r"\[METRIC\]\s+\d+\| print\('test_acc", prompt)


def test_same_signature_twice_skips_llm(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    llm = FakeLLM([VALID])
    first = _critic(llm, repo)
    second = _critic(llm, repo, last_signature=first.signature)
    assert second.status == "skipped" and not second.counts_as_attempt
    assert len(llm.prompts) == 1  # not called again


def test_new_signature_calls_llm(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    llm = FakeLLM([VALID])
    out = _critic(llm, repo, last_signature="KeyError|'x'|train.py:3")
    assert out.status == "patched" and len(llm.prompts) == 1


def test_gate_retries_invalid_then_valid(tmp_path: Path) -> None:
    wrong_original = _patch_json(BAD_LINE, "print(loss.data[1])", "print(loss.item())")
    llm = FakeLLM(["not json", wrong_original, VALID])
    out = _critic(llm, _repo(tmp_path))
    assert out.status == "patched" and out.gate_attempts == 3 and len(out.errors) == 2
    assert "does not match" in out.errors[1]
    assert "does not match" in llm.prompts[2][1]  # real validation error fed back


def test_gate_exhausted_is_invalid(tmp_path: Path) -> None:
    lr_change = _patch_json(BAD_LINE, "print(loss.data[0])", "print(loss.item() * 0.5)")
    llm = FakeLLM([lr_change] * 3)
    out = _critic(llm, _repo(tmp_path))
    assert out.status == "invalid" and out.patch is None and out.counts_as_attempt
    assert all("numeric literals changed" in e for e in out.errors)


def test_no_repo_frame_is_no_target(tmp_path: Path) -> None:
    tb = _tb(BAD_LINE).model_copy(update={"repo_frame": None})
    llm = FakeLLM([])
    out = _critic(llm, _repo(tmp_path), traceback=tb)
    assert out.status == "no_target" and llm.prompts == []


def test_patch_stack_folds_into_current_text(tmp_path: Path) -> None:
    train = "import torch\nx = torch.tensor(1.0)\na = x.data[0]\nb = x.data[0]\n"
    repo = _repo(tmp_path, train)
    first = _critic(FakeLLM(_item_patch_for_prompt), repo, traceback=_tb(3, src="a = x.data[0]"))
    assert first.patch is not None
    llm = FakeLLM(_item_patch_for_prompt)
    second = _critic(
        llm, repo, traceback=_tb(4, src="b = x.data[0]"), stack=(first.patch,),
        last_signature=first.signature,
    )  # fmt: skip
    assert second.status == "patched" and second.patch is not None
    assert "a = x.item()" in llm.prompts[0][1]  # the prompt shows the patched text
    assert "-b = x.data[0]\n+b = x.item()\n" in second.patch.diff
    assert (repo / "train.py").read_text() == train  # the clone is never patched


def test_evidence_roundtrip_and_fold(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    ok = _critic(FakeLLM([VALID]), repo)
    bad = _critic(FakeLLM(["{}"] * 3), repo, traceback=_tb(BAD_LINE, message="other"))
    ev_ok = critic_patch_evidence(ok, source="r@c", smoke_evidence_id="a" * 64)
    ev_bad = critic_patch_evidence(bad, source="r@c", smoke_evidence_id="b" * 64)
    assert (ev_ok.kind, ev_ok.authority, ev_ok.confidence) == ("critic_patch", "llm", "inferred")
    assert ev_ok.links == ("a" * 64,) and ev_ok.content["diff"] == ok.patch.diff  # type: ignore[union-attr]
    assert set(ev_ok.content) >= {"file", "diff", "rationale", "edits", "error_signature"}
    st = fold_critic_state([ev_ok, ev_bad])
    assert st.attempts == 2 and st.last_signature == bad.signature
    assert [p.patch_id for p in st.stack] == [ok.patch.patch_id]  # type: ignore[union-attr]
    skipped = _critic(FakeLLM([]), repo, last_signature=ok.signature)
    with pytest.raises(ValueError):
        critic_patch_evidence(skipped, source="r@c", smoke_evidence_id="c" * 64)


# --- loop (fake runner) ----------------------------------------------------------------------


class PatchAwareRunner:
    """probe / imports pass; the run "executes" the per-run copy: the first line with
    `.data[0]` raises IndexError at that line, otherwise exit 0."""

    def __init__(self) -> None:
        self.run_texts: list[str] = []

    def __call__(self, image, cmd, *, workdir, log_path, name, **kw):  # type: ignore[no-untyped-def]
        stage = name.rsplit("-", 1)[1]
        code, log = 0, PROBE_OK if stage == "probe" else ""
        if stage == "run":
            text = (Path(workdir) / "train.py").read_text()
            self.run_texts.append(text)
            bad = [i for i, ln in enumerate(text.split("\n"), 1) if ".data[0]" in ln]
            if bad:
                src = text.split("\n")[bad[0] - 1].strip()
                code, log = 1, "1/3 [00:01<00:02, 1.00it/s]\n" + _tb(bad[0], src=src).raw + "\n"
            else:
                log = "done\n"
        Path(log_path).write_text(log)
        return RunResult(
            exit_code=code, timed_out=False, oom_killed=False, duration_s=1.0,
            log_path=Path(log_path), name=name,
        )  # fmt: skip


def _loop(tmp_path: Path, repo: Path, llm: FakeLLM, runner: PatchAwareRunner, **kw: Any):  # type: ignore[no-untyped-def]
    def critic(tb, command, stack, last, failed):  # type: ignore[no-untyped-def]
        return run_critic(
            llm, traceback=tb, command=command, clone_dir=repo,
            repo_files=frozenset({"train.py"}), stack=stack,
            metric_line_texts=METRIC_TEXTS, last_signature=last, failed_patches=failed,
        )  # fmt: skip

    recorded: list[Any] = []
    out = smoke_loop(
        lambda s: _inputs(tmp_path, repo),
        ClassifyContext(repo_files=frozenset({"train.py"})),
        record=recorded.append,
        runner=runner,
        critic=critic,
        **kw,
    )
    assert recorded == list(out.iterations)
    return out


def test_loop_valid_patch_then_resmoke_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    runner = PatchAwareRunner()
    out = _loop(tmp_path, repo, FakeLLM([VALID]), runner)
    assert out.stop == "passed" and len(out.iterations) == 2
    first, last = out.iterations
    assert first.critic is not None and first.critic.status == "patched" and not first.capped
    assert first.result.patches_applied == ()
    assert last.result.patches_applied == (first.critic.patch.patch_id,)  # type: ignore[union-attr]
    assert last.result.details(attempt=1, log_ref="", image="x")["patches_applied"] == [
        first.critic.patch.patch_id  # type: ignore[union-attr]
    ]
    assert first.input_digest != last.input_digest  # the patch is part of the inputs
    assert "print(loss.item())" in runner.run_texts[1]
    assert (repo / "train.py").read_text() == FIXTURE_TRAIN
    assert out.critic_state.attempts == 1 and len(out.critic_state.stack) == 1


def test_loop_invalid_twice_then_valid_uses_gate_retries(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    llm = FakeLLM(["oops", _patch_json(BAD_LINE, "nope", "print(loss.item())"), VALID])
    out = _loop(tmp_path, repo, llm, PatchAwareRunner())
    assert out.stop == "passed"
    c = out.iterations[0].critic
    assert c is not None and c.gate_attempts == 3 and out.critic_state.attempts == 1


def test_loop_caps_critic_at_3(tmp_path: Path) -> None:
    train = "import torch\nx = torch.tensor(1.0)\n" + "".join(
        f"a{i} = x.data[0]\n" for i in range(5)
    )
    repo = _repo(tmp_path, train)
    llm = FakeLLM(_item_patch_for_prompt)
    out = _loop(tmp_path, repo, llm, PatchAwareRunner())
    assert out.stop == "max_critic" and len(llm.prompts) == 3
    assert [it.critic.status if it.critic else None for it in out.iterations] == [
        "patched", "patched", "patched", None,
    ]  # fmt: skip
    assert out.iterations[-1].capped and not any(it.capped for it in out.iterations[:-1])
    assert len(out.iterations[-1].result.patches_applied) == 3


USELESS = _patch_json(2, "import sys", "import sys  # noqa")  # valid, but fixes nothing


def test_loop_ineffective_patch_is_reverted_and_critic_sees_its_diff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    llm = FakeLLM([USELESS, VALID])
    runner = PatchAwareRunner()
    out = _loop(tmp_path, repo, llm, runner)
    assert out.stop == "passed" and len(llm.prompts) == 2
    first, second, third = out.iterations
    useless = first.critic.patch  # type: ignore[union-attr]
    assert useless is not None and first.reverted is None
    assert second.reverted == useless and not second.capped  # same signature -> revert
    prompt2 = llm.prompts[1][1]
    assert "did not change the error" in prompt2 and useless.diff.rstrip() in prompt2
    assert "-print(loss.data[0])" not in prompt2  # only the failed diff, not a new one
    assert "import sys  # noqa" not in runner.run_texts[2]  # dropped from the stack
    assert third.result.patches_applied == (second.critic.patch.patch_id,)  # type: ignore[union-attr]
    cs = out.critic_state
    assert cs.attempts == 2 and [p.patch_id for p in cs.stack] == [
        second.critic.patch.patch_id  # type: ignore[union-attr]
    ]
    assert cs.failed == (useless,)


def test_loop_ineffective_patches_reach_cap_capped(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    llm = FakeLLM([USELESS] * 3)
    out = _loop(tmp_path, repo, llm, PatchAwareRunner())
    assert out.stop == "max_critic" and len(llm.prompts) == 3
    assert [it.reverted is not None for it in out.iterations] == [False, True, True, True]
    assert out.iterations[-1].capped and not any(it.capped for it in out.iterations[:-1])
    assert out.critic_state.attempts == 3 and out.critic_state.stack == ()
    assert llm.prompts[2][1].count("did not change the error") == 2  # both failed diffs shown


def test_same_signature_without_revert_is_still_skipped(tmp_path: Path) -> None:
    # e.g. a resumed run after an invalid critic attempt: no patch was judged, no new evidence
    repo = _repo(tmp_path)
    sig = _critic(FakeLLM([VALID]), repo).signature
    state = CriticState(attempts=1, last_signature=sig)
    llm = FakeLLM([])
    out = _loop(tmp_path, repo, llm, PatchAwareRunner(), critic_state=state)
    assert out.stop == "no_new_evidence" and llm.prompts == [] and out.iterations[-1].capped


def test_revert_folds_out_of_the_stack(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    p1 = _critic(FakeLLM([USELESS]), repo)
    ev1 = critic_patch_evidence(p1, source="r@c", smoke_evidence_id="a" * 64)
    assert p1.patch is not None
    rv = critic_revert_evidence(
        p1.patch, source="r@c", patch_evidence_id=ev1.evidence_id, smoke_evidence_id="b" * 64
    )
    assert (rv.kind, rv.authority, rv.confidence) == ("critic_revert", "rule", "derived")
    assert rv.links == (ev1.evidence_id, "b" * 64) and rv.content["patch_id"] == p1.patch.patch_id
    st = fold_critic_state([ev1])
    assert st.pending == p1.patch and len(st.stack) == 1
    st = fold_critic_state([ev1, rv])
    assert st.stack == () and st.failed == (p1.patch,) and st.pending is None
    assert st.attempts == 1 and st.failed_for(p1.signature) == (p1.patch,)
    p2 = _critic(FakeLLM([VALID]), repo, last_signature=p1.signature, failed_patches=st.failed)
    ev2 = critic_patch_evidence(p2, source="r@c", smoke_evidence_id="b" * 64)
    st = fold_critic_state([ev1, rv, ev2])
    assert [p.patch_id for p in st.stack] == [p2.patch.patch_id] and st.attempts == 2  # type: ignore[union-attr]


def test_loop_resumes_from_stored_state_and_cap_blocks_claims(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    full = CriticState(attempts=3, last_signature="x")
    out = _loop(tmp_path, repo, FakeLLM([]), PatchAwareRunner(), critic_state=full)
    assert out.stop == "max_critic" and out.iterations[-1].capped
    it = out.iterations[-1]
    ev = make_smoke_evidence(
        passed=False, attempts=1, capped=it.capped, source="r@c", locator="smoke:x"
    )
    from reproducer.domain.contracts import ResultClaim
    from reproducer.domain.ledger import Tolerance, ToleranceConfig

    claim = ResultClaim(
        claim_id="c1", metric="test_acc", value=96.0, dataset="cifar10", split="test",
        setting=None, location="Table 1", quote="96.0",
    )  # fmt: skip
    ledger = compute_ledger(
        (claim,), {ev.evidence_id: ev}, {}, ToleranceConfig(default=Tolerance(abs=0.5))
    )
    assert ledger["c1"].status == "blocked"


# --- docker: Cutout image (torch 1.12) -------------------------------------------------------

CUTOUT_IMAGE = os.environ.get("REPRODUCER_CUTOUT_IMAGE", "reproducer/cutout:424da3bbafa8")


@pytest.fixture(scope="module")
def cutout_image() -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not found")
    r = subprocess.run(
        ["docker", "image", "inspect", CUTOUT_IMAGE], capture_output=True, check=False
    )
    if r.returncode != 0:
        pytest.skip(f"Cutout image {CUTOUT_IMAGE} not built (set REPRODUCER_CUTOUT_IMAGE)")
    return CUTOUT_IMAGE


def _docker_loop(tmp_path: Path, image: str, critic_llm: Any) -> tuple[Any, Path]:
    repo = _repo(tmp_path)

    def critic(tb, command, stack, last, failed):  # type: ignore[no-untyped-def]
        return run_critic(
            critic_llm, traceback=tb, command=command, clone_dir=repo,
            repo_files=frozenset({"train.py"}), stack=stack,
            metric_line_texts=METRIC_TEXTS, last_signature=last, failed_patches=failed,
        )  # fmt: skip

    out = smoke_loop(
        lambda s: _inputs(tmp_path, repo, image=image, third_party=("torch",)),
        ClassifyContext(repo_files=frozenset({"train.py"})),
        critic=critic,
    )
    return out, repo


@pytest.mark.docker
def test_docker_critic_fixes_index_error_in_cutout_image(cutout_image: str, tmp_path: Path) -> None:
    out, repo = _docker_loop(tmp_path, cutout_image, FakeLLM([VALID]))
    first = out.iterations[0]
    log = first.result.log_path.read_text()
    assert first.result.signal == "traceback" and first.result.progress_seen, log
    tb = first.result.traceback
    assert tb is not None and tb.exc_type == "IndexError" and "0-dim tensor" in tb.message, log
    assert tb.repo_frame is not None and tb.repo_frame.line == BAD_LINE
    assert first.classification is not None and first.classification.cls == "critic"
    assert out.stop == "passed", [it.result.log_path.read_text() for it in out.iterations]
    assert "test_acc: 1.500" in out.iterations[-1].result.log_path.read_text()
    assert (repo / "train.py").read_text() == FIXTURE_TRAIN


@pytest.mark.llm_live
@pytest.mark.docker
def test_live_critic_fixes_index_error_in_cutout_image(cutout_image: str, tmp_path: Path) -> None:
    if not (os.environ.get("REPRODUCER_LLM_CRITIC") or os.environ.get("REPRODUCER_LLM")):
        pytest.skip("set REPRODUCER_LLM_CRITIC (e.g. openrouter:anthropic/claude-haiku-5.5)")
    llm = llm_for_node("critic", Budget(max_tokens_total=30_000))
    start = time.monotonic()
    out, _ = _docker_loop(tmp_path, cutout_image, llm)
    first = out.iterations[0]
    print("\n=== live critic transcript ===")
    print(f"backend: {os.environ.get('REPRODUCER_LLM_CRITIC') or os.environ.get('REPRODUCER_LLM')}")
    print("--- traceback ---")
    print(first.result.traceback.raw if first.result.traceback else None)
    c = first.critic
    if c is not None:
        print(f"--- critic: {c.status}, gate attempts {c.gate_attempts}, signature {c.signature}")
        for e in c.errors:
            print(f"gate rejected: {e}")
        if c.patch is not None:
            print(f"rationale: {c.patch.rationale}")
            print(c.patch.diff)
    for r in llm.records():
        print(
            f"call: {r.backend}:{r.model} in={r.input_tokens} out={r.output_tokens} "
            f"latency={r.latency_s:.2f}s"
        )
    last = out.iterations[-1].result
    print(f"--- re-smoke: signal={last.signal} passed={last.passed} "
          f"patches_applied={[p[:12] for p in last.patches_applied]}")  # fmt: skip
    print(f"total wall: {time.monotonic() - start:.1f}s; stop={out.stop}: {out.reason}")
    assert c is not None and c.status == "patched" and c.patch is not None
    assert out.stop == "passed" and last.passed


# --- context eviction ------------------------------------------------------------------------


def _limited(limit: int) -> FakeLLM:
    def answer(system: str, prompt: str) -> str:
        if len(prompt) > limit:
            raise ContextTooLong(f"{len(prompt)} > {limit}")
        return VALID

    return FakeLLM(answer)


def _full_prompt() -> str:
    return critic_prompt(_tb(BAD_LINE), "python train.py", "train.py", FIXTURE_TRAIN, BAD_LINE,
                         frozenset())  # fmt: skip


def test_critic_builder_keeps_the_normal_prompt() -> None:
    build = critic_prompt_builder(
        _tb(BAD_LINE), "python train.py", "train.py", FIXTURE_TRAIN, BAD_LINE, frozenset()
    )
    assert build(None)[1] == _full_prompt() and build(None)[2] == ()


def test_critic_evicts_and_records_then_patches(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    llm = MeteredLLM(_limited(len(_full_prompt()) * 3 // 4), Budget(), "critic")
    out = _critic(llm, repo)  # type: ignore[arg-type]
    assert out.status == "patched" and out.evicted
    (rec,) = llm.records()
    assert rec.evicted == out.evicted
    ev = critic_patch_evidence(out, source="s", smoke_evidence_id="e" * 64)
    assert ev.content["evicted"] == list(out.evicted)


def test_critic_still_too_large_is_a_clean_failed_attempt(tmp_path: Path) -> None:
    out = _critic(_limited(10), _repo(tmp_path))
    assert out.status == "invalid" and out.counts_as_attempt and out.patch is None
    assert out.reason.startswith("fallback: context:")
