"""Tests for the pure parts of the `data` CLI command."""

from __future__ import annotations

from pathlib import Path

from reproducer.cli import build_parser, sandbox_proof_code


def test_sandbox_proof_code_is_ours_and_registry_only() -> None:
    code = sandbox_proof_code(["cifar10", "cifar100"], "/work/data")
    first = code.splitlines()[0]
    assert first == (
        "import torchvision.datasets as d; print(len(d.CIFAR10('/work/data', train=False, "
        "download=False)), len(d.CIFAR100('/work/data', train=False, download=False)))"
    )
    assert "open('/work/data/reproducer_write_probe', 'w')" in code
    compile(code, "<proof>", "exec")


def test_data_parser() -> None:
    args = build_parser().parse_args(["data", "--out", "runs/x"])
    assert args.out == "runs/x" and args.spec is None and args.image is None


# --- dataset_source resolution against a real RunStore -------------------------------------


def _store_with_open_unknown(tmp_path):  # type: ignore[no-untyped-def]
    from reproducer.domain.evidence import make_evidence
    from reproducer.domain.unknowns import UnknownOpened
    from reproducer.providers.store import RunStore

    store = RunStore(tmp_path)
    seed = make_evidence(
        kind="workflow", source="repo@x", locator="subtask:C", content={"x": 1},
        confidence="derived", authority="repo",
    )  # fmt: skip
    store.append_evidence(seed)
    store.append_event(
        UnknownOpened(
            unknown_id="unk:dataset_source", name="dataset_source", evidence_id=seed.evidence_id
        )
    )
    return store


def _run_update(tmp_path, calls, verified: bool):  # type: ignore[no-untyped-def]
    from reproducer.cli import update_dataset_source
    from reproducer.domain.data_plan import (
        dataset_cache_evidence,
        decide_dataset_source,
        plan_data_mounts,
    )
    from reproducer.domain.unknowns import fold_unknowns

    store = _store_with_open_unknown(tmp_path)
    for name in ("cifar10", "cifar100"):
        store.append_evidence(
            dataset_cache_evidence(
                name=name, files=[], verified=verified, split_sizes=[1],
                source="data-cache@img",
            )
        )  # fmt: skip
    cache = [e for e in store.load_evidence().values() if e.kind == "dataset_cache"]
    decision = decide_dataset_source(["CIFAR-10", "CIFAR-100"], calls, cache)
    plan = plan_data_mounts(calls, ["cifar10", "cifar100"], "/cache")
    line = update_dataset_source(store, decision, plan, [])
    state = fold_unknowns(store.load_events(), store.load_evidence())["unk:dataset_source"]
    return line, state, store


def _calls(root2):  # type: ignore[no-untyped-def]
    from reproducer.domain.data_scan import DatasetCall

    return [
        DatasetCall(path="train.py", line=1, cls="CIFAR10", root="data/", download=True),
        DatasetCall(
            path="train.py",
            line=2,
            cls="CIFAR100",
            root=root2,
            reason=None if root2 else "non-literal root: args.root",
            download=True,
        ),  # fmt: skip
    ]


def test_update_dataset_source_all_ok_resolved(tmp_path) -> None:  # type: ignore[no-untyped-def]
    line, state, store = _run_update(tmp_path, _calls("data/"), verified=True)
    assert state.status == "resolved"
    assert state.value == "torchvision:CIFAR10,CIFAR100 (official download, checksum-verified)"
    resolver = store.load_evidence()[state.resolved_by]
    assert resolver.kind == "dataset_cache" and resolver.content["name"] == "cifar10"
    assert line.startswith("resolved = ")
    # idempotent: a second run appends nothing
    from reproducer.cli import update_dataset_source
    from reproducer.domain.data_plan import DataMountPlan, DatasetSourceDecision

    n = len(store.load_events())
    again = DatasetSourceDecision(value=state.value, resolver=state.resolved_by, reasons=())
    assert "already" in update_dataset_source(store, again, DataMountPlan(), [])
    assert len(store.load_events()) == n


def test_update_dataset_source_unresolved_root_open(tmp_path) -> None:  # type: ignore[no-untyped-def]
    line, state, _ = _run_update(tmp_path, _calls(None), verified=True)
    assert state.status == "open"
    assert line.startswith("open (") and "non-literal root: args.root" in line


def test_update_dataset_source_unverified_cache_open(tmp_path) -> None:  # type: ignore[no-untyped-def]
    line, state, _ = _run_update(tmp_path, _calls("data/"), verified=False)
    assert state.status == "open" and "no verified dataset_cache evidence" in line


# --- per-run dirs: unique within the same second (regression: proof-<stamp>/work collision) --


def test_new_run_dir_same_second_unique(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from reproducer.providers import docker_runner

    monkeypatch.setattr(docker_runner.time, "strftime", lambda fmt: "20261123-174043")
    a = docker_runner.new_run_dir(tmp_path / "data", "proof")
    b = docker_runner.new_run_dir(tmp_path / "data", "proof")
    assert a != b and a.is_dir() and b.is_dir()
    assert a.name.startswith("proof-20261123-174043-") and list(a.iterdir()) == []
    assert docker_runner._NAME_RE.match(f"reproducer-data-{a.name}")


def _present_store(tmp_path):  # type: ignore[no-untyped-def]
    from reproducer.domain.data_scan import dataset_calls_evidence, find_dataset_calls
    from reproducer.domain.evidence import make_evidence
    from reproducer.providers.store import RunStore

    out = tmp_path / "run"
    store = RunStore(out)
    source = "repo:x/y@" + "0" * 40
    store.append_evidence(
        make_evidence(
            kind="repo_file", source=source, locator="inventory", content={"files": []},
            confidence="observed", authority="repo",
        )
    )  # fmt: skip
    train = (
        "from torchvision import datasets\n"
        "a = datasets.CIFAR10(root='data/', train=True, download=True)\n"
        "b = datasets.CIFAR100(root='data/', train=False, download=True)\n"
    )
    for ev in dataset_calls_evidence(find_dataset_calls({"train.py": train}), source, {}):
        store.append_evidence(ev)
    spec = tmp_path / "spec.json"
    spec.write_text(
        '{"code_links": ["https://github.com/x/y"], "datasets": ["CIFAR-10", "CIFAR-100"], '
        '"claims": [{"claim_id": "c", "metric": "error", "value": 3.0, "dataset": "CIFAR-10", '
        '"split": "test", "setting": "s", "location": "T1", "quote": "q"}], "setup": []}'
    )
    return out, spec


def test_data_rerun_all_present_same_second_prints_proof(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch, capsys
) -> None:
    """Two OVERLAPPING `data` runs in the same second (the second starts while the first's
    proof is running), everything present: both complete, use distinct fresh empty
    workdirs and print the proof. Before new_run_dir this raised FileExistsError."""
    from reproducer import cli
    from reproducer.providers import docker_runner
    from reproducer.providers.datasets import DatasetStatus
    from reproducer.providers.docker_runner import RunResult

    out, spec = _present_store(tmp_path)
    try:
        cli.ReproSpec.model_validate_json(spec.read_text())
    except Exception as e:  # keep the fixture honest about the contract
        raise AssertionError(f"fixture spec invalid: {e}") from e
    monkeypatch.setenv("REPRODUCER_DATA_CACHE", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(docker_runner.time, "strftime", lambda fmt: "20261123-174043")
    monkeypatch.setattr(cli, "image_exists", lambda tag: True)
    monkeypatch.setattr(cli, "unload_ollama", lambda: [])
    monkeypatch.setattr(
        cli,
        "ensure_datasets",
        lambda names, image, **kw: {
            n: DatasetStatus(name=n, state="present", verified=True, split_sizes=(50000, 10000))
            for n in names
        },
    )
    workdirs: list[Path] = []
    argv = ["data", "--out", str(out), "--spec", str(spec), "--image", "img:tag"]

    def fake_run(image, cmd, *, workdir, log_path, name, data_mounts, **kw):  # type: ignore[no-untyped-def]
        assert workdir.is_dir() and list(workdir.iterdir()) == []  # fresh, ours, no repo code
        assert workdir not in workdirs and name.startswith("reproducer-data-proof-")
        assert list(data_mounts.values()) == ["/work/data"]
        workdirs.append(workdir)
        if len(workdirs) == 1:  # a second `data` run starts while this proof is running
            assert cli.main(argv) == 0
            assert workdir.is_dir()  # the concurrent run did not touch ours
        log_path.write_text("10000 10000\nDATA_RO OSError\n")
        return RunResult(
            exit_code=0, timed_out=False, oom_killed=False, duration_s=0.1,
            log_path=log_path, name=name,
        )  # fmt: skip

    monkeypatch.setattr(cli, "run_container", fake_run)
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert text.count("  10000 10000") == 2
    assert text.rstrip().endswith("DATA_RO OSError")
    assert len(workdirs) == 2 and workdirs[0].parent != workdirs[1].parent
    assert not any(w.exists() for w in workdirs)  # cleaned up; logs kept
    assert all((w.parent / "run.log").exists() for w in workdirs)


def test_smoke_parser_auto_fix_flags() -> None:
    args = build_parser().parse_args(["smoke", "--out", "runs/x"])
    assert args.auto_fix is False and args.max_fixes == 5
    args = build_parser().parse_args(["smoke", "--out", "runs/x", "--auto-fix", "--max-fixes", "2"])
    assert args.auto_fix is True and args.max_fixes == 2


def test_smoke_parser_critic_and_build_timeout_flags() -> None:
    args = build_parser().parse_args(["smoke", "--out", "runs/x"])
    assert args.max_critic == 3 and args.build_timeout == 3600
    assert args.critic_budget_tokens == 60_000
    args = build_parser().parse_args(
        ["smoke", "--out", "runs/x", "--max-critic", "1", "--build-timeout", "900"]
    )
    assert args.max_critic == 1 and args.build_timeout == 900
    assert build_parser().parse_args(["env", "--out", "x"]).build_timeout == 3600
