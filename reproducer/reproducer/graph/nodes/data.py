"""data node: provision the spec datasets and prove the read-only mount (no LLM, no repo code).

scan_data: torchvision dataset calls of the repo (ast; stored as '<path>#dataset_calls'),
selected = spec datasets that the scan also found and the registry knows. provision: makes
sure they are in the shared cache (check with --network none; download only by our trusted
script in the env image) and stores dataset_cache evidence. resolve_mounts: the read-only
mount plan and the dataset_source Unknown (resolved only with verified cache evidence).
sandbox_proof: our own read-back script in a network-none sandbox with the mount.
The CLI `data` command and the setup graph both use these.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reproducer.domain.contracts import ReproSpec
from reproducer.domain.data_plan import (
    DATASET_SOURCE_UNKNOWN,
    KNOWN_DATASETS,
    DataMountPlan,
    DatasetSelection,
    DatasetSourceDecision,
    container_root,
    dataset_cache_evidence,
    decide_dataset_source,
    plan_data_mounts,
    registry_name_for_class,
    select_datasets,
)
from reproducer.domain.data_scan import (
    DatasetCallScan,
    dataset_calls_evidence,
    find_dataset_calls,
    load_dataset_calls,
)
from reproducer.domain.evidence import Evidence
from reproducer.domain.unknowns import UnknownOpened, fold_unknowns, resolve, unknown_id_for
from reproducer.graph.nodes.env import cleanup_workdir
from reproducer.providers.datasets import DatasetStatus, data_cache_dir, ensure_datasets
from reproducer.providers.docker_runner import RunResult, new_run_dir, run_container
from reproducer.providers.git import list_files, read_text
from reproducer.providers.store import RunStore

Warn = Callable[[str], None]


def _no_warn(msg: str) -> None:
    pass


def latest_repo_evidence(store: RunStore) -> tuple[str, dict[str, Evidence]]:
    """Evidence of the most recently stored inventory's source (repo@commit) only."""
    evidence = store.load_evidence()
    invs = [ev for ev in evidence.values() if ev.kind == "repo_file" and ev.locator == "inventory"]
    if not invs:
        raise ValueError(f"no inventory evidence in {store.evidence_path}; run `inventory` first")
    source = invs[-1].source
    return source, {k: ev for k, ev in evidence.items() if ev.source == source}


def scan_dataset_calls(
    out: Path,
    store: RunStore,
    source: str,
    evidence: Mapping[str, Evidence],
    warn: Warn = _no_warn,
) -> tuple[DatasetCallScan, str]:
    """Stored '#dataset_calls' evidence if any; else scan the clone's .py files (full text);
    else the stored .py heads. New scan evidence is appended to the store."""
    stored = load_dataset_calls(evidence, source)
    if stored is not None:
        return stored, "stored evidence"
    observed = {
        str(ev.content.get("path", ev.locator)): ev
        for ev in evidence.values()
        if ev.kind == "repo_file" and ev.confidence == "observed" and "#" not in ev.locator
    }
    repo_dir = out / "repo"
    texts: dict[str, str] = {}
    if repo_dir.is_dir():
        origin = "clone"
        for f in list_files(repo_dir):
            if f.path.endswith(".py") and not f.is_symlink:
                try:
                    texts[f.path] = read_text(repo_dir, f.path)[0]
                except (ValueError, OSError) as e:
                    warn(f"warning: skipped {f.path}: {e}")
    else:
        origin = "stored heads"
        texts = {
            p: ev.content["head"]
            for p, ev in observed.items()
            if p.endswith(".py") and isinstance(ev.content.get("head"), str)
        }
    scan = find_dataset_calls(texts)
    links = {p: observed[p].evidence_id for p in texts if p in observed}
    for ev in dataset_calls_evidence(scan, source, links):
        store.append_evidence(ev)
    return scan, origin


def spec_datasets(store: RunStore, spec_path: str | None) -> list[str]:
    if spec_path is not None:
        spec = ReproSpec.model_validate_json(Path(spec_path).read_text(encoding="utf-8"))
        return list(spec.datasets)
    claims = [ev for ev in store.load_evidence().values() if ev.kind == "paper_claim"]
    if not claims:
        raise ValueError(f"no paper_claim evidence in {store.root}; pass --spec")
    return list(dict.fromkeys(str(ev.content["dataset"]) for ev in claims))


def sandbox_proof_code(names: Sequence[str], root: str) -> str:
    """Our own read-back script: test-split length of each dataset (download=False), then a
    write attempt on the read-only mount. No repo code or strings."""
    lens = ", ".join(
        f"len(d.{KNOWN_DATASETS[n].cls}({root!r}, "
        + ", ".join(f"{k}={v!r}" for k, v in KNOWN_DATASETS[n].splits[-1].items())
        + ", download=False))"
        for n in names
    )
    return (
        f"import torchvision.datasets as d; print({lens})\n"
        "try:\n"
        f"    open({(root + '/reproducer_write_probe')!r}, 'w'); print('DATA_WRITABLE')\n"
        "except OSError as e:\n"
        "    print('DATA_RO', type(e).__name__)\n"
    )


def update_dataset_source(
    store: RunStore,
    decision: DatasetSourceDecision,
    plan: DataMountPlan,
    scan_evidence: Sequence[Evidence],
) -> str:
    """Resolve the dataset_source Unknown (via resolve(), resolver = the dataset_cache
    evidence) iff decision.value is set; otherwise it stays open (opened here, linked to the
    scan evidence, if a root is unresolved and it never was). Returns a status line."""
    events = store.load_events()
    evidence = store.load_evidence()
    uid = unknown_id_for(DATASET_SOURCE_UNKNOWN)
    unknowns = fold_unknowns(events, evidence)
    state = unknowns.get(uid)
    if decision.value is None:
        why = "; ".join(decision.reasons)
        if state is None and plan.unresolved:
            bad = {u.call.split(":", 1)[0] for u in plan.unresolved}
            ev = next(e for e in scan_evidence if e.locator.split("#", 1)[0] in bad)
            store.append_event(
                UnknownOpened(
                    unknown_id=uid, name=DATASET_SOURCE_UNKNOWN, evidence_id=ev.evidence_id
                )
            )
            return f"opened ({why})"
        return f"{'not opened' if state is None else state.status} ({why})"
    if state is None:
        return "not opened (nothing to resolve)"
    if state.status == "resolved":
        return f"resolved = {state.value!r} (already)"
    opened = next(e for e in events if isinstance(e, UnknownOpened) and e.unknown_id == uid)
    assert decision.resolver is not None
    store.append_event(resolve(opened, evidence[decision.resolver], decision.value))
    return f"resolved = {decision.value!r} by {decision.resolver[:12]}"


def disk_usage(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


# --- phases (the CLI prints between them) ----------------------------------------------------


@dataclass(frozen=True)
class DataScan:
    source: str
    scan: DatasetCallScan
    origin: str
    spec_datasets: tuple[str, ...]
    selection: DatasetSelection


def scan_data(
    out: Path, store: RunStore, datasets: Sequence[str], warn: Warn = _no_warn
) -> DataScan:
    source, evidence = latest_repo_evidence(store)
    scan, origin = scan_dataset_calls(out, store, source, evidence, warn)
    return DataScan(source, scan, origin, tuple(datasets), select_datasets(datasets, scan.calls))


@dataclass(frozen=True)
class Provisioned:
    name: str
    status: DatasetStatus
    evidence: Evidence | None  # dataset_cache (verified only)
    written: bool


def provision(
    store: RunStore,
    selected: Sequence[str],
    image: str,
    *,
    log_dir: Path,
    timeout_s: float,
    ensure: Callable[..., dict[str, DatasetStatus]] = ensure_datasets,
) -> list[Provisioned]:
    """ensure_datasets + dataset_cache evidence for each verified dataset. ValueError for
    names outside the registry."""
    statuses = ensure(selected, image, log_dir=log_dir, timeout_s=timeout_s)
    out: list[Provisioned] = []
    for n, st in statuses.items():
        if not st.verified:
            out.append(Provisioned(n, st, None, False))
            continue
        ev = dataset_cache_evidence(
            name=n, files=st.files, verified=True, split_sizes=st.split_sizes,
            source=f"data-cache@{image}",
        )  # fmt: skip
        out.append(Provisioned(n, st, ev, store.append_evidence(ev)))
    return out


def resolve_mounts(
    store: RunStore, ds: DataScan, provisioned: Sequence[Provisioned], cache: Path
) -> tuple[DataMountPlan, str]:
    """(mount plan of the verified selected datasets, dataset_source status line)."""
    verified = {p.name for p in provisioned if p.status.verified}
    plan = plan_data_mounts(
        ds.scan.calls, [n for n in ds.selection.selected if n in verified], str(cache)
    )
    scan_ev = [
        e for e in store.load_evidence().values()
        if e.source == ds.source and e.locator.endswith("#dataset_calls")
    ]  # fmt: skip
    decision = decide_dataset_source(
        ds.spec_datasets,
        ds.scan.calls,
        (e for e in store.load_evidence().values() if e.kind == "dataset_cache"),
    )
    return plan, update_dataset_source(store, decision, plan, scan_ev)


@dataclass(frozen=True)
class ProofResult:
    result: RunResult
    log: str
    target: str
    passed: bool


def sandbox_proof(
    image: str,
    names: Sequence[str],
    plan: DataMountPlan,
    cache: Path,
    *,
    data_dir: Path,
    timeout_s: float,
    runner: Callable[..., RunResult] = run_container,
) -> ProofResult:
    """Our read-back script, network none, cache mounted read-only at the first target, in a
    fresh EMPTY workdir (no repo code). Passed iff exit 0 and the mount is read-only."""
    target = plan.mounts[0].container
    run_dir = new_run_dir(data_dir, "proof")  # unique, created once, by us
    workdir = run_dir / "work"
    workdir.mkdir()  # empty: no repo code in the proof
    try:
        result = runner(
            image,
            ["python", "-I", "-c", sandbox_proof_code(names, target)],
            workdir=workdir,
            out_dir=run_dir / "out",
            data_mounts={str(cache): target},
            gpu=False,
            timeout_s=timeout_s,
            log_path=run_dir / "run.log",
            name=f"reproducer-data-{run_dir.name}",
        )
    finally:
        cleanup_workdir(workdir, keep=False)
    log = result.log_path.read_text(errors="replace").rstrip()
    return ProofResult(result, log, target, result.exit_code == 0 and "DATA_RO" in log)


# --- mount plan / roots for smoke ------------------------------------------------------------


def stored_mount_plan(store: RunStore, source: str) -> DataMountPlan:
    """Mount plan from stored '#dataset_calls' and verified dataset_cache evidence."""
    evidence = store.load_evidence()
    scan = load_dataset_calls(evidence, source)
    verified = sorted(
        {
            str(ev.content["name"])
            for ev in evidence.values()
            if ev.kind == "dataset_cache" and ev.content.get("verified") is True
        }
    )
    if scan is None:
        return DataMountPlan()
    return plan_data_mounts(scan.calls, verified, str(data_cache_dir().resolve()))


def dataset_roots(store: RunStore, source: str) -> dict[str, str | None]:
    """Container root -> registry dataset name, for every scanned dataset call with a literal,
    mountable root (all classes, not only the selected ones)."""
    scan = load_dataset_calls(store.load_evidence(), source)
    roots: dict[str, str | None] = {}
    for c in scan.calls if scan is not None else ():
        target, _ = container_root(c.root, c.reason)
        if target is not None:
            roots.setdefault(target, registry_name_for_class(c.cls))
    return roots


# --- graph node ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSummary:
    selected: tuple[str, ...]
    warnings: tuple[str, ...]
    datasets: tuple[str, ...]  # "<name> <state> verified=<bool>"
    mounts: tuple[str, ...]  # "host -> container (ro)"
    unresolved: tuple[str, ...]
    dataset_source: str  # status line of the dataset_source Unknown
    proof: str  # passed | failed | skipped
    ok: bool
    errors: tuple[str, ...] = field(default=())


def data_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Scan -> provision -> mount plan + dataset_source -> sandbox proof. deps: store,
    image_exists, ensure_datasets, runner, unload, data_cache_dir. data_ok iff every selected
    dataset is verified and the proof passed (or there is nothing to mount)."""
    out = Path(state["out_dir"])
    store = deps["store"]
    image = state["image_tag"]

    def failed(msg: str) -> dict[str, Any]:
        return {"data_ok": False, "setup_status": "data_failed", "stop_reason": "data_failed",
                "stop_detail": msg}  # fmt: skip

    try:
        datasets = list(state["spec"].datasets) if "spec" in state else spec_datasets(store, None)
        ds = scan_data(out, store, datasets, deps.get("warn", _no_warn))
    except (ValueError, OSError) as e:
        return failed(str(e))
    if not deps["image_exists"](image):
        return failed(f"image {image} not built")
    cache = Path(deps["data_cache_dir"]()).resolve()
    deps["unload"]()
    timeout = float(state.get("data_timeout_s", 3600))
    try:
        prov = provision(
            store, ds.selection.selected, image, log_dir=out / "data", timeout_s=timeout,
            ensure=deps["ensure_datasets"],
        )  # fmt: skip
    except ValueError as e:
        return failed(str(e))
    plan, source_status = resolve_mounts(store, ds, prov, cache)
    all_verified = all(p.status.verified for p in prov)
    proof = "skipped"
    if all_verified and plan.mounts:
        names = [p.name for p in prov if p.status.verified]
        pr = sandbox_proof(
            image, names, plan, cache, data_dir=out / "data", timeout_s=timeout,
            runner=deps["runner"],
        )  # fmt: skip
        proof = "passed" if pr.passed else "failed"
    ok = all_verified and proof != "failed"
    summary = DataSummary(
        selected=tuple(ds.selection.selected),
        warnings=tuple(ds.selection.warnings),
        datasets=tuple(
            f"{p.name} {p.status.state} verified={p.status.verified}"
            + (f" error={p.status.error}" if p.status.error else "")
            for p in prov
        ),
        mounts=tuple(f"{m.host} -> {m.container} (ro)" for m in plan.mounts),
        unresolved=tuple(f"{u.call}: {u.reason}" for u in plan.unresolved),
        dataset_source=source_status,
        proof=proof,
        ok=ok,
    )
    update: dict[str, Any] = {"data": summary, "data_ok": ok, "mount_plan": plan}
    if not ok:
        detail = "dataset not verified" if not all_verified else "sandbox proof failed"
        update.update(failed(detail), data=summary, mount_plan=plan)
    return update
