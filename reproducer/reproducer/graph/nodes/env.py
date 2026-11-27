"""env node: plan the Docker environment from stored facts and build its image (no LLM).

plan_env: stored EnvSpec (subtask A) + live GPU (nvidia-smi) + static import scan of the
entrypoints + packages added by add_package known fixes -> EnvPlan and image tag
reproducer/<run name>:<plan hash>. prepare_env stores the plan as env_plan evidence (authority
rule), the plan's assumptions as Assumption events for OPEN unknowns, and writes
<out>/env/<hash>.Dockerfile. build_env_image builds it (cached if the tag exists).
Repo code never runs on the host. The CLI `env` command and the setup graph both use these.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reproducer.domain.contracts import EnvSpec
from reproducer.domain.env_rules import (
    EnvPlan,
    GpuInfo,
    env_plan_to_evidence,
    plan_environment,
    render_dockerfile,
)
from reproducer.domain.evidence import Evidence, canonical_json
from reproducer.domain.import_scan import (
    ImportedPackage,
    ImportScan,
    normalize_pip_name,
    third_party_imports,
)
from reproducer.domain.unknowns import (
    Assumption,
    UnknownState,
    fold_unknowns,
    unknown_id_for,
)
from reproducer.graph.nodes.subtasks import SUBTASK_META_KEYS
from reproducer.providers.docker_runner import (
    BuildResult,
    build_image,
    image_size_bytes,
    image_tag,
)
from reproducer.providers.git import list_files, read_text
from reproducer.providers.gpu import query_gpu
from reproducer.providers.store import RunStore, UnknownEventT
from reproducer.rules.known_fixes import fold_fix_state


@dataclass(frozen=True)
class PlannedEnv:
    plan: EnvPlan
    gpu: GpuInfo | None
    spec_evidence: Evidence
    scan: ImportScan
    tag: str

    @property
    def digest(self) -> str:
        return self.tag.rsplit(":", 1)[1]


PlanEnvFn = Callable[[Path, bool], PlannedEnv]


def stored_env_spec(store: RunStore) -> Evidence:
    evidence = store.load_evidence()
    specs = [e for e in evidence.values() if e.kind == "env_spec" and e.locator == "subtask:A"]
    if not specs:
        raise ValueError(f"no EnvSpec (subtask A) in {store.root}; run `configure` first")
    # the same EnvSpec stored again with other node metadata (gate_attempts, evicted) is one
    # spec, as an identical record is (content-hash dedupe): the first stored one is used
    unique: dict[tuple[str, str], Evidence] = {}
    for e in specs:
        data = {k: v for k, v in e.content.items() if k not in SUBTASK_META_KEYS}
        unique.setdefault((e.source, canonical_json(data)), e)
    specs = list(unique.values())
    sources = {e.source for e in specs}
    if len(sources) > 1 or len(specs) > 1:
        raise ValueError(f"{len(specs)} EnvSpecs stored ({sorted(sources)}); expected exactly one")
    return specs[0]


def plan_env(
    out: Path, cpu_only: bool, *, gpu_query: Callable[[], GpuInfo | None] = query_gpu
) -> PlannedEnv:
    """Deterministic: stored EnvSpec + live GPU + import scan of <out>/repo -> plan and tag."""
    store = RunStore(out)
    evidence = store.load_evidence()
    spec_ev = stored_env_spec(store)
    texts = {i: evidence[i].content.get("text", "") for i in spec_ev.links if i in evidence}
    env = EnvSpec.model_validate(
        {k: v for k, v in spec_ev.content.items() if k not in SUBTASK_META_KEYS},
        context={
            "texts": texts,
            "evidence_ids": frozenset(texts),
            "aliases": {},
            "evidence_kinds": {i: evidence[i].kind for i in texts},
            "allowed_unknowns": frozenset(u["name"] for u in spec_ev.content["unknowns"]),
        },
    )
    repo_dir = out / "repo"
    files = {
        f.path: (read_text(repo_dir, f.path)[0] if f.path.endswith(".py") else "")
        for f in list_files(repo_dir)
        if not f.is_symlink
    }
    entry_paths = sorted(
        {
            str(ep["path"])
            for e in evidence.values()
            if e.kind == "entrypoints" and e.source == spec_ev.source
            for ep in e.content.get("entrypoints", [])
        }
    )
    scan = third_party_imports({p: files[p] for p in entry_paths if p in files}, files)
    # packages added by known fixes (add_package), in the order they were written
    fixes = fold_fix_state(e for e in evidence.values() if e.source == spec_ev.source)
    extra = tuple(
        ImportedPackage(pip_name=pkg, module=pkg, path="known_fix")
        for pkg in fixes.packages
        if normalize_pip_name(pkg) not in {normalize_pip_name(p.pip_name) for p in scan.packages}
    )
    if extra:
        scan = scan.model_copy(update={"packages": (*scan.packages, *extra)})
    gpu = gpu_query()
    plan = plan_environment(env, gpu, cpu_only=cpu_only, imports=scan.packages)
    return PlannedEnv(plan, gpu, spec_ev, scan, image_tag(plan, out.resolve().name.lower()))


def env_assumption_events(
    plan: EnvPlan,
    plan_evidence_id: str,
    events: Sequence[UnknownEventT],
    unknowns: Mapping[str, UnknownState],
) -> list[Assumption]:
    """One Assumption (evidence_id = the env_plan evidence) per plan assumption whose name is an
    OPEN unknown. Names never opened (or already resolved) and value-less assumptions
    ("unpinned:<pkg>") are skipped; an identical Assumption already in events is not repeated.
    Never resolves anything."""
    seen = {(e.unknown_id, e.value, e.evidence_id) for e in events if isinstance(e, Assumption)}
    out: list[Assumption] = []
    for a in plan.assumptions:
        uid = unknown_id_for(a.name)
        state = unknowns.get(uid)
        if a.value is None or state is None or state.status != "open":
            continue
        if (uid, a.value, plan_evidence_id) in seen:
            continue
        seen.add((uid, a.value, plan_evidence_id))
        out.append(Assumption(unknown_id=uid, value=a.value, evidence_id=plan_evidence_id))
    return out


@dataclass(frozen=True)
class PreparedEnv:
    planned: PlannedEnv
    evidence: Evidence  # env_plan
    written: bool  # False: the same env_plan evidence was already stored
    assumed: tuple[Assumption, ...]  # Assumption events written now
    assumed_names: tuple[str, ...]  # unknown name of each assumed event
    dockerfile: str
    dockerfile_path: Path


def prepare_env(
    out: Path, store: RunStore, pe: PlannedEnv, *, assumptions: bool = True
) -> PreparedEnv:
    """Store the env_plan evidence (+ Assumption events for open unknowns unless
    assumptions=False) and write <out>/env/<hash>.Dockerfile."""
    ev = env_plan_to_evidence(
        pe.plan, pe.spec_evidence.source, links=(pe.spec_evidence.evidence_id,), gpu=pe.gpu
    )
    written = store.append_evidence(ev)
    assumed: list[Assumption] = []
    names: list[str] = []
    if assumptions:
        events = store.load_events()
        unknowns = fold_unknowns(events, store.load_evidence())
        assumed = env_assumption_events(pe.plan, ev.evidence_id, events, unknowns)
        for event in assumed:
            store.append_event(event)
            names.append(unknowns[event.unknown_id].name)
    dockerfile = render_dockerfile(pe.plan)
    env_dir = out / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    path = env_dir / f"{pe.digest}.Dockerfile"
    path.write_text(dockerfile)
    return PreparedEnv(pe, ev, written, tuple(assumed), tuple(names), dockerfile, path)


def build_env_image(
    out: Path,
    prep: PreparedEnv,
    *,
    build_timeout: float,
    builder: Callable[..., BuildResult] = build_image,
    size_of: Callable[[str], int | None] = image_size_bytes,
) -> tuple[BuildResult, int | None]:
    """Build the prepared Dockerfile (cached if the tag exists); (result, size or None)."""
    pe = prep.planned
    result = builder(
        prep.dockerfile,
        pe.tag,
        timeout_s=build_timeout,
        log_path=out / "env" / f"build-{pe.digest}.log",
    )
    return result, (size_of(pe.tag) if result.ok else None)


def cleanup_workdir(workdir: Path, *, keep: bool) -> None:
    """Delete a per-run working copy after its container run unless keep (--keep-workdir).
    Only the copy is removed: run logs and the /out dir live next to it and are always kept."""
    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)


# --- graph node ------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvSummary:
    """One env node run, for the reproduce summary."""

    image: str
    base_image: str
    torch: str | None
    python: str | None
    cuda: str | None
    pip: tuple[str, ...]
    deviations: tuple[str, ...]
    assumptions: tuple[str, ...]
    cached: bool
    ok: bool
    build_s: float
    log: str | None
    packages_from_fixes: tuple[str, ...] = ()
    unpinned: tuple[str, ...] = ()  # packages left unpinned (value-less "unpinned:<pkg>")


def _unpinned(a: Any) -> bool:
    return a.value is None and a.name.startswith("unpinned:")


def env_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    """Plan + store + build. deps: store, plan_env(out, cpu_only), build_image, image_size.
    Failure (no EnvSpec, build failed) -> env_ok False, setup_status env_failed."""
    out = Path(state["out_dir"])
    store = deps["store"]
    builds = state.get("env_builds", 0) + 1
    try:
        pe = deps["plan_env"](out, bool(state.get("cpu_only", False)))
    except (ValueError, OSError) as e:
        return {"env_ok": False, "env_builds": builds, "setup_status": "env_failed",
                "stop_reason": "env_failed", "stop_detail": str(e)}  # fmt: skip
    prep = prepare_env(out, store, pe)
    result, _ = build_env_image(
        out,
        prep,
        build_timeout=float(state.get("build_timeout_s", 3600)),
        builder=deps["build_image"],
        size_of=deps["image_size"],
    )
    plan = pe.plan
    summary = EnvSummary(
        image=pe.tag,
        base_image=plan.base_image,
        torch=plan.torch,
        python=plan.python,
        cuda=plan.cuda,
        pip=tuple(plan.pip_packages),
        deviations=tuple(
            f"{d.field} {d.stated} -> {d.chosen}: {d.reason}" for d in plan.deviations
        ),
        assumptions=tuple(
            f"{a.name} = {a.value}: {a.reason}" for a in plan.assumptions if not _unpinned(a)
        ),
        unpinned=tuple(a.name.split(":", 1)[1] for a in plan.assumptions if _unpinned(a)),
        cached=result.cached,
        ok=result.ok,
        build_s=result.duration_s,
        log=str(result.log_path) if result.log_path is not None else None,
        packages_from_fixes=tuple(p.pip_name for p in pe.scan.packages if p.path == "known_fix"),
    )
    update: dict[str, Any] = {
        "env_plan": plan,
        "image_tag": pe.tag,
        "env_builds": builds,
        "env_log": [summary],
        "env_ok": result.ok,
    }
    if not result.ok:
        update.update(setup_status="env_failed", stop_reason="env_failed",
                      stop_detail=f"image build failed (see {result.log_path})")  # fmt: skip
    return update
