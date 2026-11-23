"""Command-line entry point.

    python -m reproducer.cli inventory <github-url> [--commit SHA] --out runs/<name>
    python -m reproducer.cli subtasks --out runs/<name>
    python -m reproducer.cli configure <github-url> [--commit SHA] --out runs/<name>

Clones into <out>/repo (never executes anything from it), classifies the files, reads the
README and requirement/docker files, and appends evidence to RunStore(<out>). Then runs
the deterministic extractors (requirements, setup.py, pyproject, environment.yml, config
keys, entrypoint candidates; ast/safe loaders only) and appends derived evidence. Shell
scripts (*.sh, at most MAX_SHELL_SCRIPTS) are stored as observed text.

`subtasks` loads that store, rebuilds the facts from stored evidence (re-running the
extractors on <out>/repo only if none are stored), and runs the gated LLM subtasks A (env),
B (entrypoints), C (workflow) sequentially with one shared Budget. Evidence, UnknownOpened
events and LLM call records are appended to the store.

`configure` runs both steps as the LangGraph configure graph (graph/build.py): clone or
reuse <out>/repo, collect evidence, run A/B/C in parallel, then one persist step writes
evidence + events. Status is configured only if all three subtasks succeed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reproducer.domain.evidence import Evidence
from reproducer.domain.inventory import RepoInventory
from reproducer.domain.unknowns import UnknownOpened
from reproducer.graph.build import build_configure_graph
from reproducer.graph.nodes.configure import (
    classify_files,
    inventory_to_evidence,
    parse_github_url,
    repo_source,
)
from reproducer.graph.nodes.subtasks import (
    MAX_SHELL_SCRIPTS,
    SUBTASK_KINDS,
    SubtaskOutcome,
    build_subtask_context,
    collect_texts,
    command_evidence,
    extract_repo_facts,
    file_evidence,
    is_shell_script,
    load_repo_facts,
    metric_line_evidence,
    run_subtask,
    version_evidence,
)
from reproducer.graph.state import merge_evidence
from reproducer.providers.git import (
    CloneError,
    clone_repo,
    head_sha,
    list_files,
    read_text,
    validate_commit,
)
from reproducer.providers.llm import Budget, BudgetExceeded, LLMError, llm_for_node
from reproducer.providers.store import RunStore


def _cmd_inventory(args: argparse.Namespace) -> int:
    try:
        parse_github_url(args.url)  # fail fast, before cloning
        if args.commit is not None:
            validate_commit(args.commit)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    repo_dir = out / "repo"
    try:
        if args.reuse_clone and repo_dir.exists():
            sha = head_sha(repo_dir)
            if args.commit is not None and sha != args.commit:
                raise CloneError(
                    f"--reuse-clone: {repo_dir} is at {sha}, not the requested --commit "
                    f"{args.commit}; use a fresh --out"
                )
            print(f"reusing existing clone {repo_dir}", file=sys.stderr)
        else:
            sha = clone_repo(args.url, repo_dir, commit=args.commit, timeout_s=args.timeout)
    except (CloneError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    inv = classify_files(list_files(repo_dir), sha)
    texts: dict[str, tuple[str, bool]] = {}
    wanted = ([inv.readme] if inv.readme else []) + [*inv.requirement_files, *inv.docker_files]
    for path in wanted:
        try:
            texts[path] = read_text(repo_dir, path)
        except (ValueError, OSError) as e:
            print(f"warning: skipped {path}: {e}", file=sys.stderr)

    source = repo_source(args.url, sha)
    evidence = inventory_to_evidence(inv, source, texts)
    store = RunStore(out)
    written = sum(store.append_evidence(ev) for ev in evidence)

    print(f"repo:          {args.url}")
    print(f"commit:        {sha}")
    print(f"files:         {len(inv.files)}")
    print(f"python:        {len(inv.python_files)}")
    print(f"notebooks:     {len(inv.notebooks)}")
    print(f"configs:       {len(inv.config_files)}")
    print(f"requirements:  {len(inv.requirement_files)} {list(inv.requirement_files)}")
    print(f"docker:        {len(inv.docker_files)} {list(inv.docker_files)}")
    print(f"submodules:    {inv.has_submodules}")
    print(f"readme:        {inv.readme}")
    print(f"evidence:      {written} written ({len(evidence)} built) -> {store.evidence_path}")

    facts = extract_repo_facts(inv, collect_texts(repo_dir, inv, args.max_py_files), source)
    derived = [ev for ev in facts if ev.confidence == "derived"]
    observed = [ev for ev in facts if ev.confidence == "observed"]
    obs_written = sum(store.append_evidence(ev) for ev in observed)
    der_written = sum(store.append_evidence(ev) for ev in derived)
    by_parser = Counter(ev.content["parser"] for ev in derived)
    counts = " ".join(f"{k}={by_parser[k]}" for k in sorted(by_parser)) or "none"
    print(f"extracted:     {counts}")
    print(
        f"derived:       {der_written} written ({len(derived)} built), "
        f"+{obs_written} source-file records"
    )

    # shell scripts: observed text, scanned for example commands by `subtasks`
    scripts = [f.path for f in inv.files if not f.is_symlink and is_shell_script(f.path)]
    sh_written = 0
    for path in scripts[:MAX_SHELL_SCRIPTS]:
        try:
            text, truncated = read_text(repo_dir, path)
        except (ValueError, OSError) as e:
            print(f"warning: skipped {path}: {e}", file=sys.stderr)
            continue
        sh_written += store.append_evidence(file_evidence(source, path, text, truncated))
    print(f"shell scripts: {len(scripts)} found, {sh_written} written (max {MAX_SHELL_SCRIPTS})")
    return 0


def _latest_repo_evidence(store: RunStore) -> tuple[str, dict[str, Evidence]]:
    """Evidence of the most recently stored inventory's source (repo@commit) only."""
    evidence = store.load_evidence()
    invs = [ev for ev in evidence.values() if ev.kind == "repo_file" and ev.locator == "inventory"]
    if not invs:
        raise ValueError(f"no inventory evidence in {store.evidence_path}; run `inventory` first")
    source = invs[-1].source
    return source, {k: ev for k, ev in evidence.items() if ev.source == source}


def _print_outcome(name: str, out: SubtaskOutcome, tokens: int, latency: float) -> None:
    line = (
        f"{name:<15} ok={out.ok} attempts={out.attempts} tokens={tokens} "
        f"latency={latency:.1f}s unknowns_opened={len(out.events)}"
    )
    if out.ok and out.kind == "B":
        paths = [e.path for e in out.value.entrypoints]  # type: ignore[union-attr]
        line += f" entrypoints={len(paths)} {paths}"
    print(line)
    v: Any = out.value
    if out.ok and out.kind == "A":
        pkgs = [p for p in (v.python, *v.packages) if p is not None]
        print(
            "  packages: " + (", ".join(f"{p.name}={p.version}({p.source})" for p in pkgs) or "-")
        )
    elif out.ok and out.kind == "B":
        for e in v.entrypoints:
            print(
                f"  {e.path} kind={e.kind} args={list(e.args_used)} example={e.example is not None}"
            )
    elif out.ok and out.kind == "C":
        print(
            f"  train={list(v.train_scripts)} eval={list(v.eval_scripts)} "
            f"configs={list(v.configs)} datasets={[d.name for d in v.datasets]} "
            f"checkpoints={len(v.checkpoints)} "
            f"metrics={[(m.path, m.metric) for m in v.metric_outputs]}"
        )
    for ev in out.events:
        print(f"  opened {ev.unknown_id}")
    for name in out.dropped_unknowns:
        print(f"  dropped {name} (version cited)")
    for i, err in enumerate(out.errors, start=1):
        label = "rejected" if out.ok else "error"
        print(f"  {label} attempt {i}: {err.splitlines()[0] if err else ''}")


def _cmd_subtasks(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    if not (out_dir / "evidence.jsonl").exists():
        print(f"error: no run store at {out_dir}; run `inventory` first", file=sys.stderr)
        return 2
    store = RunStore(out_dir)
    try:
        source, evidence = _latest_repo_evidence(store)
        facts, scan = load_repo_facts(evidence)
        if not facts:
            repo_dir = out_dir / "repo"
            if not repo_dir.is_dir():
                raise ValueError(f"no stored repo facts and no clone at {repo_dir}")
            inv_ev = next(ev for ev in evidence.values() if ev.locator == "inventory")
            inv = RepoInventory.model_validate(inv_ev.content)
            print("no stored facts: re-running extractors on the clone", file=sys.stderr)
            for ev in extract_repo_facts(inv, collect_texts(repo_dir, inv), source):
                store.append_evidence(ev)
            source, evidence = _latest_repo_evidence(store)
            facts, scan = load_repo_facts(evidence)
        versions = version_evidence(evidence)
        metric_lines = metric_line_evidence(evidence)
        commands = command_evidence(evidence)
        for ev in (*versions, *metric_lines, *commands):
            store.append_evidence(ev)
            evidence[ev.evidence_id] = ev
        # same (kind, locator, id) order as the graph state, so aliases/prompts match it
        evidence = {ev.evidence_id: ev for ev in merge_evidence(evidence.values(), None)}
        facts, scan = load_repo_facts(evidence)
        context, packed = build_subtask_context(evidence, facts, scan, max_chars=args.max_chars)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    existing = {ev.unknown_id for ev in store.load_events() if isinstance(ev, UnknownOpened)}
    budget = Budget(max_tokens_total=args.budget_tokens)
    print(f"source:         {source}")
    n_mentions = sum(len(ev.content["result"]) for ev in versions)
    print(
        f"context:        {len(context['texts'])} texts, prompt chars A/B/C = "
        f"{len(packed['A'])}/{len(packed['B'])}/{len(packed['C'])}"
    )
    print(f"candidates:     {sorted(context['entrypoint_args'])}")
    print(f"versions:       {n_mentions} mention(s) found by code")
    n_lines = sum(len(ev.content["result"]) for ev in metric_lines)
    print(f"metric lines:   {n_lines} found by code")
    n_cmds = sum(len(ev.content["result"]) for ev in commands)
    print(f"commands:       {n_cmds} found by code")
    names = {"A": "A env", "B": "B entrypoints", "C": "C workflow"}
    failed = False
    for kind in SUBTASK_KINDS:
        try:
            llm = llm_for_node(f"subtask_{kind.lower()}", budget)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        llm.on_record = store.append_call_record
        try:
            outcome = run_subtask(
                kind,
                llm,
                context,
                packed[kind],
                source=source,
                existing_unknown_ids=existing,
                max_tokens=args.max_tokens,
            )
        except (BudgetExceeded, LLMError) as e:
            print(f"{names[kind]:<15} aborted: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        for ev in outcome.evidence:
            store.append_evidence(ev)
        for event in outcome.events:
            store.append_event(event)
            existing.add(event.unknown_id)
        records = llm.records()
        tokens = sum(r.input_tokens + r.output_tokens for r in records)
        _print_outcome(names[kind], outcome, tokens, sum(r.latency_s for r in records))
        failed |= not outcome.ok
    print(f"budget:         {budget.used_tokens} tokens used")
    return 1 if failed else 0


def _cmd_configure(args: argparse.Namespace) -> int:
    try:
        parse_github_url(args.url)  # fail fast, before cloning
        if args.commit is not None:
            validate_commit(args.commit)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    budget = Budget(max_tokens_total=args.budget_tokens)
    try:  # fail on missing LLM config before any clone or call (constructors do no IO)
        for kind in SUBTASK_KINDS:
            llm_for_node(f"subtask_{kind.lower()}", budget)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    store = RunStore(Path(args.out))
    graph = build_configure_graph(
        {"store": store, "llm_for": lambda node: llm_for_node(node, budget), "budget": budget}
    )
    start = time.monotonic()
    try:
        final = graph.invoke(
            {
                "out_dir": args.out,
                "repo_url": args.url,
                "commit": args.commit,
                "max_chars": args.max_chars,
                "max_tokens": args.max_tokens,
                "status": "running",
                "configure_attempts": 0,
            }
        )
    except (CloneError, FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    wall = time.monotonic() - start
    names = {"A": "A env", "B": "B entrypoints", "C": "C workflow"}
    print(f"status:         {final['status']}")
    print(f"source:         {final['source']}")
    for o in sorted(final["outcomes"], key=lambda o: o.kind):
        print(
            f"{names[o.kind]:<15} ok={o.ok} attempts={o.attempts} tokens={o.tokens} "
            f"latency={o.latency_s:.1f}s"
        )
        for i, err in enumerate(o.errors, start=1):
            label = "rejected" if o.ok else "error"
            print(f"  {label} attempt {i}: {err.splitlines()[0] if err else ''}")
    n_state = len(final["evidence"])
    print(f"evidence:       {final['evidence_written']} written ({n_state} in state)")
    print(f"events:         {final['events_written']} written")
    print(f"budget:         {budget.used_tokens} tokens used")
    print(f"wall time:      {wall:.1f}s")
    return 0 if final["status"] == "configured" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reproducer")
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="clone a repo and record a read-only inventory")
    inv.add_argument("url", help="GitHub repository URL")
    inv.add_argument("--commit", default=None, help="full 40-char commit sha to check out")
    inv.add_argument("--out", required=True, help="run folder, e.g. runs/clip")
    inv.add_argument(
        "--reuse-clone",
        action="store_true",
        help="if <out>/repo exists, verify it is a git checkout (at --commit if given) "
        "and rebuild the inventory from it instead of cloning",
    )
    inv.add_argument("--timeout", type=int, default=300, help="clone timeout in seconds")
    inv.add_argument(
        "--max-py-files",
        type=int,
        default=50,
        help="max Python files scanned for entrypoints (train/eval/main/run names first)",
    )
    inv.set_defaults(func=_cmd_inventory)

    st = sub.add_parser("subtasks", help="run the gated LLM subtasks A/B/C on a run folder")
    st.add_argument("--out", required=True, help="run folder created by `inventory`")
    st.add_argument("--max-chars", type=int, default=12_000, help="packed prompt size cap")
    st.add_argument("--max-tokens", type=int, default=1_500, help="max output tokens per call")
    st.add_argument(
        "--budget-tokens", type=int, default=200_000, help="token cap shared by A, B and C"
    )
    st.set_defaults(func=_cmd_subtasks)

    cf = sub.add_parser("configure", help="run the configure graph (inventory + A/B/C)")
    cf.add_argument("url", help="GitHub repository URL")
    cf.add_argument("--commit", default=None, help="full 40-char commit sha to check out")
    cf.add_argument("--out", required=True, help="run folder; <out>/repo is reused if present")
    cf.add_argument("--max-chars", type=int, default=12_000, help="packed prompt size cap")
    cf.add_argument("--max-tokens", type=int, default=1_500, help="max output tokens per call")
    cf.add_argument(
        "--budget-tokens", type=int, default=200_000, help="token cap shared by A, B and C"
    )
    cf.set_defaults(func=_cmd_configure)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
