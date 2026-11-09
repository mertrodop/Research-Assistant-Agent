"""Command-line entry point.

    python -m reproducer.cli inventory <github-url> [--commit SHA] --out runs/<name>

Clones into <out>/repo (never executes anything from it), classifies the files, reads the
README and requirement/docker files, and appends evidence to RunStore(<out>).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from reproducer.graph.nodes.configure import (
    classify_files,
    inventory_to_evidence,
    parse_github_url,
    repo_source,
)
from reproducer.providers.git import (
    CloneError,
    clone_repo,
    head_sha,
    list_files,
    read_text,
    validate_commit,
)
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

    evidence = inventory_to_evidence(inv, repo_source(args.url, sha), texts)
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
    return 0


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
    inv.set_defaults(func=_cmd_inventory)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
