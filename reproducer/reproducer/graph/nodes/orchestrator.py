"""Orchestrator node (M3-S1b): one gated LLM call maps the paper's claims onto the repo, code
validates the mapping and builds one RunPlan per claim.

Order of work (every step but the LLM call is deterministic code):
  1. tolerance evidence for every claim (pre-registration: before the mapping and any run)
  2. target: entrypoint, its argparse spec and its verified commands, picked by code (the
     train script of subtask B's verified example, else the first entrypoint with a verified
     command; it must have metric lines). Per claim, code picks the verified command sharing
     the most claim_args (ties: the first; B's example first)
  3. '<entrypoint>#train_config' evidence: optimizer / scheduler / lr lines of the entrypoint
     and the local modules it imports (domain/train_config.py)
  4. input digest (paper claims + setup, repo source, ids of the evidence used). A stored
     claim_mapping with the same digest is reused (re-validated, no LLM call) unless replan.
     Otherwise gated_call(ClaimMapping): the LLM names claim flags, setup fields, the metric
     line and a conversion, and checks the non-flag setup fields against cited repo lines. It
     never writes a hyperparameter value: build_run_plan copies them from paper evidence.
  4b. plan digest = the input digest when no unknown is resolved and no lesson is applied,
     else sha256(input digest + the resolved unknowns' values and resolving evidence ids +
     the ids of the applied Reflexion lessons (lesson_applied evidence)). Applied arg /
     eval_protocol changes are folded into the plans by build_run_plan (LessonOverride).
     A human answer (or any other resolution) or an applied lesson therefore yields new run
     plans (build_run_plan uses the resolved value instead of the assumption) WITHOUT a new
     LLM call: the mapping digest is unchanged.
     run_plan evidence carries the plan digest; the current plan = newest for it.
  5. claim_mapping evidence (authority llm), then per claim: build_run_plan -> run_plan
     evidence (authority rule) with the rendered command, notes and Deviations (inconsistent
     setup checks; fidelity inferred), then Assumption events for the open unknowns the plan
     depends on (skipped if resolved or already recorded).
Repo code is only read (ast / text), never imported or executed.
"""

from __future__ import annotations

import hashlib
import posixpath
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reproducer.domain.contracts import (
    ArgOption,
    ClaimMap,
    ClaimMapping,
    ResultClaim,
    RunPlan,
    metric_line_of,
    parse_arg_value,
)
from reproducer.domain.env_rules import Deviation
from reproducer.domain.evidence import Evidence, canonical_json, make_evidence
from reproducer.domain.grounding import CITATION_MAX_QUOTE_CHARS
from reproducer.domain.import_scan import local_import_modules
from reproducer.domain.inventory import RepoInventory
from reproducer.domain.ledger import pre_registered_tolerance, tolerance_evidence
from reproducer.domain.run_plan import (
    ClaimArg,
    LessonOverride,
    PlanNote,
    SetupArg,
    build_run_plan,
    command_flags,
    render_command,
)
from reproducer.domain.run_plan import ClaimMapping as PlanMapping
from reproducer.domain.run_plan import metric_conversions as build_metric_conversions
from reproducer.domain.train_config import train_config_evidence
from reproducer.domain.unknowns import Assumption, fold_unknowns
from reproducer.graph.eviction import call_with_eviction, markdown_sections, shrink_prompt
from reproducer.graph.nodes.subtasks import load_repo_facts
from reproducer.providers.git import read_text
from reproducer.providers.llm import LLM
from reproducer.providers.store import RunStore

ORCHESTRATOR_NODE = "orchestrator"
DEFAULT_MAX_TOKENS = 6_000  # a full Cutout mapping is ~2.1-2.5k output tokens; thinking
# models count their reasoning against this limit too
CODE_PROMPT_CHARS = 12_000
_TRAIN_CONFIG_ONLY = (
    "optimizer",
    "momentum",
    "nesterov",
    "weight_decay",
    "lr_schedule",
    "data_augmentation",
    "normalization",
)


# --------------------------------------------------------------------------- target


@dataclass(frozen=True)
class PlanTarget:
    """What code picked before the LLM runs: the script, its flags and verified commands."""

    source: str  # repo@commit
    entrypoint: str
    argparse_spec: tuple[ArgOption, ...]
    commands: tuple[tuple[str, str], ...]  # ('#commands' evidence id, line); B's example first
    repo_files: frozenset[str]
    used_ids: tuple[str, ...] = ()  # evidence code read to pick this (inventory, #parsed, ...)


def _result(ev: Evidence) -> list[Mapping[str, Any]]:
    r = ev.content.get("result")
    return [x for x in r if isinstance(x, Mapping)] if isinstance(r, list) else []


def repo_evidence(evidence: Mapping[str, Evidence]) -> tuple[str, RepoInventory, dict]:
    """(source, inventory, evidence of that source) for the most recently stored inventory."""
    invs = [ev for ev in evidence.values() if ev.kind == "repo_file" and ev.locator == "inventory"]
    if not invs:
        raise ValueError("no inventory evidence: run `configure` first")
    inv_ev = invs[-1]
    repo = {k: ev for k, ev in evidence.items() if ev.source == inv_ev.source}
    return inv_ev.source, RepoInventory.model_validate(inv_ev.content), repo


def _command_lines(repo: Mapping[str, Evidence], path: str) -> list[tuple[str, str]]:
    """(evidence_id, line) of every verified command running `path`, in evidence order."""
    return [
        (ev.evidence_id, str(r["line"]))
        for ev in repo.values()
        if ev.locator.endswith("#commands") and ev.content.get("parser") == "commands"
        for r in _result(ev)
        if r.get("path") == path and isinstance(r.get("line"), str)
    ]


def metric_lines_evidence(repo: Mapping[str, Evidence], path: str) -> Evidence | None:
    return next(
        (
            ev
            for ev in repo.values()
            if ev.locator == f"{path}#metric_lines" and ev.content.get("parser") == "metric_lines"
            and _result(ev)
        ),
        None,
    )  # fmt: skip


def plan_target(evidence: Mapping[str, Evidence]) -> PlanTarget:
    """Pick the entrypoint and base command (see the module docstring). Raises ValueError if
    no entrypoint has both a verified command and metric lines."""
    source, inv, repo = repo_evidence(evidence)
    facts, scan = load_repo_facts(repo)
    candidates = {c.path: c for c in scan.candidates}
    picks: list[tuple[str, tuple[str, str] | None]] = []  # (path, B's example)
    b = [ev for ev in repo.values() if ev.kind == "entrypoints"]
    b_ids: list[str] = []
    if b:
        b_ids.append(b[-1].evidence_id)
        for e in b[-1].content.get("entrypoints", []):
            ex = e.get("example") if isinstance(e, Mapping) else None
            if e.get("kind") != "train" or not isinstance(ex, Mapping):
                continue
            pair = (str(ex.get("evidence_id")), str(ex.get("quote")))
            if pair in _command_lines(repo, str(e["path"])):
                picks.append((str(e["path"]), pair))
    picks += [(path, None) for path in sorted(candidates) if _command_lines(repo, path)]
    inv_id = next(i for i, ev in repo.items() if ev.locator == "inventory")
    for path, example in picks:
        if path in candidates and metric_lines_evidence(repo, path) is not None:
            spec = tuple(
                ArgOption.model_validate(a.model_dump()) for a in candidates[path].arguments
            )
            lines = _command_lines(repo, path)
            commands = tuple(dict.fromkeys([*([example] if example else []), *lines]))
            parsed = [ev.evidence_id for ev in facts if ev.locator == f"{path}#parsed"]
            return PlanTarget(
                source=source,
                entrypoint=path,
                argparse_spec=spec,
                commands=commands,
                repo_files=frozenset(f.path for f in inv.files),
                used_ids=tuple(sorted({inv_id, *parsed, *b_ids, *(i for i, _ in commands)})),
            )
    raise ValueError("no entrypoint with a verified command and verified metric lines")


# --------------------------------------------------------------------------- train config


def _observed_code(repo: Mapping[str, Evidence], path: str) -> Evidence | None:
    return next(
        (
            ev
            for ev in repo.values()
            if ev.kind == "repo_file" and ev.confidence == "observed" and ev.locator == path
            and isinstance(ev.content.get("head"), str)
        ),
        None,
    )  # fmt: skip


def local_module_paths(entrypoint: str, text: str, repo_files: frozenset[str]) -> list[str]:
    """Repo files of the local modules the entrypoint imports (one level, ast only)."""
    here = posixpath.dirname(entrypoint)
    out: list[str] = []
    for module in local_import_modules(entrypoint, text, set(repo_files)):
        base = module.replace(".", "/")
        for root in dict.fromkeys(("", here)):
            prefix = f"{root}/" if root else ""
            for cand in (f"{prefix}{base}.py", f"{prefix}{base}/__init__.py"):
                if cand in repo_files and cand != entrypoint and cand not in out:
                    out.append(cand)
    return out


def train_config_for(
    evidence: Mapping[str, Evidence], target: PlanTarget, repo_dir: Path | None
) -> Evidence:
    """'#train_config' evidence over the entrypoint (stored head) and its local modules (read
    from the clone when present, else their stored heads)."""
    repo = {k: ev for k, ev in evidence.items() if ev.source == target.source}
    entry_ev = _observed_code(repo, target.entrypoint)
    if entry_ev is None:
        raise ValueError(f"no stored code text for {target.entrypoint}")
    texts = {target.entrypoint: str(entry_ev.content["head"])}
    links = [entry_ev.evidence_id]
    for path in local_module_paths(target.entrypoint, texts[target.entrypoint], target.repo_files):
        stored = _observed_code(repo, path)
        if stored is not None:
            links.append(stored.evidence_id)
        if repo_dir is not None and repo_dir.is_dir():
            try:
                texts[path] = read_text(repo_dir, path)[0]
                continue
            except (ValueError, OSError):
                pass
        if stored is not None:
            texts[path] = str(stored.content["head"])
    return train_config_evidence(target.entrypoint, texts, source=target.source, links=links)


# --------------------------------------------------------------------------- context


def paper_claims(evidence: Mapping[str, Evidence]) -> dict[str, tuple[ResultClaim, Evidence]]:
    out: dict[str, tuple[ResultClaim, Evidence]] = {}
    for ev in evidence.values():
        if ev.kind == "paper_claim":
            claim = ResultClaim.model_validate({**ev.content, "location": ev.locator})
            out.setdefault(claim.claim_id, (claim, ev))
    return out


def paper_setup_evidence(evidence: Mapping[str, Evidence]) -> dict[str, Evidence]:
    out: dict[str, Evidence] = {}
    for ev in evidence.values():
        if ev.kind == "paper_setup":
            out.setdefault(str(ev.content.get("name")), ev)
    return out


_CIT = '{"evidence_id": "E1", "quote": "<text copied exactly from that source>"}'
SYSTEM = f"""You map a paper's result claims onto a Python research repo so code can run them.
Return ONLY JSON:
{{"claims": [{{"claim_id": "...",
   "claim_args": [{{"flag": "--dataset", "value": "<one of the flag's choices>"}}],
   "setup_mappings": [{{"setup_name": "<paper setup field>", "flag": "--epochs"}}],
   "metric": {{"metric_line_ref": "<alias of the verified metric lines>",
              "metric_line_quote": Citation of the one metric line printing the value,
              "conversion": "<one of the allowed conversions listed for that line>",
              "scale_citation": Citation}}}}],
 "unmapped_setup": [{{"setup_name": "...", "reason": "<= 200 chars"}}],
 "setup_checks": [{{"setup_name": "...", "repo_citation": Citation, "consistent": true,
                   "note": "<= 200 chars"}}]}}
Citation = {_CIT}; use the alias in [brackets] as evidence_id; quotes max 300 chars.
Rules:
- Every claim_id exactly once. claim_args: the choice-flags the claim itself names (dataset,
  model, ...); each value must be one of the flag's choices and named in the claim's setting,
  dataset or quote. Code runs the verified command sharing the most claim_args.
- setup_mappings: paper setup fields that a listed flag sets for that claim (e.g. a cutout
  size stated for CIFAR-10 maps only on the CIFAR-10 claim). Never write values: code copies
  them from the paper. Only listed flags.
- metric.scale_citation: quote the script code that computes the printed value, showing its
  scale (e.g. "correct / total" is a fraction, "100. * correct / total" a percent).
- setup_checks: for stated setup fields that no flag sets ({", ".join(_TRAIN_CONFIG_ONLY)}),
  cite the repo line (prefer the training config lines alias) and say whether it is
  consistent with the paper value; explain any difference in note.
- unmapped_setup: other setup fields that are neither mapped nor checked, with the reason.
- Each setup field at most once in setup_checks and in unmapped_setup."""

ORCHESTRATOR_EXAMPLES: dict[str, str] = {
    "scale_citation": _CIT,
    "repo_citation": _CIT,
    "claim_args": '{"flag": "--dataset", "value": "cifar10"}',
    "metric_line_quote": '{"evidence_id": "E2", "quote": "<the whole metric line>"}',
    "setup_mappings": '{"setup_name": "epochs", "flag": "--epochs"}',
    "unmapped_setup": '{"setup_name": "num_runs", "reason": "no flag; runs are repeated"}',
    "setup_checks": f'{{"setup_name": "weight_decay", "repo_citation": {_CIT}, '
    '"consistent": true, "note": "5e-4 in the optimizer call"}',
}


def _fmt_option(a: ArgOption) -> str:
    parts = [f"type {a.type}"] if a.type else []
    if a.action:
        parts.append(f"action {a.action}")
    parts.append(f"default {a.default!r}")
    if a.choices is not None:
        parts.append(f"choices {list(a.choices)}")
    return f"- {'/'.join(a.option_strings)}: {', '.join(parts)}"


def build_orchestrator_context(
    evidence: Mapping[str, Evidence],
    target: PlanTarget,
    *,
    train_config: Evidence | None = None,
    code_chars: int = CODE_PROMPT_CHARS,
) -> tuple[dict[str, Any], str]:
    """Validation context + prompt (pure). Citable texts, with aliases: the entrypoint's code
    (E1), its verified metric lines, its training config lines."""
    claims = {cid: c for cid, (c, _) in paper_claims(evidence).items()}
    setup = paper_setup_evidence(evidence)
    repo = {k: ev for k, ev in evidence.items() if ev.source == target.source}
    code_ev = _observed_code(repo, target.entrypoint)
    metric_ev = metric_lines_evidence(repo, target.entrypoint)
    if code_ev is None or metric_ev is None:
        raise ValueError(f"{target.entrypoint}: needs stored code and metric lines")
    code = str(code_ev.content["head"])
    texts: dict[str, str] = {code_ev.evidence_id: code}
    aliases = {"E1": code_ev.evidence_id}
    texts[metric_ev.evidence_id] = "\n".join(str(r["line"]) for r in _result(metric_ev))
    aliases["E2"] = metric_ev.evidence_id
    tc_ids: set[str] = set()
    tc_lines = _result(train_config) if train_config is not None else []
    if train_config is not None and tc_lines:
        texts[train_config.evidence_id] = "\n".join(str(r["line"]) for r in tc_lines)
        aliases["E3"] = train_config.evidence_id
        tc_ids.add(train_config.evidence_id)
    conversions = build_metric_conversions([metric_ev], claims.values())

    claim_lines = [
        f"- {c.claim_id}: {c.metric} = {c.value}{' ' + c.unit if c.unit else ''} on {c.dataset}"
        f" ({c.split or '-'}); setting: {c.setting or '-'}"
        for c in claims.values()
    ]
    setup_lines = [
        f"- {name} = {ev.content.get('value')!s}"
        if ev.content.get("value") is not None
        else f"- {name} = UNKNOWN (not stated by the paper)"
        for name, ev in setup.items()
    ]
    metric_lines = []
    for r in _result(metric_ev):
        per = conversions[metric_ev.evidence_id].get(str(r["line"]), {})
        allowed = "; ".join(f"{cid}: {sorted(v) or 'none'}" for cid, v in per.items())
        metric_lines.append(
            f'- {r.get("path")}:{r.get("line_no")} "{r["line"]}"\n  allowed conversions: '
            + (allowed or "none (no pattern can be built from this line)")
        )
    tc_text = [
        f"- {r['path']}:{r['line_no']} {r['kind']} {r['name']}: "
        f'"{str(r["line"])[:CITATION_MAX_QUOTE_CHARS]}"'
        for r in tc_lines
    ] or ["(none found)"]
    shown_code = code[:code_chars] + (
        f"\n[... {len(code) - code_chars} more chars]" if len(code) > code_chars else ""
    )
    prompt = "\n\n".join(
        [
            "## Paper claims (claim_id: metric = value on dataset (split); setting)",
            "\n".join(claim_lines),
            "## Paper setup (name = value)",
            "\n".join(setup_lines),
            f"## Entrypoint {target.entrypoint}: argparse options",
            "\n".join(_fmt_option(a) for a in target.argparse_spec if a.option_strings[0][:1]),
            "## Verified commands (code runs the one sharing the most claim_args)",
            "\n".join(line for _, line in target.commands),
            "## Verified metric output lines [E2] (found by code)",
            "\n".join(metric_lines),
            "## Training config lines [E3] (found by code)" if tc_ids else "## Training config",
            "\n".join(tc_text),
            f"## Script code {target.entrypoint} [E1]",
            shown_code,
            "Return the JSON now.",
        ]
    )
    context: dict[str, Any] = {
        "claims": claims,
        "paper_setup": {n: ev.content.get("value") for n, ev in setup.items()},
        "argparse_spec": target.argparse_spec,
        "metric_conversions": conversions,
        "code_evidence_ids": frozenset({code_ev.evidence_id}),
        "train_config_ids": frozenset(tc_ids),
        "texts": texts,
        "aliases": aliases,
        "evidence_ids": frozenset(texts),
        "repo_files": target.repo_files,
    }
    return context, prompt


_PROMPT_TAIL = "Return the JSON now."
# eviction priority by section title prefix (lowest is dropped first; default 70)
_SECTION_PRIORITY = (
    ("Paper claims", 100),
    ("Entrypoint ", 95),
    ("Verified metric output lines", 92),
    ("Paper setup", 90),
    ("Verified commands", 80),
    ("Training config", 60),
    ("Script code", 50),
)


def _section_priority(_: int, title: str) -> int:
    return next((p for prefix, p in _SECTION_PRIORITY if title.startswith(prefix)), 70)


def orchestrator_prompt_builder(prompt: str) -> Callable[[int | None], tuple[str, str, tuple]]:
    """Eviction builder: the prompt unchanged when it fits the budget (None: always); else
    its sections packed by _SECTION_PRIORITY (script code goes first, then the config)."""
    body = prompt.removesuffix(_PROMPT_TAIL).rstrip("\n")
    sections = markdown_sections(body, _section_priority)

    def build(budget: int | None) -> tuple[str, str, tuple]:
        text, evicted = shrink_prompt(prompt, sections, budget, tail=_PROMPT_TAIL)
        return SYSTEM, text, evicted

    return build


# --------------------------------------------------------------------------- plans


def pick_base_command(cm: ClaimMap, target: PlanTarget) -> tuple[str, str]:
    """The verified command sharing the most claim_args (same argparse dest and parsed value);
    ties -> the first in target.commands. Commands code cannot parse are skipped."""
    spec = target.argparse_spec
    wanted: dict[str | None, Any] = {}
    for ca in cm.claim_args:
        opt = next((a for a in spec if ca.flag in a.option_strings), None)
        if opt is not None:
            wanted[opt.dest] = parse_arg_value(opt, ca.value)
    best: tuple[int, tuple[str, str]] | None = None
    for ev_id, line in target.commands:
        try:
            pairs = command_flags(line, target.entrypoint, spec)
        except ValueError:
            continue
        have: dict[str | None, Any] = {}
        for flag, value in pairs:
            opt = next(a for a in spec if flag in a.option_strings)
            if value is not None and opt.takes_value:
                try:
                    have[opt.dest] = parse_arg_value(opt, value)
                except ValueError:
                    continue
        score = sum(1 for d, v in wanted.items() if d in have and have[d] == v)
        if best is None or score > best[0]:
            best = (score, (ev_id, line))
    if best is None:
        raise ValueError(f"no verified command of {target.entrypoint} can be parsed")
    return best[1]


def to_plan_mapping(cm: ClaimMap, target: PlanTarget, context: Mapping[str, Any]) -> PlanMapping:
    """The validated LLM mapping of one claim as build_run_plan's input: the base command
    picked by code (pick_base_command), the metric line the mapping quotes."""
    ref = cm.metric.metric_line_ref
    ev_id, line = pick_base_command(cm, target)
    return PlanMapping(
        claim_id=cm.claim_id,
        entrypoint=target.entrypoint,
        base_command_evidence_id=ev_id,
        base_command=line,
        claim_args=tuple(ClaimArg(flag=a.flag, value=a.value) for a in cm.claim_args),
        setup_args=tuple(SetupArg(setup=m.setup_name, flag=m.flag) for m in cm.setup_mappings),
        metric_line_evidence_id=ref,
        metric_line=metric_line_of(context["texts"][ref], cm.metric.metric_line_quote.quote),
        conversion=cm.metric.conversion,
    )


def input_digest(
    evidence: Mapping[str, Evidence], target: PlanTarget, context: Mapping[str, Any]
) -> str:
    """sha256 over the paper claims + setup, the repo source and the sorted ids of the evidence
    the orchestrator used (inventory, #parsed, B, #commands, code, #metric_lines,
    #train_config). Same digest -> same inputs -> the stored mapping can be reused."""
    paper = sorted(
        canonical_json({"kind": ev.kind, "locator": ev.locator, "content": ev.content})
        for ev in evidence.values()
        if ev.kind in ("paper_claim", "paper_setup")
    )
    used = sorted({*target.used_ids, *context["texts"]})
    payload = canonical_json({"paper": paper, "source": target.source, "evidence_ids": used})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


Answers = Mapping[str, tuple[str, str]]  # setup name -> (resolved value, resolving evidence id)


def resolved_answers(
    events: Sequence[Any], evidence: Mapping[str, Evidence]
) -> dict[str, tuple[str, str]]:
    """setup name -> (value, evidence id) of every RESOLVED unknown that is a paper setup
    field without a value (the only unknowns a plan can use; others, e.g. dataset_source,
    never change the plans)."""
    fields = {n for n, ev in paper_setup_evidence(evidence).items()
              if ev.content.get("value") is None}  # fmt: skip
    out: dict[str, tuple[str, str]] = {}
    for st in fold_unknowns(events, evidence).values():
        if st.name in fields and st.status == "resolved" and st.value is not None:
            if st.resolved_by:
                out[st.name] = (st.value, st.resolved_by)
    return dict(sorted(out.items()))


# lesson_applied outcomes that change a plan (stopped_no_change changes nothing)
PLAN_CHANGING_OUTCOMES = frozenset({"applied", "resolved", "reeval"})


def applied_lessons(evidence: Mapping[str, Evidence]) -> list[Evidence]:
    """lesson_applied evidence that changed a plan, in store order."""
    return [ev for ev in evidence.values() if ev.kind == "lesson_applied"
            and ev.content.get("outcome") in PLAN_CHANGING_OUTCOMES]  # fmt: skip


def lesson_overrides(evidence: Mapping[str, Evidence]) -> dict[str, tuple[LessonOverride, ...]]:
    """claim_id -> the applied arg / eval_protocol changes, in store order (an assumption
    change acts through its resolved unknown instead)."""
    out: dict[str, list[LessonOverride]] = {}
    for ev in applied_lessons(evidence):
        c = ev.content
        if c.get("kind") == "arg":
            eid = str(c.get("support_evidence_id"))
        elif c.get("kind") == "eval_protocol":
            eid = str(c.get("lesson_id"))
        else:
            continue
        out.setdefault(str(c.get("claim_id")), []).append(
            LessonOverride(
                lesson_id=str(c.get("lesson_id")),
                kind=c["kind"],
                target=str(c.get("target")),
                new_value=c.get("new_value"),
                evidence_id=eid,
            )  # fmt: skip
        )
    return {k: tuple(v) for k, v in out.items()}


def plan_digest(mapping_digest: str, answers: Answers, lesson_ids: Sequence[str] = ()) -> str:
    """The mapping digest when nothing is resolved and no lesson applied (such stores keep
    their run plans), else sha256 over it, the resolved values + evidence ids and the sorted
    ids of the applied lessons."""
    if not answers and not lesson_ids:
        return mapping_digest
    body: dict[str, Any] = {
        "input_digest": mapping_digest,
        "resolved": {k: list(v) for k, v in sorted(answers.items())},
    }
    if lesson_ids:
        body["lessons"] = sorted(lesson_ids)
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def applied_lesson_ids(evidence: Mapping[str, Evidence]) -> list[str]:
    return [str(ev.content.get("lesson_id")) for ev in applied_lessons(evidence)]


def stored_mapping(evidence: Mapping[str, Evidence], digest: str) -> Evidence | None:
    """The newest claim_mapping evidence for this input digest."""
    hits = [
        ev
        for ev in evidence.values()
        if ev.kind == "claim_mapping" and ev.content.get("input_digest") == digest
    ]
    return hits[-1] if hits else None


def current_plans(evidence: Mapping[str, Evidence], digest: str) -> dict[str, Evidence]:
    """claim_id -> the newest run_plan evidence for this input digest (store order)."""
    out: dict[str, Evidence] = {}
    for ev in evidence.values():
        if ev.kind == "run_plan" and ev.content.get("input_digest") == digest:
            out[str(ev.content["claim_id"])] = ev
    return out


def setup_check_notes(
    mapping: ClaimMapping, paper_setup: Mapping[str, Any], mapping_evidence_id: str
) -> tuple[tuple[PlanNote, ...], tuple[Deviation, ...]]:
    """Plan notes for setup checks and unmapped fields; a Deviation per inconsistent check."""
    notes: list[PlanNote] = []
    devs: list[Deviation] = []
    for chk in mapping.setup_checks:
        quote = chk.repo_citation.quote
        if chk.consistent:
            notes.append(
                PlanNote(field=chk.setup_name, status="consistent", detail=f"{chk.note} [{quote}]")
            )
            continue
        notes.append(
            PlanNote(field=chk.setup_name, status="deviation", detail=f"{chk.note} [{quote}]")
        )
        devs.append(
            Deviation(
                field=chk.setup_name,
                stated=str(paper_setup.get(chk.setup_name)),
                chosen=quote,
                reason=chk.note,
                rule_source=("setup_check", f"claim_mapping:{mapping_evidence_id[:12]}"),
            )
        )
    for u in mapping.unmapped_setup:
        notes.append(PlanNote(field=u.setup_name, status="open", detail=f"unmapped: {u.reason}"))
    return tuple(notes), tuple(devs)


def _cited_ids(mapping: ClaimMapping) -> list[str]:
    ids = {c.metric.metric_line_ref for c in mapping.claims}
    ids |= {c.metric.metric_line_quote.evidence_id for c in mapping.claims}
    ids |= {c.metric.scale_citation.evidence_id for c in mapping.claims}
    ids |= {s.repo_citation.evidence_id for s in mapping.setup_checks}
    return sorted(ids)


@dataclass(frozen=True)
class PlannedClaim:
    claim: ResultClaim
    plan: RunPlan
    argv: tuple[str, ...]
    notes: tuple[PlanNote, ...]
    deviations: tuple[Deviation, ...]
    assumptions: tuple[Assumption, ...]  # values the plan assumes (source evidence ids)
    tolerance: Evidence
    evidence: Evidence  # run_plan
    fidelity: str  # "spec_faithful" | "inferred"

    @property
    def command(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class OrchestratorOutcome:
    ok: bool
    target: PlanTarget | None
    mapping: ClaimMapping | None
    mapping_evidence: Evidence | None
    train_config: Evidence | None
    tolerances: tuple[Evidence, ...]
    plans: tuple[PlannedClaim, ...]
    events_written: tuple[Assumption, ...]
    skipped_assumptions: tuple[str, ...]  # "<unknown_id>: <why>"
    errors: tuple[str, ...]
    attempts: int  # LLM attempts (0 when the stored mapping was reused)
    input_digest: str = ""  # the mapping digest
    reused: bool = False  # the stored claim_mapping for this digest was reused (no LLM call)
    plan_digest: str = ""  # stored on the run plans (input digest + resolved unknowns)


def plan_claims(
    mapping: ClaimMapping,
    mapping_evidence: Evidence,
    evidence: Mapping[str, Evidence],
    target: PlanTarget,
    context: Mapping[str, Any],
    tolerances: Mapping[str, Evidence],
    digest: str,
    answers: Answers | None = None,
    lessons: Mapping[str, Sequence[LessonOverride]] | None = None,
) -> tuple[list[PlannedClaim], list[str]]:
    """Per claim (in mapping order): RunPlan via build_run_plan + run_plan evidence (pure).
    digest: the plan digest stored on each run_plan; answers: resolved unknowns; lessons:
    claim_id -> applied lesson changes."""
    claims = paper_claims(evidence)
    setup = paper_setup_evidence(evidence)
    paper_values = {n: ev.content.get("value") for n, ev in setup.items()}
    check_notes, devs = setup_check_notes(mapping, paper_values, mapping_evidence.evidence_id)
    planned: list[PlannedClaim] = []
    errors: list[str] = []
    for cm in mapping.claims:
        claim, claim_ev = claims[cm.claim_id]
        try:
            plan, assumptions, notes = build_run_plan(
                claim,
                to_plan_mapping(cm, target, context),
                evidence,
                target.argparse_spec,
                setup.values(),
                repo_files=target.repo_files,
                answers=answers,
                lessons=(lessons or {}).get(cm.claim_id, ()),
            )
            argv = tuple(render_command(plan, target.argparse_spec))
        except ValueError as e:  # includes pydantic ValidationError
            errors.append(f"{cm.claim_id}: {e}")
            continue
        noted = {n.field for n in notes}
        all_notes = (*notes, *(n for n in check_notes if n.field not in noted))
        inferred = (
            bool(devs)
            or bool(plan.depends_on_unknowns)
            or any(a.source == "assumption" for a in plan.args)
        )
        fidelity = "inferred" if inferred else "spec_faithful"
        tol = tolerances[cm.claim_id]
        ev = make_evidence(
            kind="run_plan",
            source=target.source,
            locator=f"run_plan:{cm.claim_id}",
            content={
                "claim_id": cm.claim_id,
                "input_digest": digest,
                "plan": plan.model_dump(mode="json"),
                "command": shlex.join(argv),
                "argv": list(argv),
                "notes": [n.model_dump(mode="json") for n in all_notes],
                "deviations": [d.model_dump(mode="json") for d in devs],
                "assumptions": [a.model_dump(mode="json") for a in assumptions],
                "fidelity": fidelity,
            },
            confidence="derived",
            authority="rule",
            links=(mapping_evidence.evidence_id, claim_ev.evidence_id, tol.evidence_id),
        )
        planned.append(
            PlannedClaim(
                claim=claim,
                plan=plan,
                argv=argv,
                notes=tuple(all_notes),
                deviations=devs,
                assumptions=assumptions,
                tolerance=tol,
                evidence=ev,
                fidelity=fidelity,
            )
        )
    return planned, errors


def _prepare(
    evidence: Mapping[str, Evidence], tolerances: Mapping[str, Evidence], repo_dir: Path | None
) -> tuple[PlanTarget, Evidence, dict[str, Evidence], dict[str, Any], str, str]:
    """(target, train_config, evidence incl. tolerances + train_config, context, prompt,
    input digest): steps 2-4 before the LLM call (pure but for reading the clone)."""
    target = plan_target(evidence)
    tc = train_config_for(evidence, target, repo_dir)
    evidence = {**evidence, **tolerances, tc.evidence_id: tc}
    context, prompt = build_orchestrator_context(evidence, target, train_config=tc)
    return target, tc, evidence, context, prompt, input_digest(evidence, target, context)


def plan_failures(
    claim_ids: Sequence[str], planned: Sequence[str], errors: Sequence[str]
) -> dict[str, str]:
    """claim_id -> why it has no plan, for every claim not in `planned` (claim order): its
    plan_claims error ("<claim_id>: <why>"), else "not in the claim mapping"."""
    out: dict[str, str] = {}
    for cid in claim_ids:
        if cid in planned:
            continue
        why = next((e[len(cid) + 2 :] for e in errors if e.startswith(f"{cid}: ")), None)
        out[cid] = why or "not in the claim mapping"
    return out


@dataclass(frozen=True)
class CurrentPlans:
    target: PlanTarget
    input_digest: str  # the plan digest (= the mapping digest when nothing is resolved)
    plans: tuple[PlannedClaim, ...]  # mapping order; each one's run_plan evidence is stored
    failed: Mapping[str, str] = field(default_factory=dict)  # claim_id -> why (partial only)


def load_current_plans(
    store: RunStore, *, repo_dir: Path | None = None, partial: bool = False
) -> CurrentPlans:
    """The current plan per claim, read-only (no LLM call, nothing written): the stored
    claim_mapping for the current input digest, re-validated and re-planned by code. Each
    plan's run_plan evidence must be the newest stored one for that digest (same inputs ->
    same evidence id). Raises ValueError when `plan` has not been run for these inputs.
    partial: a claim that does not plan is listed in `failed` instead of raising (ValueError
    only when no claim plans)."""
    evidence = store.load_evidence()
    claims = paper_claims(evidence)
    if not claims:
        raise ValueError(f"no paper_claim evidence in {store.root}: configure with --spec first")
    tolerances = {
        cid: tolerance_evidence(c, claim_evidence_id=cev.evidence_id, source=cev.source)
        for cid, (c, cev) in claims.items()
    }
    missing = sorted(cid for cid, t in tolerances.items() if t.evidence_id not in evidence)
    if missing:
        raise ValueError(f"no pre-registered tolerance for {missing}: run `plan` first")
    target, _, evidence, context, _, mdigest = _prepare(evidence, tolerances, repo_dir)
    mapping_ev = stored_mapping(evidence, mdigest)
    if mapping_ev is None:
        raise ValueError(f"no claim_mapping for the current inputs ({mdigest[:12]}): run `plan`")
    mapping = ClaimMapping.model_validate(mapping_ev.content["mapping"], context=context)
    answers = resolved_answers(store.load_events(), evidence)
    digest = plan_digest(mdigest, answers, applied_lesson_ids(evidence))
    planned, errors = plan_claims(
        mapping, mapping_ev, evidence, target, context, tolerances, digest, answers,
        lesson_overrides(evidence),
    )  # fmt: skip
    if errors and not partial:
        raise ValueError(f"stored mapping no longer plans: {errors}")
    stored = current_plans(evidence, digest)
    ok: list[PlannedClaim] = []
    for p in planned:
        cur = stored.get(p.claim.claim_id)
        if cur is None or cur.evidence_id != p.evidence.evidence_id:
            msg = "stored run_plan is not current: run `plan`"
            if not partial:
                raise ValueError(f"{p.claim.claim_id}: {msg}")
            errors.append(f"{p.claim.claim_id}: {msg}")
            continue
        ok.append(p)
    failed = plan_failures(list(claims), [p.claim.claim_id for p in ok], errors)
    if partial and not ok:
        raise ValueError(f"no claim plans: {failed}")
    return CurrentPlans(target=target, input_digest=digest, plans=tuple(ok), failed=failed)


def run_orchestrator(
    store: RunStore,
    llm: LLM | None = None,
    *,
    llm_factory: Callable[[], LLM] | None = None,
    repo_dir: Path | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 2,
    replan: bool = False,
) -> OrchestratorOutcome:
    """The whole node on a configured store (see the module docstring). Writes, in order:
    tolerance, train_config, claim_mapping, run_plan evidence, Assumption events. A stored
    claim_mapping with the same input digest is reused without an LLM call unless `replan`;
    re-running then writes nothing new (same evidence ids, events deduplicated). The LLM is
    `llm`, else made by `llm_factory` only when a call is needed (a reused mapping needs no
    LLM config). Raises
    ValueError when the store has no claims or no plannable entrypoint; BudgetExceeded /
    LLMError propagate (tolerances are already written then)."""
    evidence = store.load_evidence()
    claims = paper_claims(evidence)
    if not claims:
        raise ValueError(f"no paper_claim evidence in {store.root}: configure with --spec first")

    tolerances: dict[str, Evidence] = {}
    for cid, (claim, claim_ev) in claims.items():  # 1. pre-registration, before anything else
        tol = tolerance_evidence(
            claim, claim_evidence_id=claim_ev.evidence_id, source=claim_ev.source
        )
        store.append_evidence(tol)
        tolerances[cid] = tol

    target, tc, evidence, context, prompt, digest = _prepare(evidence, tolerances, repo_dir)
    store.append_evidence(tc)

    notes: list[str] = []
    mapping: ClaimMapping | None = None
    mapping_ev = None if replan else stored_mapping(evidence, digest)
    attempts = 0
    if mapping_ev is not None:
        try:
            mapping = ClaimMapping.model_validate(mapping_ev.content["mapping"], context=context)
        except ValueError as e:  # should not happen for the same inputs; ask the LLM again
            notes.append(f"stored mapping {mapping_ev.evidence_id[:12]} rejected: {e}")
            mapping_ev = None
    reused = mapping is not None
    if mapping is None:
        if llm is None:
            if llm_factory is None:
                raise ValueError("run_orchestrator: an LLM call is needed: pass llm or llm_factory")
            llm = llm_factory()
        call = call_with_eviction(
            llm,
            orchestrator_prompt_builder(prompt),
            schema=ClaimMapping,
            context=context,
            max_tokens=max_tokens,
            max_retries=max_retries,
            required_context=("evidence_ids", "repo_files"),
            examples=ORCHESTRATOR_EXAMPLES,
        )
        result = call.result
        attempts = result.attempts
        notes += result.errors
        if not result.ok or result.value is None:
            return OrchestratorOutcome(
                ok=False, target=target, mapping=None, mapping_evidence=None, train_config=tc,
                tolerances=tuple(tolerances.values()), plans=(), events_written=(),
                skipped_assumptions=(), errors=tuple(notes), attempts=attempts,
                input_digest=digest,
            )  # fmt: skip
        mapping = result.value
        mapping_ev = make_evidence(
            kind="claim_mapping",
            source=target.source,
            locator=f"{ORCHESTRATOR_NODE}:claim_mapping",
            content={
                "input_digest": digest,
                "mapping": mapping.model_dump(mode="json"),
                **({"evicted": list(call.evicted)} if call.evicted else {}),
                "gate_attempts": result.attempts,  # additive: tries of the gate that answered
            },
            confidence="derived",
            authority="llm",
            links=_cited_ids(mapping),
        )
        store.append_evidence(mapping_ev)
    assert mapping_ev is not None
    evidence = {**evidence, mapping_ev.evidence_id: mapping_ev}

    events = store.load_events()
    answers = resolved_answers(events, evidence)
    pdigest = plan_digest(digest, answers, applied_lesson_ids(evidence))
    planned, errors = plan_claims(
        mapping, mapping_ev, evidence, target, context, tolerances, pdigest, answers,
        lesson_overrides(evidence),
    )  # fmt: skip
    for p in planned:
        store.append_evidence(p.evidence)

    # Assumption events: open unknowns each plan depends on; none for resolved ones, none twice
    stored = {**evidence, **{p.evidence.evidence_id: p.evidence for p in planned}}
    states = fold_unknowns(events, stored)
    have = {(e.unknown_id, e.value, e.evidence_id) for e in events if isinstance(e, Assumption)}
    written: list[Assumption] = []
    skipped: list[str] = []
    for p in planned:
        for a in p.assumptions:
            state = states.get(a.unknown_id)
            if state is None:
                skipped.append(f"{a.unknown_id}: never opened")
                continue
            if state.status == "resolved":
                skipped.append(f"{a.unknown_id}: resolved")
                continue
            ev = Assumption(
                unknown_id=a.unknown_id, value=a.value, evidence_id=p.evidence.evidence_id
            )
            if (ev.unknown_id, ev.value, ev.evidence_id) in have:
                continue
            store.append_event(ev)
            have.add((ev.unknown_id, ev.value, ev.evidence_id))
            written.append(ev)
    return OrchestratorOutcome(
        ok=not errors,
        target=target,
        mapping=mapping,
        mapping_evidence=mapping_ev,
        train_config=tc,
        tolerances=tuple(tolerances.values()),
        plans=tuple(planned),
        events_written=tuple(written),
        skipped_assumptions=tuple(dict.fromkeys(skipped)),
        errors=(*notes, *errors),  # rejected attempts, then plan failures
        attempts=attempts,
        input_digest=digest,
        reused=reused,
        plan_digest=pdigest,
    )


def tolerance_summary(claim: ResultClaim) -> str:
    tol, _, reason = pre_registered_tolerance(claim)
    return f"abs={tol.abs:g} ({reason})"
