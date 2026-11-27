"""Render the store into Obsidian notes (the team's shared memory). The JSONL store stays the
source of truth; notes are DERIVED views, rebuilt deterministically from evidence, unknown
states and evaluate_claims output (same input -> byte-identical notes).

Notes (paths relative to the vault root):
  Papers/<paper>.md               repo + commit, env summary, claims table, unknowns, totals
  Claims/<paper>__<claim_id>.md   quote, tolerance, runs, inferred reasons, info, unknowns,
                                  human answers, full-run approval requests (when any)
  Runs/<run_id>.md                command, image, duration, GPU h, metrics, refs
  Unknowns/<paper>__<name>.md     origin, status, assumptions, resolution
  Lessons/<paper>__<claim>__t<k>.md  the Reflexion lesson of trial k (LLM-written, unverified):
                                  step, text, proposed change, support quote, cited evidence,
                                  what code did with it (lesson_applied), superseded lessons

Every note is YAML frontmatter (written by hand; strings escaped) followed by a managed block
between MANAGED_BEGIN and MANAGED_END. Wikilinks only point at notes rendered here.
merge_note() folds a rendered note into an existing file without touching user text.

Pure domain: no file, network or subprocess IO.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from reproducer.domain.approval import decision_for, human_resolution
from reproducer.domain.contracts import ResultClaim
from reproducer.domain.evaluate import (
    ClaimEvaluation,
    env_plan_for,
    is_aborted,
    run_plan_for,
    run_value,
)
from reproducer.domain.evidence import Evidence
from reproducer.domain.metric_format import (
    RAW_PREFIX,
    format_metric,
    format_raw,
    metric_decimals,
)
from reproducer.domain.telemetry import CallRecord, token_totals
from reproducer.domain.unknowns import (
    Assumption,
    UnknownOpened,
    UnknownResolved,
    UnknownState,
)

MANAGED_BEGIN = "<!-- reproducer:begin -->"
MANAGED_END = "<!-- reproducer:end -->"
GENERATED_BY = "reproducer"
SHORT = 12  # evidence id prefix length shown in notes

UnknownEventT = UnknownOpened | UnknownResolved | Assumption

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_REPO_RE = re.compile(r"^(?:repo:)?([^/@\s]+)/([^/@\s]+)@([0-9A-Za-z._-]+)$")
_PLAIN_YAML_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-/]*$")
_YAML_RESERVED = frozenset({"true", "false", "null", "yes", "no", "on", "off", "y", "n", "~"})
_FM_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):")

VERDICT_TEXT = {
    "within_tolerance": "within tolerance",
    "outside_tolerance": "outside tolerance",
    "no_completed_run": "no completed run",
}


# --- escaping ----------------------------------------------------------------------------------


def safe_name(text: str) -> str:
    """A file-name-safe form: runs of anything but [A-Za-z0-9._-] become '_'."""
    out = _SAFE_NAME_RE.sub("_", text).strip("._")
    return out or "_"


def yaml_scalar(value: Any) -> str:
    """One YAML scalar: null / bool / finite number plain; strings plain when unambiguous,
    else double-quoted with JSON escapes (a valid YAML double-quoted scalar)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else "null"
    text = str(value)
    if _PLAIN_YAML_RE.match(text) and text.lower() not in _YAML_RESERVED:
        return text
    return json.dumps(text, ensure_ascii=False)


def frontmatter(fields: Sequence[tuple[str, Any]]) -> str:
    lines = ["---", *(f"{k}: {yaml_scalar(v)}" for k, v in fields), "---"]
    return "\n".join(lines) + "\n"


def text(value: Any) -> str:
    """Untrusted text (paper quotes, LLM notes, repo lines) as one inert markdown line: no
    newlines, no wikilinks, no HTML comments (so it can never forge the managed markers)."""
    s = " ".join(str(value).split())
    return s.replace("[[", "\\[\\[").replace("]]", "\\]\\]").replace("<!--", "&lt;!--")


def cell(value: Any) -> str:
    """text() for a table cell: pipes escaped."""
    return text(value).replace("|", "\\|")


def code(value: Any) -> str:
    """Inline code span that survives backticks in the value."""
    s = " ".join(str(value).split())
    fence = "``" if "`" in s else "`"
    return f"{fence} {s} {fence}" if fence == "``" else f"`{s}`"


def short(evidence_id: str | None) -> str:
    return f"`{evidence_id[:SHORT]}`" if evidence_id else "-"


def ids(evidence_ids: Iterable[str]) -> str:
    return ", ".join(short(i) for i in evidence_ids)


def link(path: str, alias: str | None = None, *, in_table: bool = False) -> str:
    """[[path|alias]] to a rendered note (path without .md). In a table the alias pipe is
    escaped (Obsidian's table syntax)."""
    target = path[:-3] if path.endswith(".md") else path
    if alias is None:
        return f"[[{target}]]"
    sep = "\\|" if in_table else "|"
    return f"[[{target}{sep}{alias}]]"


# --- paths ---------------------------------------------------------------------------------------


def paper_path(paper: str) -> str:
    return f"Papers/{safe_name(paper)}.md"


def claim_path(paper: str, claim_id: str) -> str:
    return f"Claims/{safe_name(paper)}__{safe_name(claim_id)}.md"


def run_path(run_id: str) -> str:
    return f"Runs/{safe_name(run_id)}.md"


def unknown_path(paper: str, name: str) -> str:
    return f"Unknowns/{safe_name(paper)}__{safe_name(name)}.md"


def lesson_path(paper: str, claim_id: str, trial: int) -> str:
    return f"Lessons/{safe_name(paper)}__{safe_name(claim_id)}__t{trial}.md"


def lessons_by_trial(evidence: Mapping[str, Evidence], claim_id: str) -> dict[int, list[Evidence]]:
    """trial -> the claim's lesson evidence of that trial, in store order (the last one is the
    newest; earlier ones were written for other inputs: superseded / stale)."""
    out: dict[int, list[Evidence]] = {}
    for ev in evidence.values():
        t = ev.content.get("trial")
        if ev.kind == "lesson" and ev.content.get("claim_id") == claim_id and isinstance(t, int):
            out.setdefault(t, []).append(ev)
    return dict(sorted(out.items()))


def applied_of(evidence: Mapping[str, Evidence], lesson_id: str) -> Evidence | None:
    return next((ev for ev in evidence.values() if ev.kind == "lesson_applied"
                 and ev.content.get("lesson_id") == lesson_id), None)  # fmt: skip


def change_text(change: Mapping[str, Any] | None) -> str:
    if not change:
        return "none"
    val = "" if change.get("new_value") is None else f" -> {change['new_value']}"
    return f"[{change.get('kind')}] {change.get('target') or ''}{val}".rstrip()


# --- helpers -------------------------------------------------------------------------------------


def _note(fields: Sequence[tuple[str, Any]], body: Sequence[str]) -> str:
    block = "\n".join(body).strip("\n")
    return f"{frontmatter([*fields, ('generated_by', GENERATED_BY)])}\n" \
        f"{MANAGED_BEGIN}\n{block}\n{MANAGED_END}\n"  # fmt: skip


def _claims_from_evidence(evidence: Mapping[str, Evidence]) -> dict[str, ResultClaim]:
    out: dict[str, ResultClaim] = {}
    for ev in evidence.values():
        if ev.kind == "paper_claim":
            c = ResultClaim.model_validate({**ev.content, "location": ev.locator})
            out.setdefault(c.claim_id, c)
    return out


def paper_name(evidence: Mapping[str, Evidence]) -> str:
    """'paper:cutout' -> 'cutout' (the source of the first paper_claim)."""
    for ev in evidence.values():
        if ev.kind == "paper_claim":
            return ev.source.removeprefix("paper:")
    raise ValueError("no paper_claim evidence: configure with --spec first")


def repo_info(evidence: Mapping[str, Evidence]) -> tuple[str, str, str] | None:
    """(owner, name, commit) of the repo: the run_results' source, else the first repo-authored
    evidence whose source parses as owner/name@commit."""
    records = list(evidence.values())
    ordered = [ev for ev in records if ev.kind == "run_result"] + [
        ev for ev in records if ev.authority == "repo"
    ]
    for ev in ordered:
        m = _REPO_RE.match(ev.source)
        if m:
            return m[1], m[2], m[3]
    return None


def _latest(evidence: Mapping[str, Evidence], pred: Any) -> Evidence | None:
    return next((ev for ev in reversed(list(evidence.values())) if pred(ev)), None)


def _is_env_plan(ev: Evidence) -> bool:
    return ev.kind == "env_spec" and ev.locator == "env_plan"


def _preflight_for(run: Evidence, evidence: Mapping[str, Evidence]) -> Evidence | None:
    """The newest preflight stored before `run` with the run's digest."""
    digest = run.content.get("config_digest")
    found: Evidence | None = None
    for ev in evidence.values():
        if ev.evidence_id == run.evidence_id:
            break
        if ev.kind == "preflight" and ev.content.get("run_digest") == digest:
            found = ev
    return found


def _before(evidence: Mapping[str, Evidence], run: Evidence) -> list[Evidence]:
    records = list(evidence.values())
    for i, ev in enumerate(records):
        if ev.evidence_id == run.evidence_id:
            return records[:i]
    return records


def _run_status(run: Evidence, claim: ResultClaim | None) -> str:
    if is_aborted(run.content):
        return "aborted"
    if claim is not None and run_value(run.content, claim.metric) is not None:
        return "completed"
    return "failed"


def _finite(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _duration(seconds: Any) -> str:
    s = _finite(seconds)
    if s is None:
        return "-"
    h, rem = divmod(int(round(s)), 3600)
    return f"{s:.1f} s ({h}h {rem // 60:02d}m)"


def _unit(unit: str | None) -> str:
    return f" {text(unit)}" if unit else ""


def _tol_text(e: ClaimEvaluation) -> str:
    parts = [f"{k}={v}" for k, v in (("abs", e.tolerance.abs), ("rel", e.tolerance.rel))
             if v is not None]  # fmt: skip
    return " ".join(parts)


def _tol_bound(e: ClaimEvaluation) -> float:
    return max(e.tolerance.abs or 0.0, (e.tolerance.rel or 0.0) * abs(e.paper_value))


def _origin(opened: UnknownOpened, evidence: Mapping[str, Evidence]) -> str:
    ev = evidence.get(opened.evidence_id)
    return "paper" if ev is not None and ev.kind == "paper_setup" else "repo"


def _in_scope(state: UnknownState, claim_id: str) -> bool:
    """The ledger's rule: a global unknown (no claim_ids) applies to every claim."""
    return not state.claim_ids or claim_id in state.claim_ids


# --- renderer ----------------------------------------------------------------------------------


def render_vault(
    evidence: Mapping[str, Evidence],
    events: Sequence[UnknownEventT],
    unknowns: Mapping[str, UnknownState],
    evaluations: Mapping[str, ClaimEvaluation],
    *,
    call_records: Sequence[CallRecord] = (),
    run_files: Mapping[str, Sequence[str]] | None = None,
    plan_failures: Mapping[str, str] | None = None,
    max_trials: int = 3,
) -> dict[str, str]:
    """{relative_path: markdown} for the paper, its claims (in evaluations order), every
    run_result (file order) and every unknown. A run's artifact refs come from its "artifacts"
    content key; `run_files` (run_id -> out-dir-relative refs listed by the caller, this module
    does no IO) is the fallback for older run_results without one. `plan_failures` (claim_id
    -> why it has no current run plan) is shown on those claim notes. Sorted by path."""
    paper = paper_name(evidence)
    claims = _claims_from_evidence(evidence)
    missing = [cid for cid in evaluations if cid not in claims]
    if missing:
        raise ValueError(f"evaluations for claims without paper_claim evidence: {missing}")
    runs = [ev for ev in evidence.values() if ev.kind == "run_result"]
    opened: dict[str, UnknownOpened] = {}
    for e in events:
        if isinstance(e, UnknownOpened):
            opened.setdefault(e.unknown_id, e)
    files = run_files or {}

    ctx = _Ctx(paper, evidence, unknowns, opened, claims, evaluations, runs, plan_failures or {},
               max_trials)  # fmt: skip
    notes: dict[str, str] = {paper_path(paper): ctx.paper_note(call_records)}
    for cid in evaluations:
        notes[claim_path(paper, cid)] = ctx.claim_note(cid)
    for run in runs:
        notes[run_path(str(run.content.get("run_id", run.locator)))] = ctx.run_note(
            run, files.get(str(run.content.get("run_id", run.locator)), ())
        )
    for uid in sorted(unknowns, key=lambda u: unknowns[u].name):
        notes[unknown_path(paper, unknowns[uid].name)] = ctx.unknown_note(uid)
    for cid in evaluations:
        for trial, evs in lessons_by_trial(evidence, cid).items():
            notes[lesson_path(paper, cid, trial)] = ctx.lesson_note(cid, trial, evs)
    return dict(sorted(notes.items()))


class _Ctx:
    def __init__(
        self,
        paper: str,
        evidence: Mapping[str, Evidence],
        unknowns: Mapping[str, UnknownState],
        opened: Mapping[str, UnknownOpened],
        claims: Mapping[str, ResultClaim],
        evaluations: Mapping[str, ClaimEvaluation],
        runs: Sequence[Evidence],
        plan_failures: Mapping[str, str],
        max_trials: int = 3,
    ) -> None:
        self.paper = paper
        self.evidence = evidence
        self.unknowns = unknowns
        self.opened = opened
        self.claims = claims
        self.evals = evaluations
        self.runs = runs
        self.plan_failures = plan_failures
        self.max_trials = max_trials  # REPRODUCER_MAX_TRIALS (read by the caller)

    # -- small pieces --

    def claim_link(self, cid: str, *, in_table: bool = False) -> str:
        if cid in self.evals:
            return link(claim_path(self.paper, cid), cid, in_table=in_table)
        return code(cid)

    def run_link(self, run: Evidence, *, in_table: bool = False) -> str:
        rid = str(run.content.get("run_id", run.locator))
        return link(run_path(rid), rid, in_table=in_table)

    def unknown_link(self, uid: str, *, in_table: bool = False) -> str:
        name = self.unknowns[uid].name
        return link(unknown_path(self.paper, name), name, in_table=in_table)

    def fmt(self, cid: str, value: float | None) -> str:
        claim = self.claims.get(cid)
        return format_metric(value, metric_decimals(claim) if claim else 6)

    def runs_of(self, cid: str) -> list[Evidence]:
        return [r for r in self.runs if r.content.get("claim_id") == cid]

    def run_value_text(self, run: Evidence) -> str:
        claim = self.claims.get(str(run.content.get("claim_id")))
        if claim is None:
            return "-"
        v = run_value(run.content, claim.metric)
        unit = _unit(claim.unit) if v is not None else ""
        return self.fmt(claim.claim_id, v) + unit

    # -- notes --

    def paper_note(self, call_records: Sequence[CallRecord]) -> str:
        repo = repo_info(self.evidence)
        env = _latest(self.evidence, _is_env_plan)
        pre = _latest(self.evidence, lambda ev: ev.kind == "preflight")
        statuses = [e.status for e in self.evals.values()]
        verified = statuses.count("verified")
        gpu = sum(_finite(r.content.get("gpu_hours")) or 0.0 for r in self.runs)
        tok = token_totals(call_records)

        body = [f"# {text(self.paper)}", ""]
        if repo is not None and repo[0] == "local":
            body.append(f"- Repository: local snapshot {code(repo[1])} (not on GitHub)")
            body.append(f"- Commit: `{repo[2]}`")
        elif repo is not None:
            owner, name, sha = repo
            url = f"https://github.com/{owner}/{name}/tree/{sha}"
            body.append(f"- Repository: [{text(owner)}/{text(name)}]({url})")
            body.append(f"- Commit: `{sha}`")
        else:
            body.append("- Repository: -")
        paper_src = next(ev.source for ev in self.evidence.values() if ev.kind == "paper_claim")
        body.append(f"- Paper source: {code(paper_src)}")

        body += ["", "## Environment", ""]
        plan = env.content.get("plan", {}) if env is not None else {}
        plan = plan if isinstance(plan, Mapping) else {}
        if pre is not None and pre.content.get("image"):
            body.append(f"- Image: {code(pre.content['image'])}")
        if plan.get("base_image"):
            body.append(f"- Base image: {code(plan['base_image'])}")
        versions = [
            f"{k} {plan[key]}"
            for k, key in (("python", "python"), ("torch", "torch"),
                           ("torchvision", "torchvision"), ("CUDA", "cuda"))
            if plan.get(key)
        ]  # fmt: skip
        if versions:
            body.append(f"- Versions: {text(', '.join(versions))}")
        if env is None:
            body.append("- No env plan stored.")
        else:
            body.append(f"- Env plan: {short(env.evidence_id)}")
            devs = [d for d in plan.get("deviations", ()) if isinstance(d, Mapping)]
            body.append("- Deviations:" if devs else "- Deviations: none")
            for d in devs:
                why = f" ({text(d['reason'])})" if d.get("reason") else ""
                body.append(f"  - {text(d.get('field'))} {text(d.get('stated'))} -> "
                            f"{text(d.get('chosen'))}{why}")  # fmt: skip
            assumed = [
                a for a in plan.get("assumptions", ())
                if isinstance(a, Mapping) and a.get("value") is not None
            ]  # fmt: skip
            if assumed:
                body.append("- Assumptions:")
                for a in assumed:
                    body.append(f"  - {text(a.get('name'))} = {text(a.get('value'))} "
                                f"({text(a.get('reason'))})")  # fmt: skip

        body += [
            "", "## Claims", "",
            "| Claim | Paper ± tol | Ours | \\|diff\\| | Verdict | Status | Fidelity |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]  # fmt: skip
        for cid, e in self.evals.items():
            unit = _unit(e.unit).replace("|", "\\|")
            ours = "-" if e.our_value is None else f"{self.fmt(cid, e.our_value)}{unit}"
            diff = "-" if e.abs_diff is None else self.fmt(cid, e.abs_diff)
            body.append(
                f"| {self.claim_link(cid, in_table=True)} "
                f"| {self.fmt(cid, e.paper_value)}{unit} ± {self.fmt(cid, _tol_bound(e))} "
                f"| {ours} | {diff} | {VERDICT_TEXT[e.verdict]} | {e.status} "
                f"| {e.fidelity or '-'} |"
            )

        body += ["", "## Unknowns", ""]
        if self.unknowns:
            body += ["| Unknown | Origin | Status | Value |", "| --- | --- | --- | --- |"]
            for uid in sorted(self.unknowns, key=lambda u: self.unknowns[u].name):
                st = self.unknowns[uid]
                origin = _origin(self.opened[uid], self.evidence) if uid in self.opened else "-"
                body.append(f"| {self.unknown_link(uid, in_table=True)} | {origin} | {st.status} "
                            f"| {cell(self._unknown_value(st))} |")  # fmt: skip
        else:
            body.append("None.")

        body += ["", "## Runs", ""]
        for r in self.runs:
            cid = str(r.content.get("claim_id"))
            body.append(f"- {self.run_link(r)}: {self.claim_link(cid)}, "
                        f"{_run_status(r, self.claims.get(cid))}, "
                        f"{self.run_value_text(r)}")  # fmt: skip
        if not self.runs:
            body.append("No runs yet.")

        by_status = ", ".join(f"{s} {statuses.count(s)}" for s in sorted(set(statuses)))
        body += [
            "", "## Totals", "",
            f"- Claims: {len(statuses)} ({by_status or '-'})",
            f"- GPU hours: {gpu:.3f} ({len(self.runs)} run_result(s))",
            f"- Tokens: api={tok['api']} local={tok['local']} calls={tok['calls']}",
        ]  # fmt: skip
        fields = [
            ("type", "paper"),
            ("paper", self.paper),
            ("repo", f"{repo[0]}/{repo[1]}" if repo else None),
            ("commit", repo[2] if repo else None),
            ("claims_total", len(statuses)),
            ("claims_verified", verified),
            ("gpu_hours", round(gpu, 4)),
        ]
        return _note(fields, body)

    def _unknown_value(self, st: UnknownState) -> str:
        if st.status == "resolved":
            return str(st.value)
        values = list(dict.fromkeys(a.value for a in st.assumptions))
        return f"assumed {', '.join(values)}" if values else "-"

    def claim_note(self, cid: str) -> str:
        e = self.evals[cid]
        claim = self.claims[cid]
        unit = _unit(e.unit)
        f = lambda v: self.fmt(cid, v)  # noqa: E731
        bound = _tol_bound(e)
        body = [f"# {text(cid)}", "", f"Paper: {link(paper_path(self.paper), self.paper)}", ""]
        body += [f"> {text(claim.quote)}", "", f"- Location: {text(claim.location)}"]
        body.append(f"- Metric: {code(claim.metric)}{unit}; dataset {text(claim.dataset)}"
                    f", split {text(claim.split or '-')}")  # fmt: skip
        body.append(f"- Setting: {text(claim.setting or '-')}")

        body += ["", "## Result", ""]
        body.append(f"- Paper: {f(e.paper_value)}{unit} ± {f(bound)}")
        if e.our_value is None:
            body.append("- Ours: - (no completed run)")
        else:
            raw = ", ".join(f"{text(k.removeprefix(RAW_PREFIX))}={format_raw(v)}"
                            for k, v in e.our_raw.items()) or "-"  # fmt: skip
            body.append(f"- Ours: {f(e.our_value)}{unit} (raw {raw}; "
                        f"run_result {short(e.best_run_evidence_id)})")  # fmt: skip
        if e.abs_diff is not None:
            cmp = "<=" if e.within_tolerance else ">"
            body.append(f"- |diff|: {f(e.abs_diff)} {cmp} {f(bound)}")
        body.append(f"- Verdict: {VERDICT_TEXT[e.verdict]}")
        body.append(f"- Status: {e.status}")
        plan_failed = self.plan_failures.get(cid)
        if plan_failed is not None:
            body.append(f"- Plan: plan_failed: {text(plan_failed)} (no current run plan)")
        fidelity = e.fidelity or f"- (best run: {e.run_fidelity or 'no completed run'})"
        body.append(f"- Fidelity: {fidelity}")

        body += ["", "## Tolerance", ""]
        body.append(f"- {_tol_text(e)} (bound ± {f(bound)}), pre-registered "
                    f"{short(e.tolerance_evidence_id)}")  # fmt: skip
        body.append(f"- Rule: {text(e.tolerance_reason or '-')}")

        body += ["", "## Runs", ""]
        runs = self.runs_of(cid)
        body.append(f"{e.completed_runs} completed / {e.attempts} attempt(s) "
                    f"({e.aborted_runs} aborted), {e.gpu_hours:.3f} GPU h")  # fmt: skip
        body.append("")
        for r in runs:
            best = " (best)" if r.evidence_id == e.best_run_evidence_id else ""
            body.append(f"- {self.run_link(r)}{best}: {_run_status(r, claim)}, "
                        f"{self.run_value_text(r)}, exit {r.content.get('exit_code')}")  # fmt: skip
        if not runs:
            body.append("- No runs yet.")

        body += ["", "## Inferred because", ""]
        for reason in e.inferred_reasons:
            refs = f" [{ids(reason.evidence_ids)}]" if reason.evidence_ids else ""
            body.append(f"- [{reason.kind}] {text(reason.text)}{refs}")
        if not e.inferred_reasons:
            body.append("- Nothing recorded.")
        if e.info:
            body += ["", "## Info", ""]
            for reason in e.info:
                refs = f" [{ids(reason.evidence_ids)}]" if reason.evidence_ids else ""
                body.append(f"- {text(reason.text)}{refs}")

        body += ["", "## Unknowns", ""]
        used = self._best_run_inputs(e)
        scoped = sorted((u for u, st in self.unknowns.items() if _in_scope(st, cid)),
                        key=lambda u: self.unknowns[u].name)  # fmt: skip
        for uid in scoped:
            st = self.unknowns[uid]
            assumed = [a.value for a in st.assumptions if a.evidence_id in used]
            note = f", best run assumed {text(assumed[0])}" if assumed else ""
            if st.status == "resolved":
                note = f": {text(st.value)}"
                human = human_resolution(self.evidence, st)
                if human is not None:
                    note += f" (human answer: {text(human.content.get('note') or '-')})"
            body.append(f"- {self.unknown_link(uid)}: {st.status}{note}")
        if not scoped:
            body.append("- None.")

        body += self.lessons_section(cid, e)

        approvals = self.approvals_of(cid)
        if approvals:
            body += ["", "## Approvals", ""]
            body += approvals

        fields = [
            ("type", "claim"),
            ("paper", self.paper),
            ("claim_id", cid),
            ("metric", e.metric),
            ("status", e.status),
            ("fidelity", e.fidelity),
            ("verdict", e.verdict),
        ]
        if plan_failed is not None:
            fields.append(("plan_failed", plan_failed))
        return _note(fields, body)

    def lessons_section(self, cid: str, e: ClaimEvaluation) -> list[str]:
        """The claim's Reflexion lessons (newest per trial) and, for an open claim with runs,
        why it is still open."""
        trials = lessons_by_trial(self.evidence, cid)
        if not trials and (e.status == "verified" or not e.attempts):
            return []  # nothing to say (notes of stores without lessons stay unchanged)
        out = ["", "## Lessons", ""]
        for trial, evs in trials.items():
            les = evs[-1].content.get("lesson", {})
            applied = applied_of(self.evidence, evs[-1].evidence_id)
            what = f"; {applied.content.get('outcome')}" if applied is not None else ""
            out.append(f"- {link(lesson_path(self.paper, cid, trial), f'trial {trial}')}: "
                       f"{text(les.get('step'))}, change "
                       f"{text(change_text(les.get('proposed_change')))}{what}")  # fmt: skip
        if not trials:
            out.append("- None.")
        if e.status != "verified" and e.attempts:
            best = short(e.best_run_evidence_id) if e.best_run_evidence_id else "-"
            out += ["", f"Open after {e.attempts} trial(s): best run {best}, reason: "
                        f"{text(self.open_reason(cid, e, trials))}"]  # fmt: skip
        return out

    def open_reason(
        self, cid: str, e: ClaimEvaluation, trials: Mapping[int, Sequence[Evidence]]
    ) -> str:
        last = trials[max(trials)][-1] if trials else None
        applied = applied_of(self.evidence, last.evidence_id) if last is not None else None
        if applied is not None and applied.content.get("outcome") == "stopped_no_change":
            return "reflexion stopped: no grounded change"
        if e.attempts >= self.max_trials:
            return f"trial budget used ({e.attempts}/{self.max_trials})"
        if e.verdict == "outside_tolerance":
            return "outside tolerance" + (
                f"; last lesson (trial {max(trials)}) {applied.content.get('outcome')}"
                if applied is not None else ""
            )  # fmt: skip
        return VERDICT_TEXT[e.verdict]

    def lesson_note(self, cid: str, trial: int, evs: Sequence[Evidence]) -> str:
        ev = evs[-1]
        c = ev.content
        les: Mapping[str, Any] = c.get("lesson", {})
        ch: Mapping[str, Any] = les.get("proposed_change") or {}
        body = [f"# {text(cid)}: lesson of trial {trial}", "",
                f"Claim: {self.claim_link(cid)} (paper {link(paper_path(self.paper), self.paper)})",
                "", "> LLM-written, unverified: a proposal that code validated "
                    "(grounding, step order, allowed change) and may apply; not a finding.",
                "", f"- Step: {text(les.get('step'))}",
                f"- Lesson: {text(les.get('text'))}",
                f"- Change: {text(change_text(ch or None))}"]  # fmt: skip
        if ch.get("rationale"):
            body.append(f"- Rationale: {text(ch['rationale'])}")
        sup = ch.get("support") or {}
        if sup:
            src = self.evidence.get(str(sup.get("evidence_id")))
            who = f"{src.kind} ({src.authority})" if src is not None else "evidence"
            body.append(f"- Support: {who} {short(str(sup.get('evidence_id')))}: "
                        f"\"{text(sup.get('quote'))}\"")  # fmt: skip
        body.append(f"- Cited evidence: {ids(les.get('evidence_ids') or ())}")
        run_id, digest = str(c.get("run_evidence_id")), str(c.get("input_digest"))
        body.append(f"- About run_result {short(run_id)}; lesson {short(ev.evidence_id)}; "
                    f"inputs {short(digest)}")  # fmt: skip
        applied = applied_of(self.evidence, ev.evidence_id)
        body += ["", "## Applied", ""]
        if applied is None:
            body.append("- Not applied.")
        else:
            body.append(f"- {text(applied.content.get('outcome'))}: "
                        f"{text(applied.content.get('detail'))} "
                        f"(lesson_applied {short(applied.evidence_id)})")  # fmt: skip
        if len(evs) > 1:
            body += ["", "## Superseded (stale: written for other inputs, never applied)", ""]
            for old in evs[:-1]:
                body.append(f"- {short(old.evidence_id)}: "
                            f"{text(old.content.get('lesson', {}).get('text'))}")  # fmt: skip
        fields = [
            ("type", "lesson"),
            ("paper", self.paper),
            ("claim_id", cid),
            ("trial", trial),
            ("step", les.get("step")),
            ("change", ch.get("kind") or "none"),
            ("label", "LLM-written"),
        ]
        return _note(fields, body)

    def approvals_of(self, cid: str) -> list[str]:
        """One line per full_run approval request of the claim: pending / approved / rejected."""
        out: list[str] = []
        for ev in self.evidence.values():
            c = ev.content
            if ev.kind != "approval_request" or c.get("subject") != cid:
                continue
            if c.get("action") != "full_run":
                continue
            dec = decision_for(self.evidence, ev.evidence_id)
            cost = ", ".join(f"{k} {v}" for k, v in (c.get("cost") or {}).items())
            if dec is None:
                state = "pending (awaiting approval; see `queue`)"
            else:
                note = dec.content.get("note") or "-"
                state = f"{dec.content.get('decision')} by human: {text(note)}"
            body = f"- full run [{short(ev.evidence_id)}] ({text(cost or '-')}): {state}"
            out.append(body)
        return out

    def _best_run_inputs(self, e: ClaimEvaluation) -> set[str]:
        """Evidence the best run used: its links, its run_plan and its env plan."""
        best = self.evidence.get(e.best_run_evidence_id) if e.best_run_evidence_id else None
        if best is None:
            return set()
        before = _before(self.evidence, best)
        out = set(best.links)
        for ev in (run_plan_for(best, self.evidence, before),
                   env_plan_for(best, self.evidence, before)):  # fmt: skip
            if ev is not None:
                out.add(ev.evidence_id)
        return out

    def run_note(self, run: Evidence, listed: Sequence[str]) -> str:
        c = run.content
        stored = c.get("artifacts")
        artifacts = [str(a) for a in stored] if isinstance(stored, list) else list(listed)
        rid = str(c.get("run_id", run.locator))
        cid = str(c.get("claim_id"))
        claim = self.claims.get(cid)
        before = _before(self.evidence, run)
        plan = run_plan_for(run, self.evidence, before)
        env = env_plan_for(run, self.evidence, before)
        pre = _preflight_for(run, self.evidence)
        status = _run_status(run, claim)
        e = self.evals.get(cid)

        body = [f"# {text(rid)}", "", f"Claim: {self.claim_link(cid)} "
                f"(paper {link(paper_path(self.paper), self.paper)})", ""]  # fmt: skip
        if e is not None and e.best_run_evidence_id == run.evidence_id:
            body += ["Best run for this claim.", ""]
        command = (plan.content.get("command") if plan is not None else None) or (
            pre.content.get("command") if pre is not None else None
        )
        body += ["## Command", ""]
        if command:
            body += ["```bash", " ".join(str(command).split()), "```"]
        else:
            body.append("-")

        body += ["", "## Execution", ""]
        if pre is not None and pre.content.get("image"):
            body.append(f"- Image: {code(pre.content['image'])}")
        env_plan = env.content.get("plan", {}) if env is not None else {}
        if isinstance(env_plan, Mapping) and env_plan.get("base_image"):
            body.append(f"- Base image: {code(env_plan['base_image'])}")
        body.append(f"- Status: {status} (exit {c.get('exit_code')})")
        if c.get("abort_reason"):
            body.append(f"- Abort reason: {text(c['abort_reason'])}")
        body.append(f"- Duration: {_duration(c.get('duration'))}")
        gpu = _finite(c.get("gpu_hours"))
        body.append(f"- GPU hours: {'-' if gpu is None else f'{gpu:.3f}'}")
        body.append(f"- Spec-faithful: {'yes' if c.get('spec_faithful') else 'no'}"
                    f"{' (assumptions used)' if c.get('assumptions_used') else ''}")  # fmt: skip
        if c.get("diverged"):
            body.append("- Diverged: yes (non-finite metric)")

        body += ["", "## Metrics", ""]
        metrics = c.get("metrics") if isinstance(c.get("metrics"), Mapping) else {}
        if claim is not None:
            body.append(f"- Converted {code(claim.metric)} (claim precision): "
                        f"{self.run_value_text(run)}")  # fmt: skip
        for k, v in metrics.items():
            if claim is None or k != claim.metric:
                kind = "Raw" if k.startswith(RAW_PREFIX) else "Other"
                body.append(f"- {kind} {code(k)}: {format_raw(_finite(v))}")
        if not metrics:
            body.append("- No metrics recorded.")

        body += ["", "## References", ""]
        body.append(f"- Log: {code(c['log_ref']) if c.get('log_ref') else '-'}")
        for a in artifacts:
            body.append(f"- Artifact: {code(a)}")
        body.append(f"- run_result: {short(run.evidence_id)}")
        body.append(f"- run_plan: {short(plan.evidence_id if plan else None)}")
        body.append(f"- env plan: {short(env.evidence_id if env else None)}")
        body.append(f"- preflight: {short(pre.evidence_id if pre else None)}")
        body.append(f"- config digest: {short(c.get('config_digest'))}")

        fields = [
            ("type", "run"),
            ("paper", self.paper),
            ("claim_id", cid),
            ("run_id", rid),
            ("status", status),
            ("spec_faithful", bool(c.get("spec_faithful")) and not c.get("assumptions_used")),
        ]
        return _note(fields, body)

    def unknown_note(self, uid: str) -> str:
        st = self.unknowns[uid]
        op = self.opened.get(uid)
        origin = _origin(op, self.evidence) if op is not None else "-"
        body = [f"# {text(st.name)}", "", f"Paper: {link(paper_path(self.paper), self.paper)}", ""]
        body += ["## Origin", ""]
        src = self.evidence.get(op.evidence_id) if op is not None else None
        if src is not None:
            where = f" ({text(src.locator)})" if src.locator.strip() else ""
            body.append(f"- Opened from {src.kind} {short(src.evidence_id)}{where}")
            if src.kind == "paper_setup" and src.content.get("quote"):
                body.append(f"- Paper: {text(src.content['quote'])}")
        body.append(f"- Origin: {origin}")

        body += ["", "## Status", "", f"- {st.status}"]
        if st.status == "resolved":
            res = self.evidence.get(st.resolved_by or "")
            how = f" by {res.kind} {short(res.evidence_id)}" if res is not None else ""
            body.append(f"- Resolution: {text(st.value)}{how}")
            human = human_resolution(self.evidence, st)
            if human is not None:
                body.append(f"- resolved by human: {text(human.content.get('note') or '-')}")

        body += ["", "## Assumptions", ""]
        for a in st.assumptions:
            ev = self.evidence.get(a.evidence_id)
            by = f"{ev.kind} {text(ev.locator)}" if ev is not None else "evidence"
            body.append(f"- {text(a.value)} (by {by} {short(a.evidence_id)})")
        if not st.assumptions:
            body.append("- None.")

        body += ["", "## Claims", ""]
        scoped = [cid for cid in self.evals if _in_scope(st, cid)]
        for cid in scoped:
            body.append(f"- {self.claim_link(cid)}")
        if not scoped:
            body.append("- None.")

        fields = [
            ("type", "unknown"),
            ("paper", self.paper),
            ("unknown", st.name),
            ("status", st.status),
            ("origin", origin),
        ]
        return _note(fields, body)


# --- merging into an existing note ---------------------------------------------------------


def split_frontmatter(doc: str) -> tuple[list[str] | None, str]:
    """(frontmatter lines without the --- fences, rest) or (None, doc) if there is none."""
    if not doc.startswith("---\n"):
        return None, doc
    end = doc.find("\n---\n", 3)
    if end == -1:
        if doc.endswith("\n---"):  # frontmatter only, no trailing newline
            inner = doc[4:-4]
            return (inner.split("\n") if inner else []), ""
        return None, doc
    inner = doc[4:end]
    return (inner.split("\n") if inner else []), doc[end + 5 :]


def _merge_frontmatter(old: list[str], new: list[str]) -> list[str]:
    """Our keys take the new values in place; user keys (and their continuation lines) are
    kept; our keys missing from `old` are appended in our order."""
    ours = {}
    for line in new:
        m = _FM_KEY_RE.match(line)
        if m:
            ours[m[1]] = line
    out: list[str] = []
    placed: set[str] = set()
    skipping = False
    for line in old:
        m = _FM_KEY_RE.match(line)
        if m:
            skipping = m[1] in ours
            if skipping and m[1] not in placed:
                out.append(ours[m[1]])
                placed.add(m[1])
            if not skipping:
                out.append(line)
        elif not skipping:
            out.append(line)
    out += [line for k, line in ours.items() if k not in placed]
    return out


def merge_note(existing: str | None, rendered: str) -> str:
    """The file content after writing `rendered` over `existing`: text outside the managed
    block is kept; the managed block is replaced (appended if markers are missing); our
    frontmatter keys are updated, user keys kept. Raises ValueError on a damaged block (begin
    without end, end before begin, or more than one block): user text is never guessed at."""
    if existing is None:
        return rendered
    new_fm, new_rest = split_frontmatter(rendered)
    old_fm, old_rest = split_frontmatter(existing)
    assert new_fm is not None
    b, e = new_rest.index(MANAGED_BEGIN), new_rest.index(MANAGED_END) + len(MANAGED_END)
    block = new_rest[b:e]

    nb, ne = old_rest.count(MANAGED_BEGIN), old_rest.count(MANAGED_END)
    if nb == 0 and ne == 0:
        head = old_rest.rstrip("\n")
        rest = f"{head}\n\n{block}\n" if head else f"\n{block}\n"
    elif nb == 1 and ne == 1 and old_rest.index(MANAGED_BEGIN) < old_rest.index(MANAGED_END):
        ob = old_rest.index(MANAGED_BEGIN)
        oe = old_rest.index(MANAGED_END) + len(MANAGED_END)
        rest = old_rest[:ob] + block + old_rest[oe:]
    else:
        raise ValueError(
            f"damaged managed block ({nb} begin / {ne} end markers, or end before begin); "
            "fix the markers by hand"
        )
    fm = _merge_frontmatter(old_fm, new_fm) if old_fm is not None else new_fm
    return "---\n" + "\n".join(fm) + "\n---\n" + rest
