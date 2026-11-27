"""Reflect node (M4-S2): the Reflexion LESSON of one trial (Lecture 3, multi-trial Reflexion).

It writes a lesson; it never applies a change (M4-S3 does). LLM node "reflect"
(llm_for_node("reflect"): REPRODUCER_LLM_REFLECT, else REPRODUCER_LLM).

  trigger   the claim's newest run (trial = its 1-based index among the claim's run_results)
            is out_of_tolerance or diverged (rules/classify_run, via diagnose) and the breaker
            has budget left for the claim (domain/breaker). `force` (demo / testing only) also
            reflects on another outcome; the caller never stores a forced lesson.
  prompt    compact observations, never raw logs: the claim (paper value, tolerance, metric
            direction), our value + diff + direction, the CurveSummary, the plan (rendered
            command, args with sources, unknowns it depends on + assumed values), the setup
            checks with their citations, the code-recorded deviations (env / plan / known fix /
            critic patches / open unknowns / protocol, domain/evaluate.inferred_reasons), the
            previous lessons of the claim (episodic memory: text only), the fixed step order
            and the change rules; plus the citable paper / repo / human texts.
  gate      gated_call(Lesson, max_retries=2) with the validation context of contracts.Lesson
            (step order, change kind vs step, grounded support, no seed fishing, ...)
  store     lesson evidence (authority llm, derived; links the cited ids), keyed by (claim,
            trial, input digest); input digest = sha256(system + prompt + aliases). A stored
            lesson with the same key is reused without an LLM call. Changed inputs (another
            digest for the same trial) produce a fresh lesson, stored as a new one; the old one
            stays in the store and reads as stale. Only current_lesson (same key) may ever be
            applied: a stale lesson never is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from reproducer.domain.breaker import BreakerState
from reproducer.domain.contracts import (
    CHANGE_RATIONALE_MAX_CHARS,
    EVAL_PROTOCOL_TARGETS,
    LESSON_MAX_WORDS,
    LESSON_STEPS,
    ArgOption,
    Conversion,
    Lesson,
)
from reproducer.domain.curve import CurveSummary
from reproducer.domain.evaluate import inferred_reasons
from reproducer.domain.evidence import Evidence, canonical_json, make_evidence
from reproducer.domain.grounding import CITATION_MAX_QUOTE_CHARS
from reproducer.domain.metric_format import format_metric, metric_decimals
from reproducer.domain.run_plan import allowed_conversions, metric_direction
from reproducer.domain.unknowns import UnknownState, fold_unknowns
from reproducer.graph.eviction import call_with_eviction, markdown_sections, shrink_prompt
from reproducer.graph.nodes.diagnose import ClaimDiagnosis, RunDiagnosis, curve_spec_for
from reproducer.graph.nodes.orchestrator import plan_target
from reproducer.providers.llm import LLM
from reproducer.providers.store import RunStore

REFLECT_NODE = "reflect"
TRIGGER_CLASSES = frozenset({"out_of_tolerance", "diverged"})
DEFAULT_MAX_TOKENS = 2_000
MAX_RETRIES = 2
ARGPARSE_LINES_MAX = 40
WHICH_VALUES = ("last", "best")

ReflectStatus = Literal["lesson", "skipped", "failed"]


# --------------------------------------------------------------------------- trigger


def newest_run(d: ClaimDiagnosis) -> tuple[int, RunDiagnosis] | None:
    """(trial, run) of the claim's newest run_result (store order); None without runs."""
    return (len(d.runs), d.runs[-1]) if d.runs else None


def skip_reason(d: ClaimDiagnosis, breaker: BreakerState, *, force: bool = False) -> str | None:
    """Why the claim gets no lesson now, else None: no run, the newest run's outcome is not
    out_of_tolerance / diverged (unless force), or no budget left for another trial."""
    hit = newest_run(d)
    if hit is None:
        return "no run_result"
    _, r = hit
    if r.outcome.cls not in TRIGGER_CLASSES and not force:
        return f"newest run is {r.outcome.cls} (reflect only on out_of_tolerance / diverged)"
    if not breaker.can_try(d.claim.claim_id):
        why = "; ".join(breaker.tripped) or "budget exhausted"
        return f"no trial budget left ({why}): the claim stays open"
    return None


# --------------------------------------------------------------------------- inputs


def all_lessons(evidence: Mapping[str, Evidence], claim_id: str) -> list[Evidence]:
    """Every lesson evidence of the claim (stale ones included), in store order."""
    return [ev for ev in evidence.values() if ev.kind == "lesson"
            and ev.content.get("claim_id") == claim_id
            and isinstance(ev.content.get("trial"), int)]  # fmt: skip


def lessons_for(evidence: Mapping[str, Evidence], claim_id: str) -> list[Evidence]:
    """The claim's lessons, one per trial (the newest stored for it), by trial. The episodic
    memory of later trials."""
    out: dict[int, Evidence] = {}
    for ev in all_lessons(evidence, claim_id):
        out[int(ev.content["trial"])] = ev
    return [out[t] for t in sorted(out)]


def _plan_of(run: Evidence, evidence: Mapping[str, Evidence]) -> Evidence | None:
    return next((evidence[i] for i in run.links
                 if i in evidence and evidence[i].kind == "run_plan"), None)  # fmt: skip


def _argparse_spec(evidence: Mapping[str, Evidence]) -> tuple[ArgOption, ...]:
    try:
        return plan_target(evidence).argparse_spec
    except (ValueError, KeyError):
        return ()  # fail closed: every arg change is rejected


def _code_evidence(evidence: Mapping[str, Evidence], source: str, path: str) -> Evidence | None:
    return next((ev for ev in evidence.values()
                 if ev.kind == "repo_file" and ev.source == source and ev.locator == path
                 and ev.confidence == "observed" and isinstance(ev.content.get("head"), str)),
                None)  # fmt: skip


def _train_config(evidence: Mapping[str, Evidence], source: str, path: str) -> Evidence | None:
    hits = [ev for ev in evidence.values() if ev.source == source
            and ev.locator == f"{path}#train_config" and ev.content.get("result")]  # fmt: skip
    return hits[-1] if hits else None


def _quote(text: str) -> str:
    return text[:CITATION_MAX_QUOTE_CHARS]


@dataclass(frozen=True)
class Citable:
    alias: str
    evidence: Evidence
    text: str  # what citations snap onto
    shown: str  # what the prompt shows


def citable_texts(
    evidence: Mapping[str, Evidence], plan_ev: Evidence | None, source: str
) -> list[Citable]:
    """Paper [P*] (setup + claim quotes), repo [R*] (the entrypoint's argparse lines and
    training config lines), human [H*] (answers to unknowns)."""
    out: list[Citable] = []
    paper = [
        ev
        for ev in evidence.values()
        if ev.kind in ("paper_setup", "paper_claim") and str(ev.content.get("quote") or "")
    ]
    for i, ev in enumerate(paper, start=1):
        q = str(ev.content["quote"])
        what = (f"setup {ev.content.get('name')} = {ev.content.get('value')}"
                if ev.kind == "paper_setup" else f"claim {ev.content.get('claim_id')}")  # fmt: skip
        out.append(Citable(f"P{i}", ev, q, f'{what}: "{_quote(q)}"'))
    body = plan_ev.content.get("plan") if plan_ev is not None else None
    entry = str(body.get("entrypoint")) if isinstance(body, Mapping) else ""
    n = 0
    code = _code_evidence(evidence, source, entry) if entry else None
    if code is not None:
        text = str(code.content["head"])
        lines = [ln.strip() for ln in text.splitlines() if "add_argument(" in ln]
        if lines:
            n += 1
            shown = "\n".join(f"  {_quote(ln)}" for ln in lines[:ARGPARSE_LINES_MAX])
            out.append(Citable(f"R{n}", code, text, f"{entry} code, argparse lines:\n{shown}"))
    tc = _train_config(evidence, source, entry) if entry else None
    if tc is not None:
        rows = [r for r in tc.content["result"] if isinstance(r, Mapping)]
        text = "\n".join(str(r.get("line")) for r in rows)
        n += 1
        shown = "\n".join(f"  {r.get('path')}:{r.get('line_no')} {_quote(str(r.get('line')))}"
                          for r in rows)  # fmt: skip
        out.append(Citable(f"R{n}", tc, text, f"{entry} training config lines:\n{shown}"))
    humans = [ev for ev in evidence.values() if ev.kind == "human_input"]
    for i, ev in enumerate(humans, start=1):
        c = ev.content
        text = f"{c.get('unknown')} = {c.get('value')}: {c.get('note')}"
        out.append(Citable(f"H{i}", ev, text, f'human answer: "{_quote(text)}"'))
    return out


def eval_protocol_table(
    plan_body: Mapping[str, Any], claim_metric: str, claim_unit: str | None, raw_name: str | None
) -> dict[str, frozenset[str]]:
    """eval_protocol target -> allowed new values, the current value excluded: which in
    last|best; conversion among those allowed from the repo metric to the claim's."""
    ms = plan_body.get("metric_source") if isinstance(plan_body, Mapping) else None
    ms = ms if isinstance(ms, Mapping) else {}
    which = frozenset(WHICH_VALUES) - {ms.get("which")}
    conv: frozenset[Conversion] = (
        allowed_conversions(raw_name, claim_metric, claim_unit) if raw_name else frozenset()
    )
    return {
        EVAL_PROTOCOL_TARGETS[0]: which,
        EVAL_PROTOCOL_TARGETS[1]: frozenset(conv) - {ms.get("conversion")},
    }


SYSTEM = f"""You are the Reflexion step of a paper-reproduction agent. A full training run of a
paper claim finished; code measured how far our value is from the paper's. Write ONE short
lesson: what most likely explains the gap, at the earliest step of the fixed order that the
observations implicate, and at most one grounded change to try next. You propose; code
validates and a later step applies it. Return ONLY JSON:
{{"claim_id": "...", "trial": <int>, "step": "<step>",
 "evidence_ids": ["RUN", "CURVE", ...],
 "text": "<the lesson, <= {LESSON_MAX_WORDS} words>",
 "proposed_change": null | {{"kind": "eval_protocol|arg|assumption|none", "target": "...",
   "new_value": "<string>" | null,
   "support": {{"evidence_id": "<alias>", "quote": "<copied exactly>"}} | null,
   "rationale": "<= {CHANGE_RATIONALE_MAX_CHARS} chars"}}}}
Step order (localize in this order; never go back before a previous lesson's step):
{" -> ".join(LESSON_STEPS)}
Rules:
- evidence_ids: aliases in [brackets]; include RUN or CURVE.
- Never state what the paper does unless you quote it. An open unknown (see the
  authoritative unknowns list) is not stated by the paper: call it unknown / assumed.
- Previous lessons are LLM-written and unverified: trust the observations and the
  unknowns list over them.
- Only facts from the observations. Never invent values; a value must be quoted from the
  citable evidence. Within tolerance or nothing grounded to change: kind "none", target
  "none", new_value null, support null.
- eval_protocol (step eval_protocol): target and new_value only from the allowed table; a
  conversion change quotes the repo code [R*] computing the printed value (its scale).
- arg (step config_vs_paper): target an argparse flag listed, new_value a valid value;
  support must quote PAPER text [P*] stating that value.
- assumption (any step): target an open unknown the plan depends on; support quotes paper
  [P*] or repo [R*] text that states new_value.
- Never: change the metric line, the claimed value, the tolerance or the claim's own flags;
  a seed without paper text stating it (no seed fishing); more than one change."""


def _fmt(v: float | None, dec: int) -> str:
    return "-" if v is None else format_metric(v, dec)


def _curve_lines(cv: CurveSummary | None, dec: int, partial: bool) -> list[str]:
    if cv is None:
        return ["(no curve recorded)"]
    f = lambda v: _fmt(v, dec)  # noqa: E731
    slope = "-" if cv.last_k_slope is None else f"{cv.last_k_slope:+.4f}/epoch"
    plateau = "-" if cv.plateau_from_epoch is None else f"epoch {cv.plateau_from_epoch}"
    return [
        f"{cv.n_points} points from {cv.source} ({cv.column} -> {cv.metric}, {cv.conversion})"
        + ("; PARTIAL: the run stopped early" if partial else ""),
        f"final {f(cv.final)} @ epoch {cv.final_epoch}; best ({cv.direction}) {f(cv.best)} "
        f"@ epoch {cv.best_epoch}",
        f"last {cv.k}: mean {f(cv.last_k_mean)} ± {f(cv.last_k_std)}, slope {slope}",
        f"still improving: {'yes' if cv.still_improving else 'no'}; plateau from: {plateau}; "
        f"non-finite points: {'yes' if cv.nonfinite else 'no'}",
    ]


def _unknown_line(st: UnknownState) -> str:
    if st.status == "resolved":
        return f"- {st.name}: resolved = {st.value}"
    values = list(dict.fromkeys(a.value for a in st.assumptions))
    assumed = f"assumed {', '.join(values)}" if values else "nothing assumed"
    return f"- {st.name}: open (not stated by the paper; {assumed})"


@dataclass(frozen=True)
class ReflectInputs:
    claim_id: str
    trial: int
    run: Evidence
    context: dict[str, Any]
    prompt: str
    input_digest: str


def build_reflect_inputs(
    d: ClaimDiagnosis,
    evidence: Mapping[str, Evidence],
    unknowns: Mapping[str, UnknownState],
    *,
    argparse_spec: Sequence[ArgOption] | None = None,
) -> ReflectInputs:
    """Validation context + prompt + input digest for the claim's newest run (pure). Raises
    ValueError without a run. argparse_spec: default from the stored repo evidence."""
    hit = newest_run(d)
    if hit is None:
        raise ValueError(f"{d.claim.claim_id}: no run_result to reflect on")
    trial, r = hit
    claim, e, o, run = d.claim, d.evaluation, r.outcome, r.run
    dec = metric_decimals(claim)
    unit = f" {claim.unit}" if claim.unit else ""
    plan_ev = _plan_of(run, evidence)
    body: Mapping[str, Any] = plan_ev.content.get("plan", {}) if plan_ev is not None else {}
    spec = tuple(argparse_spec) if argparse_spec is not None else _argparse_spec(evidence)
    try:
        direction = metric_direction(claim.metric)
        dir_text = f"{direction}: {'higher' if direction == 'max' else 'lower'} is better"
    except ValueError:
        dir_text = "unknown direction"

    aliases: dict[str, str] = {"RUN": run.evidence_id}
    if r.curve_evidence_id:
        aliases["CURVE"] = r.curve_evidence_id
    if plan_ev is not None:
        aliases["PLAN"] = plan_ev.evidence_id
    cites = citable_texts(evidence, plan_ev, run.source)
    for c in cites:
        aliases[c.alias] = c.evidence.evidence_id

    # plan
    args = [a for a in body.get("args", ()) if isinstance(a, Mapping)]
    claim_flags = {str(a.get("flag")) for a in args if a.get("source") == "claim"}
    notes = [n for n in (plan_ev.content.get("notes", ()) if plan_ev else ())
             if isinstance(n, Mapping)]  # fmt: skip
    claim_flags |= {str(n["field"])[len("claim:") :] for n in notes
                    if str(n.get("field", "")).startswith("claim:")}  # fmt: skip
    assumed = {str(a.get("unknown_id")): str(a.get("value"))
               for a in (plan_ev.content.get("assumptions", ()) if plan_ev else ())
               if isinstance(a, Mapping)}  # fmt: skip
    deps = [str(u) for u in body.get("depends_on_unknowns", ())]
    open_deps = frozenset(
        u for u in deps if (st := unknowns.get(u)) is not None and st.status == "open"
    )
    dep_lines = [
        f"- {u}: {(unknowns[u].status if u in unknowns else 'not opened')}; assumed "
        f"{assumed.get(u, '?')}"
        for u in deps
    ] or ["(none)"]
    ms = body.get("metric_source") if isinstance(body.get("metric_source"), Mapping) else {}
    arg_lines = [f"- {a.get('flag')} {a.get('value') if a.get('value') is not None else ''}"
                 .rstrip() + f"  [{a.get('source')}]" for a in args] or ["(none)"]  # fmt: skip
    check_lines = [
        f"- {n.get('status')} {n.get('field')}: {n.get('detail')}"
        for n in notes
        if n.get("status") in ("consistent", "deviation", "assumption", "open")
        and not str(n.get("field", "")).startswith("claim:")
    ] or ["(none)"]

    # deviations + fidelity reasons recorded by code
    reasons = inferred_reasons(claim, run, evidence, unknowns, e.completed_runs)
    reason_lines = [f"- [{x.kind}] {x.text}" for x in reasons] or ["(none)"]

    # previous lessons (episodic memory)
    prev = [ev for ev in lessons_for(evidence, claim.claim_id)
            if int(ev.content["trial"]) < trial]  # fmt: skip
    # lessons of THIS trial already applied (their change left the command unchanged: no new
    # run, so this trial is reflected on again with the changed inputs)
    applied_ids = {str(ev.content.get("lesson_id")) for ev in evidence.values()
                   if ev.kind == "lesson_applied"}  # fmt: skip
    same = [ev for ev in all_lessons(evidence, claim.claim_id)
            if ev.content["trial"] == trial and ev.evidence_id in applied_ids]  # fmt: skip
    prev += same
    prev_step = str(prev[-1].content["lesson"]["step"]) if prev else None
    open_names = frozenset(st.name for st in unknowns.values() if st.status == "open")
    unk_lines = [_unknown_line(st) for st in sorted(unknowns.values(), key=lambda st: st.name)]
    unk_lines = unk_lines or ["(none)"]
    prev_lines = [f"- trial {ev.content['trial']} ({ev.content['lesson']['step']}"
                  + ("; applied, the command did not change" if ev in same else "")
                  + f"): {ev.content['lesson']['text']}" for ev in prev] or ["(none)"]  # fmt: skip

    spec_raw = curve_spec_for(run, evidence)
    table = eval_protocol_table(body, claim.metric, claim.unit,
                                spec_raw.raw_name if spec_raw else None)  # fmt: skip
    table_lines = [f"- {t}: one of {sorted(v) or 'none (no other value allowed)'}"
                   for t, v in table.items()]  # fmt: skip
    flags = [f"- {'/'.join(a.option_strings)}"
             + (f" (choices {list(a.choices)})" if a.choices else "")
             + ("" if a.takes_value else f" ({a.action}: new_value null)")
             for a in spec if a.option_strings[0].startswith("-")]  # fmt: skip
    cur_curve = r.curve_evidence_id and evidence.get(r.curve_evidence_id)
    partial = bool(cur_curve and cur_curve.content.get("partial"))

    never = sorted(claim_flags) or "-"
    diff = "-" if o.diff is None else ("+" if o.diff >= 0 else "-") + _fmt(abs(o.diff), dec)
    prompt = "\n".join(
        [
            f"## Claim {claim.claim_id} (claim_id {claim.claim_id}, trial {trial})",
            f"paper: {claim.metric} = {_fmt(claim.value, dec)}{unit} on {claim.dataset} "
            f"({claim.split or '-'}); setting: {claim.setting or '-'}",
            f"tolerance: ± {_fmt(o.tol, dec)}{unit} (pre-registered); direction {dir_text}",
            f"\n## Our run, trial {trial} [RUN]",
            f"outcome: {o.cls}; ours {_fmt(o.value, dec)}{unit}; diff (ours - paper) {diff}; "
            f"vs paper: {o.direction or '-'}",
            f"code reason: {o.reason}",
            f"trials used: {e.attempts} of the claim's budget",
            "\n## Curve" + (" [CURVE]" if r.curve_evidence_id else ""),
            *_curve_lines(r.curve, dec, partial),
            "\n## Plan" + (" [PLAN]" if plan_ev is not None else ""),
            f"command: {plan_ev.content.get('command') if plan_ev else '-'}",
            "args [source]:",
            *arg_lines,
            f"metric source: which={ms.get('which')} conversion={ms.get('conversion')}",
            "unknowns the plan depends on:",
            *dep_lines,
            "\n## Setup checks and plan notes (paper vs repo, recorded by code)",
            *check_lines,
            "\n## Deviations and fidelity reasons (env / plan / known fix / critic patch, "
            "recorded by code)",
            *reason_lines,
            "\n## Unknowns (authoritative, from evidence: open = not stated by the paper)",
            *unk_lines,
            "\n## Previous lessons for this claim (episodic memory; LLM-written, unverified)",
            *prev_lines,
            f"previous step: {prev_step or '(none)'}",
            "\n## Citable evidence (support quotes one of these)",
            *[f"[{c.alias}] {c.shown}" for c in cites],
            "\n## Allowed changes",
            "eval_protocol table:",
            *table_lines,
            f"argparse flags (arg changes; never the claim's flags {never}):",
            *(flags or ["(none known: no arg change possible)"]),
            f"open unknowns the plan depends on (assumption targets): {sorted(open_deps) or '-'}",
            "\nReturn the JSON now.",
        ]
    )
    texts = {c.evidence.evidence_id: c.text for c in cites}
    subject = {run.evidence_id, *([r.curve_evidence_id] if r.curve_evidence_id else [])}
    known = {*texts, *subject, *([plan_ev.evidence_id] if plan_ev is not None else [])}
    authority = {i: evidence[i].authority for i in known if i in evidence}
    context: dict[str, Any] = {
        "claim_id": claim.claim_id,
        "trial": trial,
        "previous_step": prev_step,
        "subject_ids": frozenset(subject),
        "evidence_ids": frozenset(known),
        "texts": texts,
        "aliases": aliases,
        "evidence_authority": authority,
        "argparse_spec": spec,
        "claim_flags": frozenset(claim_flags),
        "open_plan_unknowns": open_deps,
        "eval_protocol_table": table,
        "claim_unit": claim.unit,
        "open_unknown_names": open_names,
    }
    payload = canonical_json({"system": SYSTEM, "prompt": prompt,
                              "aliases": dict(sorted(aliases.items()))})  # fmt: skip
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return ReflectInputs(claim.claim_id, trial, run, context, prompt, digest)


_PROMPT_TAIL = "Return the JSON now."
# eviction priority by section title prefix (lowest is dropped first; default 50)
_SECTION_PRIORITY = (
    ("Claim ", 100),
    ("Our run", 98),
    ("Allowed changes", 96),
    ("Citable evidence", 94),
    ("Plan", 90),
    ("Curve", 80),
    ("Unknowns", 70),
    ("Setup checks", 60),
    ("Deviations", 55),
    ("Previous lessons", 40),
)


def _section_priority(_: int, title: str) -> int:
    return next((p for prefix, p in _SECTION_PRIORITY if title.startswith(prefix)), 50)


def reflect_prompt_builder(prompt: str) -> Callable[[int | None], tuple[str, str, tuple]]:
    """Eviction builder: the prompt unchanged when it fits the budget (None: always); else
    its sections packed by _SECTION_PRIORITY (previous lessons go first)."""
    sections = markdown_sections(prompt.removesuffix(_PROMPT_TAIL).rstrip(), _section_priority)

    def build(budget: int | None) -> tuple[str, str, tuple]:
        text, evicted = shrink_prompt(prompt, sections, budget, tail=_PROMPT_TAIL)
        return SYSTEM, text, evicted

    return build


# --------------------------------------------------------------------------- evidence


def lesson_evidence(
    lesson: Lesson,
    inputs: ReflectInputs,
    evicted: Sequence[str] = (),
    gate_attempts: int | None = None,
) -> Evidence:
    """gate_attempts (additive, left out when None): tries of the gate that answered."""
    ch = lesson.proposed_change
    support = [ch.support.evidence_id] if ch is not None and ch.support is not None else []
    return make_evidence(
        kind="lesson",
        source=inputs.run.source,
        locator=f"lesson:{lesson.claim_id}:trial{lesson.trial}",
        content={
            "claim_id": lesson.claim_id,
            "trial": lesson.trial,
            "input_digest": inputs.input_digest,
            "run_evidence_id": inputs.run.evidence_id,
            "lesson": lesson.model_dump(mode="json"),
            **({"evicted": list(evicted)} if evicted else {}),
            **({"gate_attempts": gate_attempts} if gate_attempts is not None else {}),
        },
        confidence="derived",
        authority="llm",
        links=tuple(dict.fromkeys([*lesson.evidence_ids, *support])),
    )


def current_lesson(
    evidence: Mapping[str, Evidence], claim_id: str, trial: int, input_digest: str
) -> Evidence | None:
    """The newest lesson for (claim, trial, input digest): the only one that may be applied."""
    hits = [ev for ev in all_lessons(evidence, claim_id) if ev.content["trial"] == trial
            and ev.content.get("input_digest") == input_digest]  # fmt: skip
    return hits[-1] if hits else None


def stale_lessons(
    evidence: Mapping[str, Evidence], claim_id: str, trial: int, input_digest: str
) -> list[Evidence]:
    """Lessons of (claim, trial) written for other inputs: kept in the store, never applied."""
    return [ev for ev in all_lessons(evidence, claim_id) if ev.content["trial"] == trial
            and ev.content.get("input_digest") != input_digest]  # fmt: skip


# --------------------------------------------------------------------------- node


@dataclass(frozen=True)
class ReflectOutcome:
    claim_id: str
    status: ReflectStatus
    reason: str  # why skipped / failed; "" for a lesson
    trial: int | None = None
    lesson: Lesson | None = None
    evidence: Evidence | None = None  # the stored (or, in a dry run, would-be) lesson evidence
    stored: bool = False  # appended now
    reused: bool = False  # the stored lesson for (claim, trial, input digest): no LLM call
    stale_ids: tuple[str, ...] = ()  # lessons of this trial for other inputs (never applied)
    attempts: int = 0
    errors: tuple[str, ...] = ()
    input_digest: str = ""
    forced: bool = False


def reflect_claim(
    store: RunStore,
    d: ClaimDiagnosis,
    breaker: BreakerState,
    *,
    llm: LLM | None = None,
    llm_factory: Callable[[], LLM] | None = None,
    force: bool = False,
    dry_run: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    argparse_spec: Sequence[ArgOption] | None = None,
) -> ReflectOutcome:
    """The node for one claim (see the module docstring). Appends at most one lesson evidence;
    nothing when dry_run, and a forced lesson (force on a non-triggering outcome) is never
    stored (ValueError unless dry_run). The LLM is `llm`, else made by `llm_factory` only when
    a call is needed. BudgetExceeded / LLMError propagate."""
    cid = d.claim.claim_id
    why = skip_reason(d, breaker, force=force)
    if why is not None:
        return ReflectOutcome(cid, "skipped", why)
    forced = skip_reason(d, breaker) is not None
    if forced and not dry_run:
        raise ValueError("a forced lesson is never stored: use dry_run with force")
    evidence = store.load_evidence()
    unknowns = fold_unknowns(store.load_events(), evidence)
    inputs = build_reflect_inputs(d, evidence, unknowns, argparse_spec=argparse_spec)

    old = stale_lessons(evidence, cid, inputs.trial, inputs.input_digest)
    stale = tuple(ev.evidence_id for ev in old)
    hit = current_lesson(evidence, cid, inputs.trial, inputs.input_digest)
    if hit is not None:  # same (claim, trial, inputs): re-validated, no LLM call
        try:
            lesson = Lesson.model_validate(hit.content["lesson"], context=inputs.context)
        except ValidationError:
            lesson = None  # should not happen for the same inputs: ask the LLM again
        if lesson is not None:
            return ReflectOutcome(
                cid, "lesson", "", inputs.trial, lesson, hit, reused=True, stale_ids=stale,
                input_digest=inputs.input_digest, forced=forced,
            )  # fmt: skip

    if llm is None:
        if llm_factory is None:
            raise ValueError("reflect_claim: an LLM call is needed: pass llm or llm_factory")
        llm = llm_factory()
    call = call_with_eviction(
        llm,
        reflect_prompt_builder(inputs.prompt),
        schema=Lesson,
        context=inputs.context,
        max_tokens=max_tokens,
        max_retries=MAX_RETRIES,
        required_context=("evidence_ids", "subject_ids", "open_plan_unknowns", "claim_flags"),
    )
    result = call.result
    if not result.ok or result.value is None:
        return ReflectOutcome(
            cid, "failed", f"no valid lesson after {result.attempts} attempt(s)", inputs.trial,
            attempts=result.attempts, errors=result.errors, input_digest=inputs.input_digest,
            stale_ids=stale, forced=forced,
        )  # fmt: skip
    ev = lesson_evidence(result.value, inputs, call.evicted, result.attempts)
    stored = False
    if not dry_run:
        store.append_evidence(ev)
        stored = True
    return ReflectOutcome(
        cid, "lesson", "", inputs.trial, result.value, ev, stored=stored, stale_ids=stale,
        attempts=result.attempts, errors=result.errors, input_digest=inputs.input_digest,
        forced=forced,
    )  # fmt: skip
