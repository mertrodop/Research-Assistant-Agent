"""Known fixes for recognized failure patterns (zero tokens, never a repo code patch).

fix_for(classification, FixState) -> FixAction | None:
  missing_output_dir -> prep_dir   create the dir in the per-run copy (no deviation)
  oom                -> override   halve the batch argument (never below 1); with a gradient
                                   accumulation argument it is doubled (same effective batch,
                                   no deviation); without one a Deviation records the change
  missing_module     -> add_package  the pip package (stated version if the EnvSpec states one,
                                   else unpinned + assumption: env_rules.plan_environment);
                                   changes the env plan -> new image tag -> rebuild
  missing_data_path  -> add_mount  mount the dataset cache read-only at the scanned root
None = the action would not change FixState (already applied, or batch already 1): the loop
guard then treats the failure as unfixable.
A FixState is folded from known_fix evidence in the order it was written (fold_fix_state).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reproducer.domain.env_rules import Deviation
from reproducer.domain.evidence import Evidence, canonical_json, make_evidence
from reproducer.domain.smoke_plan import Override
from reproducer.rules.classify import Classification

FixKind = Literal["prep_dir", "override", "add_package", "add_mount"]
RULE_SOURCE = ("CLAUDE.md error rulebook: exit!=0 + known pattern -> code fix, zero tokens",)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataMountFix(_Frozen):
    name: str  # registry dataset name
    container: str


class FixState(_Frozen):
    prep_dirs: tuple[str, ...] = ()  # repo-relative dirs created in each per-run copy
    overrides: tuple[Override, ...] = ()  # applied to the smoke command (in place / appended)
    packages: tuple[str, ...] = ()  # extra pip packages for the env plan
    mounts: tuple[DataMountFix, ...] = ()  # extra read-only data mounts


class FixAction(_Frozen):
    kind: FixKind
    params: dict[str, Any] = Field(default_factory=dict)
    deviation: Deviation | None = None
    reason: str


def apply_fix(state: FixState, action: FixAction) -> FixState:
    p = action.params
    if action.kind == "prep_dir":
        return state.model_copy(update={"prep_dirs": (*state.prep_dirs, str(p["dir"]))})
    if action.kind == "override":
        overrides = state.overrides
        for raw in p["overrides"]:
            ov = Override.model_validate(raw)
            overrides = (*(o for o in overrides if o.option != ov.option), ov)
        return state.model_copy(update={"overrides": overrides})
    if action.kind == "add_package":
        return state.model_copy(update={"packages": (*state.packages, str(p["package"]))})
    mount = DataMountFix(name=str(p["name"]), container=str(p["container"]))
    return state.model_copy(update={"mounts": (*state.mounts, mount)})


def fix_for(c: Classification, state: FixState) -> FixAction | None:
    """The fix for a `known` classification, or None if it would be a no-op (loop guard)."""
    if c.cls != "known":
        raise ValueError(f"no known fix for a {c.cls!r} classification")
    p = c.params
    action: FixAction | None = None
    if c.pattern == "missing_output_dir":
        d = str(p["dir"])
        if d in state.prep_dirs:
            return None
        action = FixAction(kind="prep_dir", params={"dir": d}, reason=f"create {d}/ ({c.reason})")
    elif c.pattern == "oom":
        current = int(p["current"])
        new = max(1, current // 2)
        if new == current:
            return None
        ovs = [
            Override(
                dest=str(p["dest"]), option=str(p["option"]), value=str(new), previous=str(current)
            )
        ]
        deviation: Deviation | None = None
        if "accum_option" in p:
            acc = int(p["accum_current"])
            ovs.append(
                Override(
                    dest=str(p["accum_dest"]), option=str(p["accum_option"]),
                    value=str(acc * 2), previous=str(acc),
                )
            )  # fmt: skip
            reason = f"OOM: {p['option']} {current} -> {new}, accumulation {acc} -> {acc * 2}"
        else:
            deviation = Deviation(
                field=str(p["dest"]),
                stated=str(current),
                chosen=str(new),
                reason="CUDA out of memory: batch halved; the repo has no gradient-accumulation "
                "argument, so the effective batch size changes (no code patch)",
                rule_source=(*RULE_SOURCE, "OOM: halve batch + grad accum"),
            )
            reason = f"OOM: {p['option']} {current} -> {new} (effective batch size changed)"
        action = FixAction(
            kind="override",
            params={"overrides": [o.model_dump(mode="json") for o in ovs]},
            deviation=deviation,
            reason=reason,
        )
    elif c.pattern == "missing_module":
        pkg = str(p["package"])
        if pkg in state.packages:
            return None
        action = FixAction(
            kind="add_package", params={"package": pkg, "module": p.get("module")},
            reason=f"install {pkg} for module {p.get('module')!r} (rebuilds the image)",
        )  # fmt: skip
    elif c.pattern == "missing_data_path":
        mount = DataMountFix(name=str(p["name"]), container=str(p["container"]))
        if mount in state.mounts:
            return None
        action = FixAction(
            kind="add_mount", params=mount.model_dump(mode="json"),
            reason=f"mount the {mount.name} cache read-only at {mount.container}",
        )  # fmt: skip
    else:
        raise ValueError(f"no known fix for pattern {c.pattern!r}")
    if apply_fix(state, action) == state:
        return None
    return action


# --- evidence / fold / digest --------------------------------------------------------------


def known_fix_evidence(
    action: FixAction, c: Classification, *, source: str, smoke_evidence_id: str
) -> Evidence:
    """known_fix evidence: a deterministic rule decision (authority rule, derived), linked to
    the smoke evidence that triggered it."""
    return make_evidence(
        kind="known_fix",
        source=source,
        locator=f"known_fix:{smoke_evidence_id[:12]}",
        content={
            "action": action.model_dump(mode="json"),
            "pattern": c.pattern,
            "classification_reason": c.reason,
        },
        confidence="derived",
        authority="rule",
        links=(smoke_evidence_id,),
    )


def fold_fix_state(evidence: Iterable[Evidence]) -> FixState:
    """FixState from known_fix evidence, applied in the order given (the store's file order)."""
    state = FixState()
    for ev in evidence:
        if ev.kind == "known_fix":
            state = apply_fix(state, FixAction.model_validate(ev.content["action"]))
    return state


def smoke_input_digest(*, image: str, state: FixState, patches: Sequence[str], command: str) -> str:
    """sha256 over everything that determines a smoke run's inputs."""
    payload = {
        "image": image,
        "fix_state": state.model_dump(mode="json"),
        "patches": list(patches),
        "command": command,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def count_attempts(prior_digests: Iterable[str | None], digest: str) -> int:
    """Attempts = distinct input digests (legacy records without one are not counted)."""
    return len({d for d in prior_digests if d} | {digest})
