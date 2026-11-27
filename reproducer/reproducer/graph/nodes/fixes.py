"""apply_known_fix node: the zero-token fix for a `known` smoke failure (rules/known_fixes.py).

Records the pending smoke iteration with its fix (smoke_result + known_fix evidence), folds the
fix into the in-state FixState and counts it (known_fixes_used <= MAX_KNOWN_FIXES, enforced
here as well as by route_smoke). add_package changes the env plan (route_fix -> env rebuild);
every other fix goes straight back to smoke. Never patches repo code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from reproducer.graph.nodes.smoke import record_iteration, summarize_iteration
from reproducer.graph.routing import MAX_KNOWN_FIXES
from reproducer.rules.known_fixes import apply_fix


def apply_known_fix_node(state: Mapping[str, Any], deps: Mapping[str, Any]) -> dict[str, Any]:
    used = state.get("known_fixes_used", 0)
    if used >= MAX_KNOWN_FIXES:  # route_smoke never sends us here at the cap
        raise RuntimeError(f"known-fix cap {MAX_KNOWN_FIXES} reached ({used} fixes)")
    step = state["pending"]
    action = step.action
    if action is None or step.decision.route != "known":
        raise RuntimeError("apply_known_fix without a pending known fix")
    it = replace(step.iteration, action=action)
    rec = record_iteration(deps["store"], Path(state["out_dir"]), state["smoke_setup"], it)
    summary = summarize_iteration(state["smoke_iterations"], it, rec)
    return {
        "fix_state": apply_fix(state["fix_state"], action),
        "known_fixes_used": used + 1,
        "last_fix_kind": action.kind,
        "pending": None,
        "smoke_log": [summary],
    }
