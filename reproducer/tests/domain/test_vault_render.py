"""Tests for reproducer.domain.vault_render: Cutout snapshot, determinism, wikilinks, escaping,
frontmatter and merge_note (managed block)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml
from vault_fixture import C10, C100, cutout_store, run_id

from reproducer.domain.evaluate import evaluate_claims
from reproducer.domain.unknowns import fold_unknowns
from reproducer.domain.vault_render import (
    MANAGED_BEGIN,
    MANAGED_END,
    _claims_from_evidence,
    cell,
    merge_note,
    render_vault,
    safe_name,
    split_frontmatter,
    text,
    yaml_scalar,
)

SNAPSHOT = Path(__file__).resolve().parents[1] / "fixtures" / "vault_cutout"
LINK_RE = re.compile(r"\[\[([^\]|\\]+)(?:\\?\|[^\]]*)?\]\]")


def _render() -> dict[str, str]:
    evidence, events, calls = cutout_store()
    unknowns = fold_unknowns(events, evidence)
    claims = tuple(_claims_from_evidence(evidence).values())
    evals = evaluate_claims(claims, evidence, unknowns)
    files = {run_id(C10): [f"runs/{run_id(C10)}/artifacts/logs/cifar10_resnet18.csv"]}
    # C100's current plan failed (e.g. after a re-plan); its older run stays on the note
    return render_vault(evidence, events, unknowns, evals, call_records=calls, run_files=files,
                        plan_failures={C100: PLAN_FAILED})  # fmt: skip


PLAN_FAILED = "metric line 'test_acc: %.3f' not found in train.py"


@pytest.fixture(scope="module")
def notes() -> dict[str, str]:
    return _render()


def test_cutout_snapshot(notes: dict[str, str]) -> None:
    """Golden files in tests/fixtures/vault_cutout; REPRODUCER_UPDATE_SNAPSHOTS=1 rewrites."""
    if os.environ.get("REPRODUCER_UPDATE_SNAPSHOTS") == "1":
        for rel, body in notes.items():
            (SNAPSHOT / rel).parent.mkdir(parents=True, exist_ok=True)
            (SNAPSHOT / rel).write_text(body, encoding="utf-8")
    expected = {
        p.relative_to(SNAPSHOT).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(SNAPSHOT.rglob("*.md"))
    }
    assert sorted(notes) == sorted(expected)
    for rel in notes:
        assert notes[rel] == expected[rel], rel


def test_expected_notes(notes: dict[str, str]) -> None:
    assert sorted(notes) == [
        f"Claims/cutout__{C100}.md",
        f"Claims/cutout__{C10}.md",
        "Papers/cutout.md",
        f"Runs/{run_id(C100)}.md",
        f"Runs/{run_id(C10)}.md",
        "Unknowns/cutout__cuda_version.md",
        "Unknowns/cutout__dataset_source.md",
        "Unknowns/cutout__eval_checkpoint.md",
        "Unknowns/cutout__python_version.md",
        "Unknowns/cutout__seed.md",
    ]


def test_deterministic() -> None:
    assert _render() == _render()


def test_links_resolve_to_generated_notes(notes: dict[str, str]) -> None:
    targets = {p[:-3] for p in notes}
    seen: set[str] = set()
    for rel, body in notes.items():
        for target in LINK_RE.findall(body):
            assert target in targets, f"{rel}: dangling [[{target}]]"
            seen.add(target)
    # every note but the paper is linked from somewhere; the paper from every other note
    assert seen == targets
    for rel, body in notes.items():
        if not rel.startswith("Papers/"):
            assert "[[Papers/cutout|cutout]]" in body, rel


def test_frontmatter_and_managed_block(notes: dict[str, str]) -> None:
    for rel, body in notes.items():
        fm, rest = split_frontmatter(body)
        assert fm is not None
        meta = yaml.safe_load("\n".join(fm))
        assert meta["generated_by"] == "reproducer" and meta["paper"] == "cutout"
        assert meta["type"] == rel.split("/")[0].lower().rstrip("s")
        assert rest.count(MANAGED_BEGIN) == 1 and rest.count(MANAGED_END) == 1
        assert rest.strip().startswith(MANAGED_BEGIN) and rest.strip().endswith(MANAGED_END)
    claim = yaml.safe_load("\n".join(split_frontmatter(notes[f"Claims/cutout__{C10}.md"])[0]))
    assert claim == {
        "type": "claim", "paper": "cutout", "claim_id": C10, "metric": "test_error",
        "status": "verified", "fidelity": "inferred", "verdict": "within_tolerance",
        "generated_by": "reproducer",
    }  # fmt: skip
    paper = yaml.safe_load("\n".join(split_frontmatter(notes["Papers/cutout.md"])[0]))
    assert paper["commit"] == "287f934ea5fa00d4345c2cccecf3552e2b1c33e3"  # quoted: a string
    run = yaml.safe_load("\n".join(split_frontmatter(notes[f"Runs/{run_id(C10)}.md"])[0]))
    assert (run["status"], run["spec_faithful"], run["claim_id"]) == ("completed", False, C10)
    unk = yaml.safe_load("\n".join(split_frontmatter(notes["Unknowns/cutout__seed.md"])[0]))
    assert (unk["status"], unk["origin"]) == ("open", "paper")
    ds_fm = split_frontmatter(notes["Unknowns/cutout__dataset_source.md"])[0]
    ds = yaml.safe_load("\n".join(ds_fm or []))
    assert (ds["status"], ds["origin"]) == ("resolved", "repo")


def test_paper_note_content(notes: dict[str, str]) -> None:
    p = notes["Papers/cutout.md"]
    assert "https://github.com/uoguelph-mlrg/Cutout/tree/287f934ea5fa" in p
    assert "- Image: `reproducer/cutout:424da3bbafa8`" in p
    assert "  - torch 0.4.0 -> 1.12.1 (no CUDA build for sm_89)" in p
    assert (f"| [[Claims/cutout__{C10}\\|{C10}]] | 3.99 % ± 0.26 | 4.10 % | 0.11 "
            "| within tolerance | verified | inferred |") in p  # fmt: skip
    assert "| 21.96 % ± 0.48 | 21.60 % | 0.36 |" in p
    assert "- Tokens: api=1000 local=450 calls=2" in p
    assert "- GPU hours: 3.300 (2 run_result(s))" in p


def test_claim_and_run_notes(notes: dict[str, str]) -> None:
    c = notes[f"Claims/cutout__{C10}.md"]
    assert "> ResNet18 + cutout 9.31 ± 0.18 3.99 ± 0.13" in c and "- Location: Table 1" in c
    assert "- Rule: 2 x paper std (± 0.13 after 3.99 in the claim quote)" in c
    assert "- [protocol] single run vs paper mean of 5" in c
    assert "- [[Unknowns/cutout__seed|seed]]: open, best run assumed 0" in c
    ds = "- [[Unknowns/cutout__dataset_source|dataset_source]]: resolved: torchvision:CIFAR10"
    assert ds in c
    r = notes[f"Runs/{run_id(C10)}.md"]
    assert "python train.py --dataset cifar10" in r and "Best run for this claim." in r
    assert "- Converted `test_error` (claim precision): 4.10 %" in r
    assert "- Raw `raw:test_acc`: 0.959" in r
    assert f"- Log: `runs/{run_id(C10)}/run.log`" in r
    assert "artifacts/logs/cifar10_resnet18.csv" in r
    c100 = notes[f"Runs/{run_id(C100)}.md"].split("## References")[1]
    assert "artifacts/logs/cifar100_resnet18.csv" in c100  # from the stored key


def test_stored_artifacts_win_over_listing() -> None:
    evidence, events, calls = cutout_store()
    unknowns = fold_unknowns(events, evidence)
    evals = evaluate_claims(tuple(_claims_from_evidence(evidence).values()), evidence, unknowns)
    files = {run_id(C10): ["runs/listed/c10.csv"], run_id(C100): ["runs/listed/c100.csv"]}
    notes = render_vault(evidence, events, unknowns, evals, run_files=files)
    assert "runs/listed/c10.csv" in notes[f"Runs/{run_id(C10)}.md"]  # older run: fallback
    c100 = notes[f"Runs/{run_id(C100)}.md"]
    assert "runs/listed/c100.csv" not in c100 and "cifar100_resnet18.csv" in c100


def test_unknown_notes(notes: dict[str, str]) -> None:
    seed = notes["Unknowns/cutout__seed.md"]
    assert "- Opened from paper_setup" in seed
    assert seed.count("- 0 (by run_plan run_plan:") == 2
    ds = notes["Unknowns/cutout__dataset_source.md"]
    assert "- Resolution: torchvision:CIFAR10 by dataset_cache" in ds


def test_no_paper_claim_raises() -> None:
    with pytest.raises(ValueError, match="no paper_claim"):
        render_vault({}, [], {}, {})


# --- escaping --------------------------------------------------------------------------------


def test_yaml_scalar_escapes() -> None:
    assert yaml_scalar("cutout") == "cutout"
    for value in ("a: b", "true", "123", "#x", 'q"uote', "line\nbreak", "", "- x", "null"):
        assert yaml.safe_load(f"k: {yaml_scalar(value)}") == {"k": value}
    assert yaml_scalar(None) == "null" and yaml_scalar(False) == "false"
    assert yaml_scalar(1.5) == "1.5" and yaml_scalar(float("nan")) == "null"


def test_untrusted_text_is_inert() -> None:
    evil = f"see [[Secret]] {MANAGED_END}\nnext | col"
    t = text(evil)
    assert "[[" not in t and MANAGED_END not in t and "\n" not in t
    assert "\\|" in cell(evil)
    assert safe_name("a/b c:..") == "a_b_c" and safe_name("..") == "_"


# --- merge_note ------------------------------------------------------------------------------

NEW = f"---\ntype: claim\nstatus: verified\n---\n\n{MANAGED_BEGIN}\nnew body\n{MANAGED_END}\n"


def test_merge_new_file() -> None:
    assert merge_note(None, NEW) == NEW


def test_merge_keeps_user_text_around_block() -> None:
    old = (f"---\ntype: claim\nstatus: open\ntags:\n  - mine\n---\nmy intro\n\n"
           f"{MANAGED_BEGIN}\nold body\n{MANAGED_END}\n\nmy notes after\n")  # fmt: skip
    out = merge_note(old, NEW)
    assert out == (f"---\ntype: claim\nstatus: verified\ntags:\n  - mine\n---\nmy intro\n\n"
                   f"{MANAGED_BEGIN}\nnew body\n{MANAGED_END}\n\nmy notes after\n")  # fmt: skip
    assert merge_note(out, NEW) == out  # idempotent


def test_merge_missing_markers_appends_block() -> None:
    old = "---\nauthor: me\n---\n# My own note\n\nhand-written\n"
    out = merge_note(old, NEW)
    assert out.startswith("---\nauthor: me\ntype: claim\nstatus: verified\n---\n")
    assert "# My own note\n\nhand-written\n\n" + MANAGED_BEGIN + "\nnew body\n" in out
    assert merge_note(out, NEW) == out
    plain = merge_note("just text\n", NEW)  # no frontmatter: ours is prepended
    assert plain.startswith("---\ntype: claim") and "just text\n\n" + MANAGED_BEGIN in plain


def test_merge_unchanged_is_identical() -> None:
    assert merge_note(NEW, NEW) == NEW


@pytest.mark.parametrize(
    "rest",
    [
        f"{MANAGED_BEGIN}\nno end\n",
        f"{MANAGED_END}\n{MANAGED_BEGIN}\n",
        f"{MANAGED_BEGIN}\na\n{MANAGED_END}\n{MANAGED_BEGIN}\nb\n{MANAGED_END}\n",
    ],
)
def test_merge_damaged_block_refuses(rest: str) -> None:
    with pytest.raises(ValueError, match="damaged managed block"):
        merge_note("---\ntype: claim\n---\n" + rest, NEW)


def test_c100_claim_note_present(notes: dict[str, str]) -> None:
    assert f"# {C100}" in notes[f"Claims/cutout__{C100}.md"]


def test_unplanned_claim_note_shows_plan_failed_reason(notes: dict[str, str]) -> None:
    c100 = notes[f"Claims/cutout__{C100}.md"]
    assert f"- Plan: plan_failed: {PLAN_FAILED} (no current run plan)" in c100
    assert f'plan_failed: "{PLAN_FAILED}"' in c100.split("---")[1]  # frontmatter
    c10 = notes[f"Claims/cutout__{C10}.md"]
    assert "plan_failed" not in c10
