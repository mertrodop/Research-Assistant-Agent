"""CriticPatch validators (contracts.py) and the pure patching helpers (patching.py)."""

from __future__ import annotations

import ast
from typing import Any

import pytest
from pydantic import ValidationError

from reproducer.domain.contracts import CriticPatch, LineEdit
from reproducer.domain.patching import (
    apply_line_edits,
    diff_changed_lines,
    error_signature,
    normalize_message,
    split_lines,
    unified_diff,
    value_literals,
)

TRAIN = (
    "import torch\n"  # 1
    "lr = 0.1\n"  # 2
    "print('epoch 1 progress')\n"  # 3
    "loss = torch.tensor(1.5)\n"  # 4
    "print(loss.data[0])\n"  # 5
    "print('test_acc: %.3f' % (0.5))\n"  # 6
)


def _ctx(**kw: Any) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "repo_files": {"train.py", "helper.py"},
        "failing_file": "train.py",
        "file_text": TRAIN,
        "metric_lines": {"train.py": frozenset({6})},
    }
    ctx.update(kw)
    return ctx


def _patch(edits: list[dict[str, Any]], file: str = "train.py") -> dict[str, Any]:
    return {"file": file, "rationale": "0-dim tensor: use .item()", "edits": edits}


ITEM_FIX = {
    "start_line": 5,
    "end_line": 5,
    "original": "print(loss.data[0])",
    "replacement": "print(loss.item())",
}


def _validate(data: dict[str, Any], **ctx: Any) -> CriticPatch:
    return CriticPatch.model_validate(data, context=_ctx(**ctx))


def test_valid_item_fix_accepted() -> None:
    p = _validate(_patch([ITEM_FIX]))
    after = apply_line_edits(TRAIN, p.edits)
    assert "print(loss.item())\n" in after and "data[0]" not in after
    diff = unified_diff("train.py", TRAIN, after)
    assert diff.startswith("--- a/train.py\n+++ b/train.py\n")
    assert "-print(loss.data[0])\n+print(loss.item())\n" in diff
    assert diff_changed_lines(diff) == 2


def test_missing_context_fails_closed() -> None:
    with pytest.raises(ValidationError, match="requires validation context"):
        CriticPatch.model_validate(_patch([ITEM_FIX]), context={"repo_files": {"train.py"}})


def test_wrong_file_rejected() -> None:
    with pytest.raises(ValidationError, match="failing file 'train.py' only"):
        _validate(_patch([ITEM_FIX], file="helper.py"))


def test_file_not_in_repo_rejected() -> None:
    with pytest.raises(ValidationError, match="file not in repo"):
        _validate(_patch([ITEM_FIX], file="nope.py"))


def test_original_mismatch_rejected_with_actual_text() -> None:
    bad = {**ITEM_FIX, "original": "print(loss.data[1])"}
    with pytest.raises(ValidationError, match=r"they are exactly: 'print\(loss.data\[0\]\)'"):
        _validate(_patch([bad]))
    shifted = {**ITEM_FIX, "start_line": 4, "end_line": 4}
    with pytest.raises(ValidationError, match="does not match lines 4-4"):
        _validate(_patch([shifted]))


def test_original_with_trailing_newline_is_accepted() -> None:
    _validate(_patch([{**ITEM_FIX, "original": "print(loss.data[0])\n"}]))


def test_range_beyond_file_and_overlap_rejected() -> None:
    with pytest.raises(ValidationError, match="beyond end"):
        _validate(_patch([{**ITEM_FIX, "start_line": 9, "end_line": 9}]))
    a = {
        "start_line": 4,
        "end_line": 5,
        "original": "loss = torch.tensor(1.5)\nprint(loss.data[0])",
        "replacement": "loss = torch.tensor(1.5)\nprint(loss.item())",
    }
    with pytest.raises(ValidationError, match="overlap"):
        _validate(_patch([a, ITEM_FIX]))


def test_more_than_20_changed_lines_rejected() -> None:
    text = "".join(f"x{i} = f()\n" for i in range(30))
    orig = "\n".join(f"x{i} = f()" for i in range(11))
    repl = "\n".join(f"y{i} = f()" for i in range(11))
    edit = {"start_line": 1, "end_line": 11, "original": orig, "replacement": repl}
    with pytest.raises(ValidationError, match=r"changes 22 lines .* > 20"):
        CriticPatch.model_validate(
            _patch([edit]),
            context=_ctx(file_text=text, metric_lines={}),
        )
    ok = {**edit, "end_line": 10, "original": orig.rsplit("\n", 1)[0],
          "replacement": repl.rsplit("\n", 1)[0]}  # fmt: skip
    CriticPatch.model_validate(_patch([ok]), context=_ctx(file_text=text, metric_lines={}))


def test_metric_line_overlap_rejected() -> None:
    edit = {
        "start_line": 5,
        "end_line": 6,
        "original": "print(loss.data[0])\nprint('test_acc: %.3f' % (0.5))",
        "replacement": "print(loss.item())\nprint('test_acc: %.3f' % (0.5))",
    }
    with pytest.raises(ValidationError, match=r"metric line\(s\) \[6\]"):
        _validate(_patch([edit]))


def test_numeric_literal_change_rejected() -> None:
    edit = {"start_line": 2, "end_line": 2, "original": "lr = 0.1", "replacement": "lr = 0.05"}
    with pytest.raises(ValidationError, match=r"numeric literals changed \(removed \['0.1'\], "):
        _validate(_patch([edit]))
    same_value = {**edit, "replacement": "lr = float(0.1)"}  # value kept: allowed
    _validate(_patch([same_value]))


def test_added_numeric_literal_rejected() -> None:
    edit = {**ITEM_FIX, "replacement": "print(loss.item() * 2)"}
    with pytest.raises(ValidationError, match=r"added \['2'\]"):
        _validate(_patch([edit]))


def test_syntax_breaking_patch_rejected() -> None:
    edit = {**ITEM_FIX, "replacement": "print(loss.item()"}
    with pytest.raises(ValidationError, match="does not parse"):
        _validate(_patch([edit]))


def test_noop_and_too_many_edits_rejected() -> None:
    with pytest.raises(ValidationError, match="changes nothing"):
        _validate(_patch([{**ITEM_FIX, "replacement": ITEM_FIX["original"]}]))
    with pytest.raises(ValidationError, match="at most 3"):
        _validate(_patch([ITEM_FIX] * 4))


def test_rationale_capped_at_300_chars() -> None:
    with pytest.raises(ValidationError, match="at most 300"):
        _validate({**_patch([ITEM_FIX]), "rationale": "x" * 301})


# --- helpers -------------------------------------------------------------------------------


def test_value_literals_skip_subscript_indices() -> None:
    lits = value_literals(ast.parse("a = x[0] + y[1:2, -1] + z[i] * 3 + 0.1\nb = True"))
    assert lits == {"3": 1, "0.1": 1}


def test_apply_edits_delete_insert_and_no_final_newline() -> None:
    text = "a = 1\nb = 2\nc = 3"
    e = LineEdit(start_line=3, end_line=3, original="c = 3", replacement="c = 3\nd = c")
    assert apply_line_edits(text, [e]) == "a = 1\nb = 2\nc = 3\nd = c"
    d = LineEdit(start_line=2, end_line=2, original="b = 2", replacement="")
    assert apply_line_edits(text, [d]) == "a = 1\nc = 3"
    diff = unified_diff("m.py", text, apply_line_edits(text, [e]))
    assert "\\ No newline at end of file\n" in diff
    assert split_lines("") == [] and split_lines("x\n") == ["x\n"]


def test_crlf_line_endings_kept() -> None:
    text = "a = 1\r\nb = x[0]\r\n"
    e = LineEdit(start_line=2, end_line=2, original="b = x[0]", replacement="b = x.item()")
    assert apply_line_edits(text, [e]) == "a = 1\r\nb = x.item()\r\n"


def test_error_signature_normalizes_message_and_repo_frame() -> None:
    tb = {
        "exc_type": "IndexError",
        "message": "invalid index of a 0-dim tensor at 0x7f3a2b",
        "repo_frame": {"file": "/work/train.py", "line": 5, "func": "<module>"},
    }
    assert error_signature(tb) == "IndexError|invalid index of a N-dim tensor at <addr>|train.py:5"
    other = {**tb, "message": "invalid index of a 1-dim tensor at 0x11"}
    assert error_signature(other) == error_signature(tb)  # digits / addresses masked
    moved = {**tb, "repo_frame": {"file": "train.py", "line": 9, "func": "<module>"}}
    assert error_signature(moved) != error_signature(tb)
    assert normalize_message("a   b\nsecond line") == "a b"


# --- integer axis literals (dim= / axis= / shape methods) are exempt ------------------------

AXES = (
    "import torch\n"  # 1
    "lr = 0.1\n"  # 2
    "epochs = 200\n"  # 3
    "y = x.flatten_all()\n"  # 4
    "p = torch.softmax(x)\n"  # 5
)


def _axes(line: int, original: str, replacement: str) -> CriticPatch:
    edit = {"start_line": line, "end_line": line, "original": original, "replacement": replacement}
    return CriticPatch.model_validate(_patch([edit]), context=_ctx(file_text=AXES, metric_lines={}))


def test_view_minus_one_accepted() -> None:
    _axes(4, "y = x.flatten_all()", "y = x.view(-1)")


def test_softmax_dim_keyword_accepted() -> None:
    _axes(5, "p = torch.softmax(x)", "p = torch.softmax(x, dim=1)")


def test_shape_methods_and_axis_keyword_accepted() -> None:
    _axes(4, "y = x.flatten_all()", "y = x.reshape(x.size(0), -1).permute(1, 0)")
    _axes(5, "p = torch.softmax(x)", "p = x.sum(axis=(0, 1))")


def test_hyperparameter_changes_still_rejected() -> None:
    with pytest.raises(ValidationError, match=r"removed \['0.1'\], added \['0.05'\]"):
        _axes(2, "lr = 0.1", "lr = 0.05")
    with pytest.raises(ValidationError, match=r"removed \['200'\], added \['1'\]"):
        _axes(3, "epochs = 200", "epochs = 1")


def test_float_axis_and_ints_elsewhere_not_exempt() -> None:
    with pytest.raises(ValidationError, match=r"added \['1.0'\]"):
        _axes(5, "p = torch.softmax(x)", "p = torch.softmax(x, dim=1.0)")
    with pytest.raises(ValidationError, match=r"added \['2'\]"):
        _axes(5, "p = torch.softmax(x)", "p = torch.softmax(x * 2)")
    with pytest.raises(ValidationError, match=r"added \['3'\]"):
        _axes(4, "y = x.flatten_all()", "y = x.repeat(3)")  # not a shape method


def test_value_literals_axis_exemptions() -> None:
    tree = ast.parse("a = x.view(-1, 4) + f(x, dim=2, k=5) + x.size(0) * 0.5 + x.view(2.0)")
    assert value_literals(tree) == {"5": 1, "0.5": 1, "2.0": 1}
