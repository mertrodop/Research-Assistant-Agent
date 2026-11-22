from __future__ import annotations

import pytest

from ra_agent.config import GateSettings
from ra_agent.ocr.gate import has_blockers, run_gate

GOOD = """# Introduction

Retrieval augmented generation improves factual grounding in scientific writing.
We evaluate three baselines on four datasets and report accuracy in Table 1.

| model | accuracy | f1 |
|---|---|---|
| baseline | 0.71 | 0.68 |
| ours | 0.83 | 0.80 |

The improvement is consistent across all four datasets and both metrics.

## References

Tosun, E. (2024). Grounded writing assistants. Journal of Agents, 12(3), 1-20.
Tuygan, E. (2023). Citation validation for LLMs. In Proceedings of ACL, 100-110.
"""


def test_good_page_passes():
    checks = run_gate(GOOD, GateSettings())
    assert not has_blockers(checks)


def test_empty_page_fails():
    checks = run_gate("", GateSettings())
    assert has_blockers(checks)
    assert any(c.name == "non_empty" and not c.passed for c in checks)


def test_repetition_loop_fails():
    loop = "the model output is repeated " * 40
    checks = run_gate(loop, GateSettings())
    assert any(c.name == "repetition_loop" and not c.passed for c in checks)


def test_table_column_mismatch_fails():
    broken = GOOD.replace("| ours | 0.83 | 0.80 |", "| ours | 0.83 |")
    checks = run_gate(broken, GateSettings())
    assert any(c.name == "table_columns" and not c.passed for c in checks)


def test_missing_table_header_blocks():
    broken = GOOD.replace(
        "| model | accuracy | f1 |\n|---|---|---|\n",
        "|---|---|---|\n",
    )
    checks = run_gate(broken, GateSettings())
    failed = [c for c in checks if c.name == "table_header" and not c.passed]
    assert failed and failed[0].severity.value == "blocker"
    assert has_blockers(checks)


def test_numeric_header_row_blocks():
    broken = GOOD.replace(
        "| model | accuracy | f1 |\n|---|---|---|\n| baseline | 0.71 | 0.68 |",
        "| 0.71 | 0.68 | 0.55 |\n|---|---|---|\n| baseline | 0.71 | 0.68 |",
    )
    checks = run_gate(broken, GateSettings())
    failed = [c for c in checks if c.name == "table_header" and not c.passed]
    assert failed and "numeric" in failed[0].detail
    assert has_blockers(checks)


def test_unbalanced_equation_fails():
    broken = GOOD + "\n$$\nE = mc^2\n"
    checks = run_gate(broken, GateSettings())
    assert any(c.name == "equation_delimiters" and not c.passed for c in checks)


def test_coverage_low_fails():
    checks = run_gate("short text", GateSettings(), text_layer_chars=6000)
    assert any(c.name == "coverage" and not c.passed for c in checks)


def test_reference_without_year_warns():
    broken = GOOD.replace("Tosun, E. (2024).", "Tosun, E.")
    checks = run_gate(broken, GateSettings())
    failed = [c for c in checks if c.name == "reference_fields" and not c.passed]
    assert failed and failed[0].severity.value == "warn"
    assert not has_blockers(checks)


def test_doctags_unclosed_warns():
    checks = run_gate(GOOD, GateSettings(), doctags="<text>hello")
    assert any(c.name == "doctags_balance" and not c.passed for c in checks)


def test_sparse_page_warns_instead_of_blocking():
    checks = run_gate("Thank you Q&A", GateSettings(), ink_ratio=0.01)
    failed = [c for c in checks if c.name == "non_empty" and not c.passed]
    assert failed and failed[0].severity.value == "warn"
    assert not has_blockers(checks)


def test_dense_page_with_empty_ocr_blocks():
    checks = run_gate("", GateSettings(), ink_ratio=0.15)
    failed = [c for c in checks if c.name == "non_empty" and not c.passed]
    assert failed and failed[0].severity.value == "blocker"
    assert has_blockers(checks)


@pytest.mark.parametrize("markdown", ["", "   ", "\x00\x01\x02"])
def test_garbage_is_blocked(markdown):
    assert has_blockers(run_gate(markdown, GateSettings()))
