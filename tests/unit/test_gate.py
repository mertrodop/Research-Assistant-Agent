from __future__ import annotations

import pytest

from ra_agent.config import GateSettings
from ra_agent.ocr.gate import (
    check_heading_presence,
    check_table_cells,
    has_blockers,
    run_gate,
)
from ra_agent.schema import CheckResult, PageResult, Route, localize

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


def test_misread_number_blocks_even_if_value_exists_elsewhere():
    from ra_agent.ocr.gate import check_numbers

    layer = "Score 13.2 ± 0.3 for o1 and 10.9 ± 0.9 for Claude, over 3 seeds and 20 papers."
    ocr = layer.replace("13.2 ± 0.3", "13.2 ± 0.9")  # 0.9 also appears later
    check = check_numbers(ocr, layer)
    assert not check.passed and check.severity.value == "blocker"


def test_dropped_decimal_point_blocks():
    from ra_agent.ocr.gate import check_numbers

    layer = "The judge reaches an F1 of 0.45 with a variance of 0.05 on the full set of papers."
    check = check_numbers(layer.replace("0.45", "0 45"), layer)
    assert not check.passed and "45" in check.evidence


def test_number_formatting_is_not_an_error():
    from ra_agent.ocr.gate import check_numbers

    layer = "We reviewed 1,008 studies and found 484 with errors across all fields in total."
    assert check_numbers(layer.replace("1,008", "1008"), layer).passed


def test_footnote_marker_and_spanning_cell_are_not_errors():
    from ra_agent.ocr.gate import check_numbers

    layer = "We evaluate GPT-4o,10 o1,11 and o3-mini.12 With an extended 36 hour limit the score rises."
    ocr = "We evaluate GPT-4o, 10 o1, 11 and o3-mini. 12\n\n| With an extended 36 hour limit | With an extended 36 hour limit |\n\nthe score rises."
    assert check_numbers(ocr, layer).severity.value != "blocker"


def test_table_with_repeated_rows_blocks():
    t = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    c = check_table_cells("x\n\n" + t + "\n" + t.split("\n", 2)[2])
    assert not c.passed and c.severity.value == "blocker" and "table 1" in c.detail


def test_table_with_empty_half_blocks():
    t = "| Field | Paper | n | Field | Paper | n |\n|---|---|---|---|---|---|\n" + "| ML | Raff | 255 |  |  | 30 |\n" * 4
    assert not check_table_cells(t).passed


def test_spanning_cell_repeat_is_fine():
    t = "| a | b |\n|---|---|\n| With a limit | With a limit |\n| 1 | 2 |\n"
    assert check_table_cells(t).passed


def test_missing_numbered_heading_warns():
    c = check_heading_presence("plain text only", "2 PaperBench\nIn this section ...")
    assert not c.passed and c.severity.value == "warn"
    assert check_heading_presence("## 2 PaperBench\ntext", "2 PaperBench\nIn").passed


def test_localize_names_page_engine_and_evidence():
    page = PageResult(page_no=4, route=Route.DOCLING, markdown="", engine="docling", attempt=1)
    check = CheckResult(name="numeric_agreement", passed=False, detail="3 numbers not in the text layer", evidence="0, 45, 05")
    want = "page 4: numeric_agreement: 3 numbers not in the text layer (0, 45, 05); engine docling, attempt 1"
    assert localize("paperbench", page, check) == "paperbench, " + want
    page.checks = [check]
    assert page.problems == [want]


def test_single_shared_row_across_tables_passes():
    a = "| a | b |\n|---|---|\n| N/A | N/A |\n| 1 | 2 |\n"
    b = "| c | d |\n|---|---|\n| N/A | N/A |\n| 5 | 6 |\n"
    assert check_table_cells(a + "\ntext\n\n" + b).passed


def test_duplicated_table_blocks_naming_table_2():
    t = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n"
    c = check_table_cells(t + "\ntext\n\n" + t)
    assert not c.passed and "table 2" in c.detail
