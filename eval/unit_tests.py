"""olmOCR-style unit tests on ingested notes: one binary check per known fact.

Each entry is (paper slug prefix, name, check on the note text). A failed check is a
finding about the pipeline; never loosen a check to make it pass.
"""

from __future__ import annotations

import re
from pathlib import Path

NOTE_DIR = Path(".")  # folder that holds the notes, set by run_eval before the run

CB, PB = "core-bench", "paperbench"
STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}v\d+\s+\[", re.M)  # the margin stamp, e.g. arXiv:2409.11363v1 [cs.CL]


def body(text: str) -> str:
    return text.split("\n---\n", 1)[-1]


def fm_block(text: str) -> str:
    return text.split("\n---\n", 1)[0]


def first_heading(text: str) -> str:
    return next((l for l in body(text).splitlines() if l.startswith("#")), "")


def table_after(text: str, caption: str) -> str:
    """The Markdown table rows right after the line that starts with `caption`."""
    lines = body(text).splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(caption)), None)
    if start is None:
        return ""
    rows = []
    for line in lines[start + 1 : start + 40]:
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            break
    return "\n".join(rows)


def figure_files_exist(text: str) -> bool:
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body(text))
    return bool(links) and all((NOTE_DIR / link).is_file() for link in links)


def fm(text: str, key: str) -> str:
    match = re.search(rf"^{key}: (.*)$", text, re.M)
    return match.group(1).strip() if match else ""


def common(prefix: str, title: str) -> list[tuple[str, str, object]]:
    return [
        (prefix, "title_is_first_heading", lambda t: first_heading(t).startswith(f"# {title}")),
        (prefix, "no_arxiv_stamp_line", lambda t: not STAMP.search(body(t))),
        (prefix, "no_image_placeholder", lambda t: "<!-- image -->" not in t),
        (prefix, "figure_links_resolve", figure_files_exist),
        (prefix, "no_replacement_chars", lambda t: "�" not in t),
        (prefix, "status_verified", lambda t: fm(t, "status") == "verified"),
        (prefix, "metadata_verified", lambda t: fm(t, "metadata") == "verified"),
        (prefix, "has_references_section", lambda t: "\n## References" in t),
    ]


UNIT_TESTS: list[tuple[str, str, object]] = [
    *common(CB, "CORE-Bench"),
    (CB, "table1_is_markdown_with_finance_1008",
     lambda t: "| Finance" in table_after(t, "Table 1:") and "1008" in table_after(t, "Table 1:")),
    (CB, "table5_accuracies",
     lambda t: all(v in table_after(t, "Table 5:") for v in ("60.00%", "44.44%", "21.48%", "6.67%"))),
    (CB, "table5_header_has_three_levels",
     lambda t: all(h in table_after(t, "Table 5:") for h in ("CORE-Bench-Easy", "CORE-Bench-Medium", "CORE-Bench-Hard"))),
    (CB, "section_4_5_python_vs_r", lambda t: "## 4.5 Python tasks are much easier than R" in t),
    (CB, "appendix_sections_present", lambda t: all(h in t for h in ("## A Benchmark Details", "## D Agent Details"))),
    *common(PB, "PaperBench"),
    (PB, "paperbench_code_dev_appears", lambda t: "PaperBench Code-Dev" in body(t)),
    (PB, "table4_claude_21_0", lambda t: "| CLAUDE-3.5-SONNET | 21.0" in table_after(t, "Table 4.")),
    (PB, "table6_o1_43_4", lambda t: "43.4" in table_after(t, "Table 6.")),
    (PB, "abstract_8316_tasks", lambda t: "8,316" in t.split("## Abstract", 1)[-1][:1500]),
    (PB, "section_2_6_code_dev_heading", lambda t: "## 2.6. PaperBench Code-Dev" in t),
    (PB, "code_link_preparedness", lambda t: "https://github.com/openai/preparedness" in fm_block(t)),
    (PB, "table1_has_five_rows", lambda t: len(table_after(t, "Table 1.").splitlines()) >= 7),
    (PB, "footnote_18_definition", lambda t: "[^18]: The instructions" in t),
    (CB, "code_link_core_bench", lambda t: "github.com/siegelz/core-bench" in fm_block(t)),
]
