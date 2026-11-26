from ra_agent.ocr.gate import check_table_merged_rows
from ra_agent.ocr.repair import repair_merged_rows

TABLE = (
    "| | Dev | Exec |\n|---|---|---|\n"
    "| READMEs & Docs Source code reproduce.sh | ✓ ✓ ✗ | ✓ ✗ ✓ |"
)
LAYER = "Dev\nExec\nREADMEs & Docs\n✓\n✓\nSource code\n✓\nreproduce.sh\n✗"


def test_split_into_rows():
    fixed = repair_merged_rows(TABLE, LAYER)
    assert fixed.splitlines()[2:] == [
        "| READMEs & Docs | ✓ | ✓ |",
        "| Source code | ✓ | ✗ |",
        "| reproduce.sh | ✗ | ✓ |",
    ]
    assert check_table_merged_rows(fixed).passed


def test_no_match_leaves_table_unchanged():
    assert repair_merged_rows(TABLE, "Source code\nreproduce.sh") == TABLE
    assert not check_table_merged_rows(repair_merged_rows(TABLE, "")).passed
