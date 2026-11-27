import sys

from ra_agent.cli import decide, edit_and_validate, main, parse_decision
from ra_agent.vault import Vault

PAYLOAD = {"tool": "commit_note", "args": {"paper": "x"}, "diff": "the diff"}


def fake_editor(tmp_path, body: str) -> str:
    script = tmp_path / "editor.py"
    script.write_text(f"import sys; open(sys.argv[1], 'w').write({body!r})")
    return f"{sys.executable} {script}"


def test_parse_decision():
    assert parse_decision("a") == "approve" and parse_decision(" R ") == "reject"
    assert parse_decision("e") == "edit" and parse_decision("x") is None


def test_decide_asks_again_until_valid(tmp_path, capsys):
    answers = iter(["", "what", "r"])
    vault = Vault(tmp_path / "vault", tmp_path / "staging")
    assert decide(PAYLOAD, vault, input_fn=lambda _: next(answers)) == "reject"
    assert "the diff" in capsys.readouterr().out


def test_edit_returns_validation_error(tmp_path):
    note = tmp_path / "note.md"
    assert "title" in edit_and_validate(note, fake_editor(tmp_path, "---\nfoo: 1\n---\nbody"))
    assert edit_and_validate(note, fake_editor(tmp_path, "no frontmatter")) != ""


def test_papers_lists_dev_slugs(capsys):
    assert main(["papers"]) == 0
    out = capsys.readouterr().out
    assert "paperbench-evaluating" in out and "core-bench-fostering" in out


def test_edit_then_approve(tmp_path, capsys, monkeypatch):
    vault = Vault(tmp_path / "vault", tmp_path / "staging")
    note = vault.note(vault.staging, "x")
    note.parent.mkdir(parents=True)
    note.write_text("old")
    body = (
        "---\ntitle: T\nsource_pdf: a.pdf\nsha256: abc\npages: 1\nverified_pages: 1\n"
        "status: verified\ncreated_at: now\n---\nedited"
    )
    monkeypatch.setenv("EDITOR", fake_editor(tmp_path, body))
    answers = iter(["e", "a"])
    assert decide(PAYLOAD, vault, input_fn=lambda _: next(answers)) == "approve"
    assert "edited" in note.read_text() and "edited" in capsys.readouterr().out


def test_agent_check_accepts_keyless_local_endpoint(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("models:\n  agent:\n    base_url: http://localhost:11434/v1\n    api_key_env: NO_SUCH_KEY\n")
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    monkeypatch.setattr("ra_agent.cli.run_agent", lambda *a: _done())
    assert main(["agent", "hi", "--config", str(cfg)]) == 0
    assert "Set NO_SUCH_KEY" not in capsys.readouterr().out


async def _done():
    return {"messages": [type("M", (), {"content": "ok", "name": ""})()], "outcome": "finish"}


class FakeWorkspace:
    """ingest result and stage status are set per test; commits are recorded."""

    problems, status, committed = [], "verified", []
    vault = None

    def __init__(self, settings):
        pass

    def ingest_paper(self, pdf):
        return {"ok": True, "slug": "x", "summary": "ingested", "problems": self.problems}

    def stage_note(self, slug):
        return {"ok": True, "status": self.status, "diff": "the diff", "handle": "staging/x.md"}

    def commit_note(self, slug):
        self.committed.append(slug)
        return {"ok": True, "summary": "committed x"}


def ingest(monkeypatch, tmp_path, problems, status, key="k"):
    FakeWorkspace.problems, FakeWorkspace.status, FakeWorkspace.committed = problems, status, []
    FakeWorkspace.vault = Vault(tmp_path / "vault", tmp_path / "staging")
    ran = []
    monkeypatch.setattr("ra_agent.tools.Workspace", FakeWorkspace)
    monkeypatch.setattr("ra_agent.cli.run_agent", lambda *a: ran.append(a) or _done())
    monkeypatch.setattr("ra_agent.cli.load_env_file", lambda: None)
    monkeypatch.setattr("ra_agent.cli.decide", lambda *a: "approve")  # the prompt itself is tested above
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", key)
    return main(["ingest", "/x/p.pdf"]), ran


def test_ingest_without_problems_never_runs_the_agent(monkeypatch, tmp_path):
    code, ran = ingest(monkeypatch, tmp_path, [], "verified")
    assert (code, ran, FakeWorkspace.committed) == (0, [], ["x"])


def test_ingest_with_problems_runs_the_agent_with_repair_tools_only(monkeypatch, tmp_path):
    code, ran = ingest(monkeypatch, tmp_path, ["page 3: gate failed"], "verified")
    assert code == 0 and len(ran) == 1
    task, _paper, _settings, _config, only = ran[0]
    assert "page 3: gate failed" in task and only == {"reprocess_page", "describe_figure", "check_references"}


def test_needs_review_stays_in_staging(monkeypatch, tmp_path):
    code, _ = ingest(monkeypatch, tmp_path, ["page 3: gate failed"], "needs-review")
    assert code == 1 and FakeWorkspace.committed == []
