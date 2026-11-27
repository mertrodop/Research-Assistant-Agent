"""Tests for reproducer.providers.store."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.unknowns import Assumption, UnknownOpened, unknown_id_for
from reproducer.providers.llm import Budget, FakeLLM, MeteredLLM, prompt_sha256
from reproducer.providers.store import (
    EVENTS_FILE,
    EVIDENCE_FILE,
    LLM_CALLS_FILE,
    NotARunStore,
    RunLocked,
    RunStore,
    lock_holder,
    run_lock,
)


def _ev(i: int, source: str = "repo@abc") -> Evidence:
    return make_evidence(
        kind="repo_file",
        source=source,
        locator=f"train.py:L{i}",
        content={"line": i, "text": f"x = {i}"},
        confidence="observed",
        authority="repo",
    )


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_init_creates_folder_and_files(tmp_path: Path) -> None:
    root = tmp_path / "papers" / "p1"
    RunStore(root)
    assert (root / EVIDENCE_FILE).is_file()
    assert (root / EVENTS_FILE).is_file()


def test_append_and_reload_equal(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    records = [_ev(i) for i in range(5)]
    assert all(store.append_evidence(ev) for ev in records)
    assert store.load_evidence() == {ev.evidence_id: ev for ev in records}

    ev0 = records[0]
    events = (
        UnknownOpened(unknown_id=unknown_id_for("lr"), name="lr", evidence_id=ev0.evidence_id),
        Assumption(unknown_id=unknown_id_for("lr"), value="0.1", evidence_id=ev0.evidence_id),
    )
    for e in events:
        store.append_event(e)
    assert store.load_events() == events


def test_duplicate_evidence_not_reappended(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    ev = _ev(1)
    assert store.append_evidence(ev) is True
    before = _line_count(store.evidence_path)
    assert store.append_evidence(ev) is False
    assert _line_count(store.evidence_path) == before == 1


def test_identical_events_are_both_appended(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    ev = _ev(1)
    store.append_evidence(ev)
    opened = UnknownOpened(unknown_id=unknown_id_for("lr"), name="lr", evidence_id=ev.evidence_id)
    store.append_event(opened)
    store.append_event(opened)
    assert store.load_events() == (opened, opened)


def test_tampered_line_raises_with_line_number(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    for i in range(3):
        store.append_evidence(_ev(i))
    lines = store.evidence_path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["content"]["text"] = "x = 999"  # id no longer matches content
    lines[1] = json.dumps(rec)
    store.evidence_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{EVIDENCE_FILE}:2: "):
        store.load_evidence()
    with pytest.raises(ValueError, match=rf"{EVIDENCE_FILE}:2: "):
        RunStore(tmp_path)


def test_corrupt_event_line_raises_with_line_number(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.events_path.write_text('{"event":"opened"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{EVENTS_FILE}:1: "):
        store.load_events()


def test_blank_trailing_line_ok_interior_blank_rejected(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_evidence(_ev(1))
    store.append_evidence(_ev(2))
    with open(store.evidence_path, "a", encoding="utf-8") as f:
        f.write("\n")
    assert len(store.load_evidence()) == 2

    lines = store.evidence_path.read_text(encoding="utf-8").splitlines()
    store.evidence_path.write_text(lines[0] + "\n\n" + lines[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{EVIDENCE_FILE}:2: blank line"):
        store.load_evidence()


def test_refuses_append_after_torn_last_line(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_evidence(_ev(1))
    with open(store.evidence_path, "a", encoding="utf-8") as f:
        f.write('{"evidence_id":')  # simulated torn write
    content = store.evidence_path.read_bytes()
    with pytest.raises(ValueError, match="torn"):
        store.append_evidence(_ev(2))
    assert store.evidence_path.read_bytes() == content  # nothing written, nothing truncated


def test_dedupe_survives_restart(tmp_path: Path) -> None:
    ev = _ev(1)
    assert RunStore(tmp_path).append_evidence(ev) is True
    fresh = RunStore(tmp_path)
    assert fresh.append_evidence(ev) is False
    assert _line_count(fresh.evidence_path) == 1


def test_dedupe_across_live_instances(tmp_path: Path) -> None:
    a, b = RunStore(tmp_path), RunStore(tmp_path)
    ev = _ev(1)
    assert a.append_evidence(ev) is True
    assert b.append_evidence(ev) is False  # b resyncs from disk under the lock
    assert _line_count(a.evidence_path) == 1


@pytest.mark.parametrize("shared", [True, False], ids=["shared-store", "store-per-thread"])
def test_concurrent_appends(tmp_path: Path, shared: bool) -> None:
    n_threads, n_records = 8, 20
    common = _ev(0, source="common")
    shared_store = RunStore(tmp_path)
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker(t: int) -> None:
        try:
            store = shared_store if shared else RunStore(tmp_path)
            barrier.wait()
            for i in range(n_records):
                store.append_evidence(_ev(i, source=f"t{t}"))
                store.append_evidence(common)  # every thread races on the same id
                store.append_event(
                    UnknownOpened(
                        unknown_id=unknown_id_for(f"u{t}"),
                        name=f"u{t}",
                        evidence_id=common.evidence_id,
                    )
                )
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors

    # every line parses (no interleaving), common stored exactly once
    loaded = RunStore(tmp_path).load_evidence()
    assert len(loaded) == n_threads * n_records + 1
    assert _line_count(shared_store.evidence_path) == n_threads * n_records + 1
    assert len(shared_store.load_events()) == n_threads * n_records


# --------------------------------------------------------------------------- LLM telemetry


def test_call_records_from_metered_llm_roundtrip(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    assert (tmp_path / LLM_CALLS_FILE).is_file()
    secret_prompt = "unique-secret-prompt-text-42"
    llm = MeteredLLM(
        FakeLLM(["a b", "c", "c"]), Budget(), "subtasks", on_record=store.append_call_record
    )
    llm.complete("sys", secret_prompt, max_tokens=10)
    llm.complete("sys", "other", max_tokens=10)
    llm.complete("sys", "other", max_tokens=10)  # identical record: no dedupe for telemetry

    assert store.load_call_records() == llm.records()
    assert RunStore(tmp_path).load_call_records() == llm.records()  # survives restart
    assert _line_count(tmp_path / LLM_CALLS_FILE) == 3
    raw = (tmp_path / LLM_CALLS_FILE).read_text()
    assert secret_prompt not in raw
    assert prompt_sha256("sys", secret_prompt) in raw
    assert _line_count(tmp_path / EVIDENCE_FILE) == 0  # telemetry is not evidence


def test_call_record_refuses_append_after_torn_tail(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    with open(tmp_path / LLM_CALLS_FILE, "a") as f:
        f.write('{"node":')  # simulated torn write
    llm = MeteredLLM(FakeLLM(["x"]), Budget(), "n", on_record=store.append_call_record)
    with pytest.raises(ValueError, match="torn"):
        llm.complete("s", "p", max_tokens=5)
    assert (tmp_path / LLM_CALLS_FILE).read_text() == '{"node":'


def test_corrupt_call_record_line_raises_with_line_number(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    (tmp_path / LLM_CALLS_FILE).write_text('{"node": "x"}\n')
    with pytest.raises(ValueError, match=r"llm_calls.jsonl:1"):
        store.load_call_records()


# --- run lock ------------------------------------------------------------------------------


def test_run_lock_second_acquire_fails_fast(tmp_path: Path) -> None:
    import os
    import time

    with run_lock(tmp_path) as path:
        assert path.read_text().strip() == str(os.getpid())
        t0 = time.monotonic()
        with pytest.raises(RunLocked, match=f"another `run` is in progress.*pid {os.getpid()}"):
            with run_lock(tmp_path):
                pass
        assert time.monotonic() - t0 < 1.0  # non-blocking
    with run_lock(tmp_path):  # released on exit
        pass


def test_run_lock_held_across_processes(tmp_path: Path) -> None:
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from reproducer.providers.store import RunLocked, run_lock\n"
        "try:\n"
        "    with run_lock(sys.argv[1]):\n"
        "        sys.exit(0)\n"
        "except RunLocked:\n"
        "    sys.exit(7)\n"
    )
    with run_lock(tmp_path):
        held = subprocess.run([sys.executable, "-c", code, str(tmp_path)], timeout=30)
    free = subprocess.run([sys.executable, "-c", code, str(tmp_path)], timeout=30)
    assert (held.returncode, free.returncode) == (7, 0)


def test_lock_holder_probe_is_read_only(tmp_path: Path) -> None:
    import os

    assert lock_holder(tmp_path) is None
    assert not (tmp_path / "run.lock").exists()  # the probe never creates the file
    with run_lock(tmp_path):
        assert lock_holder(tmp_path) == str(os.getpid())
        with run_lock(tmp_path / "other"):  # other stores are independent
            pass
    assert lock_holder(tmp_path) is None  # released; the stale pid file is not a holder
    with run_lock(tmp_path):  # the probe left the lock acquirable
        pass


def test_create_false_on_missing_store_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    with pytest.raises(NotARunStore, match="not a run store"):
        RunStore(root, create=False)
    assert not root.exists()
    root.mkdir()
    with pytest.raises(NotARunStore, match="not a run store"):
        RunStore(root, create=False)
    assert list(root.iterdir()) == []


def test_create_false_reads_without_creating_files(tmp_path: Path) -> None:
    RunStore(tmp_path).append_evidence(_ev(1))
    (tmp_path / EVENTS_FILE).unlink()
    (tmp_path / LLM_CALLS_FILE).unlink()
    store = RunStore(tmp_path, create=False)
    assert list(store.load_evidence()) == [_ev(1).evidence_id]
    assert store.load_events() == () and store.load_call_records() == ()
    assert sorted(p.name for p in tmp_path.iterdir()) == [EVIDENCE_FILE]
