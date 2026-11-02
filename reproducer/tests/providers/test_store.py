"""Tests for reproducer.providers.store."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from reproducer.domain.evidence import Evidence, make_evidence
from reproducer.domain.unknowns import Assumption, UnknownOpened, unknown_id_for
from reproducer.providers.store import EVENTS_FILE, EVIDENCE_FILE, RunStore


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
