# toy_critic: a SYNTHETIC paper for the ReAct critic + approval gate

**SYNTHETIC.** Nothing here is a real paper, model or result. `spec.json` and `toyrepo/`
exist only to demo and test the reproducer's crash path (smoke crash -> ReAct critic ->
human approval of the patch) end to end, on CPU in seconds, deterministically.

- `toyrepo/` is the KeyError investigation fixture of `tests/graph/test_critic_react.py`
  (made runnable with argparse flags and a per-epoch metric line): `utils/metrics.py` was
  refactored to the v2 keys `num_correct` / `num_seen`, but `train.py` still reads
  `stats["correct"]` / `stats["total"]` -> `KeyError: 'correct'` in `train.py`.
- The traceback window of `train.py` does not show the new keys: the critic has to look
  into `utils/metrics.py` (grep / read_file) and patch only the failing file, `train.py`.
- With the default approval policy every critic patch needs a human: the first `reproduce`
  stops with `awaiting_approval`; `queue` shows the request and the `approve` command.
- After approval, the re-smoke passes and the run prints `test_acc: 75.000` (3 of 4 right)
  -> `test_error` 25.00 % (`accuracy_percent_to_error_percent`) vs the paper's
  25.00 ± 0.10 -> verified (fidelity inferred: the patch is a deviation).

Run (live LLMs, Docker for the sandboxed runs):

```bash
.venv/bin/python -m reproducer.cli reproduce --spec evals/toy_critic/spec.json --out runs/toy_critic
.venv/bin/python -m reproducer.cli queue --out runs/toy_critic
# a human reviews the diff, then: approve --out runs/toy_critic <request_id> --note "..."
```
