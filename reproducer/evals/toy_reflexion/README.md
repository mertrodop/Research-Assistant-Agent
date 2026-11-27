# toy_reflexion: a SYNTHETIC paper for the Reflexion loop

**SYNTHETIC.** Nothing here is a real paper, model or result. `spec.json` and `toyrepo/`
exist only to demo and test the reproducer's Reflexion loop (evaluate -> reflect -> apply ->
re-run) on CPU in seconds, deterministically.

- `toyrepo/train.py` prints `test_err: <value>` per epoch; the value is a fixed function of
  its flags: `100 * (base[dataset] / 100 + |noise - 0.05| + 0.01 * epochs_left)` with base
  toyA = 2.0, toyB = 9.0.
- `spec.json` links the repo as `file:toyrepo`; `configure` snapshots it into
  `<out>/repo` as a one-commit git repo with a fixed author and date (same files, same sha).

Built-in gap (what Reflexion has to find):

- The README's example commands use `--noise 0.2`; the paper setup says the noise std is
  0.05, but its value is written as prose (`"0.05 (std of the Gaussian input noise)"`), which
  does not parse as the `--noise` float, so the orchestrator cannot map it (its mapping
  validator rejects it) and trial 1 runs with 0.2: toyA gives 17.00 % vs the paper's
  2.00 ± 0.10 (tolerance ± 0.20) -> out of tolerance.
- The grounded fix is an `arg` lesson `--noise 0.05` citing the paper quote "we add Gaussian
  input noise with a standard deviation of 0.05" -> trial 2: 2.00 % -> verified.
- Variant without a grounded fix: toyB's best reachable value is 9.00 % (noise 0.05) vs the
  paper's 5.00 %; after the noise fix nothing in the paper or repo explains the rest, so the
  next lesson is `none` and the claim stays open ("reflexion stopped: no grounded change")
  or the breaker stops it after 3 trials.
- `eval_checkpoint` is an open unknown (the error decreases every epoch, so last = best).

Run (live LLMs, Docker for the sandboxed runs):

```bash
.venv/bin/python -m reproducer.cli reproduce --spec evals/toy_reflexion/spec.json --out runs/toy_reflexion
```
