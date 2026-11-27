"""SYNTHETIC toy "training" script for the reproducer's Reflexion demo and tests.

Not a real model: test_err is a deterministic function of the flags (CPU only, no data,
finishes in well under a second). It exists to exercise the reproduce -> evaluate -> reflect
-> apply -> re-run loop end to end.
"""

import argparse
import time

parser = argparse.ArgumentParser(description="SYNTHETIC toy trainer (deterministic)")
parser.add_argument("--dataset", default="toyA", choices=["toyA", "toyB"])
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--noise", type=float, default=0.2, help="std of the Gaussian input noise")
args = parser.parse_args()

# the irreducible error of each toy dataset (percent)
BASE_ERR = {"toyA": 2.0, "toyB": 9.0}

for epoch in range(args.epochs):
    time.sleep(0.05)
    remaining = args.epochs - 1 - epoch
    test_err = 100.0 * (BASE_ERR[args.dataset] / 100.0 + abs(args.noise - 0.05) + 0.01 * remaining)
    print("test_err: %.2f" % (test_err), flush=True)  # noqa: UP031 (printf metric line, like many repos)
