"""SYNTHETIC toy "evaluation" script for the reproducer's critic demo and tests.

Not a real model: the accuracy is computed on a fixed 4-sample batch (CPU only, no data,
finishes in about a second). Built-in bug: utils/metrics.py was refactored to the v2 key
names (num_correct / num_seen), this script still reads the old ones -> KeyError: 'correct'.
The fix needs the helper's code, which is not in the failing file.
"""

import argparse
import sys
import time

import torch
from utils.metrics import batch_stats

parser = argparse.ArgumentParser(description="SYNTHETIC toy evaluator (deterministic)")
parser.add_argument("--dataset", default="toyC", choices=["toyC"])
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()

torch.manual_seed(0)
output = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
target = torch.tensor([1, 0, 0, 0])

for _epoch in range(args.epochs):
    for i in range(1, 4):
        sys.stderr.write("\r%d%%|##| %d/3 [00:0%d<00:0%d, 1.00it/s]" % (i * 33, i, i, 3 - i))  # noqa: UP031
        sys.stderr.flush()
        time.sleep(0.05)
    sys.stderr.write("\n")
    stats = batch_stats(output, target)
    correct, total = stats["correct"].item(), stats["total"]
    test_acc = 100.0 * correct / total
    print("test_acc: %.3f" % (test_acc), flush=True)  # noqa: UP031 (printf metric line)
