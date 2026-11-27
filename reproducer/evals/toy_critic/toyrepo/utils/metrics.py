"""Batch accuracy statistics (v2 API). SYNTHETIC toy helper."""


def batch_stats(output, target):
    """Counts for one batch as 0-dim tensors / ints; keys renamed in the v2 refactor."""
    pred = output.argmax(dim=1)
    return {"num_correct": (pred == target).sum(), "num_seen": target.size(0)}
