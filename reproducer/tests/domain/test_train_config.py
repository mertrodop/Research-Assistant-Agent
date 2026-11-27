"""Training-config pre-scan (pure: ast / regex over texts)."""

from __future__ import annotations

from reproducer.domain.train_config import find_training_config_lines, train_config_evidence

CUTOUT_LIKE = """\
import torch
from torch.optim.lr_scheduler import MultiStepLR

cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
"""

MANUAL_LR = """\
opt = optim.Adam(model.parameters(), lr=1e-3)
def adjust(optimizer, epoch):
    lr = 0.1 * (0.2 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
"""

PY2 = """\
print "training"
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.2)
"""


def test_cutout_sgd_and_multistep() -> None:
    lines = find_training_config_lines({"train.py": CUTOUT_LIKE})
    assert [(t.kind, t.name, t.line_no, t.end_line_no) for t in lines] == [
        ("optimizer", "SGD", 4, 5),
        ("scheduler", "MultiStepLR", 8, 8),
        ("scheduler", "MultiStepLR", 10, 10),
    ]
    sgd = lines[0]
    assert sgd.kwargs == {
        "lr": "args.learning_rate",
        "momentum": "0.9",
        "nesterov": "True",
        "weight_decay": "5e-4",
    }
    assert sgd.line == (
        "cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, "
        "momentum=0.9, nesterov=True, weight_decay=5e-4)"
    )  # the two source lines joined: citable as one quote
    assert lines[2].kwargs == {"milestones": "[60, 120, 160]", "gamma": "0.2"}


def test_adam_cosine_and_manual_lr_loop() -> None:
    lines = find_training_config_lines({"main.py": MANUAL_LR})
    got = [(t.kind, t.name, t.line_no) for t in lines]
    assert got == [
        ("optimizer", "Adam", 1),
        ("lr_update", "lr", 5),
        ("scheduler", "CosineAnnealingLR", 6),
    ]
    assert lines[1].kwargs == {"lr": "lr"} and lines[1].line == "param_group['lr'] = lr"
    assert lines[2].kwargs == {"T_max": "200"}


def test_unparsable_file_falls_back_to_regex() -> None:
    lines = find_training_config_lines({"old.py": PY2})
    assert [(t.kind, t.name, t.line_no) for t in lines] == [
        ("optimizer", "SGD", 2),
        ("scheduler", "MultiStepLR", 3),
    ]
    assert lines[0].kwargs == {"lr": "0.1", "momentum": "0.9", "weight_decay": "5e-4"}
    assert lines[1].kwargs == {"milestones": "[60, 120]", "gamma": "0.2"}


def test_no_false_positives_on_plain_code() -> None:
    text = "lr = args.lr\nprint('lr', lr)\nx = LRFinder\nloss = criterion(out, y)\n"
    assert find_training_config_lines({"a.py": text}) == ()


def test_evidence_covers_modules_and_is_stable() -> None:
    texts = {"train.py": CUTOUT_LIKE, "util/opt.py": MANUAL_LR}
    ev = train_config_evidence("train.py", texts, source="repo@x", links=("b", "a"))
    assert ev.locator == "train.py#train_config"
    assert (ev.kind, ev.confidence, ev.authority, ev.links) == (
        "repo_file",
        "derived",
        "repo",
        ("a", "b"),
    )
    assert ev.content["scanned"] == ["train.py", "util/opt.py"]
    assert {r["path"] for r in ev.content["result"]} == {"train.py", "util/opt.py"}
    again = train_config_evidence("train.py", dict(reversed(texts.items())), source="repo@x")
    assert again.evidence_id == ev.evidence_id  # links are not part of the id
