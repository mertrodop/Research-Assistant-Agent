"""Tests for reproducer.rules.classify (failure classification is code, never LLM)."""

from __future__ import annotations

from typing import Any

import pytest

from reproducer.domain.smoke_plan import BudgetArg
from reproducer.domain.traceback_parse import parse_last_traceback
from reproducer.rules.classify import ClassifyContext, classify, is_write_call

CUTOUT_FILES = frozenset(
    {
        "train.py", "README.md", "util/__init__.py", "util/misc.py", "util/cutout.py",
        "model/__init__.py", "model/resnet.py", "model/wide_resnet.py",
        "images/cutout_on_cifar10.jpg",
    }
)  # fmt: skip
CUTOUT_ARGS = (
    BudgetArg(option_strings=("--batch_size",), dest="batch_size", default=128),
    BudgetArg(option_strings=("--epochs",), dest="epochs", default=200),
)
CTX = ClassifyContext(
    repo_files=CUTOUT_FILES,
    arguments=CUTOUT_ARGS,
    mounted=frozenset({"/work/data"}),
    dataset_roots={"/work/data": "cifar10", "/work/svhn_root": "svhn", "/work/imgs": None},
)
CMD = "python train.py --dataset cifar10 --model resnet18 --epochs 1"

CUTOUT_LOG = """\
Traceback (most recent call last):
  File "train.py", line 165, in <module>
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)
  File "/work/util/misc.py", line 8, in __init__
    self.csv_file = open(filename, 'w')
FileNotFoundError: [Errno 2] No such file or directory: 'logs/cifar10_resnet18.csv'
"""  # noqa: E501 - the real Cutout traceback, verbatim


def _content(
    log: str, signal: str = "traceback", exit_code: int | None = 1, **kw: Any
) -> dict[str, Any]:
    tb = parse_last_traceback(log, exit_code)
    return {
        "passed": False,
        "signal": signal,
        "command": kw.pop("command", CMD),
        "traceback": None if tb is None else tb.model_dump(mode="json"),
        **kw,
    }


def _tb(line: str, exc: str, frame: str = "train.py") -> str:
    return (
        "Traceback (most recent call last):\n"
        f'  File "{frame}", line 10, in <module>\n'
        f"    {line}\n"
        f"{exc}\n"
    )


def test_real_cutout_traceback_is_missing_output_dir_logs() -> None:
    c = classify(_content(CUTOUT_LOG), CTX)
    assert (c.cls, c.pattern, c.params["dir"]) == ("known", "missing_output_dir", "logs")


TORCH_SAVE_LOG = """\
Traceback (most recent call last):
  File "train.py", line 229, in <module>
    torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
  File "/opt/conda/lib/python3.7/site-packages/torch/serialization.py", line 376, in save
    with _open_file_like(f, 'wb') as opened_file:
  File "/opt/conda/lib/python3.7/site-packages/torch/serialization.py", line 230, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/opt/conda/lib/python3.7/site-packages/torch/serialization.py", line 211, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/cifar10_resnet18.pt'
"""  # noqa: E501 - the real Cutout torch.save traceback, verbatim


def test_checkpoint_save_variant_real_multiframe_traceback() -> None:
    """The write mode is only visible in the repo frame (torch.save), not the innermost one."""
    c = classify(_content(TORCH_SAVE_LOG), CTX)
    assert (c.cls, c.pattern, c.params["dir"]) == ("known", "missing_output_dir", "checkpoints")
    assert c.params["line"].startswith("torch.save(")
    single = _tb(
        "torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')",
        "FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/cifar10_resnet18.pt'",
    )
    assert classify(_content(single), CTX).params["dir"] == "checkpoints"


def test_library_read_under_repo_call_is_not_a_write() -> None:
    log = (
        "Traceback (most recent call last):\n"
        '  File "train.py", line 5, in <module>\n'
        "    m = torch.load('ckpt/x.pt')\n"
        '  File "/opt/conda/lib/python3.7/site-packages/torch/serialization.py", line 1, in load\n'
        "    with open(name, mode) as f:\n"
        "FileNotFoundError: [Errno 2] No such file or directory: 'ckpt/x.pt'\n"
    )
    assert classify(_content(log), CTX).cls == "critic"


@pytest.mark.parametrize(
    "line",
    [
        "open(path, 'w')", 'open(p, mode="ab")', "with open(fn, 'x') as f:", "np.save(p, a)",
        "df.to_csv(p)", "plt.savefig(p)", "json.dump(obj, open(p, 'w'))", "torch.save(m, p)",
    ],
)  # fmt: skip
def test_write_calls(line: str) -> None:
    assert is_write_call(line)


@pytest.mark.parametrize("line", ["open(path)", "open(p, 'rb')", "np.load(p)", "x = 1"])
def test_read_calls(line: str) -> None:
    assert not is_write_call(line)


def test_existing_dir_or_read_is_not_missing_output_dir() -> None:
    log = _tb(
        "open('util/new.txt', 'w')",
        "FileNotFoundError: [Errno 2] No such file or directory: 'util/new.txt'",
    )
    assert classify(_content(log), CTX).cls == "critic"  # util/ exists in the clone
    log = _tb(
        "open('logs/x.csv')", "FileNotFoundError: [Errno 2] No such file or directory: 'logs/x.csv'"
    )
    assert classify(_content(log), CTX).cls == "critic"  # a read, not a write
    log = _tb(
        "open('x.csv', 'w')", "FileNotFoundError: [Errno 2] No such file or directory: 'x.csv'"
    )
    assert classify(_content(log), CTX).cls == "critic"  # cwd exists


def test_data_path_under_work_data_is_not_missing_output_dir() -> None:
    log = _tb(
        "open('data/out/x.npy', 'wb')",
        "FileNotFoundError: [Errno 2] No such file or directory: 'data/out/x.npy'",
    )
    c = classify(_content(log), CTX)
    assert (c.cls, c.pattern) == ("unfixable", "write_to_data_mount")
    log = _tb(
        "np.save('/work/data/a.npy', x)",
        "FileNotFoundError: [Errno 2] No such file or directory: '/work/data/a.npy'",
    )
    assert classify(_content(log), CTX).pattern == "write_to_data_mount"


def test_missing_data_path() -> None:
    msg = "FileNotFoundError: [Errno 2] No such file or directory: 'svhn_root/train_32x32.mat'"
    c = classify(_content(_tb("loadmat(path)", msg)), CTX)
    assert (c.cls, c.pattern, c.params["name"], c.params["container"]) == (
        "known", "missing_data_path", "svhn", "/work/svhn_root",
    )  # fmt: skip
    msg = "FileNotFoundError: [Errno 2] No such file or directory: 'imgs/a.jpg'"
    c = classify(_content(_tb("Image.open(p)", msg)), CTX)
    assert (c.cls, c.pattern) == ("unfixable", "missing_data_path")


@pytest.mark.parametrize(
    "exc",
    [
        "RuntimeError: CUDA error: unknown error",
        "RuntimeError: Unexpected error from cudaGetDeviceCount(). cuInit returned 999",
        "RuntimeError: CUDA error: no kernel image is available for execution on the device",
    ],
)
def test_cuda_init_is_infra(exc: str) -> None:
    c = classify(_content(_tb("cnn = cnn.cuda()", exc)), CTX)
    assert (c.cls, c.pattern) == ("infra", "infra")


def test_infra_signal() -> None:
    c = classify(_content("driver not loaded\n", signal="infra", exit_code=1), CTX)
    assert c.cls == "infra"


def test_oom_batch_halving_params() -> None:
    log = _tb("pred = cnn(images)", "RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB")
    c = classify(_content(log, signal="oom", command=CMD + " --batch_size 64"), CTX)
    assert (c.cls, c.pattern) == ("known", "oom")
    assert c.params == {"option": "--batch_size", "dest": "batch_size", "current": 64}
    c = classify(_content(log, signal="oom"), CTX)
    assert c.params["current"] == 128  # argparse default
    no_batch = CTX.model_copy(update={"arguments": ()})
    assert classify(_content(log, signal="oom"), no_batch).cls == "unfixable"


def test_oom_with_accumulation_arg() -> None:
    args = (*CUTOUT_ARGS, BudgetArg(option_strings=("--accum_iter",), dest="accum_iter", default=1))
    log = _tb("loss.backward()", "RuntimeError: CUDA out of memory.")
    c = classify(_content(log, signal="oom"), CTX.model_copy(update={"arguments": args}))
    assert c.params["accum_option"] == "--accum_iter" and c.params["accum_current"] == 1


def test_missing_module() -> None:
    log = _tb("import cv2", "ModuleNotFoundError: No module named 'cv2'")
    c = classify(_content(log, signal="import_error"), CTX)
    assert (c.cls, c.pattern, c.params) == (
        "known", "missing_module", {"module": "cv2", "package": "opencv-python"},
    )  # fmt: skip
    log = _tb("import util.nothere", "ModuleNotFoundError: No module named 'util.nothere'")
    assert classify(_content(log, signal="import_error"), CTX).cls == "critic"  # local module


def test_generic_error_is_critic_with_repo_frame() -> None:
    log = _tb(
        "x = model.fc.weight",
        "AttributeError: 'ResNet' object has no attribute 'fc'",
        "/work/model/resnet.py",
    )
    c = classify(_content(log), CTX)
    assert c.cls == "critic" and c.pattern is None
    assert c.params["repo_frame"]["file"] == "/work/model/resnet.py"
    assert "AttributeError" in c.reason


def test_no_traceback_is_unfixable() -> None:
    c = classify(_content("loading...\n", signal="timeout_no_progress", exit_code=None), CTX)
    assert c.cls == "unfixable"
    c = classify(_content("Killed\n", exit_code=137), CTX)
    assert c.cls == "unfixable"


def test_passed_is_rejected() -> None:
    with pytest.raises(ValueError):
        classify({"passed": True}, CTX)
