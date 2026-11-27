"""Tests for reproducer.domain.traceback_parse."""

from __future__ import annotations

from reproducer.domain.traceback_parse import RAW_MAX, Frame, parse_last_traceback

SIMPLE = """\
Namespace(dataset='cifar10')
Traceback (most recent call last):
  File "train.py", line 120, in <module>
    train_dataset = datasets.CIFAR10(root='data/', download=True)
  File "/opt/conda/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 66, in __init__
    raise RuntimeError("Dataset not found or corrupted.")
RuntimeError: Dataset not found or corrupted. You can use download=True to download it
"""

CHAINED = """\
Traceback (most recent call last):
  File "/work/util/misc.py", line 5, in load
    return d[k]
KeyError: 'x'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "train.py", line 10, in <module>
    main()
  File "/work/model/resnet.py", line 42, in forward
    out = self.layer(x)
  File "/opt/conda/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call
    return forward_call(*input, **kwargs)
ValueError: shapes do not match
"""

OOM = """\
 12%|#2        | 47/391 [00:05<00:38,  9.0it/s]\rTraceback (most recent call last):
  File "train.py", line 199, in <module>
    pred = cnn(images)
  File "/work/model/resnet.py", line 88, in forward
    out = F.relu(self.bn1(self.conv1(x)))
RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 7.79 GiB total capacity)
"""

CUDA = """\
Traceback (most recent call last):
  File "train.py", line 154, in <module>
    cnn = cnn.cuda()
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call.
"""


def test_simple_repo_frame_relative_path() -> None:
    tb = parse_last_traceback(SIMPLE)
    assert tb is not None
    assert tb.exc_type == "RuntimeError" and tb.kind == "python"
    assert tb.message.startswith("Dataset not found or corrupted.")
    assert len(tb.frames) == 2
    assert tb.repo_frame == Frame(file="train.py", line=120, func="<module>")  # innermost repo
    assert tb.raw.startswith("Traceback (most recent call last):")


def test_chained_uses_last_and_innermost_repo_frame() -> None:
    tb = parse_last_traceback(CHAINED)
    assert tb is not None
    assert (tb.exc_type, tb.message) == ("ValueError", "shapes do not match")
    assert [f.file for f in tb.frames][0] == "train.py"
    assert tb.repo_frame == Frame(file="/work/model/resnet.py", line=42, func="forward")
    assert "KeyError" not in tb.raw


def test_oom_with_tqdm_carriage_return() -> None:
    tb = parse_last_traceback(OOM)
    assert tb is not None and tb.kind == "oom" and tb.exc_type == "RuntimeError"
    assert tb.repo_frame is not None and tb.repo_frame.file == "/work/model/resnet.py"


def test_cuda_error_multiline_message() -> None:
    tb = parse_last_traceback(CUDA)
    assert tb is not None and tb.kind == "cuda"
    assert tb.message == "CUDA error: no kernel image is available for execution on the device"
    assert "asynchronously" in tb.raw  # following line kept as context, not in message


def test_interleaved_stdout_after_exception_not_in_message() -> None:
    log = (
        "Traceback (most recent call last):\n"
        '  File "/work/h.py", line 2, in boom\n'
        "ValueError: bad\n"
        "start\n"
    )
    tb = parse_last_traceback(log)
    assert tb is not None and tb.message == "bad" and tb.raw.endswith("start")


def test_data_and_site_packages_frames_are_not_repo() -> None:
    log = (
        "Traceback (most recent call last):\n"
        '  File "/work/data/evil.py", line 1, in <module>\n'
        '  File "<frozen importlib._bootstrap>", line 2, in x\n'
        "ImportError: nope\n"
    )
    tb = parse_last_traceback(log)
    assert tb is not None and tb.repo_frame is None and tb.exc_type == "ImportError"


def test_no_traceback() -> None:
    assert parse_last_traceback("epoch 1 done\ntest_acc: 0.4\n") is None
    assert parse_last_traceback("all good\n", exit_code=0) is None


def test_killed_and_137() -> None:
    tb = parse_last_traceback(" 40%|####  | 10/25\nKilled\n")
    assert tb is not None and tb.kind == "killed" and tb.exc_type == "Killed"
    tb = parse_last_traceback("training...\n", exit_code=137)
    assert tb is not None and tb.kind == "killed"


def test_nonzero_exit_without_traceback() -> None:
    log = "usage: train.py [-h]\ntrain.py: error: unrecognized arguments: --bogus\n"
    tb = parse_last_traceback(log, exit_code=2)
    assert tb is not None and tb.kind == "exit"
    assert tb.message == "train.py: error: unrecognized arguments: --bogus"


def test_exception_without_message_and_raw_cap() -> None:
    frames = "".join(f'  File "/work/m{i}.py", line {i}, in f\n    x()\n' for i in range(400))
    tb = parse_last_traceback(
        "Traceback (most recent call last):\n" + frames + "KeyboardInterrupt\n"
    )
    assert tb is not None and tb.exc_type == "KeyboardInterrupt" and tb.message == ""
    assert (
        len(tb.frames) == 400
        and tb.repo_frame is not None
        and tb.repo_frame.file == "/work/m399.py"
    )
    assert len(tb.raw.encode()) <= RAW_MAX and tb.raw.endswith("KeyboardInterrupt")


def test_header_after_progress_bar_without_newline() -> None:
    log = (
        "\r100%|##| 5/5 [00:05<00:00, 1.00it/s]Traceback (most recent call last):\n"
        '  File "/work/train.py", line 9, in <module>\n'
        "    open('ckpt/model.pt', 'wb').write(b'0')\n"
        "FileNotFoundError: [Errno 2] No such file or directory: 'ckpt/model.pt'\n"
    )
    tb = parse_last_traceback(log, exit_code=1)
    assert tb is not None and tb.kind == "python" and tb.exc_type == "FileNotFoundError"
    assert tb.raw.startswith("Traceback (most recent call last):")
