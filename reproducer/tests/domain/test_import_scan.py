from __future__ import annotations

from reproducer.domain.import_scan import (
    MODULE_TO_PIP,
    ImportedPackage,
    normalize_pip_name,
    third_party_imports,
)

# Cutout-like layout: train.py + local packages model/ and util/.
TRAIN = """\
import pdb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout

from model.resnet import ResNet18
from model.wide_resnet import WideResNet
"""
REPO = {
    "README.md": "",
    "train.py": TRAIN,
    "model/__init__.py": "",
    "model/resnet.py": "import torch\nimport torch.nn.functional as F\n",
    "model/wide_resnet.py": "import math\nimport torch\nfrom .resnet import ResNet18\n",
    "util/__init__.py": "",
    "util/misc.py": "import csv\nfrom util.helpers import deep\n",
    "util/cutout.py": "import torch\nimport numpy as np\nimport cv2\n",
    # Only reachable two levels deep (util.misc -> util.helpers): must NOT be scanned.
    "util/helpers.py": "import scipy\n",
}


def _names(scan_packages: tuple[ImportedPackage, ...]) -> list[str]:
    return [p.pip_name for p in scan_packages]


def test_cutout_like_train_py() -> None:
    scan = third_party_imports({"train.py": TRAIN}, REPO)
    assert _names(scan.packages) == ["numpy", "tqdm", "torch", "torchvision", "opencv-python"]
    by_name = {p.pip_name: p for p in scan.packages}
    assert by_name["numpy"].path == "train.py"  # first importer wins
    assert by_name["opencv-python"] == ImportedPackage(
        pip_name="opencv-python", module="cv2", path="util/cutout.py"
    )
    assert "scipy" not in by_name  # not followed beyond one level
    assert not {"pdb", "argparse", "csv", "math", "util", "model"} & set(by_name)
    assert scan.unparsed == ()


def test_module_to_pip_mapping() -> None:
    src = "import PIL.Image\nimport sklearn\nimport yaml\nfrom skimage import io\nimport cv2\n"
    scan = third_party_imports({"run.py": src}, {"run.py": src})
    assert _names(scan.packages) == [
        "pillow",
        "scikit-learn",
        "pyyaml",
        "scikit-image",
        "opencv-python",
    ]
    assert MODULE_TO_PIP["PIL"] == "pillow"


def test_entrypoint_in_subdir_sees_sibling_modules() -> None:
    repo = {
        "scripts/run.py": "import helpers\nimport requests\n",
        "scripts/helpers.py": "import pandas\n",
    }
    scan = third_party_imports({"scripts/run.py": repo["scripts/run.py"]}, repo)
    assert _names(scan.packages) == ["pandas", "requests"]  # depth-first at the import line


def test_unparseable_file_reported_not_guessed() -> None:
    py2 = "import numpy\nprint 'hello'\n"
    scan = third_party_imports({"old.py": py2}, {"old.py": py2})
    assert scan.packages == () and scan.unparsed == ("old.py",)


def test_function_level_and_relative_imports() -> None:
    src = "from . import local\ndef f():\n    import matplotlib.pyplot as plt\n"
    scan = third_party_imports({"pkg/run.py": src}, {"pkg/run.py": src, "pkg/local.py": ""})
    assert _names(scan.packages) == ["matplotlib"]


def test_normalize_pip_name() -> None:
    assert normalize_pip_name("Scikit_Learn") == normalize_pip_name("scikit-learn")
