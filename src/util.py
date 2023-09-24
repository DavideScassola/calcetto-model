import fnmatch
import importlib.util
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


def load_json(file: str | Path) -> dict:
    with open(file) as json_file:
        d = json.load(json_file)
    return d


def store_json(d: dict, *, file: str | Path):
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


def file_name(file: str | Path) -> str:
    return str(file).split("/")[-1]


def load_module(path: str | Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("module", path)
    if spec == None:
        raise ValueError(f"{path} is not a valid module path")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def create_experiment_folder(*, path: Path, postfix: str | None = None) -> Path:
    postfix = f"_{postfix}" if postfix else ""
    folder_name = Path(datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f") + postfix)
    experiment_folder = path / folder_name
    os.makedirs(experiment_folder)
    return experiment_folder


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def logistic(
    x: torch.Tensor | np.ndarray, *, k: float = 1.0
) -> torch.Tensor | np.ndarray:
    f = torch.exp if isinstance(x, torch.Tensor) else np.exp
    return 1 / (1 + f(-k * x))  # type: ignore


def make_p_more_extreme(
    x: torch.Tensor | np.ndarray, *, k: float = 1.0
) -> torch.Tensor | np.ndarray:
    y0 = logistic(0.0 - 0.5, k=k)
    y1 = logistic(1.0 - 0.5, k=k)
    return (logistic(x - 0.5, k=k) - y0) / (y1 - y0)
