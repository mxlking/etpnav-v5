from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_torch_checkpoint(path: str | Path, *args: Any, **kwargs: Any):
    load_kwargs = dict(kwargs)
    load_kwargs.setdefault("weights_only", False)
    try:
        return torch.load(str(path), *args, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        return torch.load(str(path), *args, **load_kwargs)
