#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

# We use memory-pinning in CPU and async transfer between CPU and GPU to ensure
# input data transfer is not a bottleneck
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
            for i in starting_idxs
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
            for i in starting_idxs
        ]
    )
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
