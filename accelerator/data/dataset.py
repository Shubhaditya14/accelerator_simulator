"""Character-level dataset utilities for tiny Shakespeare."""

from __future__ import annotations

from typing import Iterator

import numpy as np


def batch_iterator(data: np.ndarray, block_size: int, batch_size: int) -> Iterator[np.ndarray]:
    """Yield random batches of token sequences."""
    while True:
        idx = np.random.randint(0, len(data) - block_size - 1, (batch_size,))
        batch = np.stack([data[i : i + block_size] for i in idx])
        yield batch
