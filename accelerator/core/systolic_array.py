"""Cycle-accurate N x N systolic array simulator."""

from __future__ import annotations

import numpy as np

from accelerator.core.pe import ProcessingElement


class SystolicArray:
    """Weight-stationary systolic array with output-stationary accumulators."""

    def __init__(self, size: int) -> None:
        """Create an N x N array of processing elements."""
        self.size = int(size)
        self.pes = [[ProcessingElement() for _ in range(self.size)] for _ in range(self.size)]

    def reset(self) -> None:
        """Reset all processing elements in the array."""
        for row in self.pes:
            for pe in row:
                pe.reset()

    def matmul(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
        """Multiply two INT8 matrices using cycle-accurate simulation."""
        assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D matrices."
        assert a.dtype == np.int8 and b.dtype == np.int8, "Inputs must be INT8."
        m, k_a = a.shape
        k_b, n = b.shape
        assert k_a == k_b, "Inner dimensions must match."
        assert m <= self.size and n <= self.size, "Matrix dimensions exceed array size."

        self.reset()
        a_pipe = [[0 for _ in range(self.size)] for _ in range(self.size)]
        b_pipe = [[0 for _ in range(self.size)] for _ in range(self.size)]

        # Skewing example for K=3:
        # cycle 0: feed A[0,0] to row0, B[0,0] to col0
        # cycle 1: feed A[0,1], A[1,0] and B[1,0], B[0,1]
        # cycle 2: feed A[0,2], A[1,1], A[2,0] and B[2,0], B[1,1], B[0,2]

        total_cycles = m + n + k_a - 2
        for t in range(total_cycles):
            left_in = [0 for _ in range(self.size)]
            top_in = [0 for _ in range(self.size)]

            for i in range(m):
                k_idx = t - i
                if 0 <= k_idx < k_a:
                    left_in[i] = int(a[i, k_idx])

            for j in range(n):
                k_idx = t - j
                if 0 <= k_idx < k_a:
                    top_in[j] = int(b[k_idx, j])

            for i in range(self.size):
                a_pipe[i][0] = left_in[i]
            for j in range(self.size):
                b_pipe[0][j] = top_in[j]

            next_a = [[0 for _ in range(self.size)] for _ in range(self.size)]
            next_b = [[0 for _ in range(self.size)] for _ in range(self.size)]

            for i in range(self.size):
                for j in range(self.size):
                    a_out, b_out = self.pes[i][j].tick(int(a_pipe[i][j]), int(b_pipe[i][j]))
                    if j + 1 < self.size:
                        next_a[i][j + 1] = int(a_out)
                    if i + 1 < self.size:
                        next_b[i + 1][j] = int(b_out)

            a_pipe = next_a
            b_pipe = next_b

        result = np.zeros((m, n), dtype=np.int32)
        for i in range(m):
            for j in range(n):
                result[i, j] = int(self.pes[i][j].accumulator)

        expected = (a.astype(np.int32) @ b.astype(np.int32))
        if not np.allclose(result, expected, atol=0, rtol=0):
            diff = np.max(np.abs(result - expected))
            raise AssertionError(
                f"Systolic array output mismatch: max diff {diff}"
            )
        return result, total_cycles

    def tiled_matmul(self, a: np.ndarray, b: np.ndarray, tile_size: int) -> tuple[np.ndarray, int]:
        """Multiply matrices larger than the array using tiling."""
        assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D matrices."
        assert a.dtype == np.int8 and b.dtype == np.int8, "Inputs must be INT8."
        m, k_a = a.shape
        k_b, n = b.shape
        assert k_a == k_b, "Inner dimensions must match."
        tile = int(tile_size)
        assert tile <= self.size, "Tile size must fit within the array."

        result = np.zeros((m, n), dtype=np.int32)
        total_cycles = 0
        for i0 in range(0, m, tile):
            for j0 in range(0, n, tile):
                tile_acc = np.zeros((min(tile, m - i0), min(tile, n - j0)), dtype=np.int32)
                for k0 in range(0, k_a, tile):
                    a_tile = a[i0:i0 + tile, k0:k0 + tile]
                    b_tile = b[k0:k0 + tile, j0:j0 + tile]
                    partial, cycles = self.matmul(a_tile, b_tile)
                    tile_acc += partial
                    total_cycles += cycles
                result[i0:i0 + tile_acc.shape[0], j0:j0 + tile_acc.shape[1]] = tile_acc
        return result, total_cycles
