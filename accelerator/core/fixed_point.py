"""Fixed-point quantization utilities for INT8 simulation."""

from __future__ import annotations

import numpy as np


_OVERFLOW_COUNT = 0


def _int_range(bits: int) -> tuple[int, int]:
    """Return the signed integer range for the given bit width."""
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    return qmin, qmax


def reset_overflow_counter() -> None:
    """Reset the global overflow counter."""
    global _OVERFLOW_COUNT
    _OVERFLOW_COUNT = 0


def get_overflow_count() -> int:
    """Get the number of overflows observed so far."""
    return _OVERFLOW_COUNT


def simulate_overflow(value: int, bits: int = 8) -> int:
    """Clip to the signed INT range and track overflow count."""
    global _OVERFLOW_COUNT
    qmin, qmax = _int_range(bits)
    clipped = int(max(min(int(value), qmax), qmin))
    if clipped != int(value):
        _OVERFLOW_COUNT += 1
    return clipped


def quantize_tensor(tensor: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float, int]:
    """Quantize a float tensor to INT8 using symmetric min/max scaling."""
    qmin, qmax = _int_range(bits)
    t_min = float(np.min(tensor))
    t_max = float(np.max(tensor))
    max_abs = max(abs(t_min), abs(t_max))
    if max_abs == 0.0:
        scale = 1.0
        zero_point = 0
        quantized = np.zeros_like(tensor, dtype=np.int8)
        return quantized, scale, zero_point

    scale = max_abs / float(qmax)
    zero_point = 0
    quantized = np.round(tensor / scale)
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
    return quantized, float(scale), int(zero_point)


def dequantize_tensor(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize an INT8 tensor back to float using scale and zero-point."""
    return (quantized.astype(np.float32) - float(zero_point)) * float(scale)


def requantize(
    acc_int32: np.ndarray,
    input_scale: float,
    weight_scale: float,
    output_scale: float,
) -> np.ndarray:
    """Requantize INT32 accumulators to INT8 using scales."""
    scaled = acc_int32.astype(np.float32) * float(input_scale) * float(weight_scale)
    quantized = np.round(scaled / float(output_scale))
    qmin, qmax = _int_range(8)
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
    return quantized
