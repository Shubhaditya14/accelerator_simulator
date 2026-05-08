"""Quantization utilities for model weights and calibration."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from accelerator.core.fixed_point import quantize_tensor


def calibrate(model: Any, dataloader: Any) -> dict[str, tuple[float, int]]:
    """Compute per-layer symmetric scales using min/max calibration."""
    model.eval()
    layer_stats: dict[str, list[float]] = {}

    for batch in dataloader:
        with np.errstate(all="ignore"):
            _ = model(batch)

        for name, param in model.named_parameters():
            if param.dim() >= 2:
                tensor = param.detach().cpu().numpy()
                layer_stats.setdefault(name, [])
                layer_stats[name].append(float(np.min(tensor)))
                layer_stats[name].append(float(np.max(tensor)))

    scales: dict[str, tuple[float, int]] = {}
    for name, values in layer_stats.items():
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        max_abs = max(abs(min_val), abs(max_val))
        if max_abs == 0.0:
            scale = 1.0
        else:
            scale = max_abs / 127.0
        scales[name] = (float(scale), 0)
    return scales


def quantize_model(model: Any) -> dict[str, tuple[np.ndarray, float, int]]:
    """Quantize model parameters to INT8 and return per-layer data."""
    quantized: dict[str, tuple[np.ndarray, float, int]] = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            weights = param.detach().cpu().numpy()
            q, scale, zero_point = quantize_tensor(weights, bits=8)
            quantized[name] = (q, scale, zero_point)
    return quantized


def export_weights(quantized_model: dict[str, tuple[np.ndarray, float, int]], output_dir: str) -> None:
    """Export quantized weights to raw INT8 binaries plus metadata."""
    os.makedirs(output_dir, exist_ok=True)
    metadata: dict[str, dict[str, Any]] = {}
    for name, (q_weights, scale, zero_point) in quantized_model.items():
        safe_name = name.replace(".", "_")
        bin_path = os.path.join(output_dir, f"{safe_name}.bin")
        q_weights.astype(np.int8).tofile(bin_path)
        metadata[name] = {
            "file": f"{safe_name}.bin",
            "shape": list(q_weights.shape),
            "scale": float(scale),
            "zero_point": int(zero_point),
        }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_weights(weight_dir: str) -> dict[str, tuple[np.ndarray, float, int]]:
    """Load quantized weights and metadata from disk."""
    meta_path = os.path.join(weight_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    weights: dict[str, tuple[np.ndarray, float, int]] = {}
    for name, entry in metadata.items():
        bin_path = os.path.join(weight_dir, entry["file"])
        data = np.fromfile(bin_path, dtype=np.int8)
        shape = tuple(entry["shape"])
        weights[name] = (data.reshape(shape), float(entry["scale"]), int(entry["zero_point"]))
    return weights
