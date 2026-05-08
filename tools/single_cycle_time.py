"""Estimate wall-clock time per simulated cycle."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from accelerator.core.quantization import load_weights
from accelerator.model.quantized_transformer import QuantizedTransformer
from accelerator.model.transformer import TransformerConfig


def _load_config(checkpoint_path: str) -> tuple[dict[str, int], str]:
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config")
        if isinstance(config, dict):
            return config, "checkpoint"
    return TransformerConfig().__dict__, "default"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure wall time per simulated accelerator cycle",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("accelerator", "model", "checkpoint.pt"),
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--weights-dir",
        default=os.path.join("accelerator", "model", "quantized_weights"),
        help="Directory with quantized weights",
    )
    parser.add_argument("--array-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--clock-mhz", type=float, default=12.0)
    args = parser.parse_args()

    meta_path = os.path.join(args.weights_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit(
            "Quantized weights not found. Run accelerator/main.py first."
        )

    weights = load_weights(args.weights_dir)
    config, source = _load_config(args.checkpoint)

    vocab_size = int(config["vocab_size"])
    block_size = int(config["block_size"])
    seq_len = block_size if args.seq_len is None else int(args.seq_len)
    if seq_len > block_size:
        raise SystemExit("seq-len must be <= block_size")

    model = QuantizedTransformer(weights, config, args.array_size, args.tile_size)
    idx = np.random.randint(0, vocab_size, (args.batch, seq_len))

    start = time.perf_counter()
    _ = model.forward(idx)
    end = time.perf_counter()

    wall_s = end - start
    cycles = model.cycle_counter.total_cycles()
    wall_ms = wall_s * 1000.0

    print(f"Config source: {source}")
    print(f"Cycles: {cycles}")
    print(f"Wall time: {wall_ms:.3f} ms")

    if cycles <= 0:
        print("No cycles recorded.")
        return

    wall_ns_per_cycle = (wall_s / cycles) * 1e9
    sim_ns_per_cycle = 1e3 / args.clock_mhz
    slowdown = wall_ns_per_cycle / sim_ns_per_cycle
    sim_latency_ms = cycles / (args.clock_mhz * 1e6) * 1000.0

    print(f"Wall time per simulated cycle: {wall_ns_per_cycle:.3f} ns")
    print(f"Sim cycle time @ {args.clock_mhz:.2f} MHz: {sim_ns_per_cycle:.3f} ns")
    print(f"Slowdown vs {args.clock_mhz:.2f} MHz: {slowdown:.1f}x")
    print(f"Estimated simulated latency: {sim_latency_ms:.3f} ms")


if __name__ == "__main__":
    main()
