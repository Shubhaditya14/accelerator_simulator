"""Benchmark CPU baseline vs simulated accelerator."""

from __future__ import annotations

import time

import numpy as np
import torch

from accelerator.model.transformer import load_transformer, TransformerConfig
from accelerator.core.quantization import load_weights
from accelerator.model.quantized_transformer import QuantizedTransformer


def _cpu_baseline(model: torch.nn.Module, inputs: np.ndarray, runs: int) -> dict[str, float]:
    """Measure CPU-only inference latency for the FP32 model."""
    torch.backends.mps.enabled = False
    model.eval()
    torch.set_num_threads(1)
    timings = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(torch.from_numpy(inputs))
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)
    mean = float(np.mean(timings))
    std = float(np.std(timings))
    throughput = 1000.0 / mean if mean > 0 else 0.0
    return {"mean_ms": mean, "std_ms": std, "throughput": throughput}


def _accelerator_sim(
    weights_dir: str,
    inputs: np.ndarray,
    runs: int,
    array_size: int,
    tile_size: int,
) -> dict[str, float | int]:
    """Measure simulated accelerator wall-clock and estimated latency."""
    weights = load_weights(weights_dir)
    config = TransformerConfig().__dict__
    model = QuantizedTransformer(weights, config, array_size, tile_size)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.forward(inputs)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    mean = float(np.mean(timings))
    std = float(np.std(timings))
    throughput = 1000.0 / mean if mean > 0 else 0.0
    cycles = model.cycle_counter.total_cycles()
    logs = [
        {
            "name": log.name,
            "cycles": log.cycles,
            "input_shape": log.input_shape,
            "output_shape": log.output_shape,
        }
        for log in model.cycle_counter.logs
    ]
    return {
        "mean_ms": mean,
        "std_ms": std,
        "throughput": throughput,
        "cycles": cycles,
        "latency_ms_12mhz": model.cycle_counter.estimate_latency_ms(12.0),
        "latency_ms_100mhz": model.cycle_counter.estimate_latency_ms(100.0),
        "logs": logs,
    }


def run_benchmark(
    checkpoint_path: str,
    weights_dir: str,
    runs: int = 10,
    array_size: int = 32,
    tile_size: int = 32,
    accuracy: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """Benchmark CPU and simulated accelerator on identical inputs."""
    config = TransformerConfig()
    inputs = np.random.randint(0, config.vocab_size, (1, config.block_size))

    print("Running CPU baseline (PyTorch, CPU only)...")
    fp32_model = load_transformer(checkpoint_path, device="cpu")
    cpu_stats = _cpu_baseline(fp32_model, inputs, runs)

    print("Running simulated accelerator benchmark...")
    sim_stats = _accelerator_sim(weights_dir, inputs, runs, array_size, tile_size)
    sim_stats["accuracy"] = accuracy

    print("| Metric              | CPU (M1 Max)  | FPGA Sim (12MHz) | FPGA Sim (100MHz) |")
    print("|---------------------|---------------|------------------|-------------------|")
    print(
        f"| Latency (ms)        | {cpu_stats['mean_ms']:.2f}        | {sim_stats['latency_ms_12mhz']:.2f}             | {sim_stats['latency_ms_100mhz']:.2f}              |"
    )
    print(
        f"| Throughput (inf/s)  | {cpu_stats['throughput']:.2f}        | {1000.0 / sim_stats['latency_ms_12mhz']:.2f}             | {1000.0 / sim_stats['latency_ms_100mhz']:.2f}              |"
    )
    print(
        f"| Cycle count         | N/A           | {sim_stats['cycles']}            | {sim_stats['cycles']}             |"
    )
    print(
        f"| Accuracy            | baseline      | {sim_stats['accuracy'] * 100:.2f}%             | {sim_stats['accuracy'] * 100:.2f}%              |"
    )

    return {"cpu": cpu_stats, "sim": sim_stats, "logs": sim_stats["logs"]}
