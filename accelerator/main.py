"""Entry point for the accelerator simulation pipeline."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import os

import numpy as np

from accelerator.benchmark.benchmark import run_benchmark
from accelerator.benchmark.visualize import plot_results
from accelerator.core.quantization import export_weights, quantize_model
from accelerator.model.transformer import load_transformer, train_transformer
from accelerator.simulation.accelerator_inference import run_accelerator_inference


def main() -> None:
    """Run training, quantization, inference, benchmarking, and visualization."""
    parser = argparse.ArgumentParser(description="Systolic array accelerator simulator")
    parser.add_argument("--skip-training", action="store_true", help="Skip training if checkpoint exists")
    args = parser.parse_args()

    np.random.seed(42)

    checkpoint_path = os.path.join("accelerator", "model", "checkpoint.pt")
    data_dir = os.path.join("accelerator", "data")
    weights_dir = os.path.join("accelerator", "model", "quantized_weights")

    if os.path.exists(checkpoint_path) and args.skip_training:
        print("Checkpoint found, skipping training.")
        model = load_transformer(checkpoint_path, device="cpu")
    else:
        model = train_transformer(checkpoint_path, data_dir)

    print("Quantizing model...")
    quantized = quantize_model(model)
    export_weights(quantized, weights_dir)
    print("Quantization complete.")

    stats = run_accelerator_inference(checkpoint_path, weights_dir)
    print("Note: Python simulation wall-clock is slower than CPU; use cycle-based latency.")
    print(f"Estimated latency at 12MHz: {stats['latency_ms_12mhz']:.2f} ms")

    benchmark_results = run_benchmark(checkpoint_path, weights_dir, accuracy=stats["accuracy"])
    benchmark_results["logs"] = stats["logs"]
    plot_results(benchmark_results, os.path.join("accelerator", "benchmark", "plots"))

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
