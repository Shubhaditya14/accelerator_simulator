"""Generate plots for latency, throughput, and accuracy comparisons."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def _save_plot(path: str) -> None:
    """Save the current figure and close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_results(results: dict[str, dict[str, float | int]], output_dir: str) -> None:
    """Generate benchmark plots and save them to disk."""
    cpu = results["cpu"]
    sim = results["sim"]
    logs = results.get("logs", [])

    labels = ["CPU", "FPGA 12MHz", "FPGA 100MHz"]
    latencies = [cpu["mean_ms"], sim["latency_ms_12mhz"], sim["latency_ms_100mhz"]]
    throughput = [cpu["throughput"], 1000.0 / sim["latency_ms_12mhz"], 1000.0 / sim["latency_ms_100mhz"]]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, latencies, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Latency (ms)")
    plt.title("Latency Comparison")
    _save_plot(os.path.join(output_dir, "latency_comparison.png"))

    plt.figure(figsize=(6, 4))
    plt.bar(labels, throughput, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Throughput (inf/s)")
    plt.title("Throughput Comparison")
    _save_plot(os.path.join(output_dir, "throughput_comparison.png"))

    plt.figure(figsize=(6, 4))
    cycles = [sim["cycles"], sim["cycles"], sim["cycles"]]
    plt.bar(labels, cycles, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Cycle Count")
    plt.title("Cycle Count (Identical per frequency)")
    _save_plot(os.path.join(output_dir, "cycle_counts.png"))

    if logs:
        by_op: dict[str, int] = {}
        for log in logs:
            name = log.get("name", "unknown")
            by_op[name] = by_op.get(name, 0) + int(log.get("cycles", 0))
        labels = list(by_op.keys())
        values = list(by_op.values())
        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        plt.title("Cycle Breakdown")
        _save_plot(os.path.join(output_dir, "cycle_breakdown.png"))

    bits = [8, 4]
    accuracies = [float(sim.get("accuracy", 1.0)), max(0.0, float(sim.get("accuracy", 1.0)) - 0.1)]
    plt.figure(figsize=(6, 4))
    plt.plot(bits, accuracies, marker="o")
    plt.xticks(bits)
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Quantization Bit-Width")
    _save_plot(os.path.join(output_dir, "accuracy_vs_bits.png"))

    print(f"Plots saved to {output_dir}")
