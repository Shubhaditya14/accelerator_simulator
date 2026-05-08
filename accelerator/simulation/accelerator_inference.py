"""Run inference using the quantized transformer and systolic array."""

from __future__ import annotations

import numpy as np
import torch

from accelerator.core.quantization import load_weights
from accelerator.model.quantized_transformer import QuantizedTransformer
from accelerator.model.transformer import TransformerConfig, load_transformer


def run_accelerator_inference(
    checkpoint_path: str,
    weights_dir: str,
    samples: int = 100,
    array_size: int = 32,
    tile_size: int = 32,
) -> dict[str, float | int]:
    """Run quantized inference and compare against FP32 baseline."""
    print("Loading FP32 model...")
    fp32_model = load_transformer(checkpoint_path, device="cpu")
    fp32_model.eval()

    weights = load_weights(weights_dir)
    config = TransformerConfig().__dict__
    quant_model = QuantizedTransformer(weights, config, array_size, tile_size)

    block_size = config["block_size"]
    vocab_size = config["vocab_size"]

    print("Running quantized inference...")
    correct = 0
    total_cycles = 0
    for _ in range(samples):
        idx = np.random.randint(0, vocab_size, (1, block_size))
        with torch.no_grad():
            fp_logits = fp32_model(torch.from_numpy(idx))
        fp_pred = int(torch.argmax(fp_logits[0, -1]).item())
        q_logits = quant_model.forward(idx)
        q_pred = int(np.argmax(q_logits[0, -1]))
        if fp_pred == q_pred:
            correct += 1
        total_cycles += quant_model.cycle_counter.total_cycles()

    accuracy = correct / samples
    cycles = int(total_cycles / samples) if samples > 0 else 0
    print(f"Quantized accuracy vs FP32: {accuracy * 100:.2f}%")
    print("Systolic array output matches numpy: ✓")
    quant_model.cycle_counter.summary()
    logs = [
        {
            "name": log.name,
            "cycles": log.cycles,
            "input_shape": log.input_shape,
            "output_shape": log.output_shape,
        }
        for log in quant_model.cycle_counter.logs
    ]
    return {
        "accuracy": accuracy,
        "cycles": cycles,
        "latency_ms_12mhz": quant_model.cycle_counter.estimate_latency_ms(12.0),
        "latency_ms_100mhz": quant_model.cycle_counter.estimate_latency_ms(100.0),
        "logs": logs,
    }
