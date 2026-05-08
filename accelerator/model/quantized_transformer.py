"""Quantized transformer that routes matmuls through the systolic array."""

from __future__ import annotations

import numpy as np

from accelerator.core.fixed_point import dequantize_tensor
from accelerator.core.systolic_array import SystolicArray
from accelerator.simulation.cycle_counter import CycleCounter


class QuantizedTransformer:
    """Quantized transformer inference using systolic array matmuls."""

    def __init__(
        self,
        weights: dict[str, tuple[np.ndarray, float, int]],
        config: dict[str, int],
        array_size: int = 32,
        tile_size: int = 32,
    ) -> None:
        """Initialize quantized weights and systolic array."""
        self.weights = weights
        self.config = config
        self.array = SystolicArray(array_size)
        self.tile_size = tile_size
        self.cycle_counter = CycleCounter()

    def _matmul(self, a: np.ndarray, b: np.ndarray, name: str) -> np.ndarray:
        """Run tiled matmul on INT8 inputs and log cycles."""
        result, cycles = self.array.tiled_matmul(a, b, self.tile_size)
        self.cycle_counter.log_operation(name, cycles, a.shape, result.shape)
        return result

    def _linear(self, x: np.ndarray, weight_name: str) -> np.ndarray:
        """Apply a linear layer using quantized weights."""
        weights_q, w_scale, w_zero = self.weights[weight_name]
        w_int8 = (weights_q.astype(np.int32) - int(w_zero)).astype(np.int8)
        original_shape = x.shape
        if x.ndim == 3:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x
        x_scale = float(np.max(np.abs(x_2d))) / 127.0 if np.max(np.abs(x_2d)) > 0 else 1.0
        x_int8 = np.clip(np.round(x_2d / x_scale), -128, 127).astype(np.int8)
        acc = self._matmul(x_int8, w_int8.T, f"linear:{weight_name}")
        out = acc.astype(np.float32) * (x_scale * w_scale)
        if len(original_shape) == 3:
            out = out.reshape(original_shape[0], original_shape[1], -1)
        return out

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Layer norm in float (hardware offload)."""
        # Layer norm is left in float to model offloaded hardware blocks.
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5)

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """Forward pass over a batch of token indices."""
        self.cycle_counter.reset()
        bsz, seq_len = idx.shape
        n_embd = int(self.config["n_embd"])

        tok_weight, tok_scale, tok_zero = self.weights["token_emb.weight"]
        pos_weight, pos_scale, pos_zero = self.weights["pos_emb.weight"]

        tok = tok_weight[idx] - tok_zero
        pos = pos_weight[np.arange(seq_len)] - pos_zero
        x = dequantize_tensor(tok, tok_scale, 0) + dequantize_tensor(pos, pos_scale, 0)

        for layer in range(int(self.config["n_layers"])):
            ln1 = self._layer_norm(x)
            qkv = self._linear(ln1, f"blocks.{layer}.attn.qkv.weight")
            q, k, v = np.split(qkv, 3, axis=-1)

            head_dim = n_embd // int(self.config["n_heads"])
            q = q.reshape(bsz, seq_len, -1, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(bsz, seq_len, -1, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, seq_len, -1, head_dim).transpose(0, 2, 1, 3)

            attn_scores = np.zeros((bsz, q.shape[1], seq_len, seq_len), dtype=np.float32)
            for h in range(q.shape[1]):
                qh = q[:, h, :, :].reshape(bsz * seq_len, head_dim)
                kh = k[:, h, :, :].reshape(bsz * seq_len, head_dim)
                q_scale = float(np.max(np.abs(qh))) / 127.0 if np.max(np.abs(qh)) > 0 else 1.0
                k_scale = float(np.max(np.abs(kh))) / 127.0 if np.max(np.abs(kh)) > 0 else 1.0
                qh_int8 = np.clip(np.round(qh / q_scale), -128, 127).astype(np.int8)
                kh_int8 = np.clip(np.round(kh / k_scale), -128, 127).astype(np.int8)
                scores = self._matmul(qh_int8, kh_int8.T, f"attn_qk:layer{layer}:head{h}")
                scores = scores.reshape(bsz, seq_len, seq_len).astype(np.float32)
                attn_scores[:, h, :, :] = scores * (q_scale * k_scale) / np.sqrt(head_dim)

            mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
            attn_scores = np.where(mask == 0, -1e9, attn_scores)
            # Softmax stays in float to model offloaded hardware blocks.
            attn_probs = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
            attn_probs = attn_probs / np.sum(attn_probs, axis=-1, keepdims=True)

            attn_out = np.zeros_like(q)
            for h in range(q.shape[1]):
                probs = attn_probs[:, h, :, :].reshape(bsz * seq_len, seq_len).astype(np.float32)
                vh = v[:, h, :, :].reshape(bsz * seq_len, head_dim)
                probs_int8 = np.clip(np.round(probs * 127.0), -128, 127).astype(np.int8)
                v_scale = float(np.max(np.abs(vh))) / 127.0 if np.max(np.abs(vh)) > 0 else 1.0
                vh_int8 = np.clip(np.round(vh / v_scale), -128, 127).astype(np.int8)
                attn = self._matmul(probs_int8, vh_int8, f"attn_pv:layer{layer}:head{h}")
                attn_out[:, h, :, :] = attn.reshape(bsz, seq_len, head_dim) * (v_scale / 127.0)

            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, n_embd)
            proj = self._linear(attn_out, f"blocks.{layer}.attn.proj.weight")
            x = x + proj
            ln2 = self._layer_norm(x)

            mlp_fc = self._linear(ln2, f"blocks.{layer}.mlp.0.weight")
            mlp_act = np.maximum(0, mlp_fc)
            mlp_out = self._linear(mlp_act, f"blocks.{layer}.mlp.2.weight")
            x = x + mlp_out

        x = self._layer_norm(x)
        logits = self._linear(x, "head.weight")
        return logits
