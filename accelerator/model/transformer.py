"""Tiny character-level transformer for baseline FP32 inference."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class TransformerConfig:
    """Configuration for the tiny transformer."""

    vocab_size: int = 65
    n_embd: int = 64
    n_heads: int = 4
    n_layers: int = 2
    block_size: int = 64


class TinyTransformer(nn.Module):
    """Minimal transformer with causal self-attention."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize embeddings, blocks, and output head."""
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.n_embd, config.n_heads)
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=False)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for token indices."""
        bsz, seq_len = idx.shape
        assert seq_len <= self.config.block_size, "Sequence too long."
        pos = torch.arange(0, seq_len, device=idx.device)
        tok = self.token_emb(idx)
        pos = self.pos_emb(pos)[None, :, :]
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, context: torch.Tensor, max_tokens: int) -> torch.Tensor:
        """Generate tokens autoregressively from a context."""
        self.eval()
        out = context
        for _ in range(max_tokens):
            idx_cond = out[:, -self.config.block_size :]
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            out = torch.cat([out, next_token], dim=1)
        return out


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, n_embd: int, n_heads: int) -> None:
        """Initialize the block sublayers."""
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.attn = CausalSelfAttention(n_embd, n_heads)
        self.ln_2 = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP with residual connections."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Causal self-attention with multi-head projections."""

    def __init__(self, n_embd: int, n_heads: int) -> None:
        """Initialize attention projections and dimensions."""
        super().__init__()
        assert n_embd % n_heads == 0, "Embedding dimension must be divisible by heads."
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal attention outputs."""
        bsz, seq_len, n_embd = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, n_embd)
        return self.proj(out)


def download_shakespeare(data_dir: str) -> str:
    """Download the tiny Shakespeare dataset and return the file path."""
    os.makedirs(data_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    out_path = os.path.join(data_dir, "tiny_shakespeare.txt")
    if os.path.exists(out_path):
        return out_path
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return out_path


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-level vocabulary mappings."""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    """Encode text to integer token ids."""
    return np.array([stoi[c] for c in text], dtype=np.int64)


def decode(tokens: np.ndarray, itos: dict[int, str]) -> str:
    """Decode integer token ids back to text."""
    return "".join([itos[int(t)] for t in tokens])


def train_transformer(
    checkpoint_path: str,
    data_dir: str,
    steps: int = 2000,
    target_loss: float = 1.5,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    device: str | None = None,
) -> TinyTransformer:
    """Train the tiny transformer and save a checkpoint."""
    torch.manual_seed(42)
    np.random.seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    text_path = download_shakespeare(data_dir)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    config = TransformerConfig()
    model = TinyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Training tiny transformer...")
    for step in tqdm(range(steps), desc="train", ncols=80):
        idx = torch.randint(0, len(data) - config.block_size - 1, (batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(data[i : i + config.block_size])
                for i in idx.tolist()
            ]
        ).to(device)
        y = torch.stack(
            [
                torch.from_numpy(data[i + 1 : i + config.block_size + 1])
                for i in idx.tolist()
            ]
        ).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 200 == 0:
            print(f"Step {step + 1} | loss {loss.item():.4f}")
        if loss.item() <= target_loss:
            print(f"Target loss reached at step {step + 1}: {loss.item():.4f}")
            break

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "stoi": stoi,
            "itos": itos,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return model


def load_transformer(checkpoint_path: str, device: str | None = None) -> TinyTransformer:
    """Load transformer from checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TransformerConfig(**checkpoint["config"])
    model = TinyTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_vocab(checkpoint_path: str) -> tuple[dict[str, int], dict[int, str]]:
    """Load vocabulary mappings from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint["stoi"], checkpoint["itos"]
