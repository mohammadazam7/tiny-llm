"""The model: a decoder-only transformer small enough to read in one sitting.

Token embeddings plus learned position embeddings go through ``layers`` pre-norm blocks.
Each block is causal multi-head self-attention followed by a GELU MLP, both added back
to the residual stream. A final layer norm and a linear head give one logit per
vocabulary entry at every position. The head shares its weight matrix with the token
embedding, so a token is read out with the same vectors it is read in with.

Attention is written out by hand instead of calling the library routine, so the mask is
visible in the code, the tests can check it, and the weights can be returned for
inspection.

Shapes in comments: B batch, T tokens in the sequence, W width, H heads, V vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MLP_RATIO = 4
INIT_STD = 0.02


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context: int = 128
    layers: int = 4
    heads: int = 4
    width: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context", "layers", "heads", "width"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1")
        if self.width % self.heads:
            raise ConfigError(
                f"width {self.width} is not divisible by heads {self.heads}; "
                "every head needs the same number of dimensions"
            )
        if not 0 <= self.dropout < 1:
            raise ConfigError("dropout must be at least 0 and below 1")

    def parameter_count(self) -> int:
        """Trainable parameters, worked out by hand so a test can hold the model to it.

        Embeddings: V*W for tokens (the output head reuses them and has no bias) and
        context*W for positions. One block: two layer norms (2W each), the query, key
        and value projection (3W*W + 3W), the attention output projection (W*W + W)
        and the MLP (W*4W + 4W going up, 4W*W + W coming down), which is 12*W*W + 13*W.
        The final layer norm adds 2W.
        """
        w = self.width
        embeddings = self.vocab_size * w + self.context * w
        block = 12 * w * w + 13 * w
        return embeddings + self.layers * block + 2 * w


class CausalSelfAttention(nn.Module):
    """Every position looks at itself and at earlier positions, never at later ones."""

    mask: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        # True where position i (row) may look at position j (column): j <= i.
        allowed = torch.tril(torch.ones(config.context, config.context, dtype=torch.bool))
        self.register_buffer("mask", allowed, persistent=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """The attention output (B, T, W) and the attention weights (B, H, T, T)."""
        batch, tokens, width = x.shape
        per_head = width // self.heads
        q, k, v = (
            part.view(batch, tokens, self.heads, per_head).transpose(1, 2)  # B, H, T, W/H
            for part in self.qkv(x).split(width, dim=2)
        )
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(per_head)  # B, H, T, T
        # A masked score of -inf becomes a weight of exactly 0 after the softmax.
        scores = scores.masked_fill(~self.mask[:tokens, :tokens], float("-inf"))
        weights = scores.softmax(dim=-1)
        mixed = self.attention_dropout(weights) @ v  # B, H, T, W/H
        out = mixed.transpose(1, 2).reshape(batch, tokens, width)
        return self.residual_dropout(self.proj(out)), weights


class Block(nn.Module):
    """Pre-norm: normalise, transform, add back. The residual path is never normalised,
    which keeps gradients flowing through deep stacks without a warmup trick."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.width)
        self.up = nn.Linear(config.width, MLP_RATIO * config.width)
        self.down = nn.Linear(MLP_RATIO * config.width, config.width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attended, weights = self.attention(self.norm_attention(x))
        x = x + attended
        x = x + self.dropout(self.down(F.gelu(self.up(self.norm_mlp(x)))))
        return x, weights


class TinyLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight  # tied: one matrix, two uses
        self._initialise()

    def _initialise(self) -> None:
        """Small normal weights and zero biases. The two layers that write into the
        residual stream in each block start smaller by sqrt(2 * layers), so the sum of
        all their contributions has about the same scale whatever the depth."""
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        scaled = INIT_STD / math.sqrt(2 * self.config.layers)
        for block in self.blocks:
            assert isinstance(block, Block)
            nn.init.normal_(block.attention.proj.weight, mean=0.0, std=scaled)
            nn.init.normal_(block.down.weight, mean=0.0, std=scaled)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _run(self, tokens: Tensor) -> tuple[Tensor, list[Tensor]]:
        if tokens.dim() != 2:
            raise ValueError(
                f"expected token ids of shape (batch, tokens), got {tuple(tokens.shape)}"
            )
        length = tokens.shape[1]
        if length > self.config.context:
            raise ValueError(
                f"sequence of {length} tokens is longer than the context of {self.config.context}"
            )
        positions = torch.arange(length, device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        attention: list[Tensor] = []
        for block in self.blocks:
            x, weights = block(x)
            attention.append(weights)
        return self.head(self.norm(x)), attention

    def forward(self, tokens: Tensor) -> Tensor:
        """Logits of shape (B, T, V): at each position, a score for every next token."""
        return self._run(tokens)[0]

    def attention(self, tokens: Tensor) -> list[Tensor]:
        """The attention weights of every layer, each (B, H, T, T), for looking inside."""
        return self._run(tokens)[1]

    def loss(self, tokens: Tensor, targets: Tensor) -> Tensor:
        """Mean cross-entropy between the prediction at each position and ``targets``,
        which is ``tokens`` shifted one place to the left."""
        logits = self(tokens)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
