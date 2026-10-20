# Griffin, Section 2.3 "Temporal-mixing blocks": global MQA and local (sliding-window) MQA.
# Official code: LocalAttentionBlock and _apply_rope in
# https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/modules.py#L53-L90
# https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/modules.py#L294-L475
import math
from typing import NamedTuple

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


def apply_rope(
    x: Float[Tensor, "*batch T D_head"],
    positions: Int[Tensor, " T"],
    max_wavelength: int = 10_000,
) -> Float[Tensor, "*batch T D_head"]:
    """Rotary Position Embedding (RoPE, Su et al., 2021) on the last dim of x.

    x: queries or keys. positions: position of each token.
    Pair i = (x[i], x[i + D_head/2]) is rotated by the angle positions * theta_i,
    with theta_i = max_wavelength ** (-2 i / D_head). The rotation keeps the norm of x,
    and q_m . k_n then depends only on m - n: RoPE encodes the relative position.
    The official code rotates only the first half of each head and leaves the rest as is.
    """
    device, dtype = x.device, x.dtype
    D_head = x.shape[-1]  # width of one head: the last dim of q or k, 128 in the paper
    half = D_head // 2  # each pair (x[i], x[i + half]) is one rotation plane
    i = torch.arange(half, device=device)
    # the angle stays in float32, as in the official code: in bfloat16 it would be too coarse
    theta: Float[Tensor, " half"] = max_wavelength ** (-2 * i / D_head)
    angle: Float[Tensor, "T half"] = positions[:, None] * theta
    sin, cos = torch.sin(angle).to(dtype), torch.cos(angle).to(dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class KVCache(NamedTuple):
    """Cache of one local MQA block for generation.

    k, v: keys (after RoPE) and values of the last L = min(n, window_size) tokens.
    n: number of tokens seen so far, which is the position of the next token.
    """
    k: Float[Tensor, "B 1 L D_head"]
    v: Float[Tensor, "B 1 L D_head"]
    n: int


class Local_MQA_block(nn.Module):
    """Local (sliding-window) multi-query attention, Griffin Section 2.3.

    MQA (Shazeer, 2019): H query heads share one key head and one value head, so the
    KV cache is H times smaller than in MHA. Paper: D_head = 128 and H = D / D_head, so
    D must be a multiple of 128; RoPE and no absolute position embedding; the window
    size is 1024 tokens (Section 3). Simple version: the KV cache keeps the last
    window_size tokens by slicing (the official code uses a ring buffer), and the full
    (T, S) score matrix is computed and masked, with S = cached tokens + T.
    """

    def __init__(self, D: int, D_head: int = 128, window_size: int = 1024):
        super().__init__()
        assert D % D_head == 0, "D must be a multiple of D_head"
        self.D = D
        self.D_head = D_head
        self.H = D // D_head
        self.window_size = window_size
        # same as the official code: no bias on q, k, v, bias on the final projection
        self.Wq = nn.Linear(D, D, bias=False)  # H heads of D_head: H * D_head = D
        self.Wk = nn.Linear(D, D_head, bias=False)  # one key head, shared by all query heads
        self.Wv = nn.Linear(D, D_head, bias=False)  # one value head
        self.Wo = nn.Linear(D, D, bias=True)

    def forward(self, x: Float[Tensor, "B T D"],
                cache: KVCache | None = None,
                ) -> tuple[Float[Tensor, "B T D"], KVCache]:
        # Cache for generation: K and V of the last window_size tokens, and n, the number of tokens before x.
        # Prefill: cache = None, n = 0. Decode: x holds one token, at position n.
        device, dtype = x.device, x.dtype
        B, T, _ = x.shape
        n = 0 if cache is None else cache.n
        positions: Int[Tensor, " T"] = torch.arange(n, n + T, device=device)

        q: Float[Tensor, "B H T D_head"] = self.Wq(x).view(B, T, self.H, self.D_head).transpose(1, 2)
        k: Float[Tensor, "B 1 T D_head"] = self.Wk(x).unsqueeze(1)  # the one head, broadcast over H below
        v: Float[Tensor, "B 1 T D_head"] = self.Wv(x).unsqueeze(1)
        q, k = apply_rope(q, positions), apply_rope(k, positions)

        # S keys: the cached tokens, already rotated at their own positions, then the T tokens of x
        if cache is not None:
            k: Float[Tensor, "B 1 S D_head"] = torch.cat([cache.k, k], dim=2)
            v: Float[Tensor, "B 1 S D_head"] = torch.cat([cache.v, v], dim=2)
        S = k.shape[2]
        k_positions: Int[Tensor, " S"] = torch.arange(n + T - S, n + T, device=device)

        scores: Float[Tensor, "B H T S"] = q @ k.transpose(-1, -2) / math.sqrt(self.D_head)
        # token t sees token s when s <= t (causal) and s >= t - window_size (local)
        t, s = positions[:, None], k_positions[None, :]
        mask: Bool[Tensor, "T S"] = (s <= t) & (s >= t - self.window_size)
        scores = scores.masked_fill(~mask, -torch.inf)
        probs: Float[Tensor, "B H T S"] = torch.softmax(scores.float(), dim=-1).to(dtype)  # softmax in float32, as the official code
        y: Float[Tensor, "B H T D_head"] = probs @ v
        y: Float[Tensor, "B T D"] = y.transpose(1, 2).reshape(B, T, self.D)  # concatenate the heads
        y = self.Wo(y)

        # the next token sees window_size tokens back. k[:, :, -window_size:] is a view:
        # a cache that holds it keeps the whole k in memory.
        k_cache: Float[Tensor, "B 1 L D_head"] = k[:, :, -self.window_size:].clone()
        v_cache: Float[Tensor, "B 1 L D_head"] = v[:, :, -self.window_size:].clone()
        new_cache = KVCache(k_cache, v_cache, n + T)

        return y, new_cache
