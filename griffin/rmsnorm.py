# https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/layers.py#L30-L73
# float32 computation as in Gemma: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L166-L190
import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """RMS Norm, as in the official code: y = x / sqrt(mean(x^2) + eps) * (1 + scale)."""

    def __init__(self, width: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.width = width
        self.eps = eps

        # Parameters.
        self.scale = nn.Parameter(torch.empty([self.width], device=device, dtype=dtype))

        # Initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.scale)

    def forward(self, x: Tensor) -> Tensor:
        # Compute in float32 (float64 stays float64) and cast back, as gemma_pytorch does with x.float():
        # the backward pass needs (var + eps)^(-3/2), which is 1e9 for a zero vector and overflows float16.
        dtype = torch.promote_types(x.dtype, torch.float32)
        x_hi = x.to(dtype)
        var = torch.mean(torch.square(x_hi), axis=-1, keepdims=True)
        normed_x = x_hi * torch.rsqrt(var + self.eps)

        scale = torch.reshape(self.scale, [1 for _ in range(x.ndim - 1)] + [-1])

        return (normed_x * (scale.to(dtype) + 1)).to(x.dtype)
