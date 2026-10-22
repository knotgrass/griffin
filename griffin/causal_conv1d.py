import torch
from jaxtyping import Float
from torch import Tensor, nn


class Temporal_Conv1D(nn.Module):
    """Causal depthwise Conv1D (Griffin, Fig. 2c).

    B: batch size, T: sequence length, D: feature dimension,
    P = kernel_size - 1: the past inputs kept as the cache.
    """

    def __init__(self, D: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.k = kernel_size
        # A separable 1D convolution:
        # - Input channels = output channels = D
        # - groups = D makes it depthwise (channel-wise) convolution.
        self.conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=kernel_size,
            groups=D,
            bias=False,
            padding=0,
        )

    def forward(self, x: Float[Tensor, "B T D"],
                state: Float[Tensor, "B P D"] | None = None,
                ) -> tuple[Float[Tensor, "B T D"], Float[Tensor, "B P D"]]:
        # Cache for generation: state holds the last P inputs of the previous call.
        # Prefill: state = None, the pad is P zeros. Decode: x holds one token, the pad is the cache.
        P = self.k - 1
        if state is None:
            state: Float[Tensor, "B P D"] = x.new_zeros(x.shape[0], P, x.shape[2])

        # pad LEFT only: causal
        x: Float[Tensor, "B T_pad D"] = torch.cat([state, x], dim=1)
        # x[:, -P:] is a view: a cache that holds it keeps the whole x in memory.
        new_state: Float[Tensor, "B P D"] = x[:, -P:].clone()

        x: Float[Tensor, "B D T_pad"] = x.transpose(1, 2)
        x: Float[Tensor, "B D T"] = self.conv(x)
        x: Float[Tensor, "B T D"] = x.transpose(1, 2)

        return x, new_state
