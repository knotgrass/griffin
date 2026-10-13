from jaxtyping import Float, Float32
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Literal
from .rmsnorm import RMSNorm


class Gated_MLP_block(nn.Module):
    def __init__(
        self,
        D: int,
        expansion_factor: int = 3,
        approximate: Literal["none", "tanh"] = "none",
    ) -> None:
        super().__init__()
        self.D = D
        self.M = expansion_factor
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=D, out_features=D*self.M)
        self.p2 = nn.Linear(in_features=D, out_features=D*self.M)
        self.p3 = nn.Linear(in_features=D*self.M, out_features=D)

    def forward(self, x:Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)

        return y


class Temporal_Conv1D(nn.Module):
    """Causal depthwise Conv1D (Griffin, Fig. 2c).

    B: batch size, T: sequence length, D: feature dimension.
    """

    def __init__(self, D: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
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

    def forward(self, x: Float[Tensor, "B T D"]) -> Float[Tensor, "B T D"]:
        x: Float[Tensor, "B D T"] = x.transpose(1, 2)
        # pad LEFT only: causal
        x: Float[Tensor, "B D T_pad"] = F.pad(x, (self.kernel_size - 1, 0))
        x: Float[Tensor, "B D T"] = self.conv(x)
        x: Float[Tensor, "B T D"] = x.transpose(1, 2)
        return x


class Real_Gated_Linear_Recurrent_Unit(nn.Module):
    c = 8.0

    def __init__(self, D: int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.D = D

        # (4) is element-wise: h_t, i_t and x_t have the same width
        self.Wa = nn.Parameter(torch.empty(D, D, **factory_kwargs))
        self.Wx = nn.Parameter(torch.empty(D, D, **factory_kwargs))
        self.ba = nn.Parameter(torch.empty(D, **factory_kwargs))
        self.bx = nn.Parameter(torch.empty(D, **factory_kwargs))
        self.Lambda = nn.Parameter(torch.empty(D, **factory_kwargs))  # Λ
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # https://tinyurl.com/lecuninit
        nn.init.normal_(self.Wa, mean=0, std=1 / (self.D ** 0.5))
        nn.init.normal_(self.Wx, mean=0, std=1 / (self.D ** 0.5))

        # init bias
        # https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/layers.py#L117-L120
        nn.init.zeros_(self.ba)
        nn.init.zeros_(self.bx)

        # init Λ
        nn.init.uniform_(self.Lambda, a=0.9, b=0.999)
        with torch.no_grad():
            self.Lambda.copy_(-torch.log(self.Lambda ** (-1. / self.c) - 1.))

    def forward(self, x:Float32[Tensor, "batch_size sequence_length dim"]
                ) -> Float32[Tensor, "batch_size sequence_length dim"]:

        # Paper, Section 2.4: "The layer has gates on both the input x and the
        # recurrent weight a. However, neither gate depends on the recurrent
        # state h_{t-1}, which ensures that the computation can be executed
        # efficiently on device."
        # So (1), (2) and (6) run on all t at once. Only (4) needs the loop.
        r = torch.sigmoid(F.linear(x, self.Wa, self.ba))  # (1), (batch_size, sequence_length, D)
        i = torch.sigmoid(F.linear(x, self.Wx, self.bx))  # (2)

        # TODO (3)
        # a = torch.sigmoid(self.Lambda) ** (self.c * r)

        # TODO Appendix A - https://github.com/kyegomez/Griffin/issues/6
        log_a = - self.c * F.softplus(-self.Lambda, beta=1, threshold=20) * r  # (6)
        a = torch.exp(log_a)
        b = torch.sqrt(1 - a**2) * (i * x)  # the input term of (4)

        ht = x.new_zeros(x.shape[0], self.D)  # (batch_size, D)
        y = []
        for at, bt in zip(a.unbind(1), b.unbind(1)):  # at, bt: (batch_size, D)
            ht = at * ht + bt  # (4)
            y.append(ht)

        y = torch.stack(y, dim=1)  # (batch_size, sequence_length, D)

        return y


class RGLRU(Real_Gated_Linear_Recurrent_Unit): ...


class Recurrent_block(nn.Module):
    def __init__(self, D:int, D_rnn:int | None = None,
                 approximate:Literal['none', 'tanh']='none'):
        super().__init__()
        D_rnn = D_rnn or int(round(4 * D / 3))
        self.D = D
        self.D_rnn = D_rnn
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p2 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p3 = nn.Linear(in_features=D_rnn, out_features=D)
        self.separableConv1D = Temporal_Conv1D(D_rnn, kernel_size=4)
        self.rglru = RGLRU(D_rnn)

    def forward(self, x:Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)
        x2 = self.separableConv1D(x2)
        x2 = self.rglru(x2)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)
        return y


class Residual_block(nn.Module):
    def __init__(self, D:int):
        super().__init__()
        self.tmb_norm = RMSNorm(d=D)
        self.tmb = Recurrent_block(D, D_rnn=int(round(4 * D / 3)))
        self.mlp_norm = RMSNorm(d=D)
        self.mlp = Gated_MLP_block(D, expansion_factor=3)

    def forward(self, x:Tensor) -> Tensor:
        x1 = self.tmb_norm(x)
        x1 = self.tmb(x1)

        y1 = x + x1

        x2 = self.mlp_norm(y1)
        x2 = self.mlp(x2)

        y2 = y1 + x2

        return y2
