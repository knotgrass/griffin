import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
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


class SqrtBoundDerivative(torch.autograd.Function):
    """sqrt(x) whose derivative is bounded by max_gradient (N1).

    d sqrt(x) / dx = 1 / (2 sqrt(x)) = 1 / sqrt(4x) goes to infinity when x goes to 0.
    In (4), x = 1 - a_t^2 is 0 when a_t rounds to 1 (bfloat16, log a_t > -1e-3,
    before N2) or when r_t is exactly 0 (log a_t = 0, also after N2). The backward
    pass then gives inf, and inf * 0 gives NaN further down the graph.

    The backward pass here returns grad / sqrt(max(4x, 1 / max_gradient^2)), so
    1 / sqrt(4x) <= max_gradient. The forward pass is a normal sqrt. Same as the
    official code, which uses _MAX_SQRT_GRADIENT = 1000.0:
    https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/layers.py#L216-L230
    """

    @staticmethod
    def forward(ctx, x: Tensor, max_gradient: float = 1000.0) -> Tensor:
        ctx.save_for_backward(x)
        ctx.max_gradient = max_gradient
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clamp(4.0 * x, min=1.0 / ctx.max_gradient**2)
        return grad_output / torch.sqrt(clipped_x_times_4), None  # None: no grad for max_gradient


class Real_Gated_Linear_Recurrent_Unit(nn.Module):
    c = 8.0

    def __init__(self, D: int, numerically_stable: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.D = D
        # Paper, Section 2.4: "For numerical stability, in practice we compute
        # a^{c r_t} in log-space (see Appendix A)." That is (6) in forward.
        # The paper stops there. This flag adds two fixes to the sqrt term of (4),
        # see input_term:
        # True:  take 1 - a_t**2 from log a_t (N2, ours) and bound the derivative
        #        of the sqrt (N1, from the official recurrentgemma code).
        # False: sqrt(1 - a_t**2) as written in (4), kept to compare.
        self.numerically_stable = numerically_stable

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

    def input_term(self, log_a: Tensor, a: Tensor, i: Tensor, x: Tensor) -> Tensor:
        """The input term of (4): sqrt(1 - a_t^2) * (i_t * x_t), for all t at once.

        In the official code, x * gate_x is `gated_x`, sqrt(1 - a_square) is
        `multiplier`, and the product is `normalized_x`, under the comment
        "Apply gamma normalization to the input":
        https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/layers.py#L346-L353
        The name comes from the LRU paper (Orvieto et al., 2023), where the factor
        sqrt(1 - |lambda|^2) on the input is the normalization parameter gamma.
        """
        if self.numerically_stable:
            # N2: 1 - a**2 = 1 - exp(2 log a) = -expm1(2 log a), with expm1(x) = exp(x) - 1.
            # Note the minus sign: expm1(2 log a) itself is a**2 - 1 <= 0.
            #
            # Why not 1 - a**2: a_t is close to 1 when the recurrence gate r_t is close
            # to 0. Just below 1, float32 numbers are 2**-24 ~ 6e-8 apart, so a_t and
            # a_t**2 keep only a few of the 24 bits that log a_t has, and 1 - a**2
            # returns mostly rounding error. Measured in float32 with log a = -1e-7:
            # 1 - a**2 = 2.384e-7 (19% off), -expm1(2 log a) = 2.000e-7 (4e-6% off).
            # With log a = -1e-8, 1 - a**2 is exactly 0, so the sqrt below gets an
            # infinite derivative. In bfloat16 this already happens at log a = -1e-3.
            #
            # expm1 computes exp(x) - 1 directly, so the value never passes through a
            # number near 1, and the factor 2 is exact in floating point. The official
            # code uses 1 - torch.exp(2 * log_a) and bounds the sqrt derivative instead.
            one_minus_a2 = -torch.expm1(2 * log_a)
            # N1: 1 / (2 sqrt(x)) is infinite at x = 0; bound it at 1000, see SqrtBoundDerivative.
            multiplier = SqrtBoundDerivative.apply(one_minus_a2)
        else:
            one_minus_a2 = 1 - a**2  # as written in (4): loses its digits when a_t is close to 1
            multiplier = torch.sqrt(one_minus_a2)  # derivative inf at 0, so NaN gradients in bfloat16
        return multiplier * (i * x)

    def forward(self, x:Float32[Tensor, "batch_size sequence_length dim"],
                h0:Float32[Tensor, "batch_size dim"] | None = None,
                ) -> tuple[Float32[Tensor, "batch_size sequence_length dim"], Float32[Tensor, "batch_size dim"]]:
        # Cache for generation: h0 is h_T of the previous call.
        # Prefill: h0 = None, the loop starts from h_0 = 0 and runs over the prompt.
        # Decode: x holds one token, h0 is the cache, the loop runs once.

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
        b = self.input_term(log_a, a, i, x)  # sqrt(1 - a_t^2) * (i_t * x_t)

        ht = x.new_zeros(x.shape[0], self.D) if h0 is None else h0  # (batch_size, D)
        y = []
        for at, bt in zip(a.unbind(1), b.unbind(1)):  # at, bt: (batch_size, D)
            ht = at * ht + bt  # (4)
            y.append(ht)

        y = torch.stack(y, dim=1)  # (batch_size, sequence_length, D)

        # ht = h_T, the cache for the next call. It equals y[:, -1], but y[:, -1] is a view:
        # a cache that holds it keeps the whole y in memory.
        return y, ht


class RGLRU(Real_Gated_Linear_Recurrent_Unit): ...
