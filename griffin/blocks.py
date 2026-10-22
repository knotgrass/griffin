from typing import Literal, NamedTuple

from jaxtyping import Float
from torch import Tensor, nn

from .attention import KVCache, Local_MQA_block
from .causal_conv1d import Temporal_Conv1D
from .rglru import RGLRU
from .rmsnorm import RMSNorm


class Gated_MLP_block(nn.Module):
    def __init__(
        self,
        D: int,
        expansion_factor: int = 3,
        approximate: Literal["none", "tanh"] = "tanh",  # the official code uses gelu(x, approximate="tanh")
    ) -> None:
        super().__init__()
        self.D = D
        self.M = expansion_factor
        # https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/modules.py#L289-L291
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


class RecurrentCache(NamedTuple):
    """Cache of one recurrent block for generation.

    conv_state: the last kernel_size - 1 = 3 inputs of the Conv1D. h: the state h_T of the RG-LRU.
    """
    conv_state: Float[Tensor, "B P D_rnn"]
    h: Float[Tensor, "B D_rnn"]


class Recurrent_block(nn.Module):
    def __init__(self, D:int, D_rnn:int | None = None,
                 approximate:Literal['none', 'tanh']='tanh'):  # the official code uses gelu(x, approximate="tanh")
        super().__init__()
        D_rnn = D_rnn or int(round(4 * D / 3))
        self.D = D
        self.D_rnn = D_rnn
        # https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/modules.py#L289-L291
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p2 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p3 = nn.Linear(in_features=D_rnn, out_features=D)
        self.separableConv1D = Temporal_Conv1D(D_rnn, kernel_size=4)
        self.rglru = RGLRU(D_rnn)

    def forward(self, x:Tensor, cache:RecurrentCache | None = None) -> tuple[Tensor, RecurrentCache]:
        # Cache for generation: the Conv1D inputs and the RG-LRU state of the previous call. Prefill: cache = None.
        conv_state = None if cache is None else cache.conv_state
        h = None if cache is None else cache.h

        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)
        x2, conv_state = self.separableConv1D(x2, state=conv_state)
        x2, h = self.rglru(x2, h0=h)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)
        new_cache = RecurrentCache(conv_state, h)
        return y, new_cache


class Residual_block(nn.Module):
    def __init__(self, D:int, temporal_block_type:Literal['recurrent', 'attention']='recurrent',
                 D_rnn:int | None = None, M:int = 3, H:int | None = None, window_size:int = 1024):
        super().__init__()
        self.temporal_norm = RMSNorm(D)
        # Paper, Section 2.3: the temporal-mixing block is the recurrent block or local MQA.
        # Griffin stacks two residual blocks with 'recurrent', then one with 'attention' (Section 3).
        # D_rnn, M (MLP expansion), H (attention heads) and window_size are the columns of Table 2,
        # see GriffinConfig; the defaults are the rules of the paper: D_rnn = 4D/3, M = 3, D_head = 128.
        if temporal_block_type == 'recurrent':
            self.temporal_block = Recurrent_block(D, D_rnn=D_rnn)
        elif temporal_block_type == 'attention':
            if H is None:
                D_head = 128
            else:
                assert D % H == 0, "D must be a multiple of the number of heads H"
                D_head = D // H
            self.temporal_block = Local_MQA_block(D, D_head=D_head, window_size=window_size)
        else:
            raise ValueError(f"temporal_block_type must be 'recurrent' or 'attention', got {temporal_block_type!r}")
        self.mlp_norm = RMSNorm(D)
        self.mlp = Gated_MLP_block(D, expansion_factor=M)

    def forward(self, x:Tensor, cache:RecurrentCache | KVCache | None = None) -> tuple[Tensor, RecurrentCache | KVCache]:
        x1 = self.temporal_norm(x)
        x1, new_cache = self.temporal_block(x1, cache=cache)  # RecurrentCache or KVCache, as the block

        y1 = x + x1

        x2 = self.mlp_norm(y1)
        x2 = self.mlp(x2)

        y2 = y1 + x2

        return y2, new_cache
