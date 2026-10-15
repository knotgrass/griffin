# Griffin, Section 2 "Model Architecture", Section 3 "Griffin" and Table 2 for the sizes.
# Official code: Griffin in griffin.py, Embedder in modules.py, presets with the block pattern in common.py
# https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/griffin.py#L150-L200
# https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/torch/modules.py#L946-L1003
# https://github.com/google-deepmind/recurrentgemma/blob/2efa84dac0e68e63547a27a18fa943c98f1c312e/recurrentgemma/common.py#L44-L76
import itertools

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

from .config import GriffinConfig
from .modules import Residual_block
from .rmsnorm import RMSNorm


class Griffin(nn.Module):
    """Paper, Section 2: embedding -> N residual blocks -> RMSNorm -> linear layer whose weights are
    shared with the embedding -> softmax over the V tokens. All sizes come from a GriffinConfig,
    e.g. Griffin(GriffinConfig.from_table2("7B", V=32_000)).

    Section 3: the residual blocks follow the pattern recurrent, recurrent, attention (local MQA).
    Table 2: depth N = 12 (100M to 400M), 24 (1.3B, 3B), 32 (7B), 40 (14B). The official presets
    take the first N of the cycled pattern, so N = 32 is 10 triples plus 2 recurrent blocks.
    Not here: the sqrt(D) scaling of the embedding and the soft cap of the logits. RecurrentGemma
    uses both; the paper presets of the official code have both off.
    """

    def __init__(self, cfg: GriffinConfig):
        super().__init__()
        self.cfg = cfg  # V, D, N and the other columns of Table 2, see config.py
        self.embedding = nn.Embedding(cfg.V, cfg.D)  # its weight is also the output layer, see forward
        # official Embedder: std = 1 / sqrt(D), so the shared output layer gives logits of order 1
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / cfg.D ** 0.5)
        pattern = itertools.cycle(['recurrent', 'recurrent', 'attention'])  # as the official presets
        self.blocks = nn.ModuleList([
            Residual_block(cfg.D, temporal_block_type=t, D_rnn=cfg.D_rnn, M=cfg.M, H=cfg.H,
                           window_size=cfg.window_size)
            for t in itertools.islice(pattern, cfg.N)
        ])
        self.final_norm = RMSNorm(cfg.D)

    def forward(self, tokens: Int[Tensor, "B T"]) -> Float[Tensor, "B T V"]:
        x: Float[Tensor, "B T D"] = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits: Float[Tensor, "B T V"] = F.linear(x, self.embedding.weight)  # x @ E^T, the shared weights
        return logits

    def probabilities(self, tokens: Int[Tensor, "B T"]) -> Float[Tensor, "B T V"]:
        """The last step of the paper: softmax over the V tokens.

        Training uses F.cross_entropy on the logits of forward instead; it applies log-softmax itself.
        """
        probs = torch.softmax(self(tokens), dim=-1)
        return probs
