# Griffin, Table 2 (Appendix C): "Key model hyper-parameters considered for different model sizes."
from dataclasses import dataclass

# Table 2, without the training tokens column. D_head = D / H = 128 in every row.
TABLE_2 = {
    "100M": {"D": 768, "D_rnn": 1024, "N": 12, "M": 3, "H": 6},
    "200M": {"D": 1024, "D_rnn": 1536, "N": 12, "M": 3, "H": 8},
    "400M": {"D": 1536, "D_rnn": 2048, "N": 12, "M": 3, "H": 12},
    "1.3B": {"D": 2048, "D_rnn": 2560, "N": 24, "M": 3, "H": 16},
    "3B": {"D": 3072, "D_rnn": 4096, "N": 24, "M": 3, "H": 24},
    "7B": {"D": 4096, "D_rnn": 5632, "N": 32, "M": 3, "H": 32},
    "14B": {"D": 5120, "D_rnn": 8192, "N": 40, "M": 3, "H": 40},
}


@dataclass
class GriffinConfig:
    """The columns of Table 2, plus the vocabulary size and the local attention window."""

    V: int  # vocabulary size: number of distinct tokens, also the width of the output layer; not in Table 2
    D: int  # model width: size of a token embedding and of the residual stream
    N: int  # depth: number of residual blocks
    D_rnn: int | None = None  # RNN width; None gives the 4D/3 of Section 3, Table 2 has the exact values
    M: int = 3  # MLP expansion factor, 3 for every size
    H: int | None = None  # attention heads; None gives D / 128, so D_head = 128 as in Section 2.3
    window_size: int = 1024  # local attention window, Section 3

    @classmethod
    def from_table2(cls, size: str, V: int) -> "GriffinConfig":
        """A config with the row `size` of Table 2, e.g. GriffinConfig.from_table2("7B", V=32_000)."""
        return cls(V=V, **TABLE_2[size])
