"""Model configuration for ZeroMem.

Kept as a plain dataclass so every other module (training, inference,
tokenizer) imports the same source of truth instead of re-declaring
magic numbers.
"""

from dataclasses import dataclass


@dataclass
class ZeroMemConfig:
    # --- vocabulary ---
    vocab_size: int = 16384

    # --- transformer shape ---
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1408          # SwiGLU hidden dim (~8/3 * d_model, rounded to 64)
    max_seq_len: int = 2048

    # --- rotary position embeddings ---
    rope_theta: float = 10000.0

    # --- regularization ---
    dropout: float = 0.0

    # --- special token ids (set by the tokenizer at training time; these
    # defaults are placeholders and MUST match zeromem/tokenizer/special_tokens.py) ---
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    chunk_sep_token_id: int = 3
    know_token_id: int = 4
    unsure_token_id: int = 5
    refuse_token_id: int = 6
    done_token_id: int = 7

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
