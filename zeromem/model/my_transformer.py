"""ZeroMem: a from-scratch decoder-only transformer.

Architecture is the modern Llama/Qwen-style recipe (RMSNorm, RoPE, SwiGLU,
tied embeddings) — deliberately, so what we learn here transfers directly
to reasoning about the Qwen side of the pipeline later.

The one architectural addition beyond a stock GPT clone is chunk-native
attention (see `build_chunk_attention_mask` below): retrieved chunks are
prevented from attending to each other, so information can't leak between
sources, while the query/answer segment can see everything. This is the
published Block-Attention scheme (arXiv:2409.15355) — see BLUEPRINT.md for
why we're not claiming to have invented the masking idea itself.

Run this file directly for a shape/gradient smoke test:
    python -m zeromem.model.my_transformer
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from zeromem.model.config import ZeroMemConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer norm. Cheaper than LayerNorm (no mean/bias term)
    and what Llama/Qwen/Mistral all use."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability regardless of the input dtype.
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms
        return (x * self.weight).to(dtype)


# ---------------------------------------------------------------------------
# Rotary position embeddings (RoPE)
# ---------------------------------------------------------------------------
def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    """Returns complex-valued rotation frequencies, shape (max_seq_len, head_dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: (batch, n_heads, seq_len, head_dim). Rotates each adjacent pair of
    dims by the precomputed angle for that position."""
    b, h, t, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(b, h, t, d // 2, 2))
    freqs_cis = freqs_cis[:t].view(1, 1, t, d // 2)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).reshape(b, h, t, d)
    return x_out.type_as(x)


# ---------------------------------------------------------------------------
# Chunk-native attention mask (Block-Attention scheme)
# ---------------------------------------------------------------------------
def build_chunk_attention_mask(chunk_ids: torch.Tensor) -> torch.Tensor:
    """Build an additive attention bias enforcing chunk isolation + causality.

    chunk_ids: (batch, seq_len) long tensor.
        0   = "global" segment (question/answer tokens) — attends to
              everything before it, like normal causal attention.
        >0  = a retrieved-chunk token, tagged with that chunk's id — may attend
              to earlier tokens of the SAME chunk and to earlier GLOBAL tokens
              (the question, when it is placed first), but never to another
              chunk. That is what stops chunk 1 leaking into chunk 2's
              representation while still letting every chunk know what it is
              being asked about (the Fusion-in-Decoder idea: question + one
              passage encoded together, passages isolated from each other).

    Returns an additive float mask of shape (batch, 1, seq_len, seq_len):
    0.0 where attention is allowed, -inf where it is blocked. Add this
    directly to attention logits before softmax.
    """
    b, t = chunk_ids.shape
    device = chunk_ids.device

    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))  # (t, t)

    query_is_global = (chunk_ids == 0).unsqueeze(2)          # (b, t, 1) — is row i global?
    key_is_global = (chunk_ids == 0).unsqueeze(1)            # (b, 1, t) — is column j global?
    same_chunk = chunk_ids.unsqueeze(2) == chunk_ids.unsqueeze(1)  # (b, t, t) — chunk_ids[i] == chunk_ids[j]

    # Row i may see column j if: causal AND (i is global OR j is global OR i,j share a chunk id)
    allowed = causal.unsqueeze(0) & (query_is_global | key_is_global | same_chunk)

    mask = torch.zeros(b, t, t, dtype=torch.float32, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(1)  # (b, 1, t, t) — broadcasts over heads


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: ZeroMemConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim

        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # attn_mask is an additive bias (0 / -inf) that already encodes both
        # causality and chunk isolation — see build_chunk_attention_mask.
        # is_causal must be False here since masking is handled explicitly.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# SwiGLU feed-forward
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: ZeroMemConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: ZeroMemConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Pre-norm residual blocks (norm -> sublayer -> add), not post-norm —
        # this is what makes deep transformers trainable without warmup tricks.
        x = x + self.attn(self.attn_norm(x), freqs_cis, attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# ZeroMem
# ---------------------------------------------------------------------------
class ZeroMem(nn.Module):
    def __init__(self, cfg: ZeroMemConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight  # weight tying

        freqs_cis = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_embed.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        chunk_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        input_ids: (batch, seq_len) token ids.
        chunk_ids: (batch, seq_len) optional — see build_chunk_attention_mask.
                   Pass None during stage-1 pretraining (plain causal LM);
                   pass real chunk tags during stage-2 citation fine-tuning.
        targets:   (batch, seq_len) optional — next-token labels for loss.
        """
        b, t = input_ids.shape
        assert t <= self.cfg.max_seq_len, (
            f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        if chunk_ids is not None:
            attn_mask = build_chunk_attention_mask(chunk_ids)
        else:
            attn_mask = None  # SDPA's is_causal fast path handles plain causal masking below

        x = self.tok_embed(input_ids)
        for block in self.blocks:
            if attn_mask is not None:
                x = block(x, self.freqs_cis, attn_mask)
            else:
                x = self._forward_plain_causal_block(block, x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )
        return logits, loss

    def _forward_plain_causal_block(self, block: Block, x: torch.Tensor) -> torch.Tensor:
        """Stage-1 pretraining path: no chunk structure yet, just standard
        causal attention via SDPA's built-in fast path (no explicit mask
        materialized, cheaper than building one full of zeros)."""
        b, t, _ = x.shape
        x_norm = block.attn_norm(x)
        q = block.attn.wq(x_norm).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = block.attn.wk(x_norm).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        v = block.attn.wv(x_norm).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        q = apply_rope(q, self.freqs_cis)
        k = apply_rope(k, self.freqs_cis)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=block.attn.dropout if self.training else 0.0, is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        x = x + block.attn.wo(out)
        x = x + block.ffn(block.ffn_norm(x))
        return x

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Greedy/sampled autoregressive generation. No KV cache yet — fine
        for smoke-testing and short citation answers; add a cache before
        this needs to serve real traffic."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if (next_id == self.cfg.eos_token_id).all():
                break
        return input_ids


if __name__ == "__main__":
    # Smoke test: shapes, forward, backward, and chunk-mask isolation all work.
    cfg = ZeroMemConfig()
    model = ZeroMem(cfg)
    print(f"ZeroMem params: {model.num_params():,} total, "
          f"{model.num_params(exclude_embeddings=True):,} excluding tied embeddings")

    batch, seq_len = 2, 32
    input_ids = torch.randint(8, cfg.vocab_size, (batch, seq_len))  # avoid special-token ids
    targets = torch.randint(8, cfg.vocab_size, (batch, seq_len))

    # Plain causal pretraining path.
    logits, loss = model(input_ids, targets=targets)
    assert logits.shape == (batch, seq_len, cfg.vocab_size)
    loss.backward()
    print(f"plain causal forward+backward OK, loss={loss.item():.3f}")

    # Chunk-native path: first 10 tokens = chunk 1, next 10 = chunk 2, last 12 = global.
    chunk_ids = torch.cat([
        torch.ones(batch, 10, dtype=torch.long),
        torch.full((batch, 10), 2, dtype=torch.long),
        torch.zeros(batch, 12, dtype=torch.long),
    ], dim=1)
    model.zero_grad()
    logits, loss = model(input_ids, chunk_ids=chunk_ids, targets=targets)
    loss.backward()
    print(f"chunk-native forward+backward OK, loss={loss.item():.3f}")

    # Verify isolation directly: a token in chunk 1 must not be able to see chunk 2.
    mask = build_chunk_attention_mask(chunk_ids)
    chunk1_to_chunk2 = mask[0, 0, 5, 15].item()  # row=chunk1 token, col=chunk2 token
    global_to_chunk2 = mask[0, 0, 25, 15].item()  # row=global token, col=chunk2 token
    assert chunk1_to_chunk2 == float("-inf"), "chunk isolation broken: chunk 1 can see chunk 2"
    assert global_to_chunk2 == 0.0, "global segment should see all prior chunks"
    print("chunk isolation verified: chunk-1 cannot see chunk-2, global segment sees both")

    # Question-first layout (Fusion-in-Decoder style): [question x4][chunk1 x6][chunk2 x6][answer x4]
    fid_ids = torch.tensor([[0] * 4 + [1] * 6 + [2] * 6 + [0] * 4])
    m = build_chunk_attention_mask(fid_ids)[0, 0]
    assert m[6, 1].item() == 0.0, "chunk 1 tokens must see the question that comes before them"
    assert m[12, 1].item() == 0.0, "chunk 2 tokens must see the question too"
    assert m[12, 6].item() == float("-inf"), "chunk 2 must still NOT see chunk 1"
    assert m[6, 12].item() == float("-inf"), "chunk 1 must not see the future / chunk 2"
    assert m[17, 6].item() == 0.0 and m[17, 12].item() == 0.0, "answer tokens see every chunk"
    print("question-first layout verified: chunks see the question, never each other; answer sees all")
