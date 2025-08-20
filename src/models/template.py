from __future__ import annotations
from typing import Optional
import math
import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1)]


class SimpleBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ARModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([SimpleBlock(d_model, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.LongTensor, attn_mask: Optional[torch.Tensor] = None):
        h = self.emb(x)
        h = self.pos(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.lm_head(h)

    def loss_fn(self, batch) -> tuple[torch.Tensor, dict]:
        x, y = batch["x"].to(self.lm_head.weight.device), batch["y"].to(self.lm_head.weight.device)
        logits = self.forward(x)
        vocab = logits.size(-1)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), y.view(-1), ignore_index=0
        )
        return loss, {}

    @torch.no_grad()
    def generate(self, prompt_ids: torch.LongTensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        device = next(self.parameters()).device
        out = prompt_ids.clone().to(device)
        for _ in range(max_new_tokens):
            logits = self.forward(out)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, idx = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                probs = torch.softmax(v, dim=-1)
                next_token = idx.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            out = torch.cat([out, next_token], dim=1)
            if (next_token == 0).all():
                break
        return out


def build_model(vocab_size: int, args):
    return ARModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
