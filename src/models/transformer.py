from __future__ import annotations
import torch
from torch import nn


def causal_mask(sz: int, device):
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()


class TransformerAR(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        n_head: int,
        dim_ff: int,
        dropout: float = 0.1,
        rope: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 10000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("_mask_cache", torch.empty(0), persistent=False)

    def forward(self, x: torch.LongTensor, attn_mask=None):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :T]
        if self._mask_cache.shape[0] < T:
            self._mask_cache = causal_mask(T, x.device)
        mask = self._mask_cache[:T, :T]
        h = self.encoder(h, mask=mask)
        h = self.ln_f(h)
        return self.lm_head(h)

    def loss_fn(self, batch):
        x, y = batch["x"].to(self.lm_head.weight.device), batch["y"].to(self.lm_head.weight.device)
        logits = self.forward(x)
        vocab = logits.size(-1)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=0)
        return loss, {}

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        device = next(self.parameters()).device
        out = prompt_ids.to(device)
        for _ in range(max_new_tokens):
            logits = self.forward(out)
            nxt = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, idx = torch.topk(nxt, min(top_k, nxt.size(-1)))
                probs = torch.softmax(v, dim=-1)
                next_token = idx.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = torch.softmax(nxt, dim=-1)
                next_token = torch.multinomial(probs, 1)
            out = torch.cat([out, next_token], dim=1)
            if (next_token == 0).all():
                break
        return out


def build_model(vocab_size: int, args):
    return TransformerAR(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dim_ff=getattr(args, "dim_ff", args.d_model * 4),
        dropout=args.dropout,
        rope=args.rope,
    )
