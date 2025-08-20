from __future__ import annotations
import torch
from torch import nn

try:
    from mamba_ssm import Mamba  # type: ignore
    HAVE_MAMBA = True
except Exception:
    HAVE_MAMBA = False


class GRUBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.ln(out)


class MambaLike(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([GRUBlock(d_model) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, attn_mask=None):
        h = self.emb(x)
        for layer in self.layers:
            h = layer(h)
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
    if HAVE_MAMBA:
        class MambaWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, args.d_model)
                self.blocks = nn.ModuleList([Mamba(d_model=args.d_model) for _ in range(args.n_layer)])
                self.ln_f = nn.LayerNorm(args.d_model)
                self.lm_head = nn.Linear(args.d_model, vocab_size, bias=False)

            def forward(self, x, attn_mask=None):
                h = self.emb(x)
                for blk in self.blocks:
                    h = blk(h)
                h = self.ln_f(h)
                return self.lm_head(h)

            def loss_fn(self, batch):
                x, y = batch["x"].to(self.lm_head.weight.device), batch["y"].to(self.lm_head.weight.device)
                logits = self.forward(x)
                vocab = logits.size(-1)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, vocab), y.view(-1), ignore_index=0
                )
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

        return MambaWrapper()
    else:
        return MambaLike(vocab_size, d_model=args.d_model, n_layer=args.n_layer, dropout=args.dropout)
