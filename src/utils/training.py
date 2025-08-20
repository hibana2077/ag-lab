from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional
from . import metrics as M


class WarmupCosine:
    def __init__(self, optimizer, warmup: int, max_steps: int, min_lr: float = 0.0):
        self.opt = optimizer
        self.warmup = warmup
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        for i, g in enumerate(self.opt.param_groups):
            if self.step_num < self.warmup:
                g["lr"] = self.base_lrs[i] * self.step_num / max(1, self.warmup)
            else:
                progress = (self.step_num - self.warmup) / max(1, self.max_steps - self.warmup)
                cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
                g["lr"] = self.min_lr + (self.base_lrs[i] - self.min_lr) * cosine.item()
        return [g["lr"] for g in self.opt.param_groups]


def train_one_epoch(model, loader: DataLoader, opt, sched, scaler, device, args) -> Dict:
    model.train()
    losses = []
    opt.zero_grad(set_to_none=True)
    autocast_dtype = (
        torch.float16 if args.precision == "fp16" else torch.bfloat16 if args.precision == "bf16" else None
    )
    for step, batch in enumerate(loader):
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            loss, _ = model.loss_fn(batch)
            loss = loss / args.grad_accum
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % args.grad_accum == 0:
            if args.grad_clip:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        losses.append(loss.item() * args.grad_accum)
    return {"loss": float(sum(losses) / len(losses))}


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, args, task_meta, tokenizer, save_samples: Optional[str] = None):
    model.eval()
    all_losses = []
    preds = []
    refs = []
    needle_records = []
    autocast_dtype = (
        torch.float16 if args.precision == "fp16" else torch.bfloat16 if args.precision == "bf16" else None
    )
    for batch in loader:
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            loss, _ = model.loss_fn(batch)
        all_losses.append(loss.item())
        x = batch["x"].to(device)
        logits = model(x)
        pred_ids = torch.argmax(logits, dim=-1)
        for ids_ref, ids_pred, meta, ref_text in zip(batch["y"], pred_ids, batch["metas"], batch["texts"]):
            pred_text = tokenizer.decode(ids_pred.tolist())
            preds.append(pred_text)
            refs.append(ref_text)
            if task_meta["name"] == "needle" and "needle" in meta:
                needle = meta["needle"]
                gap = meta["gap"]
                after_hash = pred_text.split("#")[-1][: len(needle)] if "#" in pred_text else ""
                needle_records.append({"gap": gap, "correct": after_hash == needle})
    mean_loss = sum(all_losses) / max(1, len(all_losses))
    result = {"loss": mean_loss, "ppl": M.perplexity(mean_loss), "bpc": M.bpc(mean_loss, base=2)}
    if task_meta["name"] in {"addition", "dyck", "needle", "reverse", "rle", "caesar", "trigo", "fibonacci", "poly", "fourier_mix"}:
        result["em"] = M.exact_match(preds, refs)
    if task_meta["name"] == "dyck":
        result["max_depth_acc"] = M.max_depth_acc(preds, refs)
    if task_meta["name"] == "needle":
        result["distance_curve"] = M.hit_rate_by_gap(needle_records)
    if save_samples:
        with open(save_samples, "a", encoding="utf-8") as f:
            for p, r in list(zip(preds, refs))[:10]:
                f.write(f"REF: {r}\nPRED:{p}\n---\n")
    return result


def load_checkpoint_partial(model, path: str, map_location=None):
    try:
        ckpt = torch.load(path, map_location=map_location)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        return ckpt
    except Exception as e:
        print(f"[warn] failed to load checkpoint: {e}")
        return None
