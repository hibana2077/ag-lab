import argparse
import json
import math
import os
import random
import time
from typing import Dict

import torch
from torch import nn

from dataset import (
    tiny_shakespeare,
    dyck,
    addition,
    needle,
    reverse,
    rle,
    caesar,
    trigo,
    fibonacci,
    poly,
    fourier_mix,
)
from src.models import transformer as mdl_transformer
from src.models import mamba as mdl_mamba
from src.models import template as mdl_template
from src.utils.training import train_one_epoch, evaluate, load_checkpoint_partial, WarmupCosine
from src.utils.viz import plot_loss, plot_ppl, plot_distance_curve


TASKS = {
    "tiny_shakespeare": tiny_shakespeare.build_loaders,
    "dyck": dyck.build_loaders,
    "addition": addition.build_loaders,
    "needle": needle.build_loaders,
    "reverse": reverse.build_loaders,
    "rle": rle.build_loaders,
    "caesar": caesar.build_loaders,
    "trigo": trigo.build_loaders,
    "fibonacci": fibonacci.build_loaders,
    "poly": poly.build_loaders,
    "fourier_mix": fourier_mix.build_loaders,
}
MODELS = {
    "transformer": mdl_transformer.build_model,
    "mamba": mdl_mamba.build_model,
    "template": mdl_template.build_model,
}


def parse_args():
    ap = argparse.ArgumentParser(description="Unified benchmark for small autoregressive models")
    ap.add_argument("--task", required=True, choices=TASKS.keys())
    ap.add_argument("--model", required=True, choices=MODELS.keys())
    # data sizes
    ap.add_argument("--context_len", type=int, default=512)
    ap.add_argument("--train_size", type=int, default=5000)
    ap.add_argument("--val_size", type=int, default=500)
    ap.add_argument("--test_size", type=int, default=500)
    # model hyperparams
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rope", action="store_true", help="Use rotary embedding where supported")
    # optimization
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # misc
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--vis_only", action="store_true")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume / evaluate")
    ap.add_argument("--sample_every", type=int, default=0, help="If >0, generate samples every N epochs")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or f"runs/{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    # Build data + tokenizer
    tokenizer, loaders, task_meta = TASKS[args.task](args)
    vocab_size = tokenizer.vocab_size

    # Build model
    model: nn.Module = MODELS[args.model](vocab_size=vocab_size, args=args).to(device)

    start_epoch = 1
    history: Dict[str, list] = {"loss_curve": [], "ppl_curve": []}

    if args.resume:
        ckpt_path = args.resume if os.path.isfile(args.resume) else os.path.join(args.resume, "ckpt.pt")
        if os.path.isfile(ckpt_path):
            meta = load_checkpoint_partial(model, ckpt_path, map_location=device)
            if meta and "history" in meta:
                history = meta["history"]
            if meta and "epoch" in meta:
                start_epoch = int(meta["epoch"]) + 1
            print(f"[resume] Loaded checkpoint from {ckpt_path}")
        else:
            print(f"[warn] resume path not found: {ckpt_path}")

    # Visualization only
    if args.vis_only:
        metrics_path = os.path.join(out_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            print(f"No metrics.json at {metrics_path}")
            return
        metrics_all = json.load(open(metrics_path, "r", encoding="utf-8"))
        history = {k: metrics_all.get(k, []) for k in ["loss_curve", "ppl_curve"]}
        plot_loss(history["loss_curve"], os.path.join(out_dir, "figures/loss.png"))
        plot_ppl(history["ppl_curve"], os.path.join(out_dir, "figures/ppl.png"))
        if "distance_curve" in metrics_all:
            plot_distance_curve(metrics_all["distance_curve"], os.path.join(out_dir, "figures/distance_curve.png"))
        print("[vis] figures updated")
        return

    # Eval only
    if args.eval_only:
        eval_metrics = evaluate(
            model,
            loaders["val"],
            device,
            args,
            task_meta,
            tokenizer,
        )
        print(json.dumps(eval_metrics, indent=2, ensure_ascii=False))
        return

    # Optimizer + scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(loaders["train"]) / max(1, args.grad_accum))
    sched = WarmupCosine(opt, warmup=args.warmup_steps, max_steps=total_steps)
    # No GradScaler per user request; we still use autocast for forward pass.

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    for epoch in range(start_epoch, args.epochs + 1):
        logs = train_one_epoch(model, loaders["train"], opt, sched, device, args)
        val_metrics = evaluate(model, loaders["val"], device, args, task_meta, tokenizer)
        history["loss_curve"].append((epoch, float(logs["loss"])) )
        ppl = val_metrics.get("ppl")
        if ppl is not None:
            history["ppl_curve"].append((epoch, float(ppl)))
        # Safe formatting for ppl (can be None); avoid conditional inside format specifier
        ppl_str = f"{ppl:.2f}" if (ppl is not None) else "-1"
        print(f"[E{epoch}] train_loss={logs['loss']:.4f} val_ppl={ppl_str}")

        # Optional sample
        if args.sample_every and (epoch % args.sample_every == 0):
            prompt = random.choice(task_meta.get("sample_prompts", ["Hello"]))[: args.context_len // 4]
            with torch.no_grad():
                input_ids = tokenizer.encode(prompt)
                input_tensor = torch.tensor([input_ids], device=device)
                generated = model.generate(
                    input_tensor,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.8,
                    top_k=50,
                )
            decoded = tokenizer.decode(generated[0].tolist())
            with open(os.path.join(out_dir, "samples.txt"), "a", encoding="utf-8") as sf:
                sf.write(f"\n[E{epoch}] prompt: {prompt}\n{decoded}\n")

    # Final test + save
    test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        args,
        task_meta,
        tokenizer,
        save_samples=os.path.join(out_dir, "samples.txt"),
    )
    results = {
        "history": history,
        "final_test": test_metrics,
        "args": vars(args),
    }
    if "distance_curve" in test_metrics:
        plot_distance_curve(
            test_metrics["distance_curve"], os.path.join(out_dir, "figures/distance_curve.png")
        )
    plot_loss(history["loss_curve"], os.path.join(out_dir, "figures/loss.png"))
    plot_ppl(history["ppl_curve"], os.path.join(out_dir, "figures/ppl.png"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs,
            "vocab": tokenizer.vocab,
            "args": vars(args),
            "history": history,
        },
        os.path.join(out_dir, "ckpt.pt"),
    )
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
