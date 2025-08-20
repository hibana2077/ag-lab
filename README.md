# ag-lab

Minimal educational autoregressive (char-level) modeling lab featuring:

- Unified `benchmark.py` CLI (train / eval / vis)
- Datasets: Tiny Shakespeare, Dyck parentheses, Addition, Needle-in-haystack
- Models: Transformer (causal), Mamba (fallback GRU), Template (residual MLP)
- Utilities: tokenizer, training loop (AMP, warmup+cosine, grad-accum & clip), metrics, visualization

## Quick Start

Install deps (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

Train a tiny transformer on Tiny Shakespeare (3 epochs):

```bash
python benchmark.py --task tiny_shakespeare --model transformer --epochs 3
```

Run Needle task with Mamba fallback (GRU):

```bash
python benchmark.py --task needle --model mamba --context_len 256 --epochs 3
```

Visualize existing run (after training):

```bash
python benchmark.py --task tiny_shakespeare --model transformer --vis_only --out_dir runs/<timestamp>
```

## Outputs

```text
runs/<timestamp>/
	config.json
	ckpt.pt
	metrics.json
	samples.txt
	figures/
		loss.png
		ppl.png
		distance_curve.png (needle only)
```

## Extending

Add new dataset under `dataset/` returning `(tokenizer, loaders, task_meta)`; add model under `src/models/` exposing `build_model` with `loss_fn` & `generate`.

Enjoy hacking! 🧪
