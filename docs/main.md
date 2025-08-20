# 1) 目錄結構

```
your-repo/
├─ benchmark.py                # CLI：train / eval / vis
├─ requirements.txt            # 依賴（最小集合）
├─ dataset/
│  ├─ __init__.py
│  ├─ tiny_shakespeare.py      # 小型文字語料
│  ├─ dyck.py                  # 平衡括號（長依賴/堆疊記憶）
│  ├─ addition.py              # 長整數逐位加法（計算型長鏈）
│  ├─ needle.py                # Needle-in-a-Haystack 檢索
│  ├─ reverse.py               # Reverse 序列反轉（拷貝與位置對應）
│  ├─ rle.py                   # Run-Length Encoding 壓縮
│  ├─ caesar.py                # Caesar Cipher 加密
│  ├─ trigo.py                 # Trigonometric Functions 三角函數預測
│  ├─ fibonacci.py             # Fibonacci 數列生成
│  ├─ poly.py                  # Polynomial Functions 多項式預測
│  ├─ fourier_mix.py           # Fourier Mix 頻域混合
│  └─ common.py                # 基底 Dataset、collate、亂數種子
├─ src/
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ transformer.py        # 以 torch.nn.TransformerEncoder 為核心
│  │  ├─ mamba.py              # 嘗試匯入 mamba-ssm；否則退化到簡化版 SSM/GRU
│  │  └─ template.py           # 你要開發的新「自回歸層」範本
│  └─ utils/
│     ├─ tokenizer.py          # CharTokenizer（含BOS/EOS/PAD）
│     ├─ training.py           # 訓練/eval loop、儲存/載入
│     ├─ metrics.py            # CE、bpc、perplexity、EM、距離曲線
│     └─ viz.py                # Matplotlib 圖表輸出（loss/acc/距離曲線）
└─ runs/                       # 輸出（自動建立）
   └─ {time_stamp}/
      ├─ config.json
      ├─ ckpt.pt
      ├─ metrics.json
      ├─ samples.txt
      └─ figures/
         ├─ loss.png
         ├─ ppl.png
         └─ distance_curve.png
```

---

# 2) `benchmark.py`（單檔完成 train / eval / vis）

**用途**

* 以同一套 CLI 參數在不同資料集與模型間切換。
* 訓練結束自動跑 eval、存 **metrics.json** 與圖表。

**CLI（建議）**

```
python benchmark.py \
  --task {tiny_shakespeare|dyck|addition|needle} \
  --model {transformer|mamba|template} \
  --context_len 512 --d_model 384 --n_layer 6 --n_head 6 \
  --lr 3e-4 --batch_size 32 --epochs 5 \
  --precision bf16 \
  --out_dir runs/ts_gpt_small
```

**最低需要支援的參數**

* 資料：`--task`, `--context_len`, `--train_size`, `--val_size`, `--test_size`
* 模型：`--model`, `--d_model`, `--n_layer`, `--n_head`（無頭時忽略）, `--dropout`
* 優化：`--lr`, `--warmup_steps`, `--epochs`, `--batch_size`, `--grad_clip`, `--weight_decay`
* 其他：`--precision {fp32|fp16|bf16}`, `--seed`, `--out_dir`, `--eval_only`, `--vis_only`

**核心流程（概念規格）**

1. 解析參數、設 seed、建 out\_dir。
2. 依 `--task` 取得 `train/val/test` DataLoader（字元級 tokenizer 內建）。
3. 依 `--model` 實例化模型；統一 forward 介面：

   * `loss, aux = model.loss_fn(batch)`（封裝 shift/label smoothing 等）
   * `logits = model(x, attn_mask=...)`
   * `generate(prompt, max_new_tokens, temperature, top_k)`
4. Optimizer + Cosine decay（含 warmup）。
5. 訓練：梯度累積（視 batch\_size）、AMP、grad clip。
6. 驗證：通用指標（CE、bpc、ppl）＋任務特定（例如 Dyck 的 **exact-match**、Needle 的 **命中率@距離**）。
7. 保存：`ckpt.pt`（含 tokenizer/args/state\_dict）、`metrics.json`、`samples.txt`（每任務 5\~10 組樣例）。
8. 視覺化：`viz.py` 畫 `loss.png`、`ppl.png`、可選 `distance_curve.png`（Needle/Dyck）。

---

# 3) Dataset 子模組（共通介面）

**共同原則**

* **字元級 tokenizer**：訓練前先掃描 `train` 文本建立 vocab；特殊符號：`<pad>=0,<bos>,<eos>`.
* **因果 LM 格式**：batch 輸入 `x`（含 `<bos>` 與正文），標籤 `y` 為 `x` 右移一位，`<pad>` 位置忽略。
* `__getitem__` 回傳（`ids`, `length`, 其他任務 meta），由 `common.collate_fn` 統一 pad & mask。

**任務最小規格**

* `tiny_shakespeare.py`：載入內建小文本（附在檔案內或從網路複製文本貼進常數字串），切塊為固定長度。
* `dyck.py`：動態產生平衡括號字串；參數：`max_len`, `max_depth`, `noise_prob`。
* `addition.py`：產生 `"a+b=" -> "sum"` 的對映；訓練以逐字輸出 `sum`（同樣當作下一字預測）。
* `needle.py`：在長上下文中插入 key 並要求模型複寫 key；保存 key 的插入位置以便畫距離曲線。

---

# 4) 模型層（可替換）

## 4.1 共通基底 `template.py`

* 目的：讓你快速替換成**新自回歸層**而不動到訓練/資料程式。
* 介面（建議）：

  ```python
  class ARModel(nn.Module):
      def __init__(self, vocab_size:int, d_model:int, **kwargs): ...
      def forward(self, x: torch.LongTensor, attn_mask=None) -> torch.Tensor:
          """回傳 logits: (B, T, vocab)"""
      def loss_fn(self, batch) -> tuple[torch.Tensor, dict]:
          """封裝 CE 損失與任務特定監控"""
      @torch.no_grad()
      def generate(self, prompt_ids, max_new_tokens:int, temperature:float=1.0, top_k:int|None=None) -> torch.LongTensor:
          ...
  ```
* 內含：嵌入層、位置編碼（可選 ALiBi/Rotary）、投影頭 `lm_head`。

## 4.2 Transformer（`transformer.py`）

* 使用 `torch.nn.TransformerEncoderLayer` + `TransformerEncoder`，以 **causal mask**（上三角）限制未來資訊。
* 重要參數：`n_layer, n_head, d_model, dim_ffn, dropout, rope=False`。
* 支援 KV-cache 的 **generate**（逐步步進）。

## 4.3 Mamba（`mamba.py`）

* 嘗試：

  1. `try: from mamba_ssm import Mamba`（若你選用外部套件），
  2. `except:` fallback 到**簡化版**：以 `nn.GRU` 或小型 SSM 區塊（1D depthwise conv + gating）模擬線性時間遞迴，保留 `forward/generate` 介面一致。
* 重點：**增量解碼狀態是固定大小**（與序列長度無關），以凸顯 Mamba 類模型的部署優勢。

---

# 5) Utils

## 5.1 `tokenizer.py`

* `CharTokenizer.build(corpus:str, extra_tokens=["<pad>","<bos>","<eos>"])`
* `encode(str)->List[int]`, `decode(List[int])->str`
* 保存/載入 `vocab.json`。

## 5.2 `training.py`

* `train_one_epoch(model, loader, opt, sched, scaler, device, args) -> dict_logs`
* `evaluate(model, loader, device, args) -> metrics_dict`
* AMP（fp16/bf16）、grad clip、儲存/載入 checkpoint。

## 5.3 `metrics.py`

* 通用：`cross_entropy`, `bpc`, `perplexity`
* 任務特定：

  * Dyck：`exact_match`, `max_depth_acc`
  * Addition：`exact_match`
  * Needle：`hit_rate_by_gap`（輸入 key 與插入距離，輸出距離→命中率 dict，供畫曲線）

## 5.4 `viz.py`

* `plot_loss(history, out_path)`
* `plot_ppl(history, out_path)`
* `plot_distance_curve(distance2acc, out_path)`（Needle/Dyck）

---

# 6) `requirements.txt`（最小集合）

```
torch>=2.2
numpy
tqdm
matplotlib
```

> 可選：`mamba-ssm`（若採用官方/社群實作）、`rich`（更好看的 logger）。

---

# 7) 成品輸出（自動）

* `runs/{time_stamp}/config.json`：完整 CLI 與模型超參數。
* `runs/{time_stamp}/ckpt.pt`：`state_dict` + `tokenizer.vocab` + `args`。
* `runs/{time_stamp}/metrics.json`：train/val/test 指標（含任務特定）。
* `runs/{time_stamp}/samples.txt`：每任務的生成樣例（例如：Dyck/Needle 的示例）。
* `runs/{time_stamp}/figures/*.png`：學習曲線與（若有）距離曲線。

---

# 8) 範例：關鍵檔案「最小骨架」

## 8.1 `benchmark.py`（簡化骨架）

```python
import argparse, json, os, time, torch
from src.utils.training import train_one_epoch, evaluate
from src.utils.viz import plot_loss, plot_ppl
from src.utils.tokenizer import CharTokenizer
from dataset import tiny_shakespeare, dyck, addition, needle
from src.models import transformer as mdl_transformer
from src.models import mamba as mdl_mamba
from src.models import template as mdl_template

TASKS = {
    "tiny_shakespeare": tiny_shakespeare.build_loaders,
    "dyck": dyck.build_loaders,
    "addition": addition.build_loaders,
    "needle": needle.build_loaders,
}
MODELS = {
    "transformer": mdl_transformer.build_model,
    "mamba": mdl_mamba.build_model,
    "template": mdl_template.build_model,
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS.keys())
    ap.add_argument("--model", required=True, choices=MODELS.keys())
    ap.add_argument("--context_len", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--precision", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--vis_only", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = args.out_dir or f"runs/{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)

    # 1) 建 DataLoader 與 tokenizer
    tok, loaders, task_meta = TASKS[args.task](args)
    vocab_size = tok.vocab_size

    # 2) 建模型
    model = MODELS[args.model](vocab_size=vocab_size, args=args).to("cuda" if torch.cuda.is_available() else "cpu")

    # 3) 只有評估/視覺化模式
    if args.vis_only:
        hist = json.load(open(os.path.join(out_dir, "metrics.json")))
        plot_loss(hist["loss_curve"], os.path.join(out_dir, "figures/loss.png"))
        plot_ppl(hist["ppl_curve"], os.path.join(out_dir, "figures/ppl.png"))
        return
    if args.eval_only:
        metrics = evaluate(model, loaders["val"], device="cuda" if torch.cuda.is_available() else "cpu", args=args, task_meta=task_meta)
        print(json.dumps(metrics, indent=2))
        return

    # 4) 正式訓練 → 驗證 → 儲存 → 視覺化
    hist = {"loss_curve": [], "ppl_curve": []}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision=="fp16"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(1, args.epochs+1):
        logs = train_one_epoch(model, loaders["train"], opt, sched, scaler, device, args)
        val = evaluate(model, loaders["val"], device, args, task_meta)
        hist["loss_curve"].append((epoch, logs["loss"]))
        hist["ppl_curve"].append((epoch, val.get("ppl", None)))
        print(f"[E{epoch}] train_loss={logs['loss']:.4f} val_ppl={val.get('ppl', -1):.2f}")

    # 最終測試與輸出
    test = evaluate(model, loaders["test"], device, args, task_meta, save_samples=os.path.join(out_dir,"samples.txt"))
    json.dump({"final_test": test, **hist, "args": vars(args)}, open(os.path.join(out_dir,"metrics.json"),"w"), indent=2)
    torch.save({"model": model.state_dict(), "args": vars(args), "vocab": tok.vocab}, os.path.join(out_dir,"ckpt.pt"))
    os.makedirs(os.path.join(out_dir,"figures"), exist_ok=True)
    plot_loss(hist["loss_curve"], os.path.join(out_dir,"figures/loss.png"))
    plot_ppl(hist["ppl_curve"], os.path.join(out_dir,"figures/ppl.png"))

if __name__ == "__main__":
    main()
```

> 其餘檔案（`dataset/*.py`, `src/models/*.py`, `src/utils/*.py`）只要依據上述介面填入即可；若你要，我也可以直接幫你產出**可執行的最小骨架程式碼**。

---

# 9) 快速 smoke test（建議）

* Transformer + TinyShakespeare（char）
  `python benchmark.py --task tiny_shakespeare --model transformer --epochs 3`
* Mamba（fallback/或外部套件）+ Needle（L=512）
  `python benchmark.py --task needle --model mamba --context_len 512 --epochs 3`
* Template（你的層）+ Dyck（max\_depth=8）
  `python benchmark.py --task dyck --model template --epochs 3`

