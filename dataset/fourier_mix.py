"""Fourier mixture value task.
Format: mix:a1,b1,a2,b2|x=<x>#<y> where y = a1*sin(b1*x)+a2*cos(b2*x) rounded.
Coefficients small integers; x in [0, 2pi] represented as float with 2 decimals.
"""
import random, math
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

def gen_example():
    a1 = random.randint(-2,2)
    b1 = random.randint(1,4)
    a2 = random.randint(-2,2)
    b2 = random.randint(1,4)
    x = random.uniform(0, 2*math.pi)
    y = a1*math.sin(b1*x) + a2*math.cos(b2*x)
    y_r = f"{y:.3f}"
    x_s = f"{x:.2f}"
    return f"mix:{a1},{b1},{a2},{b2}|x={x_s}#{y_r}", {"x": x}


def make_split(size):
    texts, metas = [], []
    for _ in range(size):
        t, m = gen_example()
        texts.append(t)
        metas.append(m)
    return texts, metas


def build_loaders(args):
    train, mtr = make_split(args.train_size)
    val, mv = make_split(args.val_size)
    test, mte = make_split(args.test_size)
    tokenizer = build_tokenizer_from_texts(CharTokenizer, train)
    ds_train = TextDataset(train, mtr, tokenizer, args.context_len)
    ds_val = TextDataset(val, mv, tokenizer, args.context_len)
    ds_test = TextDataset(test, mte, tokenizer, args.context_len)
    loaders = {
        "train": DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
    }
    task_meta = {"name": "fourier_mix", "sample_prompts": ["mix:1,1,1,2|x=0.50#", "mix:-1,2,0,1|x=3.14#"]}
    return tokenizer, loaders, task_meta
