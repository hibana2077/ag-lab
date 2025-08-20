"""Trigonometric symbolic value prediction.
Format: sin(<deg>)=...
We restrict degrees to common angles 0,30,45,60,90 etc.
Model must output value in simplified form (e.g. sqrt(3)/2).
"""
import random
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

ANGLES = [0,30,45,60,90,120,135,150,180]
VALUES_SIN = {
    0: "0",
    30: "1/2",
    45: "sqrt(2)/2",
    60: "sqrt(3)/2",
    90: "1",
    120: "sqrt(3)/2",
    135: "sqrt(2)/2",
    150: "1/2",
    180: "0",
}
VALUES_COS = {
    0: "1",
    30: "sqrt(3)/2",
    45: "sqrt(2)/2",
    60: "1/2",
    90: "0",
    120: "-1/2",
    135: "-sqrt(2)/2",
    150: "-sqrt(3)/2",
    180: "-1",
}

FUNCS = ["sin", "cos"]

def gen_example():
    f = random.choice(FUNCS)
    a = random.choice(ANGLES)
    v = VALUES_SIN[a] if f=="sin" else VALUES_COS[a]
    return f"{f}({a})={v}", {"f": f, "angle": a}


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
    task_meta = {"name": "trigo", "sample_prompts": ["sin(30)=", "cos(45)="]}
    return tokenizer, loaders, task_meta
