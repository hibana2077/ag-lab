"""Polynomial coefficient prediction.
Format: poly:a,b,c|x=<x>#<y> where y = ax^2+bx+c.
"""
import random
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

def gen_example():
    a = random.randint(-3,3)
    b = random.randint(-5,5)
    c = random.randint(-5,5)
    x = random.randint(-5,5)
    y = a*x*x + b*x + c
    return f"poly:{a},{b},{c}|x={x}#{y}", {"a":a,"b":b,"c":c,"x":x}


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
    task_meta = {"name": "poly", "sample_prompts": ["poly:1,2,3|x=4#", "poly:-1,0,2|x=3#"]}
    return tokenizer, loaders, task_meta
