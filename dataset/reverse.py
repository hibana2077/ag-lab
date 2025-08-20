"""Reverse sequence task.

Sample format: <seq>#<reversed_seq>
Model must learn to copy and reverse after delimiter.
"""
import random
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

ALPH = "abcdefghijklmnopqrstuvwxyz"


def gen_example(max_len: int):
    L = random.randint(4, max(4, min(20, max_len // 4)))
    s = "".join(random.choice(ALPH) for _ in range(L))
    return f"{s}#{s[::-1]}", {"src": s}


def make_split(args, size):
    texts, metas = [], []
    for _ in range(size):
        t, m = gen_example(args.context_len - 2)
        texts.append(t)
        metas.append(m)
    return texts, metas


def build_loaders(args):
    train, mtr = make_split(args, args.train_size)
    val, mv = make_split(args, args.val_size)
    test, mte = make_split(args, args.test_size)
    tokenizer = build_tokenizer_from_texts(CharTokenizer, train)
    ds_train = TextDataset(train, mtr, tokenizer, args.context_len)
    ds_val = TextDataset(val, mv, tokenizer, args.context_len)
    ds_test = TextDataset(test, mte, tokenizer, args.context_len)
    loaders = {
        "train": DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
    }
    task_meta = {"name": "reverse", "sample_prompts": ["abcd#", "xyz#"]}
    return tokenizer, loaders, task_meta
