"""Needle-in-a-haystack retrieval task."""
import random
import string
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

ALPH = string.ascii_lowercase + " "


def gen_example(context_len: int, needle_len: int = 6):
    needle = "".join(random.choice(string.ascii_lowercase) for _ in range(needle_len))
    prefix_len = random.randint(10, max(10, context_len - needle_len * 2 - 5))
    prefix = "".join(random.choice(ALPH) for _ in range(prefix_len))
    sep = "#"
    seq = f"{prefix}{needle}{sep}{needle}"
    gap = len(prefix)
    return seq, {"needle": needle, "gap": gap}


def make_split(args, size):
    texts = []
    metas = []
    usable_len = args.context_len - 2
    for _ in range(size):
        t, m = gen_example(usable_len)
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
    task_meta = {"name": "needle", "sample_prompts": ["abcde#", "xyz#"]}
    return tokenizer, loaders, task_meta
