"""Long integer addition dataset: 'a+b=' followed by sum."""
import random
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer


def gen_example(n_digits=4):
    a = random.randint(0, 10**n_digits - 1)
    b = random.randint(0, 10**n_digits - 1)
    s = a + b
    sample = f"{a}+{b}={s}"
    return sample, {"a": a, "b": b, "s": s}


def make_split(args, size):
    texts = []
    metas = []
    for _ in range(size):
        t, m = gen_example(n_digits=max(2, min(6, args.context_len // 8)))
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
    task_meta = {"name": "addition", "sample_prompts": ["12+34=", "999+1="]}
    return tokenizer, loaders, task_meta
