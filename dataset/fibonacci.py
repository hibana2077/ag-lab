"""Fibonacci continuation task.
Format: fib<n>=<value> or sequence generation: <n1>,<n2>,...,<nk>#<next>
We'll implement simple next value prediction using small n.
"""
import random
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

MAX_N = 20

# precompute
F = [0,1]
for i in range(2, MAX_N+2):
    F.append(F[-1]+F[-2])

def gen_example():
    k = random.randint(3, 7)
    start = random.randint(0, 5)
    seq = F[start:start+k]
    nxt = F[start+k]
    seq_str = ",".join(str(x) for x in seq)
    return f"{seq_str}#{nxt}", {"seq": seq}


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
    task_meta = {"name": "fibonacci", "sample_prompts": ["0,1,1#", "2,3,5,8#"]}
    return tokenizer, loaders, task_meta
