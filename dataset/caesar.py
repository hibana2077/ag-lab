"""Caesar cipher shift task.
Format: <plain>@<shift>#<cipher>
Model must output cipher after '#'.
"""
import random
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

ALPH = "abcdefghijklmnopqrstuvwxyz"

def caesar(s: str, k: int) -> str:
    return "".join(ALPH[(ALPH.index(c)+k)%26] for c in s)


def gen_example(max_len: int):
    L = random.randint(4, max(4, min(20, max_len // 5)))
    s = "".join(random.choice(ALPH) for _ in range(L))
    k = random.randint(1, 25)
    return f"{s}@{k}#{caesar(s,k)}", {"src": s, "k": k}


def make_split(args, size):
    texts, metas = [], []
    for _ in range(size):
        t, m = gen_example(args.context_len - 6)
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
    task_meta = {"name": "caesar", "sample_prompts": ["abc@3#", "hello@5#"]}
    return tokenizer, loaders, task_meta
