"""Run-length encoding (RLE) task.
Format: <input>|<rle>
Example: aaabbc -> aaabbc|a3b2c1
"""
import random
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

ALPH = "aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz"

def rle_encode(s: str) -> str:
    if not s:
        return ""
    out = []
    cur = s[0]
    count = 1
    for c in s[1:]:
        if c == cur:
            count += 1
        else:
            out.append(f"{cur}{count}")
            cur = c
            count = 1
    out.append(f"{cur}{count}")
    return "".join(out)


def gen_example(max_len: int):
    L = random.randint(4, max(4, min(30, max_len // 4)))
    s = "".join(random.choice(ALPH) for _ in range(L))
    return f"{s}|{rle_encode(s)}", {"src": s}


def make_split(args, size):
    texts, metas = [], []
    for _ in range(size):
        t, m = gen_example(args.context_len - 4)
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
    task_meta = {"name": "rle", "sample_prompts": ["aaabb|", "hhhhii|"]}
    return tokenizer, loaders, task_meta
