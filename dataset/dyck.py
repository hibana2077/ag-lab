"""Dyck language (balanced parentheses) synthetic dataset."""
import random
from torch.utils.data import DataLoader
from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

PAIRS = ["()", "[]", "{}", "<>"]


def gen_sequence(max_len=128, max_depth=8, noise_prob=0.0):
    target_len = random.randint(max_len // 2, max_len)
    stack = []
    s = []
    depth_reached = 0
    while len(s) < target_len:
        if stack and (random.random() < 0.5 or len(stack) >= max_depth):
            open_idx = stack.pop()
            s.append(PAIRS[open_idx][1])
        else:
            open_idx = random.randrange(len(PAIRS))
            stack.append(open_idx)
            s.append(PAIRS[open_idx][0])
            depth_reached = max(depth_reached, len(stack))
        if not stack and len(s) >= target_len * 0.8:
            break
    while stack:
        s.append(PAIRS[stack.pop()][1])
    seq = "".join(s)
    if noise_prob and random.random() < noise_prob:
        pos = random.randrange(len(seq))
        orig = seq[pos]
        choices = [c for c in "()[]{}<>" if c != orig]
        seq = seq[:pos] + random.choice(choices) + seq[pos + 1 :]
    return seq, depth_reached


def make_split(args, size):
    texts = []
    metas = []
    for _ in range(size):
        seq, depth = gen_sequence(max_len=args.context_len - 2, max_depth=8, noise_prob=0.0)
        metas.append({"depth": depth})
        texts.append(seq)
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
    task_meta = {"name": "dyck", "sample_prompts": ["(([[{{", "<[{("]}
    return tokenizer, loaders, task_meta
