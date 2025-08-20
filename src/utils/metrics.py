from __future__ import annotations
import math
from typing import List, Dict


def cross_entropy(loss_val: float):
    return loss_val


def bpc(loss_val: float, base: int):
    return loss_val / math.log(2)


def perplexity(loss_val: float):
    try:
        return math.exp(loss_val)
    except OverflowError:
        return float("inf")


def exact_match(preds: List[str], refs: List[str]):
    correct = sum(p == r for p, r in zip(preds, refs))
    return correct / max(1, len(refs))


def max_depth(s: str):
    depth = 0
    max_d = 0
    pairs = {')': '(', ']': '[', '}': '{', '>': '<'}
    for ch in s:
        if ch in '([{<':
            depth += 1
            max_d = max(max_d, depth)
        elif ch in pairs:
            depth = max(0, depth - 1)
    return max_d


def max_depth_acc(preds: List[str], refs: List[str]):
    correct = sum(max_depth(p) == max_depth(r) for p, r in zip(preds, refs))
    return correct / max(1, len(refs))


def hit_rate_by_gap(records: List[Dict]):
    from collections import defaultdict
    stats = defaultdict(lambda: [0, 0])
    for r in records:
        g = r["gap"]
        stats[g][1] += 1
        if r["correct"]:
            stats[g][0] += 1
    out = {int(g): stats[g][0] / stats[g][1] for g in stats}
    return dict(sorted(out.items(), key=lambda kv: kv[0]))
