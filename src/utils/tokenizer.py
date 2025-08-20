from __future__ import annotations
import json
from typing import List

SPECIAL = ["<pad>", "<bos>", "<eos>"]


class CharTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def build(cls, corpus: str, extra_tokens: List[str] = SPECIAL):
        chars = sorted(set(corpus))
        vocab = list(extra_tokens) + chars
        seen = set()
        dedup = []
        for ch in vocab:
            if ch not in seen:
                seen.add(ch)
                dedup.append(ch)
        return cls(dedup)

    def encode(self, s: str) -> List[int]:
        ids = [self.stoi["<bos>"]] + [self.stoi.get(ch, self.stoi["<pad>"]) for ch in s] + [self.stoi["<eos>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i == self.stoi["<eos>"]:
                break
            if i in self.itos and self.itos[i] not in SPECIAL:
                out.append(self.itos[i])
        return "".join(out)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)
