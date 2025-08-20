import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any

PAD_TOKEN = "<pad>"
PAD_ID = 0


@dataclass
class Sample:
    text: str
    meta: Dict[str, Any]


class TextDataset(Dataset):
    def __init__(self, texts: List[str], metas: List[Dict[str, Any]], tokenizer, context_len: int):
        assert len(texts) == len(metas)
        self.texts = texts
        self.metas = metas
        self.tokenizer = tokenizer
        self.context_len = context_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        meta = self.metas[idx]
        ids = self.tokenizer.encode(text)
        if len(ids) > self.context_len:
            ids = ids[: self.context_len]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "length": len(ids),
            "text": text,
            "meta": meta,
        }


def collate_fn(batch):
    max_len = max(item["length"] for item in batch)
    pad_id = 0
    B = len(batch)
    x = torch.full((B, max_len), pad_id, dtype=torch.long)
    y = torch.full((B, max_len), pad_id, dtype=torch.long)
    lengths = []
    texts = []
    metas = []
    for i, item in enumerate(batch):
        ids = item["ids"]
        l = item["length"]
        lengths.append(l)
        texts.append(item["text"])
        metas.append(item["meta"])
        x[i, :l] = ids
        if l > 1:
            y[i, : l - 1] = ids[1:]
            y[i, l - 1] = pad_id
    attn_mask = (x != pad_id).long()
    return {
        "x": x,
        "y": y,
        "lengths": lengths,
        "texts": texts,
        "metas": metas,
        "attn_mask": attn_mask,
    }


def build_tokenizer_from_texts(tokenizer_cls, texts: List[str]):
    corpus = "".join(texts)
    return tokenizer_cls.build(corpus)
