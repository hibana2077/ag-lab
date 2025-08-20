"""Tiny Shakespeare dataset wrapper.

Primary source: Hugging Face dataset ``tiny_shakespeare``.
Falls back to a short embedded snippet if the ``datasets`` package or network
access is unavailable so that training code still runs in offline mode.

Public functions intentionally mirror the other dataset modules:
    - ``make_splits(args)``
    - ``build_loaders(args)``

``args`` is expected to expose attributes: ``context_len``, ``train_size``,
``val_size``, ``test_size``, ``batch_size`` (same contract as before).
"""

from typing import List
import random
from torch.utils.data import DataLoader

from .common import TextDataset, collate_fn, build_tokenizer_from_texts
from src.utils.tokenizer import CharTokenizer

# Minimal fallback snippet (small so tokenizer builds quickly offline)
FALLBACK_SNIPPET = (
    "FIRST CITIZEN: Before we proceed any further, hear me speak.\n"
    "ALL: Speak, speak.\n"
    "FIRST CITIZEN: You are all resolved rather to die than to famish?\n"
    "ALL: Resolved. resolved.\n"
    "FIRST CITIZEN: First, you know Caius Marcius is chief enemy to the people.\n"
)


def _load_corpus_text() -> str:
    """Return full corpus text.

    Tries the Hugging Face ``tiny_shakespeare`` dataset. If unavailable (either
    because ``datasets`` isn't installed or there is no network), returns the
    fallback snippet repeated to approximate length for experiments.
    """
    try:  # pragma: no cover - defensive import
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("tiny_shakespeare")  # usually only a 'train' split
        # Some variants have a single long line in 'train'. Join all rows just in case.
        texts: List[str] = ds["train"]["text"]  # type: ignore[index]
        corpus = "\n".join(texts)
        # Strip any trailing whitespace to make chunking deterministic.
        corpus = corpus.strip("\n")
        if len(corpus) < 1000:  # Very unlikely, but keep a minimum length
            corpus = (corpus + "\n" + FALLBACK_SNIPPET) * 5
        return corpus
    except Exception:  # noqa: BLE001 - broad on purpose: offline / missing pkg
        # Provide a deterministic pseudo-corpus by repeating fallback.
        return FALLBACK_SNIPPET * 50


def _chunk_corpus(corpus: str, context_len: int) -> List[str]:
    """Split corpus into non-overlapping chunks sized for examples.

    We subtract 2 to leave room for BOS/EOS or special tokens (mirroring the
    original implementation).
    Very short tail chunks ( < 1/3 of main chunk length ) are dropped.
    """
    chunk_len = max(1, context_len - 2)
    chunks = [corpus[i : i + chunk_len] for i in range(0, len(corpus), chunk_len)]
    if chunks and len(chunks[-1]) < chunk_len // 3:
        chunks.pop()
    return chunks


def make_splits(args):  # preserves previous signature
    corpus = _load_corpus_text()
    chunks = _chunk_corpus(corpus, args.context_len)
    random.shuffle(chunks)
    train = chunks[: args.train_size]
    val = chunks[args.train_size : args.train_size + args.val_size]
    test = chunks[args.train_size + args.val_size : args.train_size + args.val_size + args.test_size]

    def metas(n):
        return [{} for _ in range(n)]

    return train, metas(len(train)), val, metas(len(val)), test, metas(len(test))


def build_loaders(args):  # preserves previous signature
    train, mtr, val, mv, test, mte = make_splits(args)
    tokenizer = build_tokenizer_from_texts(CharTokenizer, train)
    ds_train = TextDataset(train, mtr, tokenizer, args.context_len)
    ds_val = TextDataset(val, mv, tokenizer, args.context_len)
    ds_test = TextDataset(test, mte, tokenizer, args.context_len)
    loaders = {
        "train": DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
    }
    task_meta = {
        "name": "tiny_shakespeare",
        "sample_prompts": ["ROMEO:", "JULIET:", "HAMLET:", "FIRST CITIZEN:"],
    }
    return tokenizer, loaders, task_meta
