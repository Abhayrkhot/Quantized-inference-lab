"""Evaluation and calibration data, following the GPTQ paper's protocol.

* WikiText-2 (raw) test split: documents joined with ``"\\n\\n"``, tokenized once, cut into
  non-overlapping windows of ``seqlen`` tokens (the tail is dropped).
* C4 validation: the first 1100 documents of validation shard 0 joined with spaces; the first
  ``256 * seqlen`` tokens, cut into 256 windows.
* Calibration: ``n`` windows from C4 train shard 0. Each one is a random ``seqlen`` slice of a
  random document that is longer than ``seqlen`` tokens (seeded).

Tokenized windows are cached under ``$QIL_DATA/cache/qlab`` keyed by tokenizer and protocol.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import torch

C4_TRAIN = "en/c4-train.00000-of-01024.json.gz"
C4_VAL = "en/c4-validation.00000-of-00008.json.gz"


def _cache_dir() -> Path:
    root = Path(os.environ.get("QIL_DATA", Path.home() / "qildata")) / "cache" / "qlab"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cached(key: str, build):
    tag = hashlib.sha1(key.encode()).hexdigest()[:16]
    p = _cache_dir() / f"{tag}.pt"
    if p.exists():
        return torch.load(p)
    t = build()
    torch.save(t, p)
    return t


def _tok_id(tokenizer) -> str:
    return f"{tokenizer.name_or_path}|{len(tokenizer)}"


def _windows(ids: torch.Tensor, seqlen: int, limit: int | None = None) -> torch.Tensor:
    n = ids.numel() // seqlen
    if limit is not None:
        n = min(n, limit)
    return ids[: n * seqlen].view(n, seqlen).clone()


def wikitext2(tokenizer, seqlen: int = 2048, split: str = "test") -> torch.Tensor:
    def build():
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        return _windows(ids, seqlen)

    return _cached(f"wikitext2|{split}|{seqlen}|{_tok_id(tokenizer)}", build)


def c4_val(tokenizer, seqlen: int = 2048, n: int = 256) -> torch.Tensor:
    def build():
        from datasets import load_dataset

        ds = load_dataset("allenai/c4", data_files={"validation": C4_VAL}, split="validation")
        ids = tokenizer(" ".join(ds[:1100]["text"]), return_tensors="pt").input_ids[0]
        return _windows(ids, seqlen, limit=n)

    return _cached(f"c4val|{seqlen}|{n}|{_tok_id(tokenizer)}", build)


def c4_calib(tokenizer, n: int = 128, seqlen: int = 2048, seed: int = 0) -> torch.Tensor:
    def build():
        from datasets import load_dataset

        ds = load_dataset("allenai/c4", data_files={"train": C4_TRAIN}, split="train")
        rng = random.Random(seed)
        out = []
        while len(out) < n:
            doc = ds[rng.randrange(len(ds))]["text"]
            ids = tokenizer(doc, return_tensors="pt").input_ids[0]
            if ids.numel() <= seqlen:
                continue
            start = rng.randrange(ids.numel() - seqlen)
            out.append(ids[start : start + seqlen])
        return torch.stack(out)

    return _cached(f"c4calib|{n}|{seqlen}|{seed}|{_tok_id(tokenizer)}", build)


DATASETS = {"wikitext2": wikitext2, "c4": c4_val}
