from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from unlearning_project.benchmarks.base import BenchmarkAdapter
from unlearning_project.common import write_json, write_jsonl


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def _tokenize_for_hashing(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _hashed_ngram_embedding(text: str, dim: int, ngram_range: tuple[int, int]) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize_for_hashing(text)
    if not tokens:
        return vector
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        for start in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[start : start + n])
            bucket = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16) % dim
            vector[bucket] += 1.0 / math.sqrt(n)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def _encode_texts_hashed(texts: list[str], dim: int, ngram_range: tuple[int, int]) -> np.ndarray:
    return np.stack([_hashed_ngram_embedding(text, dim=dim, ngram_range=ngram_range) for text in texts])


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-8)
    return summed / counts


def _encode_texts_transformer(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    torch_dtype: str,
    device: str | None,
    trust_remote_code: bool,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=_dtype_from_name(torch_dtype),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    if device:
        model.to(device)
    elif torch.cuda.is_available():
        model.to("cuda")

    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch)
        pooled = _mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        pooled = F.normalize(pooled.float(), dim=-1)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def compute_text_embeddings(texts: list[str], cfg: dict[str, Any]) -> np.ndarray:
    backend = str(cfg.get("backend", "hashed_ngram"))
    if backend == "hashed_ngram":
        dim = int(cfg.get("hash_dim", 2048))
        min_n = int(cfg.get("min_ngram", 1))
        max_n = int(cfg.get("max_ngram", 2))
        return _encode_texts_hashed(texts, dim=dim, ngram_range=(min_n, max_n))
    if backend == "transformer_embedding":
        model_name = str(cfg["encoder_model_name"])
        return _encode_texts_transformer(
            texts=texts,
            model_name=model_name,
            batch_size=int(cfg.get("batch_size", 8)),
            max_length=int(cfg.get("max_length", 256)),
            torch_dtype=str(cfg.get("torch_dtype", "float16")),
            device=cfg.get("device"),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        )
    raise ValueError(f"Unknown augmentation backend: {backend}")


def build_geometry_augmentations(
    adapter: BenchmarkAdapter,
    forget_samples: list[dict[str, Any]],
    retain_candidates: list[dict[str, Any]],
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    strategy = str(cfg.get("strategy", "none"))
    if strategy == "none":
        manifest = {
            "enabled": False,
            "strategy": "none",
            "augmented_retain_path": None,
            "augmented_count": 0,
        }
        write_json(manifest, output_dir / "manifest.json")
        return manifest

    if strategy != "geometry_retain_neighbors":
        raise ValueError(f"Unknown augmentation strategy: {strategy}")

    if not forget_samples or not retain_candidates:
        manifest = {
            "enabled": True,
            "strategy": strategy,
            "reason": "empty_forget_or_retain_pool",
            "augmented_retain_path": str(output_dir / "augmented_retain.jsonl"),
            "augmented_count": 0,
        }
        write_jsonl([], output_dir / "augmented_retain.jsonl")
        write_json(manifest, output_dir / "manifest.json")
        return manifest

    per_forget = int(cfg.get("neighbors_per_forget", 2))
    candidate_limit = cfg.get("candidate_pool_limit")
    if candidate_limit is not None:
        retain_candidates = retain_candidates[: int(candidate_limit)]

    forget_texts = [adapter.sample_to_embedding_text(sample) for sample in forget_samples]
    retain_texts = [adapter.sample_to_embedding_text(sample) for sample in retain_candidates]
    forget_embeddings = compute_text_embeddings(forget_texts, cfg)
    retain_embeddings = compute_text_embeddings(retain_texts, cfg)

    scores = forget_embeddings @ retain_embeddings.T
    augmented_rows: list[dict[str, Any]] = []
    linkage: list[dict[str, Any]] = []
    used_pairs: set[tuple[int, int]] = set()
    for forget_idx, forget_sample in enumerate(forget_samples):
        ranking = np.argsort(scores[forget_idx])[::-1]
        added = 0
        for candidate_idx in ranking.tolist():
            pair = (forget_idx, candidate_idx)
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            candidate = dict(retain_candidates[candidate_idx])
            score = float(scores[forget_idx, candidate_idx])
            candidate["augmentation_metadata"] = {
                "strategy": strategy,
                "similarity": score,
                "source_forget_id": adapter.sample_id(forget_sample),
                "source_forget_index": forget_idx,
            }
            augmented_rows.append(candidate)
            linkage.append(
                {
                    "forget_index": forget_idx,
                    "forget_id": adapter.sample_id(forget_sample),
                    "candidate_index": candidate_idx,
                    "candidate_id": adapter.sample_id(retain_candidates[candidate_idx]),
                    "similarity": score,
                }
            )
            added += 1
            if added >= per_forget:
                break

    augmented_path = output_dir / "augmented_retain.jsonl"
    linkage_path = output_dir / "neighbor_links.json"
    write_jsonl(augmented_rows, augmented_path)
    write_json(linkage, linkage_path)

    manifest = {
        "enabled": True,
        "strategy": strategy,
        "backend": cfg.get("backend", "hashed_ngram"),
        "augmented_retain_path": str(augmented_path),
        "neighbor_links_path": str(linkage_path),
        "augmented_count": len(augmented_rows),
        "retain_candidate_count": len(retain_candidates),
        "forget_count": len(forget_samples),
        "neighbors_per_forget": per_forget,
    }
    write_json(manifest, output_dir / "manifest.json")
    return manifest
