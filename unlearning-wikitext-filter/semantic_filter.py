"""Semantic Filtering Pipeline: WikiText Collateral Damage Detection.

This module answers two key design questions:

  Q1 – How to search at the representation level?
       Run the base model M with output_hidden_states=True. Extract the hidden
       state at the last transformer layer and the last real-token position
       (the decoder-model equivalent of [EOS]).  This gives h_M(x) ∈ R^d.

       - Forget anchor:  V_forget = mean_{x ∈ D_forget} h_M(x)
         Centroid of the forget set in representation space.
       - Per-passage score:  score(w) = cosine_sim(h_M(w), V_forget)
       - Vulnerable subset:  D* = { w ∈ Wikitext : score(w) ≥ θ }

       Because WikiText is general encyclopedic text and D_forget is TOFU
       fictional-author biographies, any passage ending up in D* is "unrelated"
       by topic but shares the same representational neighbourhood — it is
       genuine collateral damage, not near-duplicate forget content.

  Q2 – How to search efficiently across hundreds of WikiText lines?
       Two-level strategy:
         a) Batch encoding — feed all passages through the model in batches
            (e.g. batch_size=32), reducing GPU passes from N to N/B.
         b) FAISS index — after encoding, build a FAISS IndexFlatIP over
            L2-normalised embeddings (inner product = cosine similarity).
            Retrieval is a single nearest-neighbour query: O(N·d) build once,
            O(d) per query.  For WikiText-2 test (~2 000 passages) this is
            negligible; scales to millions with IVF/HNSW variants.

Reference: §4 of "Protecting Model Utility from Unlearning: A Semantic
Filtering Approach" (Su, Yan, Le 2026).

Usage:
  python semantic_filter.py
  python semantic_filter.py --threshold 0.3 --topk 300 --layer -1
  python semantic_filter.py --unlearned_model <hf_id> --threshold 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# Q1 Answer — Representation-level embedding extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_embeddings(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    layer: int = -1,
    max_length: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Extract the last-token hidden state at a specified transformer layer.

    For decoder-only models (Llama), the last non-padding token is the
    representation equivalent of the [EOS] position in the paper:

        h_M(x) = hidden_states[layer][ last_real_token_position ]

    All texts are processed in batches; the model is run once per batch.
    Right-padding is used so the last *real* token index = sum(attention_mask) − 1.

    Args:
        layer: transformer layer to extract from.
               -1 = last layer (default), -2 = second-to-last, etc.

    Returns:
        float32 ndarray of shape (len(texts), hidden_size)
    """
    model.eval()
    all_embs: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors, each (B, T, D)
        hidden = out.hidden_states[layer]  # (B, T, D)

        # Last real (non-padding) token index per sequence
        last_real = enc["attention_mask"].sum(dim=1) - 1  # (B,)

        for b, pos in enumerate(last_real):
            emb = hidden[b, pos.item(), :].float().cpu().numpy()
            all_embs.append(emb)

        del out, hidden, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        done = min(start + batch_size, len(texts))
        logger.info("  Encoded %d / %d", done, len(texts))

    return np.array(all_embs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Semantic anchor V_forget  (paper §4 Step 1)
# ─────────────────────────────────────────────────────────────────────────────


def build_semantic_anchor(
    model: torch.nn.Module,
    tokenizer,
    forget_texts: list[str],
    batch_size: int = 16,
    layer: int = -1,
    device: str = "cpu",
) -> np.ndarray:
    """Compute V_forget = mean_{x ∈ D_forget} h_M(x).

    This is the centroid of the forget set in the model's representation
    space — the "semantic signature" of the knowledge to be removed.

    Returns:
        float32 ndarray of shape (hidden_size,)
    """
    logger.info("Step 1 — Building semantic anchor from %d forget texts", len(forget_texts))
    H = extract_embeddings(
        model,
        tokenizer,
        forget_texts,
        batch_size=batch_size,
        layer=layer,
        device=device,
    )
    V_forget = H.mean(axis=0)
    logger.info(
        "  V_forget: shape=%s  norm=%.4f", V_forget.shape, float(np.linalg.norm(V_forget))
    )
    return V_forget


# ─────────────────────────────────────────────────────────────────────────────
# Q2 Answer — WikiText chunking + FAISS-based efficient retrieval
# ─────────────────────────────────────────────────────────────────────────────


def load_wikitext_passages(
    split: str = "test",
    chunk_tokens: int = 128,
    stride_tokens: int = 64,
    tokenizer=None,
    min_chars: int = 50,
) -> list[str]:
    """Load WikiText-2-raw-v1 and slice into overlapping token-length passages.

    Efficiency note (Q2a):
        Chunking converts the raw corpus into a fixed-size passage list.
        A sliding window (chunk_tokens wide, stride_tokens step) is used so
        every passage fits in one model forward pass and boundary context is
        not lost.  WikiText-2 test ≈ 245 K tokens → ~2 000 passages at
        chunk=128, stride=64.

    Args:
        chunk_tokens:  passage width in tokens.
        stride_tokens: step between successive windows; overlap = chunk − stride.
        min_chars:     minimum character count per raw line (drops blank lines
                       and short headers).

    Returns:
        List of decoded passage strings.
    """
    logger.info("Loading WikiText-2-raw-v1 (%s split) …", split)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    raw_lines = [
        row["text"]
        for row in ds
        if len(row["text"].strip()) >= min_chars
        and not row["text"].strip().startswith("=")
    ]
    all_text = " ".join(raw_lines)

    if tokenizer is not None:
        token_ids = tokenizer.encode(all_text, add_special_tokens=False)
        passages: list[str] = []
        for s in range(0, len(token_ids) - chunk_tokens, stride_tokens):
            chunk_ids = token_ids[s : s + chunk_tokens]
            text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if text:
                passages.append(text)
        logger.info(
            "  %d passages from %d tokens (chunk=%d, stride=%d)",
            len(passages),
            len(token_ids),
            chunk_tokens,
            stride_tokens,
        )
    else:
        # Word-level fallback when no tokenizer is available
        words = all_text.split()
        half = chunk_tokens // 2
        passages = [
            " ".join(words[i : i + chunk_tokens])
            for i in range(0, len(words) - chunk_tokens, half)
        ]
        logger.info("  %d passages (word-level fallback)", len(passages))

    return passages


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS IndexFlatIP over L2-normalised embeddings.

    Efficiency note (Q2b):
        After L2 normalisation, dot-product == cosine similarity.
        IndexFlatIP performs exact cosine search.
        Build cost: O(N·d) — one-time.
        Query cost: O(N·d) per query vector — but N ≈ 2 000 for WikiText-2,
        so this is negligible (< 1 ms on CPU).

        For larger corpora (WikiText-103, ≈ 1 M passages) replace with
        faiss.IndexIVFFlat or IndexHNSWFlat for sub-linear query time.

    Returns:
        (faiss.Index, normalised_embeddings)
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Install faiss-cpu: pip install faiss-cpu") from exc

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalised = (embeddings / norms).astype(np.float32)

    index = faiss.IndexFlatIP(normalised.shape[1])
    index.add(normalised)
    logger.info(
        "  FAISS IndexFlatIP: %d vectors, dim=%d", index.ntotal, normalised.shape[1]
    )
    return index, normalised


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Vulnerable subset retrieval  (paper §4 Step 2)
# ─────────────────────────────────────────────────────────────────────────────


def retrieve_vulnerable_subset(
    index,
    passages: list[str],
    V_forget: np.ndarray,
    topk: int = 500,
    threshold: float = 0.0,
) -> tuple[list[str], np.ndarray]:
    """Retrieve WikiText passages most similar to V_forget.

        score(w) = cosine_sim(h_M(w), V_forget)
        D* = { w ∈ Wikitext : score(w) ≥ θ }

    Steps:
        1. L2-normalise V_forget (same space as the indexed embeddings).
        2. Query FAISS for the top-K nearest passages in O(N·d).
        3. Filter by cosine similarity ≥ θ.

    Returns:
        (vulnerable_passages, scores) — sorted by descending similarity.
    """
    v = V_forget.astype(np.float32).reshape(1, -1)
    v = v / (np.linalg.norm(v) + 1e-8)

    k_actual = min(topk, len(passages))
    scores, indices = index.search(v, k_actual)
    scores = scores[0]    # (k_actual,)
    indices = indices[0]  # (k_actual,)

    mask = scores >= threshold
    vulnerable = [passages[i] for i in indices[mask]]
    final_scores = scores[mask]

    logger.info(
        "  Retrieved %d passages (θ=%.3f, top-%d queried)",
        len(vulnerable),
        threshold,
        k_actual,
    )
    return vulnerable, final_scores


# ─────────────────────────────────────────────────────────────────────────────
# Irrelevance check — sim(D*, D_forget) ≤ δ at the content level
# ─────────────────────────────────────────────────────────────────────────────


def check_irrelevance(
    vulnerable_passages: list[str],
    forget_texts: list[str],
    delta: float = 0.3,
) -> dict:
    """Verify content-level irrelevance: D* must not be near-duplicate forget content.

    Uses word-level Jaccard similarity as a lightweight proxy for lexical overlap.
    A passage is "suspicious" if its Jaccard overlap with the forget vocabulary
    exceeds δ.  In practice this should be near zero for Wikitext vs. TOFU.

    Returns:
        dict with mean_jaccard, max_jaccard, fraction_exceeding_delta
    """
    forget_vocab: set[str] = set()
    for t in forget_texts:
        forget_vocab.update(t.lower().split())

    overlaps: list[float] = []
    for p in vulnerable_passages:
        words = set(p.lower().split())
        if not words:
            overlaps.append(0.0)
            continue
        overlaps.append(len(words & forget_vocab) / len(words | forget_vocab))

    arr = np.array(overlaps)
    result = {
        "mean_jaccard": float(arr.mean()) if len(arr) else 0.0,
        "max_jaccard": float(arr.max()) if len(arr) else 0.0,
        "fraction_exceeding_delta": float((arr > delta).mean()) if len(arr) else 0.0,
        "delta": delta,
        "n_passages": len(vulnerable_passages),
    }
    logger.info(
        "  Irrelevance — mean_jaccard=%.4f  max=%.4f  frac>%.2f: %.4f",
        result["mean_jaccard"],
        result["max_jaccard"],
        delta,
        result["fraction_exceeding_delta"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Side Effect Index (SEI)  (paper §4 Step 3)
# ─────────────────────────────────────────────────────────────────────────────


def compute_avg_ppl(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 256,
    device: str = "cpu",
) -> float:
    """Compute average per-token perplexity of model over a text list."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        # Mask padding tokens so they don't contribute to loss
        labels = ids.clone()
        labels[mask == 0] = -100

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, labels=labels)

        # out.loss = mean NLL over non-masked label positions in the batch.
        # n_label_tokens = total real tokens minus one per sequence (shifted labels).
        n_label_tokens = int((mask.sum() - mask.shape[0]).item())
        total_nll += out.loss.item() * n_label_tokens
        total_tokens += n_label_tokens

        del out, ids, mask, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(np.exp(total_nll / max(total_tokens, 1)))


def compute_sei(
    base_model: torch.nn.Module,
    unlearned_model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    device: str = "cpu",
) -> dict:
    """Compute SEI(S) = Avg_PPL_M'(S) / Avg_PPL_M(S).

    SEI > 1.0  →  perplexity increased after unlearning (collateral damage)
    SEI ≈ 1.0  →  no detectable utility loss on S

    Returns:
        dict with ppl_base, ppl_unlearned, sei
    """
    logger.info("Step 3 — Computing SEI over %d passages", len(texts))
    ppl_base = compute_avg_ppl(base_model, tokenizer, texts, batch_size=batch_size, device=device)
    logger.info("  PPL_base = %.4f", ppl_base)
    ppl_unlearned = compute_avg_ppl(
        unlearned_model, tokenizer, texts, batch_size=batch_size, device=device
    )
    logger.info("  PPL_unlearned = %.4f", ppl_unlearned)
    sei = ppl_unlearned / ppl_base
    logger.info("  SEI = %.4f", sei)
    return {"ppl_base": ppl_base, "ppl_unlearned": ppl_unlearned, "sei": sei}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Semantic Filtering Pipeline — identifies WikiText passages vulnerable "
            "to unlearning collateral damage via representation-level similarity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base_model",
        default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        help="HuggingFace ID of the base (pre-unlearning) model",
    )
    p.add_argument(
        "--unlearned_model",
        default=None,
        help="HuggingFace ID of the unlearned model (enables Step 3 SEI computation)",
    )
    p.add_argument(
        "--forget_dataset",
        default="locuslab/TOFU",
        help="HuggingFace dataset ID for the forget set",
    )
    p.add_argument("--forget_split", default="forget10")
    p.add_argument(
        "--forget_question_field",
        default="question",
        help="Text field containing the question/prompt in the forget dataset",
    )
    p.add_argument(
        "--forget_answer_field",
        default="answer",
        help="Text field containing the answer in the forget dataset (concatenated with question)",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer layer for hidden-state extraction (-1 = last, -2 = penultimate, …)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Encoding batch size (reduce if OOM)",
    )
    p.add_argument(
        "--chunk_tokens",
        type=int,
        default=128,
        help="WikiText passage width in tokens",
    )
    p.add_argument(
        "--stride_tokens",
        type=int,
        default=64,
        help="Sliding-window stride (overlap = chunk − stride)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Cosine similarity threshold θ for D* (higher = stricter, smaller D*)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=500,
        help="Top-K candidates to retrieve from FAISS before threshold filtering",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output_semantic_filter"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    logger.info("Device: %s", device)

    # ── Load base model & tokenizer ──────────────────────────────────────────
    logger.info("Loading base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if "cuda" in device else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype
    ).to(device)

    # ── Build forget-set text list ───────────────────────────────────────────
    logger.info("Loading forget set: %s / %s", args.forget_dataset, args.forget_split)
    forget_ds = load_dataset(args.forget_dataset, split=args.forget_split)
    cols = forget_ds.column_names

    if args.forget_answer_field in cols:
        forget_texts = [
            row[args.forget_question_field] + " " + row[args.forget_answer_field]
            for row in forget_ds
        ]
    else:
        forget_texts = [row[args.forget_question_field] for row in forget_ds]
    logger.info("Forget set: %d samples", len(forget_texts))

    # ── Step 1 — Semantic anchor ─────────────────────────────────────────────
    V_forget = build_semantic_anchor(
        base_model,
        tokenizer,
        forget_texts,
        batch_size=args.batch_size,
        layer=args.layer,
        device=device,
    )
    np.save(args.output_dir / "V_forget.npy", V_forget)
    logger.info("Saved V_forget → %s/V_forget.npy", args.output_dir)

    # ── Step 2a — Chunk WikiText ─────────────────────────────────────────────
    logger.info("Step 2 — Chunking WikiText …")
    passages = load_wikitext_passages(
        split="test",
        chunk_tokens=args.chunk_tokens,
        stride_tokens=args.stride_tokens,
        tokenizer=tokenizer,
    )

    # ── Step 2b — Batch-encode all passages (Q2 answer) ─────────────────────
    logger.info("Encoding %d WikiText passages (batch_size=%d) …", len(passages), args.batch_size)
    wiki_embs = extract_embeddings(
        base_model,
        tokenizer,
        passages,
        batch_size=args.batch_size,
        layer=args.layer,
        device=device,
    )
    np.save(args.output_dir / "wiki_embeddings.npy", wiki_embs)
    logger.info("Saved wiki_embeddings → %s/wiki_embeddings.npy", args.output_dir)

    # ── Step 2c — FAISS index + retrieval ───────────────────────────────────
    index, _ = build_faiss_index(wiki_embs)
    D_star, scores = retrieve_vulnerable_subset(
        index, passages, V_forget, topk=args.topk, threshold=args.threshold
    )

    # ── Irrelevance verification ─────────────────────────────────────────────
    irrelevance = check_irrelevance(D_star, forget_texts)

    # ── Save D* ──────────────────────────────────────────────────────────────
    output: dict = {
        "config": {
            "base_model": args.base_model,
            "layer": args.layer,
            "chunk_tokens": args.chunk_tokens,
            "stride_tokens": args.stride_tokens,
            "threshold": args.threshold,
            "topk": args.topk,
        },
        "stats": {
            "n_total_passages": len(passages),
            "n_vulnerable": len(D_star),
            "fraction_vulnerable": len(D_star) / max(len(passages), 1),
        },
        "irrelevance": irrelevance,
        "D_star": [
            {"text": text, "cosine_sim": float(s)} for text, s in zip(D_star, scores)
        ],
    }
    tag = f"layer{args.layer}_theta{args.threshold:.2f}"
    out_path = args.output_dir / f"D_star_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Saved D* (%d passages) → %s", len(D_star), out_path)

    # ── Step 3 (optional) — SEI ──────────────────────────────────────────────
    if args.unlearned_model and D_star:
        logger.info("Loading unlearned model: %s", args.unlearned_model)
        unlearned_model = AutoModelForCausalLM.from_pretrained(
            args.unlearned_model, torch_dtype=dtype
        ).to(device)
        sei_result = compute_sei(
            base_model,
            unlearned_model,
            tokenizer,
            D_star,
            batch_size=args.batch_size,
            device=device,
        )
        sei_path = args.output_dir / f"sei_{tag}.json"
        with open(sei_path, "w") as f:
            json.dump(sei_result, f, indent=2)
        logger.info("SEI(D*) = %.4f  → %s", sei_result["sei"], sei_path)


if __name__ == "__main__":
    main()
