"""
Task Vector Heat Map for Pareto-Optimal Unlearning Models.

Task vector τ = θ_model − θ_base (model minus base model).
Computes pairwise cosine similarity between task vectors from:
  - retain90 (reference)
  - unlearned models (GradDiff pareto variants from pareto_comparison.json)

Reference: explore_open_unlearning_models.ipynb, Task Arithmetic Figure 5 style.

Usage:
  python task_vector_heatmap.py
  python task_vector_heatmap.py --pareto_json ../unlearning-wikitext-filter/output_pareto/pareto_comparison.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

# Param patterns: attention and MLP weights (Llama architecture)
ATTN_PATTERNS = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]
MLP_PATTERNS = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
MLP_PATTERNS_ALT = ["mlp.fc1.weight", "mlp.fc2.weight", "fc1.weight", "fc2.weight"]


def get_param_patterns():
    return ATTN_PATTERNS + MLP_PATTERNS + MLP_PATTERNS_ALT


def param_matches(name: str) -> bool:
    for p in get_param_patterns():
        if p in name:
            return True
    return False


def load_state_dict(model_id: str, revision=None, device: str = "cpu") -> dict:
    """Load model and return state_dict in float32 on CPU."""
    kwargs = {"torch_dtype": torch.float32}
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = model.to(device)
    return {k: v.cpu().float().numpy() for k, v in model.state_dict().items()}


def compute_task_vector(base_sd: dict, model_sd: dict) -> dict:
    """τ_k = θ_k(model) − θ_k(base) for matching params."""
    delta = {}
    for k in base_sd:
        if k not in model_sd:
            continue
        if not param_matches(k):
            continue
        b = base_sd[k]
        m = model_sd[k]
        if b.shape != m.shape:
            continue
        delta[k] = (m - b).astype(np.float64)
    return delta


def flatten_to_vector(delta: dict) -> np.ndarray:
    """Flatten task vector dict to 1D vector for cosine similarity."""
    keys = sorted(delta.keys())
    parts = [delta[k].flatten() for k in keys]
    return np.concatenate(parts).astype(np.float64)


def cosine_similarity_matrix(vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Pairwise cosine similarity matrix. Returns (n, n), diagonal = 1."""
    names = list(vectors.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i, na in enumerate(names):
        v_i = vectors[na].astype(np.float64)
        norm_i = np.linalg.norm(v_i)
        if norm_i < 1e-20:
            mat[i, i] = 1.0
            continue
        v_i = v_i / norm_i
        for j in range(i, n):
            nb = names[j]
            v_j = vectors[nb].astype(np.float64)
            norm_j = np.linalg.norm(v_j)
            if norm_j < 1e-20:
                sim = 1.0 if i == j else 0.0
            else:
                sim = float(np.dot(v_i, v_j / norm_j))
            mat[i, j] = mat[j, i] = sim
    return mat


def plot_heatmap(
    sim_matrix: np.ndarray,
    labels: list[str],
    path: Path,
    title: str = "Task Vector Cosine Similarity",
    cmap: str = "RdYlBu_r",
    vmin: float = -1,
    vmax: float = 1,
):
    """Plot heatmap similar to Task Arithmetic Figure 5."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = sim_matrix[i, j]
            color = "white" if (val > 0.5 if vmin >= 0 else abs(val) > 0.5) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_pareto_models(pareto_json_path: Path) -> tuple[str, dict[str, str]]:
    """
    Load base model id and {label: model_id} from pareto_comparison.json.
    Returns (base_model_id, {label: model_id}) including retain90 and unlearned models.
    """
    with open(pareto_json_path) as f:
        data = json.load(f)

    base_id = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
    models = {}
    for item in data:
        label = item["label"]
        model_id = item["model_id"]
        models[label] = model_id
    return base_id, models


def main():
    parser = argparse.ArgumentParser(
        description="Task vector heat map: retain90 + unlearned models vs base"
    )
    default_pareto = Path(__file__).parent.parent / "unlearning-wikitext-filter" / "output_pareto" / "pareto_comparison.json"
    parser.add_argument(
        "--pareto_json",
        type=Path,
        default=default_pareto,
        help="Path to pareto_comparison.json",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./output"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.pareto_json.exists():
        raise FileNotFoundError(f"Pareto JSON not found: {args.pareto_json}")

    base_id, model_dict = load_pareto_models(args.pareto_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Task Vector Heat Map (retain90 + unlearned vs base)")
    print("τ = θ_model − θ_base")
    print("=" * 60)

    print(f"\nLoading base model: {base_id}")
    base_sd = load_state_dict(base_id, device=args.device)

    task_vectors = {}
    for label, model_id in model_dict.items():
        print(f"  Loading {label}: {model_id}")
        try:
            model_sd = load_state_dict(model_id, device=args.device)
        except Exception as e:
            print(f"    Failed: {e}")
            continue
        delta = compute_task_vector(base_sd, model_sd)
        vec = flatten_to_vector(delta)
        task_vectors[label] = vec
        print(f"    Task vector dim: {vec.size}")

    if len(task_vectors) < 2:
        print("Need at least 2 task vectors. Exiting.")
        return

    print("\nComputing cosine similarity matrix...")
    sim_matrix = cosine_similarity_matrix(task_vectors)
    labels = list(task_vectors.keys())

    print("\nCosine similarity matrix:")
    print(" " * 14 + " ".join(f"{l:>10}" for l in labels))
    for i, la in enumerate(labels):
        row = " ".join(f"{sim_matrix[i, j]:10.3f}" for j in range(len(labels)))
        print(f"{la:>12}  {row}")

    plot_path = args.output_dir / "task_vector_heatmap.png"
    plot_heatmap(
        sim_matrix,
        labels,
        plot_path,
        title="Task Vector Cosine Similarity\n(retain90 + GradDiff pareto models vs base)",
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
    )

    mask = ~np.eye(len(labels), dtype=bool)
    off_diag = sim_matrix[mask]
    summary = {
        "base_model": base_id,
        "models": labels,
        "cosine_similarity_matrix": sim_matrix.tolist(),
        "mean_off_diag": float(np.mean(off_diag)),
        "min_off_diag": float(np.min(off_diag)),
        "max_off_diag": float(np.max(off_diag)),
    }
    with open(args.output_dir / "task_vector_similarity.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean off-diagonal cosine sim: {summary['mean_off_diag']:.4f}")
    print(f"Min / Max off-diagonal: {summary['min_off_diag']:.4f} / {summary['max_off_diag']:.4f}")
    print(f"\nSaved heatmap to {plot_path}")


if __name__ == "__main__":
    main()
