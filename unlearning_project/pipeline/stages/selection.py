from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_project.benchmarks.base import BenchmarkAdapter
from unlearning_project.common import write_json

IGNORE_INDEX = -100


def sample_forget_examples(
    samples: list[dict[str, Any]],
    n_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    if n_samples <= 0 or n_samples >= len(samples):
        return samples, list(range(len(samples)))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=generator)[:n_samples].tolist()
    return [samples[index] for index in indices], indices


def tokenize_sample(
    tokenizer,
    prompt_messages: list[dict[str, str]],
    response_messages: list[dict[str, str]],
    max_length: int,
) -> dict[str, torch.Tensor]:
    full_chat = prompt_messages + response_messages
    full_ids = tokenizer.apply_chat_template(
        full_chat,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    prompt_length = len(prompt_ids)
    labels = [IGNORE_INDEX] * prompt_length + full_ids[prompt_length:]
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attention_mask = [1] * len(full_ids) + [0] * (max_length - len(full_ids))
    full_ids = full_ids + [pad_id] * (max_length - len(full_ids))
    labels = labels + [IGNORE_INDEX] * (max_length - len(labels))
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def get_selected_param_names(model: torch.nn.Module, scope: str) -> list[str]:
    named_params = list(model.named_parameters())
    prefix = "model.model.layers"
    if not any(name.startswith(prefix) for name, _ in named_params):
        prefix = "model.layers"
    last_idx = -1
    for name, _ in named_params:
        match = re.search(rf"{re.escape(prefix)}\.(\d+)", name)
        if match:
            last_idx = max(last_idx, int(match.group(1)))
    if last_idx < 0:
        raise ValueError("Could not determine final transformer block")
    layer_prefix = f"{prefix}.{last_idx}"
    if scope == "last_layer_mlp":
        selected = [name for name, _ in named_params if layer_prefix in name and "mlp" in name]
    elif scope == "last_block":
        selected = [name for name, _ in named_params if layer_prefix in name]
    elif scope == "lm_head":
        selected = [name for name, _ in named_params if "lm_head" in name]
    elif scope == "down_proj_only":
        selected = [
            name
            for name, _ in named_params
            if layer_prefix in name and "mlp.down_proj" in name
        ]
    else:
        raise ValueError(f"Unknown param scope: {scope}")
    if not selected:
        raise ValueError(f"No parameters matched scope '{scope}' in final block")
    return sorted(selected)


def extract_per_sample_gradients(
    model: torch.nn.Module,
    tokenizer,
    adapter: BenchmarkAdapter,
    samples: list[dict[str, Any]],
    param_names: list[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    param_set = set(param_names)
    gradients: list[torch.Tensor] = []
    model.train()
    for sample in tqdm(samples, desc="Extracting gradients"):
        model.zero_grad()
        batch = tokenize_sample(
            tokenizer,
            adapter.sample_to_prompt_messages(sample),
            adapter.sample_to_response_messages(sample),
            max_length=max_length,
        )
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        outputs.loss.backward()
        parts: list[torch.Tensor] = []
        for name, param in model.named_parameters():
            if name in param_set and param.grad is not None:
                parts.append(param.grad.detach().float().flatten())
        if not parts:
            raise RuntimeError("No gradients were collected for the selected parameters")
        gradients.append(torch.cat(parts).cpu())
    return torch.stack(gradients)


def compute_svd_rank(G_f: torch.Tensor, energy_threshold: float = 0.9) -> dict[str, Any]:
    U, S, Vh = torch.linalg.svd(G_f.float(), full_matrices=False)
    singular_values = S.numpy()
    energy = singular_values**2
    total_energy = energy.sum()
    cumulative = energy.cumsum()
    intrinsic_rank = 1
    for k in range(1, len(singular_values) + 1):
        if cumulative[k - 1] >= energy_threshold * total_energy:
            intrinsic_rank = k
            break
    return {
        "U": U,
        "S": singular_values,
        "Vh": Vh,
        "r_g": intrinsic_rank,
        "total_energy": float(total_energy),
    }


def filter_samples_top_k_energy(
    U: torch.Tensor,
    S: np.ndarray,
    samples: list[dict[str, Any]],
    k: int,
    n_keep: int,
) -> tuple[list[dict[str, Any]], list[int], list[float]]:
    singular_values = torch.from_numpy(S).float()
    k = min(k, U.shape[1])
    projection = U[:, :k] * singular_values[:k]
    energy = (projection**2).sum(dim=1)
    _, indices = torch.sort(energy, descending=True)
    n_keep = min(n_keep, len(samples))
    kept_indices = indices[:n_keep].tolist()
    filtered = [samples[index] for index in kept_indices]
    return filtered, kept_indices, energy.tolist()


def filter_samples_redundancy(
    G_f: torch.Tensor,
    samples: list[dict[str, Any]],
    n_keep: int,
    similarity_threshold: float = 0.95,
) -> tuple[list[dict[str, Any]], list[int]]:
    n_rows = G_f.shape[0]
    norms = (G_f**2).sum(dim=1).sqrt()
    norms = torch.clamp(norms, min=1e-8)
    normalized = G_f / norms.unsqueeze(1)
    kept: list[int] = []
    for _ in range(min(n_keep, n_rows)):
        if not kept:
            index = norms.argmax().item()
        else:
            similarities = (normalized @ normalized[kept].T).abs().max(dim=1).values
            candidates = (similarities < similarity_threshold).nonzero(as_tuple=True)[0]
            candidate_ids = [item.item() for item in candidates if item.item() not in kept]
            if not candidate_ids:
                break
            index = max(candidate_ids, key=lambda item: norms[item].item())
        kept.append(index)
    return [samples[index] for index in kept], kept


def save_plots(
    singular_values: np.ndarray,
    intrinsic_rank: int,
    output_dir: Path,
    title_prefix: str,
) -> dict[str, str]:
    decay_path = output_dir / "singular_value_decay.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(1, len(singular_values) + 1), singular_values + 1e-20, linewidth=1)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title(f"{title_prefix} singular value decay")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(decay_path, dpi=150)
    plt.close(fig)

    energy = singular_values**2
    cumulative = energy.cumsum() / energy.sum()
    energy_path = output_dir / "cumulative_energy.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cumulative) + 1), cumulative, linewidth=1)
    ax.axhline(0.9, color="r", linestyle="--", label="90% energy")
    ax.axvline(intrinsic_rank, color="g", linestyle=":", label=f"r_g={intrinsic_rank}")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative energy ratio")
    ax.set_title(f"{title_prefix} cumulative energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(energy_path, dpi=150)
    plt.close(fig)
    return {
        "singular_value_decay": str(decay_path),
        "cumulative_energy": str(energy_path),
    }


def copy_forget_dataset(
    adapter: BenchmarkAdapter,
    selected_dir: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    selected_samples = adapter.load_forget_dataset()
    selected_path = selected_dir / adapter.selected_output_filename()
    adapter.write_selected_dataset(selected_samples, selected_path)
    metadata = {
        "selection_enabled": False,
        "reason": "selection stage skipped",
        "source_forget_path": str(adapter.resolve_config_path("forget_path")),
        "selected_count": len(selected_samples),
        "sampled_count": len(selected_samples),
    }
    write_json(metadata, metadata_path)
    return {
        "selected_path": str(selected_path),
        "metadata_path": str(metadata_path),
        "selected_count": len(selected_samples),
        "sampled_count": len(selected_samples),
    }


def run_selection_stage(
    adapter: BenchmarkAdapter,
    selection_cfg: dict[str, Any],
    selected_dir: Path,
    artifacts_dir: Path,
) -> dict[str, Any]:
    forget_samples = adapter.load_forget_dataset()
    sampled_samples, sampled_indices = sample_forget_examples(
        forget_samples,
        int(selection_cfg.get("n_samples", 0)),
        int(selection_cfg.get("seed", 42)),
    )

    dtype_name = str(selection_cfg.get("torch_dtype", "bfloat16"))
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    model_name = str(selection_cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(selection_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=selection_cfg.get("device_map", "auto"),
        trust_remote_code=bool(selection_cfg.get("trust_remote_code", False)),
    )

    param_scope = str(selection_cfg.get("param_scope", "down_proj_only"))
    param_names = get_selected_param_names(model, param_scope)
    parameter_dimension = sum(model.get_parameter(name).numel() for name in param_names)
    device = next(model.parameters()).device
    G_f = extract_per_sample_gradients(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        samples=sampled_samples,
        param_names=param_names,
        device=device,
        max_length=int(selection_cfg.get("max_length", 512)),
    )
    n_samples = G_f.shape[0]

    svd_result = compute_svd_rank(
        G_f,
        energy_threshold=float(selection_cfg.get("energy_threshold", 0.9)),
    )
    singular_values = svd_result["S"]
    intrinsic_rank = int(svd_result["r_g"])
    effective_rank = intrinsic_rank / max(1, n_samples)
    report = {
        "benchmark": adapter.benchmark_name(),
        "model_name": model_name,
        "n_total_forget_samples": len(forget_samples),
        "n_sampled_for_svd": n_samples,
        "sampled_indices": sampled_indices,
        "parameter_scope": param_scope,
        "parameter_dimension": parameter_dimension,
        "gradient_rank": intrinsic_rank,
        "effective_gradient_rank": effective_rank,
        "gradient_rank_ratio": intrinsic_rank / parameter_dimension,
        "top_10_singular_values": singular_values[: min(10, len(singular_values))].tolist(),
    }
    report_path = artifacts_dir / "gradient_svd_report.json"
    write_json(report, report_path)
    plot_paths = save_plots(singular_values, intrinsic_rank, artifacts_dir, adapter.benchmark_name())

    filter_strategy = str(selection_cfg.get("filter_strategy", "top_k_energy"))
    metadata: dict[str, Any] = {
        "selection_enabled": True,
        "source_forget_path": str(adapter.resolve_config_path("forget_path")),
        "filter_strategy": filter_strategy,
        "sampled_indices": sampled_indices,
        "report_path": str(report_path),
    }

    if filter_strategy == "none":
        selected_samples = sampled_samples
        kept_indices = list(range(len(sampled_samples)))
    else:
        n_keep = int(selection_cfg.get("filter_n") or intrinsic_rank)
        n_keep = min(n_keep, n_samples)
        if filter_strategy == "top_k_energy":
            filter_k = int(selection_cfg.get("filter_k") or intrinsic_rank)
            selected_samples, kept_indices, energies = filter_samples_top_k_energy(
                svd_result["U"],
                singular_values,
                sampled_samples,
                k=filter_k,
                n_keep=n_keep,
            )
            metadata["filter_k"] = filter_k
            metadata["projection_energies"] = [float(energies[index]) for index in kept_indices]
        elif filter_strategy == "redundancy":
            selected_samples, kept_indices = filter_samples_redundancy(
                G_f,
                sampled_samples,
                n_keep=n_keep,
                similarity_threshold=float(selection_cfg.get("similarity_threshold", 0.95)),
            )
            metadata["similarity_threshold"] = float(selection_cfg.get("similarity_threshold", 0.95))
        else:
            raise ValueError(f"Unknown filter strategy: {filter_strategy}")
        metadata["n_keep_requested"] = n_keep

    metadata["kept_indices"] = kept_indices
    metadata["selected_count"] = len(selected_samples)
    metadata["sampled_count"] = len(sampled_samples)

    selected_path = selected_dir / adapter.selected_output_filename()
    metadata_path = selected_dir / "selection_meta.json"
    adapter.write_selected_dataset(selected_samples, selected_path)
    write_json(metadata, metadata_path)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "selected_path": str(selected_path),
        "metadata_path": str(metadata_path),
        "report_path": str(report_path),
        "selected_count": len(selected_samples),
        "sampled_count": len(sampled_samples),
        "plot_paths": plot_paths,
    }
