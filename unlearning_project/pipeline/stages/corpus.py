from __future__ import annotations

from pathlib import Path
from typing import Any

from unlearning_project.benchmarks.base import BenchmarkAdapter
from unlearning_project.common import write_json, write_jsonl


def _dedupe_samples(
    adapter: BenchmarkAdapter,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        identity = adapter.sample_id(sample)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(sample)
    return deduped


def prepare_corpora(
    adapter: BenchmarkAdapter,
    selected_path: Path,
    augmentation_manifest: dict[str, Any],
    corpora_dir: Path,
    corpus_cfg: dict[str, Any],
) -> dict[str, Any]:
    forget_samples = adapter.load_dataset(selected_path)
    retain_base = adapter.build_retain_dataset(forget_samples)
    augmented_path_value = augmentation_manifest.get("augmented_retain_path")
    augmented_samples: list[dict[str, Any]] = []
    if augmented_path_value:
        augmented_path = Path(augmented_path_value)
        if augmented_path.exists():
            augmented_samples = adapter.load_dataset(augmented_path)

    merge_mode = str(corpus_cfg.get("retain_merge_mode", "append"))
    retain_merged = list(retain_base)
    retain_merged.extend(augmented_samples)
    if merge_mode == "dedupe_by_identity":
        retain_merged = _dedupe_samples(adapter, retain_merged)

    forget_rows = [{"text": adapter.sample_to_training_text(sample)} for sample in forget_samples]
    retain_base_rows = [{"text": adapter.sample_to_training_text(sample)} for sample in retain_base]
    augmented_rows = [{"text": adapter.sample_to_training_text(sample)} for sample in augmented_samples]
    retain_merged_rows = [{"text": adapter.sample_to_training_text(sample)} for sample in retain_merged]

    forget_path = corpora_dir / adapter.forget_corpus_filename()
    retain_base_path = corpora_dir / "retain-base.jsonl"
    retain_augmented_path = corpora_dir / "retain-augmented.jsonl"
    retain_merged_path = corpora_dir / adapter.retain_corpus_filename()
    write_jsonl(forget_rows, forget_path)
    write_jsonl(retain_base_rows, retain_base_path)
    write_jsonl(augmented_rows, retain_augmented_path)
    write_jsonl(retain_merged_rows, retain_merged_path)

    manifest = {
        "selected_path": str(selected_path),
        "forget_corpus_path": str(forget_path),
        "retain_base_corpus_path": str(retain_base_path),
        "retain_augmented_corpus_path": str(retain_augmented_path),
        "retain_merged_corpus_path": str(retain_merged_path),
        "forget_count": len(forget_rows),
        "retain_base_count": len(retain_base_rows),
        "retain_augmented_count": len(augmented_rows),
        "retain_merged_count": len(retain_merged_rows),
        "retain_merge_mode": merge_mode,
    }
    write_json(manifest, corpora_dir / "manifest.json")
    return manifest
