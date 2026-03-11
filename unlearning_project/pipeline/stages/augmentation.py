from __future__ import annotations

from pathlib import Path
from typing import Any

from unlearning_project.augmentation import build_geometry_augmentations
from unlearning_project.benchmarks.base import BenchmarkAdapter
from unlearning_project.common import write_json, write_jsonl


def run_augmentation_stage(
    adapter: BenchmarkAdapter,
    selected_path: Path,
    augmentation_cfg: dict[str, Any],
    augmentation_dir: Path,
) -> dict[str, Any]:
    forget_samples = adapter.load_dataset(selected_path)
    retain_candidates = adapter.build_retain_dataset(forget_samples)
    write_jsonl(retain_candidates, augmentation_dir / "retain_candidates.jsonl")

    if not augmentation_cfg.get("enabled", False):
        manifest = {
            "enabled": False,
            "strategy": "none",
            "selected_path": str(selected_path),
            "retain_candidate_path": str(augmentation_dir / "retain_candidates.jsonl"),
            "augmented_retain_path": None,
            "augmented_count": 0,
        }
        write_json(manifest, augmentation_dir / "manifest.json")
        return manifest

    manifest = build_geometry_augmentations(
        adapter=adapter,
        forget_samples=forget_samples,
        retain_candidates=retain_candidates,
        cfg=augmentation_cfg,
        output_dir=augmentation_dir,
    )
    manifest["selected_path"] = str(selected_path)
    manifest["retain_candidate_path"] = str(augmentation_dir / "retain_candidates.jsonl")
    write_json(manifest, augmentation_dir / "manifest.json")
    return manifest
