from __future__ import annotations

from pathlib import Path
from typing import Any

from unlearning_project.common import write_json
from unlearning_project.evaluation.behavioral import (
    run_external_adjacent_eval,
    summarize_adjacent_eval,
)
from unlearning_project.evaluation.representation import (
    run_external_representation_shift,
    summarize_representation_shift,
)


def run_analysis_stage(
    analysis_cfg: dict[str, Any],
    analysis_dir: Path,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}

    behavioral_cfg = analysis_cfg.get("behavioral", {})
    if behavioral_cfg.get("enabled", False):
        adjacent_cfg = behavioral_cfg.get("adjacent", {})
        mode = adjacent_cfg.get("mode", "summarize_existing")
        if mode == "summarize_existing":
            input_path = Path(adjacent_cfg["input_json"]).expanduser().resolve()
            outputs["behavioral"] = summarize_adjacent_eval(input_path, analysis_dir / "behavioral")
        elif mode == "run_external_script":
            script_path = Path(adjacent_cfg["script_path"]).expanduser().resolve()
            outputs["behavioral"] = run_external_adjacent_eval(
                script_path=script_path,
                output_dir=analysis_dir / "behavioral",
                extra_args=[str(item) for item in adjacent_cfg.get("args", [])],
            )
        else:
            raise ValueError(f"Unsupported behavioral analysis mode: {mode}")

    representation_cfg = analysis_cfg.get("representation", {})
    if representation_cfg.get("enabled", False):
        shift_cfg = representation_cfg.get("shift", {})
        mode = shift_cfg.get("mode", "summarize_existing")
        if mode == "summarize_existing":
            input_path = Path(shift_cfg["summary_json"]).expanduser().resolve()
            outputs["representation"] = summarize_representation_shift(
                input_path,
                analysis_dir / "representation",
            )
        elif mode == "run_external_script":
            script_path = Path(shift_cfg["script_path"]).expanduser().resolve()
            outputs["representation"] = run_external_representation_shift(
                script_path=script_path,
                benchmark=str(shift_cfg["benchmark"]),
                output_dir=analysis_dir / "representation",
                extra_args=[str(item) for item in shift_cfg.get("args", [])],
            )
        else:
            raise ValueError(f"Unsupported representation analysis mode: {mode}")

    write_json(outputs, analysis_dir / "manifest.json")
    return outputs
