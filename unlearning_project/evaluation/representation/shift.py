from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from unlearning_project.common import load_json, write_json


def summarize_representation_shift(summary_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_json(summary_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {summary_path}")

    summary = {
        "analysis_type": "representation_shift",
        "source_path": str(summary_path),
        "benchmark": payload.get("benchmark"),
        "input": payload.get("input", {}),
        "last_layer": payload.get("summary", {}).get("last_layer"),
        "most_all_drift_layer": payload.get("summary", {}).get("most_all_drift_layer"),
        "most_prompt_drift_layer": payload.get("summary", {}).get("most_prompt_drift_layer"),
        "most_answer_drift_layer": payload.get("summary", {}).get("most_answer_drift_layer"),
        "mean_all_layers": payload.get("summary", {}).get("mean_all_layers"),
    }
    write_json(summary, output_dir / "representation_shift_summary.json")

    lines = [
        "# Representation Shift Summary",
        "",
        f"- Source file: `{summary_path}`",
        f"- Benchmark: `{summary.get('benchmark')}`",
    ]
    mean_all_layers = summary.get("mean_all_layers") or {}
    if mean_all_layers:
        lines.extend(
            [
                f"- Mean all-layer cosine: `{mean_all_layers.get('all_cosine')}`",
                f"- Mean prompt cosine: `{mean_all_layers.get('prompt_cosine')}`",
                f"- Mean answer cosine: `{mean_all_layers.get('answer_cosine')}`",
            ]
        )
    last_layer = summary.get("last_layer") or {}
    if last_layer:
        lines.extend(
            [
                "",
                "## Last Layer",
                "",
                f"- Layer: `{last_layer.get('layer_name')}`",
                f"- All cosine: `{last_layer.get('all_cosine')}`",
                f"- Prompt cosine: `{last_layer.get('prompt_cosine')}`",
                f"- Answer cosine: `{last_layer.get('answer_cosine')}`",
            ]
        )
    (output_dir / "representation_shift_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def run_external_representation_shift(
    script_path: Path,
    benchmark: str,
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    extra_args = extra_args or []
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--benchmark",
        benchmark,
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]
    log_path = output_dir / "representation_shift.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Representation shift script failed with exit code {result.returncode}. "
            f"See log: {log_path}"
        )

    summary_candidates = sorted(output_dir.glob("*_summary.json"))
    if not summary_candidates:
        raise FileNotFoundError(f"No representation shift summary found in {output_dir}")
    return summarize_representation_shift(summary_candidates[-1], output_dir)
