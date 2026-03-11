from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from unlearning_project.common import load_json, write_json


def _safe_score_drop(row: dict[str, Any]) -> float:
    value = row.get("score_drop")
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def summarize_adjacent_eval(adjacent_eval_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_json(adjacent_eval_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {adjacent_eval_path}")

    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    scored_rows = [
        row
        for row in results
        if isinstance(row, dict) and row.get("score_drop") is not None
    ]
    top_drops = sorted(
        scored_rows,
        key=_safe_score_drop,
        reverse=True,
    )[:10]

    summary = {
        "analysis_type": "adjacent_side_effects",
        "source_path": str(adjacent_eval_path),
        "dataset": payload.get("dataset"),
        "n_samples": payload.get("n_samples", len(results)),
        "n_affected": payload.get("n_affected"),
        "mean_score_before": payload.get("mean_score_before"),
        "mean_score_after": payload.get("mean_score_after"),
        "mean_score_drop": payload.get("mean_score_drop"),
        "n_degraded": payload.get("n_degraded"),
        "n_unchanged": payload.get("n_unchanged"),
        "n_improved": payload.get("n_improved"),
        "top_drops": [
            {
                "prompt": row.get("prompt"),
                "score_before": row.get("score_before"),
                "score_after": row.get("score_after"),
                "score_drop": row.get("score_drop"),
                "affected": row.get("affected"),
            }
            for row in top_drops
        ],
    }
    write_json(summary, output_dir / "adjacent_summary.json")

    lines = [
        "# Adjacent Side-Effect Summary",
        "",
        f"- Source file: `{adjacent_eval_path}`",
        f"- Dataset: `{summary.get('dataset')}`",
        f"- Samples: `{summary.get('n_samples')}`",
        f"- Affected prompts: `{summary.get('n_affected')}`",
        f"- Mean score before: `{summary.get('mean_score_before')}`",
        f"- Mean score after: `{summary.get('mean_score_after')}`",
        f"- Mean score drop: `{summary.get('mean_score_drop')}`",
        "",
        "## Top Drops",
        "",
    ]
    for row in summary["top_drops"]:
        lines.append(
            f"- `{row['prompt']}` | before={row['score_before']} after={row['score_after']} drop={row['score_drop']}"
        )
    (output_dir / "adjacent_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_external_adjacent_eval(
    script_path: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    extra_args = extra_args or []
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script_path), "--output-dir", str(output_dir), *extra_args]
    log_path = output_dir / "adjacent_eval.log"
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
            f"Adjacent eval script failed with exit code {result.returncode}. "
            f"See log: {log_path}"
        )
    summary_path = output_dir / "adjacent_eval.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No adjacent_eval.json found in {output_dir}")
    return summarize_adjacent_eval(summary_path, output_dir)
