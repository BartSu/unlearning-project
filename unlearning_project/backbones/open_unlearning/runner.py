from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from unlearning_project.benchmarks.base import BenchmarkAdapter
from unlearning_project.common import (
    copy_directory,
    dict_to_overrides,
    find_free_port,
    load_json,
    render_command,
    resolve_path,
    safe_symlink,
    slugify,
    write_json,
)


def run_logged_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env_updates: dict[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ.copy()
    if env_updates:
        environment.update(env_updates)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}. See log: {log_path}"
        )
    return {
        "command": command,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "returncode": result.returncode,
    }


def run_train(
    adapter: BenchmarkAdapter,
    open_unlearning_cfg: dict[str, Any],
    forget_corpus_path: Path,
    retain_corpus_path: Path,
    train_dir: Path,
) -> dict[str, Any]:
    train_cfg = open_unlearning_cfg.get("train", {})
    open_root = resolve_path(open_unlearning_cfg["root"])
    if open_root is None or not open_root.exists():
        raise FileNotFoundError(f"open-unlearning root not found: {open_unlearning_cfg['root']}")

    if not train_cfg.get("enabled", True):
        model_locator = train_cfg.get("existing_model_path")
        if not model_locator:
            raise ValueError("Training disabled but no existing model path was provided")
        resolved_model_path = resolve_path(model_locator, open_root)
        model_is_path = bool(resolved_model_path and resolved_model_path.exists())
        if model_is_path:
            safe_symlink(resolved_model_path, train_dir / "model")
        manifest = {
            "skipped": True,
            "model_path": str(resolved_model_path if model_is_path else model_locator),
            "model_is_path": model_is_path,
            "reason": "training disabled in config",
        }
        write_json(manifest, train_dir / "manifest.json")
        return manifest

    accelerate_cfg = open_unlearning_cfg.get("accelerate", {})
    experiment = train_cfg.get("experiment") or adapter.open_unlearning_train_experiment()
    if not experiment:
        raise ValueError("No open-unlearning train experiment configured")
    task_name = str(train_cfg["task_name"])
    entrypoint = str(train_cfg.get("entrypoint", "src/train.py"))
    config_name = str(train_cfg.get("config_name", "unlearn.yaml"))

    output_dir = resolve_path(train_cfg.get("output_dir"), train_dir) or (train_dir / "open_unlearning_output")
    output_dir = output_dir.resolve()

    command = ["accelerate", "launch"]
    if accelerate_cfg.get("num_processes") is not None:
        command.extend(["--num_processes", str(accelerate_cfg["num_processes"])])
    if accelerate_cfg.get("config_file"):
        command.extend(["--config_file", str(accelerate_cfg["config_file"])])
    command.extend(["--main_process_port", str(find_free_port())])
    command.extend([entrypoint, f"--config-name={config_name}"])

    overrides = {
        "experiment": experiment,
        "task_name": task_name,
        "paths.output_dir": str(output_dir),
    }
    overrides.update(adapter.open_unlearning_train_overrides(forget_corpus_path, retain_corpus_path))
    overrides.update(train_cfg.get("overrides", {}))
    command.extend(dict_to_overrides(overrides))

    log_path = train_dir / "train.log"
    command_info = run_logged_command(command, cwd=open_root, log_path=log_path)
    safe_symlink(output_dir, train_dir / "model")
    manifest = {
        **command_info,
        "task_name": task_name,
        "model_path": str(output_dir),
        "experiment": experiment,
    }
    write_json(manifest, train_dir / "manifest.json")
    return manifest


def load_builtin_eval_summary(eval_output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries: dict[str, Any] = {}
    raw_outputs: dict[str, Any] = {}
    for summary_file in sorted(eval_output_dir.glob("*_SUMMARY.json")):
        summaries[summary_file.stem] = load_json(summary_file)
    for eval_file in sorted(eval_output_dir.glob("*_EVAL.json")):
        raw_outputs[eval_file.stem] = load_json(eval_file)
    flat_summary: dict[str, Any] = {}
    if len(summaries) == 1:
        flat_summary = next(iter(summaries.values()))
    else:
        for prefix, payload in summaries.items():
            if isinstance(payload, dict):
                for key, value in payload.items():
                    flat_summary[f"{prefix}/{key}"] = value
    return flat_summary, raw_outputs


def run_builtin_eval(
    adapter: BenchmarkAdapter,
    open_unlearning_cfg: dict[str, Any],
    model_path: Path | str,
    eval_dir: Path,
) -> dict[str, Any]:
    evaluation_cfg = open_unlearning_cfg.get("evaluation", {})
    builtin_cfg = evaluation_cfg.get("builtin", {})
    if not builtin_cfg.get("enabled", True):
        manifest = {"skipped": True, "reason": "builtin evaluation disabled in config"}
        write_json(manifest, eval_dir / "builtin_manifest.json")
        return manifest

    open_root = resolve_path(open_unlearning_cfg["root"])
    if open_root is None or not open_root.exists():
        raise FileNotFoundError(f"open-unlearning root not found: {open_unlearning_cfg['root']}")

    experiment = builtin_cfg.get("experiment") or adapter.open_unlearning_eval_experiment()
    if not experiment:
        raise ValueError("No open-unlearning eval experiment configured")

    task_name = str(builtin_cfg["task_name"])
    entrypoint = str(builtin_cfg.get("entrypoint", "src/eval.py"))
    config_name = str(builtin_cfg.get("config_name", "eval.yaml"))
    output_dir = resolve_path(builtin_cfg.get("output_dir"), eval_dir) or (eval_dir / "open_unlearning_output")
    output_dir = output_dir.resolve()

    command = [sys.executable, entrypoint, f"--config-name={config_name}"]
    overrides = {
        "experiment": experiment,
        "task_name": task_name,
        "paths.output_dir": str(output_dir),
    }
    overrides.update(adapter.open_unlearning_eval_overrides(model_path))
    overrides.update(builtin_cfg.get("overrides", {}))
    command.extend(dict_to_overrides(overrides))

    log_path = eval_dir / "builtin_eval.log"
    command_info = run_logged_command(command, cwd=open_root, log_path=log_path)

    copied_eval_dir = eval_dir / "open_unlearning"
    copy_directory(output_dir, copied_eval_dir)
    summary, raw_outputs = load_builtin_eval_summary(output_dir)
    normalized = adapter.normalize_eval_summary(summary)

    write_json(summary, eval_dir / "builtin_summary.json")
    write_json(normalized, eval_dir / "summary.json")
    write_json(raw_outputs, eval_dir / "builtin_raw_metrics.json")

    manifest = {
        **command_info,
        "task_name": task_name,
        "experiment": experiment,
        "eval_output_dir": str(output_dir),
        "copied_eval_dir": str(copied_eval_dir),
        "normalized_summary_path": str(eval_dir / "summary.json"),
    }
    write_json(manifest, eval_dir / "builtin_manifest.json")
    return manifest


def run_external_eval_hooks(
    hooks_cfg: list[dict[str, Any]],
    eval_dir: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    external_root = eval_dir / "external"
    external_root.mkdir(parents=True, exist_ok=True)
    for hook_cfg in hooks_cfg:
        if not hook_cfg.get("enabled", False):
            continue
        hook_name = str(hook_cfg["name"])
        hook_dir = external_root / slugify(hook_name)
        hook_dir.mkdir(parents=True, exist_ok=True)
        output_json = hook_dir / "metrics.json"
        render_context = {
            **context,
            "hook_dir": hook_dir,
            "output_json": output_json,
        }
        command_value = hook_cfg["command"]
        if isinstance(command_value, str):
            command_parts = shlex.split(command_value)
        else:
            command_parts = [str(part) for part in command_value]
        command = render_command(command_parts, render_context)
        cwd = resolve_path(hook_cfg.get("cwd"), Path.cwd()) or Path.cwd()
        env_updates = {key: str(value) for key, value in hook_cfg.get("env", {}).items()}
        env_updates["MODEL_PATH"] = str(context["model_path"])
        env_updates["RUN_DIR"] = str(context["run_dir"])
        env_updates["OUTPUT_JSON"] = str(output_json)
        manifest = run_logged_command(command, cwd=cwd, log_path=hook_dir / "hook.log", env_updates=env_updates)
        metrics = load_json(output_json) if output_json.exists() else {}
        result = {**manifest, "metrics": metrics, "output_json": str(output_json)}
        write_json(result, hook_dir / "manifest.json")
        results[hook_name] = result
    if results:
        write_json(results, eval_dir / "external_hooks.json")
    return results
