from __future__ import annotations

import copy
import json
import os
import re
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    selected_dir: Path
    augmentation_dir: Path
    corpora_dir: Path
    train_dir: Path
    eval_dir: Path
    analysis_dir: Path
    reports_dir: Path
    logs_dir: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: dict[str, Any] | list[Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return records
        results = payload.get("results")
        if isinstance(results, list):
            return results
        return [payload]
    raise ValueError(f"Unsupported dataset format in {path}")


def resolve_path(path_value: str | Path | None, base_dir: Path = PROJECT_DIR) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "run"


def default_run_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{slugify(prefix)}-{timestamp}"


def build_run_layout(output_root: Path, run_name: str) -> RunLayout:
    run_dir = ensure_dir((output_root / run_name).resolve())
    return RunLayout(
        run_dir=run_dir,
        selected_dir=ensure_dir(run_dir / "selected"),
        augmentation_dir=ensure_dir(run_dir / "augmentation"),
        corpora_dir=ensure_dir(run_dir / "corpora"),
        train_dir=ensure_dir(run_dir / "train"),
        eval_dir=ensure_dir(run_dir / "eval"),
        analysis_dir=ensure_dir(run_dir / "analysis"),
        reports_dir=ensure_dir(run_dir / "reports"),
        logs_dir=ensure_dir(run_dir / "logs"),
    )


def value_to_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def dict_to_overrides(values: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    for key, value in values.items():
        if isinstance(value, dict):
            nested = dict_to_overrides(
                {f"{key}.{nested_key}": nested_value for nested_key, nested_value in value.items()}
            )
            overrides.extend(nested)
            continue
        overrides.append(f"{key}={value_to_override(value)}")
    return overrides


def find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def copy_directory(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ensure_dir(dst.parent)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def safe_symlink(target: Path, link_path: Path) -> None:
    ensure_dir(link_path.parent)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    try:
        os.symlink(target, link_path, target_is_directory=target.is_dir())
    except OSError:
        write_json({"target": str(target)}, link_path.with_suffix(".pointer.json"))


def render_template(value: str, context: dict[str, Any]) -> str:
    rendered_context = {key: str(item) for key, item in context.items()}
    return value.format(**rendered_context)


def render_command(parts: list[str], context: dict[str, Any]) -> list[str]:
    return [render_template(part, context) for part in parts]
