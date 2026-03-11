from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from unlearning_project.backbones.open_unlearning import (
    run_builtin_eval,
    run_external_eval_hooks,
    run_train,
)
from unlearning_project.benchmarks import create_adapter
from unlearning_project.common import (
    PROJECT_DIR,
    build_run_layout,
    deep_merge,
    default_run_name,
    dump_yaml,
    load_json,
    load_yaml,
    resolve_path,
    safe_symlink,
    write_json,
)
from unlearning_project.pipeline.stages import (
    copy_forget_dataset,
    prepare_corpora,
    run_analysis_stage,
    run_augmentation_stage,
    run_selection_stage,
)

STAGES = ["select", "augment", "prepare", "train", "eval", "analysis", "report"]


def set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def apply_cli_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = deep_merge({}, config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must look like key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        set_nested_value(updated, key, value)
    return updated


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    benchmark_base: dict[str, Any] = {}
    benchmark_config_path = config.get("benchmark_config")
    if benchmark_config_path:
        benchmark_path = resolve_path(benchmark_config_path, config_path.parent)
        if benchmark_path is not None and not benchmark_path.exists():
            benchmark_path = resolve_path(benchmark_config_path, PROJECT_DIR)
        if benchmark_path is None:
            raise ValueError(f"Could not resolve benchmark config: {benchmark_config_path}")
        benchmark_base = load_yaml(benchmark_path)
    benchmark_override = config.get("benchmark", {})
    config["benchmark"] = deep_merge(benchmark_base, benchmark_override)
    config["_config_path"] = str(config_path)
    return config


def stage_index(stage_name: str | None) -> int:
    if stage_name is None:
        return 0
    if stage_name not in STAGES:
        raise ValueError(f"Unknown stage '{stage_name}'. Known stages: {', '.join(STAGES)}")
    return STAGES.index(stage_name)


def hydrate_previous_state(
    start_index: int,
    adapter,
    layout,
    state: dict[str, Any],
) -> None:
    if start_index > stage_index("select"):
        selected_path = layout.selected_dir / adapter.selected_output_filename()
        if not selected_path.exists():
            raise FileNotFoundError(
                f"Cannot resume from later stage; selected dataset missing: {selected_path}"
            )
        state["selection"] = {
            "selected_path": str(selected_path),
            "metadata_path": str(layout.selected_dir / "selection_meta.json"),
        }
        metadata_path = layout.selected_dir / "selection_meta.json"
        if metadata_path.exists():
            state["selection"].update(load_json(metadata_path))
    if start_index > stage_index("augment"):
        manifest_path = layout.augmentation_dir / "manifest.json"
        if manifest_path.exists():
            state["augmentation"] = load_json(manifest_path)
    if start_index > stage_index("prepare"):
        manifest_path = layout.corpora_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot resume from later stage; corpus manifest missing: {manifest_path}"
            )
        state["corpora"] = load_json(manifest_path)
    if start_index > stage_index("train"):
        manifest_path = layout.train_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot resume from eval/report; train manifest missing: {manifest_path}"
            )
        state["train"] = load_json(manifest_path)
    if start_index > stage_index("eval"):
        summary_path = layout.eval_dir / "summary.json"
        if summary_path.exists():
            state["eval_summary"] = load_json(summary_path)
    if start_index > stage_index("analysis"):
        manifest_path = layout.analysis_dir / "manifest.json"
        if manifest_path.exists():
            state["analysis"] = load_json(manifest_path)


def build_report(layout, state: dict[str, Any], config: dict[str, Any], adapter) -> dict[str, Any]:
    report = {
        "run_name": state["run_name"],
        "run_dir": str(layout.run_dir),
        "benchmark": adapter.benchmark_name(),
        "config_path": config.get("_config_path"),
        "selection": state.get("selection", {}),
        "augmentation": state.get("augmentation", {}),
        "corpora": state.get("corpora", {}),
        "train": state.get("train", {}),
        "eval_summary": state.get("eval_summary", {}),
        "analysis": state.get("analysis", {}),
        "external_evaluations": state.get("external_evaluations", {}),
    }
    write_json(report, layout.run_dir / "run_manifest.json")
    write_json(report, layout.reports_dir / "run_summary.json")

    lines = [
        "# Run Summary",
        "",
        f"- Run name: {report['run_name']}",
        f"- Benchmark: {report['benchmark']}",
        f"- Run directory: {report['run_dir']}",
    ]
    selection = report["selection"]
    if selection:
        lines.extend(
            [
                f"- Selected dataset: {selection.get('selected_path', 'n/a')}",
                f"- Sampled count: {selection.get('sampled_count', 'n/a')}",
                f"- Selected count: {selection.get('selected_count', 'n/a')}",
            ]
        )
    augmentation = report["augmentation"]
    if augmentation:
        lines.extend(
            [
                f"- Augmentation strategy: {augmentation.get('strategy', 'n/a')}",
                f"- Augmented retain count: {augmentation.get('augmented_count', 'n/a')}",
            ]
        )
    corpora = report["corpora"]
    if corpora:
        lines.extend(
            [
                f"- Forget corpus: {corpora.get('forget_corpus_path', 'n/a')}",
                f"- Retain merged corpus: {corpora.get('retain_merged_corpus_path', 'n/a')}",
            ]
        )
    train = report["train"]
    if train:
        lines.append(f"- Model path: {train.get('model_path', 'n/a')}")
    eval_summary = report["eval_summary"]
    if eval_summary:
        lines.extend(["", "## Builtin Eval", ""])
        for key, value in sorted(eval_summary.items()):
            lines.append(f"- {key}: {value}")
    analysis = report["analysis"]
    if analysis:
        lines.extend(["", "## Analysis", ""])
        behavioral = analysis.get("behavioral", {})
        if behavioral:
            lines.append(f"- Adjacent mean score drop: {behavioral.get('mean_score_drop')}")
            lines.append(f"- Adjacent affected prompts: {behavioral.get('n_affected')}")
        representation = analysis.get("representation", {})
        if representation:
            mean_all_layers = representation.get("mean_all_layers", {})
            lines.append(
                "- Representation mean all-layer cosine: "
                f"{mean_all_layers.get('all_cosine')}"
            )

    summary_path = layout.reports_dir / "run_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_pipeline(
    config: dict[str, Any],
    run_name_override: str | None = None,
    from_stage: str | None = None,
    until_stage: str | None = None,
) -> dict[str, Any]:
    benchmark_cfg = config["benchmark"]
    adapter = create_adapter(benchmark_cfg, project_dir=PROJECT_DIR)

    run_cfg = config.get("run", {})
    run_name = run_name_override or run_cfg.get("name") or default_run_name(adapter.benchmark_name())
    output_root = resolve_path(run_cfg.get("output_root", "runs"), PROJECT_DIR)
    if output_root is None:
        raise ValueError("Could not resolve output root")
    layout = build_run_layout(output_root, run_name)
    snapshot_config = deep_merge(config, {"run": {"name": run_name}})
    dump_yaml(snapshot_config, layout.run_dir / "config.snapshot.yaml")

    start_index = stage_index(from_stage)
    end_index = stage_index(until_stage) if until_stage is not None else len(STAGES) - 1
    if start_index > end_index:
        raise ValueError("--from-stage must come before --until-stage")

    state: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(layout.run_dir),
    }
    hydrate_previous_state(start_index, adapter, layout, state)

    if start_index <= stage_index("select") <= end_index:
        selection_cfg = config.get("selection", {})
        if selection_cfg.get("enabled", True):
            state["selection"] = run_selection_stage(
                adapter=adapter,
                selection_cfg=selection_cfg,
                selected_dir=layout.selected_dir,
                artifacts_dir=layout.selected_dir,
            )
        else:
            state["selection"] = copy_forget_dataset(
                adapter=adapter,
                selected_dir=layout.selected_dir,
                metadata_path=layout.selected_dir / "selection_meta.json",
            )

    if start_index <= stage_index("augment") <= end_index:
        selected_path = Path(state["selection"]["selected_path"])
        state["augmentation"] = run_augmentation_stage(
            adapter=adapter,
            selected_path=selected_path,
            augmentation_cfg=config.get("augmentation", {}),
            augmentation_dir=layout.augmentation_dir,
        )

    if start_index <= stage_index("prepare") <= end_index:
        selected_path = Path(state["selection"]["selected_path"])
        state["corpora"] = prepare_corpora(
            adapter=adapter,
            selected_path=selected_path,
            augmentation_manifest=state.get("augmentation", {}),
            corpora_dir=layout.corpora_dir,
            corpus_cfg=config.get("corpora", {}),
        )

    if start_index <= stage_index("train") <= end_index:
        state["train"] = run_train(
            adapter=adapter,
            open_unlearning_cfg=config.get("open_unlearning", {}),
            forget_corpus_path=Path(state["corpora"]["forget_corpus_path"]),
            retain_corpus_path=Path(state["corpora"]["retain_merged_corpus_path"]),
            train_dir=layout.train_dir,
        )

    if start_index <= stage_index("eval") <= end_index:
        train_manifest = state.get("train", {})
        model_locator = train_manifest.get("model_path")
        if not model_locator:
            raise ValueError("No model path available for evaluation")
        maybe_model_path = Path(str(model_locator))
        if maybe_model_path.exists():
            safe_symlink(maybe_model_path, layout.eval_dir / "model")
        builtin_manifest = run_builtin_eval(
            adapter=adapter,
            open_unlearning_cfg=config.get("open_unlearning", {}),
            model_path=str(model_locator),
            eval_dir=layout.eval_dir,
        )
        state["builtin_eval"] = builtin_manifest
        summary_path = layout.eval_dir / "summary.json"
        if summary_path.exists():
            state["eval_summary"] = load_json(summary_path)
        external_hooks = config.get("open_unlearning", {}).get("evaluation", {}).get("external_hooks", [])
        state["external_evaluations"] = run_external_eval_hooks(
            hooks_cfg=external_hooks,
            eval_dir=layout.eval_dir,
            context={
                "model_path": str(model_locator),
                "run_dir": layout.run_dir,
                "open_unlearning_root": resolve_path(config.get("open_unlearning", {}).get("root"), PROJECT_DIR),
            },
        )

    if start_index <= stage_index("analysis") <= end_index:
        state["analysis"] = run_analysis_stage(
            analysis_cfg=config.get("analysis", {}),
            analysis_dir=layout.analysis_dir,
        )

    if start_index <= stage_index("report") <= end_index:
        state["report"] = build_report(layout, state, config, adapter)

    return state


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the unified pre-unlearning augmentation -> unlearning -> analysis pipeline."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--run-name", default=None, help="Override run name")
    parser.add_argument("--from-stage", default=None, choices=STAGES, help="Resume from stage")
    parser.add_argument("--until-stage", default=None, choices=STAGES, help="Stop after stage")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dotted paths",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config_path = resolve_path(args.config, PROJECT_DIR)
    if config_path is None:
        raise ValueError(f"Could not resolve config path: {args.config}")
    config = load_experiment_config(config_path)
    if args.set:
        config = apply_cli_overrides(config, args.set)
    run_pipeline(
        config=config,
        run_name_override=args.run_name,
        from_stage=args.from_stage,
        until_stage=args.until_stage,
    )


if __name__ == "__main__":
    main()
