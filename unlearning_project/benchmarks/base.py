from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from unlearning_project.common import PROJECT_DIR, load_records, resolve_path, write_jsonl


class BenchmarkAdapter(ABC):
    name = "base"

    def __init__(self, config: dict[str, Any], project_dir: Path = PROJECT_DIR):
        self.config = config
        self.project_dir = project_dir

    def benchmark_name(self) -> str:
        return str(self.config.get("benchmark_name", self.name))

    def resolve_config_path(self, key: str) -> Path:
        value = self.config.get(key)
        if value is None:
            raise ValueError(f"Benchmark config missing required path: {key}")
        resolved = resolve_path(value, self.project_dir)
        if resolved is None:
            raise ValueError(f"Could not resolve benchmark path: {key}")
        return resolved

    @abstractmethod
    def load_dataset(self, path: Path) -> list[dict[str, Any]]:
        raise NotImplementedError

    def load_forget_dataset(self) -> list[dict[str, Any]]:
        return self.load_dataset(self.resolve_config_path("forget_path"))

    def load_full_dataset(self) -> list[dict[str, Any]]:
        return self.load_dataset(self.resolve_config_path("full_path"))

    @abstractmethod
    def sample_to_prompt_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def sample_to_response_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def sample_to_training_text(self, sample: dict[str, Any]) -> str:
        raise NotImplementedError

    def sample_to_embedding_text(self, sample: dict[str, Any]) -> str:
        return self.sample_to_training_text(sample)

    @abstractmethod
    def sample_id(self, sample: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_retain_dataset(self, forget_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def write_selected_dataset(self, samples: list[dict[str, Any]], path: Path) -> None:
        write_jsonl(samples, path)

    def selected_output_filename(self) -> str:
        return str(self.config.get("selected_output_filename", "forget_selected.jsonl"))

    def forget_corpus_filename(self) -> str:
        return str(self.config.get("forget_corpus_filename", "forget-corpus.jsonl"))

    def retain_corpus_filename(self) -> str:
        return str(self.config.get("retain_corpus_filename", "retain-corpus.jsonl"))

    def open_unlearning_train_experiment(self) -> str | None:
        return self.config.get("open_unlearning", {}).get("train", {}).get("experiment")

    def open_unlearning_eval_experiment(self) -> str | None:
        return self.config.get("open_unlearning", {}).get("eval", {}).get("experiment")

    def open_unlearning_train_overrides(
        self,
        forget_corpus_path: Path,
        retain_corpus_path: Path,
    ) -> dict[str, Any]:
        return {}

    def open_unlearning_eval_overrides(self, model_path: Path | str) -> dict[str, Any]:
        return {"model.model_args.pretrained_model_name_or_path": str(model_path)}

    def normalize_eval_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in summary.items():
            normalized[str(key)] = float(value) if isinstance(value, (int, float)) else value
        return normalized

    def load_generic_records(self, path: Path) -> list[dict[str, Any]]:
        return load_records(path)

    def question_key(self) -> str:
        return str(self.config.get("question_key", "question"))

    def answer_key(self) -> str:
        return str(self.config.get("answer_key", "answer"))

    def choices_key(self) -> str | None:
        value = self.config.get("choices_key")
        return str(value) if value else None

    def include_choices(self) -> bool:
        return bool(self.config.get("include_choices", False))

    def system_prompt(self) -> str:
        return str(self.config.get("system_prompt", "You are a helpful assistant."))

    def render_question(self, sample: dict[str, Any]) -> str:
        question = str(sample[self.question_key()])
        choices_key = self.choices_key()
        if not self.include_choices() or not choices_key:
            return question
        choices = sample.get(choices_key)
        if not isinstance(choices, list) or not choices:
            return question
        lines = [f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)]
        return f"Question: {question}\nChoices:\n" + "\n".join(lines)

    def stable_identity(self, sample: dict[str, Any]) -> str:
        identity = {
            "question": sample.get(self.question_key()),
            "answer": sample.get(self.answer_key()),
        }
        choices_key = self.choices_key()
        if choices_key:
            identity["choices"] = sample.get(choices_key)
        return json.dumps(identity, ensure_ascii=False, sort_keys=True)
