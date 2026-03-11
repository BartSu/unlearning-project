from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter


class WMDPAdapter(BenchmarkAdapter):
    name = "wmdp"

    def load_dataset(self, path: Path) -> list[dict[str, Any]]:
        return self.load_generic_records(path)

    def sample_to_prompt_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.render_question(sample)},
        ]

    def sample_to_response_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        answer = sample[self.answer_key()]
        if isinstance(answer, list):
            answer = answer[0]
        return [{"role": "assistant", "content": str(answer)}]

    def sample_to_training_text(self, sample: dict[str, Any]) -> str:
        answer = sample[self.answer_key()]
        if isinstance(answer, list):
            answer = answer[0]
        return f"{self.render_question(sample)}\nAnswer: {answer}"

    def sample_id(self, sample: dict[str, Any]) -> str:
        return self.stable_identity(sample)

    def build_retain_dataset(self, forget_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        full_samples = self.load_full_dataset()
        forget_ids = {self.sample_id(sample) for sample in forget_samples}
        return [sample for sample in full_samples if self.sample_id(sample) not in forget_ids]

    def forget_corpus_filename(self) -> str:
        default_name = f"{self.config.get('data_split', 'default')}-forget-corpus.jsonl"
        return str(self.config.get("forget_corpus_filename", default_name))

    def retain_corpus_filename(self) -> str:
        default_name = f"{self.config.get('data_split', 'default')}-retain-corpus.jsonl"
        return str(self.config.get("retain_corpus_filename", default_name))

    def open_unlearning_train_overrides(
        self,
        forget_corpus_path: Path,
        retain_corpus_path: Path,
    ) -> dict[str, Any]:
        data_split = self.config.get("data_split", "bio")
        return {
            "data_split": data_split,
            "data.forget.WMDP_forget.args.hf_args.data_files": str(forget_corpus_path),
            "data.retain.WMDP_retain.args.hf_args.data_files": str(retain_corpus_path),
        }

    def open_unlearning_eval_overrides(self, model_path: Path | str) -> dict[str, Any]:
        overrides = super().open_unlearning_eval_overrides(model_path)
        overrides["data_split"] = self.config.get("data_split", "bio")
        return overrides
