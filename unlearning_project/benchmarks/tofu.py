from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkAdapter


class TOFUAdapter(BenchmarkAdapter):
    name = "tofu"

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
        return f"Question: {sample[self.question_key()]}\nAnswer: {answer}"

    def sample_id(self, sample: dict[str, Any]) -> str:
        return self.stable_identity(sample)

    def build_retain_dataset(self, forget_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        full_samples = self.load_full_dataset()
        forget_ids = {self.sample_id(sample) for sample in forget_samples}
        return [sample for sample in full_samples if self.sample_id(sample) not in forget_ids]
