from __future__ import annotations

import os
from dataclasses import dataclass, field

from .operators import AttendOperator, TransformOperator
from .types import ModelAdapter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(slots=True)
class RuleBasedModel:
    name: str = "rule-based-baseline"
    attend: AttendOperator = field(default_factory=AttendOperator)
    transform: TransformOperator = field(default_factory=TransformOperator)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        del system_prompt, max_new_tokens
        inline_expression = self.transform.extract_inline_expression(prompt)
        if inline_expression:
            value = self.transform.solve_math(inline_expression)
            if value is not None:
                if float(value).is_integer():
                    return str(int(value))
                return f"{value:.10g}"

        lower = prompt.lower()
        if any(keyword in lower for keyword in ("write", "draft", "story", "email")):
            return "Here is a concise draft based on your prompt."

        facts = [line[2:].strip() for line in prompt.splitlines() if line.startswith("- ")]
        if facts:
            question = prompt.split("Question:", 1)[-1].strip() if "Question:" in prompt else prompt
            picked = self.attend.select(question, facts, top_k=1)
            if picked:
                return self.transform.extract_fact_answer(question, picked[0])

        numbers = self.transform.extract_numbers(prompt)
        if numbers:
            return str(int(numbers[0])) if numbers[0].is_integer() else f"{numbers[0]:.10g}"
        return prompt.strip().splitlines()[-1]


@dataclass(slots=True)
class TransformersTextModel:
    model_name: str
    max_new_tokens: int = 96
    name: str = ""
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _is_encoder_decoder: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.model_name

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        config = AutoConfig.from_pretrained(self.model_name)
        self._is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._is_encoder_decoder:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()
        torch.set_grad_enabled(False)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        self._ensure_loaded()

        assert self._model is not None
        assert self._tokenizer is not None

        combined_prompt = prompt
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{prompt}"

        encoded = self._tokenizer(
            combined_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        generation = self._model.generate(
            **encoded,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(self._tokenizer, "pad_token_id", None),
            eos_token_id=getattr(self._tokenizer, "eos_token_id", None),
        )
        decoded = self._tokenizer.decode(generation[0], skip_special_tokens=True).strip()

        if not self._is_encoder_decoder and decoded.startswith(combined_prompt):
            decoded = decoded[len(combined_prompt) :].strip()
        return decoded


def build_model_adapter(model_name: str | None) -> ModelAdapter:
    if model_name is None or model_name in {"rule", "rule-based", "rule_based"}:
        return RuleBasedModel()
    return TransformersTextModel(model_name=model_name)
