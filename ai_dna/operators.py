from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .types import ExecutionContext, ModelAdapter

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")
_EXPR_RE = re.compile(r"(?<!\w)([\d\.\s\+\-\*\/\(\)]+)(?!\w)")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_text(text))


def _safe_eval_expression(expression: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
    )
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
    return float(eval(compile(tree, "<ai-dna>", "eval"), {"__builtins__": {}}, {}))


@dataclass(slots=True)
class AttendOperator:
    def select(self, query: str, candidates: Sequence[str], top_k: int = 3) -> list[str]:
        if not candidates:
            return []

        query_tokens = set(tokenize(query))
        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            candidate_tokens = set(tokenize(candidate))
            overlap = len(query_tokens & candidate_tokens)
            density = overlap / max(len(candidate_tokens), 1)
            bonus = 0.25 if normalize_text(query) in normalize_text(candidate) else 0.0
            score = overlap + density + bonus
            if score > 0:
                scored.append((score, candidate))

        if not scored:
            return list(candidates[:top_k])

        scored.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in scored[:top_k]]


@dataclass(slots=True)
class TransformOperator:
    def extract_numbers(self, text: str) -> list[float]:
        return [float(match) for match in _NUMBER_RE.findall(text)]

    def extract_inline_expression(self, text: str) -> str | None:
        matches = []
        for raw_match in _EXPR_RE.findall(text):
            candidate = raw_match.strip()
            if not candidate:
                continue
            if not any(symbol in candidate for symbol in "+-*/"):
                continue
            if not any(char.isdigit() for char in candidate):
                continue
            matches.append(candidate)
        if not matches:
            return None
        matches.sort(key=len, reverse=True)
        return matches[0]

    def infer_math_operation(self, text: str) -> str | None:
        lower = normalize_text(text)
        if any(keyword in lower for keyword in ("average", "mean")):
            return "average"
        if any(keyword in lower for keyword in ("shared equally", "equally", "divided", "per", "quotient")):
            return "divide"
        if any(keyword in lower for keyword in ("left", "remain", "remaining", "difference", "fewer", "less", "after")):
            return "subtract"
        if any(keyword in lower for keyword in ("times", "product", "each", "every", "double", "triple")):
            return "multiply"
        if any(keyword in lower for keyword in ("greater", "larger", "highest", "maximum", "max")):
            return "max"
        if any(keyword in lower for keyword in ("smaller", "lowest", "minimum", "min")):
            return "min"
        if any(keyword in lower for keyword in ("sum", "total", "altogether", "combined", "more", "in all")):
            return "add"
        return None

    def solve_math(self, text: str) -> float | None:
        expression = self.extract_inline_expression(text)
        if expression:
            return _safe_eval_expression(expression)

        numbers = self.extract_numbers(text)
        operation = self.infer_math_operation(text)
        if not numbers or operation is None:
            return None

        if operation == "add":
            return float(sum(numbers))
        if operation == "subtract":
            return float(numbers[0] - sum(numbers[1:]))
        if operation == "multiply":
            product = 1.0
            for number in numbers:
                product *= number
            return float(product)
        if operation == "divide":
            value = numbers[0]
            for number in numbers[1:]:
                value /= number
            return float(value)
        if operation == "average":
            return float(sum(numbers) / len(numbers))
        if operation == "max":
            return float(max(numbers))
        if operation == "min":
            return float(min(numbers))
        return None

    def infer_task_type(self, prompt: str, support_context: Sequence[str] = ()) -> str:
        lower = normalize_text(prompt)
        if support_context:
            return "context_qa"
        if any(keyword in lower for keyword in ("write", "draft", "story", "article", "email", "blog", "copy")):
            return "writing"
        if self.solve_math(prompt) is not None or any(
            keyword in lower
            for keyword in ("calculate", "how many", "how much", "total", "sum", "average")
        ):
            return "math"
        if any(keyword in lower for keyword in ("analyze", "compare", "reason", "why", "how")):
            return "reasoning"
        return "qa"

    def extract_keywords(self, text: str, limit: int = 6) -> list[str]:
        counts = Counter(
            token
            for token in tokenize(text)
            if len(token) > 2 and token not in _STOP_WORDS and not token.isdigit()
        )
        return [token for token, _ in counts.most_common(limit)]

    def summarize_facts(self, facts: Sequence[str]) -> str:
        return " ".join(fact.strip() for fact in facts if fact.strip())

    def extract_fact_answer(self, question: str, fact: str) -> str:
        clean_fact = fact.strip().rstrip(".")
        lower_question = normalize_text(question)

        if "which city" in lower_question or "where" in lower_question:
            for pattern in (r"\bin ([A-Za-z][A-Za-z0-9 -]+)$", r"\bat ([A-Za-z][A-Za-z0-9 -]+)$"):
                match = re.search(pattern, clean_fact)
                if match:
                    return match.group(1).strip()

        if "when" in lower_question:
            match = re.search(r"\bon ([A-Za-z0-9 ,:-]+)$", clean_fact)
            if match:
                return match.group(1).strip()
            match = re.search(r"\bis ([A-Za-z0-9 ,:-]+)$", clean_fact)
            if match:
                return match.group(1).strip()

        for connector in (" is ", " are ", " was ", " were "):
            if connector in clean_fact and any(word in lower_question for word in ("which", "what", "who")):
                return clean_fact.split(connector, 1)[0].strip()

        return clean_fact

    def build_outline(self, context: ExecutionContext) -> list[str]:
        outline = [f"Classify task as {context.task_type}"]
        if context.focused_items:
            outline.append("Use attended evidence")
        if context.task_type == "math":
            outline.append("Compute exact numeric answer")
            outline.append("Return concise result")
        elif context.task_type == "context_qa":
            outline.append("Extract answer from supporting facts")
        elif context.task_type == "writing":
            outline.append("Create a short structure before drafting")
        else:
            outline.append("Answer directly and keep it grounded")
        return outline

    def build_model_prompt(self, context: ExecutionContext) -> str:
        sections = [
            "You are an AI-DNA gene execution endpoint.",
            f"Task type: {context.task_type}",
            f"User prompt: {context.prompt}",
        ]
        if context.focused_items:
            sections.append("Focused facts:\n- " + "\n- ".join(context.focused_items))
        if context.plan:
            sections.append("Plan:\n- " + "\n- ".join(context.plan))
        if context.answer is not None:
            sections.append(f"Current intermediate answer: {context.answer}")
        sections.append("Return the best final answer.")
        return "\n\n".join(sections)


@dataclass(slots=True)
class ControlOperator:
    def should_use_model(self, context: ExecutionContext) -> bool:
        if context.task_type in {"writing", "qa", "reasoning"}:
            return True
        return context.answer is None


@dataclass(slots=True)
class GenerateOperator:
    def format_value(self, value: float | int | str) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (float, int)):
            if math.isclose(float(value), round(float(value)), abs_tol=1e-9):
                return str(int(round(float(value))))
            return f"{float(value):.10g}"
        return str(value)

    def synthesize(self, context: ExecutionContext, model: ModelAdapter | None = None) -> str:
        if context.answer is not None:
            return self.format_value(context.answer)
        if model is not None:
            return model.generate(context.prompt)
        if context.focused_items:
            return context.focused_items[0]
        if context.plan:
            return " | ".join(context.plan)
        return context.prompt
