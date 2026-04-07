from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .operators import AttendOperator, ControlOperator, GenerateOperator, TransformOperator
from .types import ExecutionContext, ModelAdapter

CodonExecutor = Callable[[ExecutionContext, "RuntimeServices"], None]


@dataclass(slots=True)
class RuntimeServices:
    attend: AttendOperator = field(default_factory=AttendOperator)
    transform: TransformOperator = field(default_factory=TransformOperator)
    control: ControlOperator = field(default_factory=ControlOperator)
    generate: GenerateOperator = field(default_factory=GenerateOperator)
    model: ModelAdapter | None = None


@dataclass(frozen=True, slots=True)
class CodonDefinition:
    sequence: str
    name: str
    description: str
    executor: CodonExecutor


def _semantic_understanding(context: ExecutionContext, services: RuntimeServices) -> None:
    focused = services.attend.select(context.prompt, context.support_context, top_k=3)
    context.focused_items = focused or context.focused_items
    keywords = services.transform.extract_keywords(context.prompt)
    context.memory["keywords"] = keywords
    context.memory["intent"] = " ".join(keywords) or context.prompt.strip()
    context.log(f"[TCG] semantic focus={focused or 'self'} keywords={keywords}")


def _logic_reasoning(context: ExecutionContext, services: RuntimeServices) -> None:
    if not context.plan:
        context.plan = services.transform.build_outline(context)
    if context.task_type == "context_qa" and context.focused_items and context.answer is None:
        context.answer = services.transform.extract_fact_answer(context.prompt, context.focused_items[0])
        context.answer_kind = "text"
    context.log(f"[ATC] plan={context.plan}")


def _planning_decision(context: ExecutionContext, services: RuntimeServices) -> None:
    if not context.plan:
        context.plan = services.transform.build_outline(context)
    context.metadata["use_model"] = services.control.should_use_model(context)
    context.log(f"[CGA] use_model={context.metadata['use_model']}")


def _text_generation(context: ExecutionContext, services: RuntimeServices) -> None:
    if context.answer is not None:
        context.answer = services.generate.synthesize(context)
        context.log(f"[GAT] generated from intermediate answer={context.answer}")
        return

    if context.metadata.get("use_model") and services.model is not None:
        prompt = services.transform.build_model_prompt(context)
        context.answer = services.model.generate(prompt)
        context.answer_kind = "text"
        context.log("[GAT] generated via model adapter")
        return

    if context.focused_items:
        context.answer = services.transform.extract_fact_answer(context.prompt, context.focused_items[0])
    else:
        context.answer = services.generate.synthesize(context)
    context.answer_kind = "text"
    context.log(f"[GAT] generated fallback answer={context.answer}")


def _arithmetic_calculation(context: ExecutionContext, services: RuntimeServices) -> None:
    value = services.transform.solve_math(context.prompt)
    if value is None:
        context.log("[ACC] no solvable math expression detected")
        return
    formatted = services.generate.format_value(value)
    context.tool_outputs["calculation"] = formatted
    context.answer = formatted
    context.answer_kind = "numeric"
    context.log(f"[ACC] exact calculation={formatted}")


def _memory_storage(context: ExecutionContext, services: RuntimeServices) -> None:
    context.memory["prompt"] = context.prompt
    if context.support_context:
        context.memory["support_context"] = list(context.support_context)
    if context.focused_items:
        context.memory["focused_items"] = list(context.focused_items)
    context.log("[TGG] memory updated")


DEFAULT_CODONS: dict[str, CodonDefinition] = {
    "ATC": CodonDefinition("ATC", "logic_reasoning", "Build reasoning steps and extract grounded answers.", _logic_reasoning),
    "TCG": CodonDefinition(
        "TCG",
        "semantic_understanding",
        "Attend to the prompt and supporting context, then store intent cues.",
        _semantic_understanding,
    ),
    "CGA": CodonDefinition("CGA", "planning_decision", "Choose the next execution path and whether to invoke a model.", _planning_decision),
    "GAT": CodonDefinition("GAT", "text_generation", "Produce the final natural-language answer.", _text_generation),
    "ACC": CodonDefinition("ACC", "arithmetic_calculation", "Compute an exact arithmetic result.", _arithmetic_calculation),
    "TGG": CodonDefinition("TGG", "memory_storage", "Store prompt and attended context in working memory.", _memory_storage),
}
