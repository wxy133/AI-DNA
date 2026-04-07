from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ModelAdapter(Protocol):
    name: str

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        ...


@dataclass(slots=True)
class TaskSample:
    sample_id: str
    task_type: str
    prompt: str
    expected: str
    support_context: tuple[str, ...] = ()
    scenario: str = "default"


@dataclass(slots=True)
class ExecutionContext:
    prompt: str
    task_type: str
    scenario: str = "default"
    support_context: tuple[str, ...] = ()
    expected: str | None = None
    memory: dict[str, Any] = field(default_factory=dict)
    focused_items: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    answer: str | None = None
    answer_kind: str = "text"
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    traces: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def log(self, message: str) -> None:
        self.traces.append(message)


@dataclass(slots=True)
class AgentResult:
    output: str
    task_type: str
    active_genes: tuple[str, ...]
    traces: tuple[str, ...]
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
