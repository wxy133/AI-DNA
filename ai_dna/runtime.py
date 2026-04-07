from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .codons import RuntimeServices
from .genome import Genome, build_default_genome
from .models import RuleBasedModel
from .parser import DNAParser
from .types import AgentResult, ExecutionContext, ModelAdapter, TaskSample


@dataclass(slots=True)
class AIDNAAgent:
    genome: Genome = field(default_factory=build_default_genome)
    model: ModelAdapter | None = None
    parser: DNAParser = field(default_factory=DNAParser)
    services: RuntimeServices = field(init=False)

    def __post_init__(self) -> None:
        self.services = RuntimeServices(model=self.model)

    def classify_task(self, prompt: str, support_context: Iterable[str] = ()) -> str:
        return self.services.transform.infer_task_type(prompt, tuple(support_context))

    def execute_gene(self, gene_name: str, sequence: str, context: ExecutionContext) -> None:
        codons = self.parser.decode(sequence)
        context.log(f"Gene<{gene_name}> activate codons={[codon.sequence for codon in codons]}")
        for codon in codons:
            codon.executor(context, self.services)

    def run(
        self,
        prompt_or_sample: str | TaskSample,
        task_type: str | None = None,
        support_context: tuple[str, ...] = (),
        scenario: str = "default",
    ) -> AgentResult:
        if isinstance(prompt_or_sample, TaskSample):
            sample = prompt_or_sample
            prompt = sample.prompt
            task_type = sample.task_type
            support_context = sample.support_context
            scenario = sample.scenario
            expected = sample.expected
        else:
            prompt = prompt_or_sample
            expected = None

        resolved_task_type = task_type or self.classify_task(prompt, support_context)
        context = ExecutionContext(
            prompt=prompt,
            task_type=resolved_task_type,
            scenario=scenario,
            support_context=tuple(support_context),
            expected=expected,
        )

        active_genes = self.genome.active_genes(resolved_task_type, scenario=scenario)
        if not active_genes:
            raise RuntimeError(f"No genes available for task type: {resolved_task_type}")

        for gene in active_genes:
            self.execute_gene(gene.name, gene.sequence, context)
            if context.answer is not None and gene.name == "math_gene":
                break

        if context.answer is None:
            context.answer = self.services.generate.synthesize(context, self.model)
            context.log("[FINAL] synthesized without explicit terminal codon")

        result = AgentResult(
            output=str(context.answer).strip(),
            task_type=resolved_task_type,
            active_genes=tuple(gene.name for gene in active_genes),
            traces=tuple(context.traces),
            metrics={"activated_genes": float(len(active_genes))},
            metadata={
                "plan": list(context.plan),
                "focused_items": list(context.focused_items),
                "tool_outputs": dict(context.tool_outputs),
            },
        )
        return result

    def run_baseline(self, sample: TaskSample, model: ModelAdapter | None = None) -> str:
        baseline_model = model or self.model or RuleBasedModel()
        prompt = sample.prompt
        if sample.support_context:
            prompt = (
                "Use the following context to answer the question.\n\n"
                + "\n".join(f"- {fact}" for fact in sample.support_context)
                + f"\n\nQuestion: {sample.prompt}"
            )
        return baseline_model.generate(prompt).strip()
