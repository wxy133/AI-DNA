from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Gene:
    name: str
    sequence: str
    description: str
    task_types: frozenset[str]
    priority: int = 0
    activation_bias: float = 1.0
    epigenetic_marks: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def activation_score(self, scenario: str) -> float:
        return self.activation_bias * self.epigenetic_marks.get(scenario, 1.0)


@dataclass(slots=True)
class RegulationRule:
    task_type: str
    gene_name: str
    order: int = 0
    weight: float = 1.0
    scenario: str = "default"


@dataclass(slots=True)
class Genome:
    name: str
    genes: dict[str, Gene]
    regulations: list[RegulationRule]
    max_active_genes: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "Genome":
        return copy.deepcopy(self)

    def active_genes(self, task_type: str, scenario: str = "default") -> list[Gene]:
        candidates: list[tuple[float, int, Gene]] = []

        for rule in self.regulations:
            if rule.task_type not in {task_type, "*"}:
                continue
            if rule.scenario not in {scenario, "*", "default"}:
                continue
            gene = self.genes.get(rule.gene_name)
            if gene is None:
                continue
            score = rule.weight * gene.activation_score(scenario)
            candidates.append((score, rule.order, gene))

        if not candidates:
            for gene in self.genes.values():
                if task_type in gene.task_types or "*" in gene.task_types:
                    candidates.append((gene.activation_score(scenario), gene.priority, gene))

        candidates.sort(key=lambda item: (-item[0], item[1], -item[2].priority))
        selected: list[Gene] = []
        seen: set[str] = set()
        for _, _, gene in candidates:
            if gene.name in seen:
                continue
            selected.append(gene)
            seen.add(gene.name)
            if len(selected) >= self.max_active_genes:
                break
        return selected

    def apply_epigenetic_tuning(self, scenario: str, adjustments: dict[str, float]) -> None:
        for gene_name, multiplier in adjustments.items():
            if gene_name in self.genes:
                self.genes[gene_name].epigenetic_marks[scenario] = multiplier


def build_default_genome() -> Genome:
    genes = {
        "math_gene": Gene(
            name="math_gene",
            sequence="TGGACCATCGAT",
            description="Stores context, computes exact math, reasons briefly, then generates the answer.",
            task_types=frozenset({"math"}),
            priority=3,
            activation_bias=1.3,
        ),
        "qa_gene": Gene(
            name="qa_gene",
            sequence="TCGATCCGAGAT",
            description="Understands the prompt, reasons, plans, then answers.",
            task_types=frozenset({"qa", "reasoning"}),
            priority=2,
            activation_bias=1.0,
        ),
        "context_gene": Gene(
            name="context_gene",
            sequence="TCGTGGATCGAT",
            description="Attends to support context, stores it, reasons over it, then answers.",
            task_types=frozenset({"context_qa"}),
            priority=3,
            activation_bias=1.2,
        ),
        "writer_gene": Gene(
            name="writer_gene",
            sequence="TCGCGAGATATC",
            description="Builds a short plan and drafts written content.",
            task_types=frozenset({"writing"}),
            priority=1,
            activation_bias=1.0,
        ),
    }

    regulations = [
        RegulationRule("math", "math_gene", order=0, weight=1.8),
        RegulationRule("math", "qa_gene", order=1, weight=0.9),
        RegulationRule("context_qa", "context_gene", order=0, weight=1.7),
        RegulationRule("context_qa", "qa_gene", order=1, weight=0.8),
        RegulationRule("qa", "qa_gene", order=0, weight=1.4),
        RegulationRule("reasoning", "qa_gene", order=0, weight=1.3),
        RegulationRule("writing", "writer_gene", order=0, weight=1.5),
        RegulationRule("*", "qa_gene", order=5, weight=0.4),
    ]
    return Genome(name="default_ai_dna", genes=genes, regulations=regulations, max_active_genes=2)


def build_unoptimized_genome() -> Genome:
    genome = build_default_genome()
    genome.name = "unoptimized_ai_dna"
    genome.regulations = [
        RegulationRule("math", "writer_gene", order=0, weight=1.6),
        RegulationRule("math", "qa_gene", order=1, weight=1.2),
        RegulationRule("math", "math_gene", order=2, weight=0.3),
        RegulationRule("context_qa", "qa_gene", order=0, weight=1.1),
        RegulationRule("context_qa", "context_gene", order=1, weight=0.6),
        RegulationRule("writing", "writer_gene", order=0, weight=1.5),
        RegulationRule("*", "qa_gene", order=5, weight=0.5),
    ]
    genome.max_active_genes = 1
    return genome
