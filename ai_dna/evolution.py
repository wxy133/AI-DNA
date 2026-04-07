from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .genome import Gene, Genome

NUCLEOTIDES = ("A", "T", "C", "G")


def mutate_sequence(
    sequence: str,
    mutation_rate: float = 0.1,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random()
    letters = list(sequence)
    for index, base in enumerate(letters):
        if rng.random() < mutation_rate:
            letters[index] = rng.choice([candidate for candidate in NUCLEOTIDES if candidate != base])

    if rng.random() < mutation_rate:
        operation = rng.choice(("insert", "delete"))
        if operation == "insert":
            letters.insert(rng.randrange(len(letters) + 1), rng.choice(NUCLEOTIDES))
        elif len(letters) > 3:
            del letters[rng.randrange(len(letters))]
    return "".join(letters)


def crossover_sequences(
    left: str,
    right: str,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    if len(left) != len(right):
        raise ValueError("Crossover requires equal-length sequences.")
    rng = rng or random.Random()
    pivot = rng.randrange(1, len(left))
    return left[:pivot] + right[pivot:], right[:pivot] + left[pivot:]


def mutate_genome(genome: Genome, rng: random.Random | None = None, mutation_rate: float = 0.1) -> Genome:
    rng = rng or random.Random()
    mutated = genome.clone()

    if rng.random() < 0.55:
        gene_name = rng.choice(list(mutated.genes))
        gene = mutated.genes[gene_name]
        mutated.genes[gene_name] = Gene(
            name=gene.name,
            sequence=mutate_sequence(gene.sequence, mutation_rate=mutation_rate, rng=rng),
            description=gene.description,
            task_types=gene.task_types,
            priority=gene.priority,
            activation_bias=max(0.2, gene.activation_bias + rng.uniform(-0.25, 0.25)),
            epigenetic_marks=dict(gene.epigenetic_marks),
            metadata=dict(gene.metadata),
        )
    else:
        rule = rng.choice(mutated.regulations)
        rule.weight = max(0.1, rule.weight + rng.uniform(-1.0, 1.0))
        rule.order = max(0, rule.order + rng.choice((-1, 0, 1)))

    if rng.random() < 0.7:
        candidate = rng.choice(list(mutated.genes))
        mutated.apply_epigenetic_tuning("evolve", {candidate: rng.uniform(0.5, 5.0)})

    return mutated


def crossover_genomes(left: Genome, right: Genome, rng: random.Random | None = None) -> Genome:
    rng = rng or random.Random()
    child = left.clone()
    child.name = f"{left.name}_x_{right.name}"

    for gene_name, gene in child.genes.items():
        if gene_name not in right.genes:
            continue
        sibling = right.genes[gene_name]
        if len(gene.sequence) != len(sibling.sequence):
            continue
        if rng.random() < 0.5:
            left_seq, _ = crossover_sequences(gene.sequence, sibling.sequence, rng=rng)
            gene.sequence = left_seq

    for index, rule in enumerate(child.regulations):
        if index < len(right.regulations) and rng.random() < 0.5:
            rule.weight = right.regulations[index].weight
            rule.order = right.regulations[index].order
    return child


@dataclass(slots=True)
class EvaluatedGenome:
    genome: Genome
    fitness: float


def natural_selection(
    population: list[Genome],
    scorer: Callable[[Genome], float],
    keep_top_k: int,
) -> list[EvaluatedGenome]:
    evaluated = [EvaluatedGenome(genome=genome, fitness=scorer(genome)) for genome in population]
    evaluated.sort(key=lambda item: item.fitness, reverse=True)
    return evaluated[:keep_top_k]


def evolve_population(
    seed: Genome,
    scorer: Callable[[Genome], float],
    generations: int = 5,
    population_size: int = 8,
    mutation_rate: float = 0.1,
    rng_seed: int = 7,
) -> EvaluatedGenome:
    rng = random.Random(rng_seed)
    population = [seed.clone()]
    while len(population) < population_size:
        population.append(mutate_genome(seed, rng=rng, mutation_rate=mutation_rate))

    best = EvaluatedGenome(genome=seed.clone(), fitness=scorer(seed))
    for _ in range(generations):
        elites = natural_selection(population, scorer=scorer, keep_top_k=max(2, population_size // 2))
        if elites[0].fitness >= best.fitness:
            best = elites[0]

        next_population = [elite.genome.clone() for elite in elites[:2]]
        while len(next_population) < population_size:
            parent_a = rng.choice(elites).genome
            if rng.random() < 0.5:
                parent_b = rng.choice(elites).genome
                child = crossover_genomes(parent_a, parent_b, rng=rng)
            else:
                child = parent_a.clone()
            child = mutate_genome(child, rng=rng, mutation_rate=mutation_rate)
            next_population.append(child)
        population = next_population

    return best
