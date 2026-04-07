from __future__ import annotations

import argparse

from .benchmarks import DEFAULT_BENCHMARK, run_benchmark
from .evolution import evolve_population
from .genome import build_default_genome, build_unoptimized_genome
from .models import build_model_adapter
from .runtime import AIDNAAgent
from .types import TaskSample


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-DNA prototype CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single prompt through the AI-DNA agent.")
    run_parser.add_argument("--prompt", required=True)
    run_parser.add_argument("--task-type")
    run_parser.add_argument("--context", action="append", default=[])
    run_parser.add_argument("--scenario", default="default")
    run_parser.add_argument("--model")

    demo_parser = subparsers.add_parser("demo", help="Run a small built-in demo.")
    demo_parser.add_argument("--model")

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare a baseline model with AI-DNA.")
    benchmark_parser.add_argument("--model")
    benchmark_parser.add_argument("--limit", type=int)

    evolve_parser = subparsers.add_parser("evolve", help="Run a short evolutionary search over the genome.")
    evolve_parser.add_argument("--generations", type=int, default=5)
    evolve_parser.add_argument("--population", type=int, default=8)
    evolve_parser.add_argument("--mutation-rate", type=float, default=0.12)

    return parser


def _create_agent(model_name: str | None, optimized: bool = True) -> AIDNAAgent:
    genome = build_default_genome() if optimized else build_unoptimized_genome()
    model = build_model_adapter(model_name)
    return AIDNAAgent(genome=genome, model=model)


def _print_demo(agent: AIDNAAgent) -> None:
    demo_samples = [
        TaskSample("demo-math", "math", "18 pencils and 9 more are combined. How many pencils is that?", "27"),
        TaskSample(
            "demo-context",
            "context_qa",
            "Which city hosts the summit?",
            "Lisbon",
            support_context=("The summit opening is in Lisbon.", "The workshop takes place in Porto."),
        ),
        TaskSample("demo-writing", "writing", "Write a two-sentence launch announcement for a robotics newsletter.", ""),
    ]

    for sample in demo_samples:
        result = agent.run(sample)
        print(f"[{sample.sample_id}] {sample.prompt}")
        print(f"Output: {result.output}")
        print(f"Genes: {', '.join(result.active_genes)}")
        print()


def _print_benchmark(agent: AIDNAAgent, limit: int | None = None) -> None:
    samples = DEFAULT_BENCHMARK[:limit] if limit else DEFAULT_BENCHMARK
    report = run_benchmark(agent=agent, samples=samples)
    print(f"Baseline model: {report.model_name}")
    print(f"Baseline accuracy: {report.baseline_accuracy:.2%}")
    print(f"AI-DNA accuracy: {report.ai_dna_accuracy:.2%}")
    print()
    for item in report.per_sample:
        print(f"[{item.sample_id}] expected={item.expected}")
        print(f"  baseline: {item.baseline_output} ({item.baseline_score:.0f})")
        print(f"  ai-dna : {item.ai_dna_output} ({item.ai_dna_score:.0f})")


def _print_evolution(args: argparse.Namespace) -> None:
    seed = build_unoptimized_genome()
    evolve_samples = tuple(
        TaskSample(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            prompt=sample.prompt,
            expected=sample.expected,
            support_context=sample.support_context,
            scenario="evolve",
        )
        for sample in DEFAULT_BENCHMARK[:4]
    )

    def scorer(genome) -> float:
        agent = AIDNAAgent(genome=genome, model=build_model_adapter("rule-based"))
        report = run_benchmark(agent=agent, samples=evolve_samples)
        return report.ai_dna_accuracy

    before = scorer(seed)
    best = evolve_population(
        seed=seed,
        scorer=scorer,
        generations=args.generations,
        population_size=args.population,
        mutation_rate=args.mutation_rate,
    )
    after = scorer(best.genome)
    print(f"Seed fitness: {before:.2%}")
    print(f"Best fitness: {best.fitness:.2%}")
    print(f"Best genome: {best.genome.name}")
    print(f"Re-scored best fitness: {after:.2%}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        agent = _create_agent(args.model)
        result = agent.run(
            args.prompt,
            task_type=args.task_type,
            support_context=tuple(args.context),
            scenario=args.scenario,
        )
        print(result.output)
        print(f"genes={', '.join(result.active_genes)}")
        for trace in result.traces:
            print(trace)
        return

    if args.command == "demo":
        agent = _create_agent(args.model)
        _print_demo(agent)
        return

    if args.command == "benchmark":
        agent = _create_agent(args.model)
        _print_benchmark(agent, limit=args.limit)
        return

    if args.command == "evolve":
        _print_evolution(args)
        return


if __name__ == "__main__":
    main()
