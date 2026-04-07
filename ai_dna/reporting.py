from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .benchmarks import DEFAULT_BENCHMARK, run_benchmark
from .evolution import crossover_genomes, mutate_genome, natural_selection
from .genome import build_default_genome, build_unoptimized_genome
from .models import build_model_adapter
from .runtime import AIDNAAgent
from .types import TaskSample

DEFAULT_REPORT_MODELS: tuple[str, ...] = (
    "rule-based",
    "google/flan-t5-small",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
)


@dataclass(slots=True)
class ReportArtifacts:
    output_dir: Path
    summary: list[dict[str, float | str]]
    evolution_history: list[dict[str, float | int]]
    files: dict[str, Path]


def _load_plotting_libs():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError(
            "Reporting dependencies are missing. Install them with `pip install -e .[report]`."
        ) from exc
    return plt, pd, sns


def _write_markdown_report(
    output_dir: Path,
    summary_rows: list[dict[str, float | str]],
    evolution_history: list[dict[str, float | int]],
) -> Path:
    lines = [
        "# AI-DNA Experiment Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Benchmark Summary",
        "",
        "| Model | Baseline | AI-DNA | Gain |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['baseline_accuracy']:.0%} | {row['ai_dna_accuracy']:.0%} | {row['improvement']:.0%} |"
        )

    last = evolution_history[-1]
    lines.extend(
        [
            "",
            "## Evolution Summary",
            "",
            f"- Final best fitness: {last['best_so_far']:.0%}",
            f"- Best generation fitness: {last['best_fitness']:.0%}",
            f"- Final elite average: {last['avg_elite_fitness']:.0%}",
            "",
            "## Artifacts",
            "",
            "- `accuracy_comparison.png`",
            "- `evolution_curve.png`",
            "- `ai_dna_dashboard.png`",
            "- `benchmark_summary.csv`",
            "- `benchmark_details.csv`",
            "- `evolution_history.csv`",
            "- `experiment_summary.json`",
        ]
    )

    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_experiment_report(
    output_dir: str | Path = "outputs",
    model_names: Sequence[str] = DEFAULT_REPORT_MODELS,
    generations: int = 8,
    population_size: int = 10,
    mutation_rate: float = 0.12,
    rng_seed: int = 7,
) -> ReportArtifacts:
    plt, pd, sns = _load_plotting_libs()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    summary_rows: list[dict[str, float | str]] = []
    detail_rows: list[dict[str, float | str]] = []

    for model_name in model_names:
        model = build_model_adapter(model_name)
        agent = AIDNAAgent(genome=build_default_genome(), model=model)
        report = run_benchmark(agent=agent, samples=DEFAULT_BENCHMARK)
        summary_rows.append(
            {
                "model": model_name,
                "baseline_accuracy": report.baseline_accuracy,
                "ai_dna_accuracy": report.ai_dna_accuracy,
                "improvement": report.ai_dna_accuracy - report.baseline_accuracy,
            }
        )
        for item in report.per_sample:
            detail_rows.append(
                {
                    "model": model_name,
                    "sample_id": item.sample_id,
                    "task_type": item.task_type,
                    "expected": item.expected,
                    "baseline_output": item.baseline_output,
                    "ai_dna_output": item.ai_dna_output,
                    "baseline_score": item.baseline_score,
                    "ai_dna_score": item.ai_dna_score,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    summary_df.to_csv(output_path / "benchmark_summary.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(output_path / "benchmark_details.csv", index=False, encoding="utf-8-sig")

    plot_df = summary_df.melt(
        id_vars=["model"],
        value_vars=["baseline_accuracy", "ai_dna_accuracy"],
        var_name="mode",
        value_name="accuracy",
    )
    plot_df["mode"] = plot_df["mode"].map(
        {"baseline_accuracy": "Baseline", "ai_dna_accuracy": "AI-DNA"}
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    sns.barplot(data=plot_df, x="model", y="accuracy", hue="mode", palette=["#c44e52", "#55a868"], ax=ax)
    ax.set_title("AI-DNA Accuracy Gain Across Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", labels=[f"{bar.get_height()*100:.0f}%" for bar in container])
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    accuracy_chart = output_path / "accuracy_comparison.png"
    fig.savefig(accuracy_chart, bbox_inches="tight")
    plt.close(fig)

    math_samples = tuple(
        TaskSample(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            prompt=sample.prompt,
            expected=sample.expected,
            support_context=sample.support_context,
            scenario="evolve",
        )
        for sample in DEFAULT_BENCHMARK
        if sample.task_type == "math"
    )

    seed = build_unoptimized_genome()

    def scorer(genome) -> float:
        agent = AIDNAAgent(genome=genome, model=build_model_adapter("rule-based"))
        report = run_benchmark(agent=agent, samples=math_samples)
        return report.ai_dna_accuracy

    import random

    rng = random.Random(rng_seed)
    population = [seed.clone()]
    while len(population) < population_size:
        population.append(mutate_genome(seed, rng=rng, mutation_rate=mutation_rate))

    history: list[dict[str, float | int]] = []
    best_overall = scorer(seed)
    for generation in range(generations + 1):
        evaluated = natural_selection(population, scorer=scorer, keep_top_k=max(2, population_size // 2))
        generation_best = evaluated[0].fitness
        generation_avg = sum(item.fitness for item in evaluated) / len(evaluated)
        best_overall = max(best_overall, generation_best)
        history.append(
            {
                "generation": generation,
                "best_fitness": generation_best,
                "avg_elite_fitness": generation_avg,
                "best_so_far": best_overall,
            }
        )
        if generation == generations:
            break
        next_population = [elite.genome.clone() for elite in evaluated[:2]]
        while len(next_population) < population_size:
            parent_a = rng.choice(evaluated).genome
            if rng.random() < 0.5:
                parent_b = rng.choice(evaluated).genome
                child = crossover_genomes(parent_a, parent_b, rng=rng)
            else:
                child = parent_a.clone()
            child = mutate_genome(child, rng=rng, mutation_rate=mutation_rate)
            next_population.append(child)
        population = next_population

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path / "evolution_history.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    sns.lineplot(data=history_df, x="generation", y="best_fitness", marker="o", linewidth=2.5, label="Generation best", ax=ax)
    sns.lineplot(data=history_df, x="generation", y="avg_elite_fitness", marker="o", linewidth=2.5, label="Elite average", ax=ax)
    sns.lineplot(data=history_df, x="generation", y="best_so_far", marker="o", linewidth=2.5, linestyle="--", label="Best so far", ax=ax)
    ax.set_title("AI-DNA Evolution Search on Math Tasks")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness / Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    evolution_chart = output_path / "evolution_curve.png"
    fig.savefig(evolution_chart, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=180)
    sns.barplot(data=plot_df, x="model", y="accuracy", hue="mode", palette=["#c44e52", "#55a868"], ax=axes[0])
    axes[0].set_title("Baseline vs AI-DNA")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=12)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.0f%%", labels=[f"{bar.get_height()*100:.0f}%" for bar in container])

    sns.lineplot(data=history_df, x="generation", y="best_fitness", marker="o", linewidth=2.5, label="Generation best", ax=axes[1])
    sns.lineplot(data=history_df, x="generation", y="best_so_far", marker="o", linewidth=2.5, linestyle="--", label="Best so far", ax=axes[1])
    axes[1].set_title("Evolution Improvement")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Fitness / Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    plt.tight_layout()
    dashboard = output_path / "ai_dna_dashboard.png"
    fig.savefig(dashboard, bbox_inches="tight")
    plt.close(fig)

    payload = {"summary": summary_rows, "evolution_history": history}
    summary_json = output_path / "experiment_summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = _write_markdown_report(output_path, summary_rows, history)

    return ReportArtifacts(
        output_dir=output_path,
        summary=summary_rows,
        evolution_history=history,
        files={
            "summary_csv": output_path / "benchmark_summary.csv",
            "details_csv": output_path / "benchmark_details.csv",
            "evolution_csv": output_path / "evolution_history.csv",
            "accuracy_chart": accuracy_chart,
            "evolution_chart": evolution_chart,
            "dashboard": dashboard,
            "summary_json": summary_json,
            "report_markdown": report_md,
        },
    )
