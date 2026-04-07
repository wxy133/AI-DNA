from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .models import RuleBasedModel
from .runtime import AIDNAAgent
from .types import ModelAdapter, TaskSample

DEFAULT_BENCHMARK: tuple[TaskSample, ...] = (
    TaskSample(
        sample_id="math-add-word",
        task_type="math",
        prompt="Lena had 12 shells and found 5 more. How many shells does she have now?",
        expected="17",
    ),
    TaskSample(
        sample_id="math-subtract-word",
        task_type="math",
        prompt="A box has 30 cookies. If 7 are eaten, how many remain?",
        expected="23",
    ),
    TaskSample(
        sample_id="math-multiply-word",
        task_type="math",
        prompt="Each pack has 6 batteries. How many batteries are in 4 packs?",
        expected="24",
    ),
    TaskSample(
        sample_id="math-divide-word",
        task_type="math",
        prompt="24 marbles are shared equally among 6 kids. How many marbles does each kid get?",
        expected="4",
    ),
    TaskSample(
        sample_id="context-planet",
        task_type="context_qa",
        prompt="Which planet is closest to the Sun?",
        expected="Mercury",
        support_context=(
            "Mercury is the closest planet to the Sun.",
            "Venus is the hottest planet because of its atmosphere.",
        ),
    ),
    TaskSample(
        sample_id="context-schedule",
        task_type="context_qa",
        prompt="When is the Project Nova launch?",
        expected="Friday",
        support_context=(
            "The demo rehearsal is on Wednesday.",
            "Project Nova launches on Friday.",
        ),
    ),
)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().replace(".", " ").replace(",", " ").split())


def _extract_first_number(text: str) -> str | None:
    for token in text.replace(",", " ").split():
        try:
            value = float(token)
        except ValueError:
            continue
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return None


def score_prediction(sample: TaskSample, prediction: str) -> float:
    expected = _normalize_text(sample.expected)
    normalized = _normalize_text(prediction)

    if sample.task_type == "math":
        predicted_number = _extract_first_number(prediction)
        return 1.0 if predicted_number == expected else 0.0

    return 1.0 if expected in normalized else 0.0


@dataclass(slots=True)
class SampleComparison:
    sample_id: str
    task_type: str
    expected: str
    baseline_output: str
    ai_dna_output: str
    baseline_score: float
    ai_dna_score: float


@dataclass(slots=True)
class BenchmarkReport:
    model_name: str
    baseline_accuracy: float
    ai_dna_accuracy: float
    per_sample: list[SampleComparison]


def run_benchmark(
    agent: AIDNAAgent,
    samples: tuple[TaskSample, ...] = DEFAULT_BENCHMARK,
    baseline_model: ModelAdapter | None = None,
) -> BenchmarkReport:
    fallback_baseline = baseline_model or agent.model or RuleBasedModel()
    comparisons: list[SampleComparison] = []

    for sample in samples:
        baseline_output = agent.run_baseline(sample, fallback_baseline)
        ai_dna_output = agent.run(sample).output
        comparisons.append(
            SampleComparison(
                sample_id=sample.sample_id,
                task_type=sample.task_type,
                expected=sample.expected,
                baseline_output=baseline_output,
                ai_dna_output=ai_dna_output,
                baseline_score=score_prediction(sample, baseline_output),
                ai_dna_score=score_prediction(sample, ai_dna_output),
            )
        )

    return BenchmarkReport(
        model_name=fallback_baseline.name,
        baseline_accuracy=mean(item.baseline_score for item in comparisons),
        ai_dna_accuracy=mean(item.ai_dna_score for item in comparisons),
        per_sample=comparisons,
    )
