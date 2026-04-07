import unittest

from ai_dna.benchmarks import DEFAULT_BENCHMARK, run_benchmark
from ai_dna.genome import build_default_genome
from ai_dna.models import RuleBasedModel
from ai_dna.runtime import AIDNAAgent


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = AIDNAAgent(genome=build_default_genome(), model=RuleBasedModel())

    def test_math_gene_solves_word_problem(self) -> None:
        result = self.agent.run(
            "24 marbles are shared equally among 6 kids. How many marbles does each kid get?",
            task_type="math",
        )
        self.assertEqual(result.output, "4")
        self.assertIn("math_gene", result.active_genes)

    def test_context_gene_uses_support_context(self) -> None:
        result = self.agent.run(
            "Which planet is closest to the Sun?",
            task_type="context_qa",
            support_context=(
                "Mercury is the closest planet to the Sun.",
                "Venus is the hottest planet because of its atmosphere.",
            ),
        )
        self.assertIn("Mercury", result.output)
        self.assertIn("context_gene", result.active_genes)

    def test_benchmark_beats_rule_baseline(self) -> None:
        report = run_benchmark(self.agent, samples=DEFAULT_BENCHMARK)
        self.assertGreater(report.ai_dna_accuracy, report.baseline_accuracy)


if __name__ == "__main__":
    unittest.main()
