import unittest

from ai_dna.evolution import crossover_sequences, mutate_sequence


class EvolutionTests(unittest.TestCase):
    def test_mutation_keeps_valid_bases(self) -> None:
        mutated = mutate_sequence("TGGACCATCGAT", mutation_rate=0.5)
        self.assertTrue(set(mutated).issubset({"A", "T", "C", "G"}))

    def test_crossover_preserves_length(self) -> None:
        left, right = crossover_sequences("TGGACCATCGAT", "TCGATCCGAGAT")
        self.assertEqual(len(left), 12)
        self.assertEqual(len(right), 12)


if __name__ == "__main__":
    unittest.main()
