from __future__ import annotations

from dataclasses import dataclass, field

from .codons import CodonDefinition, DEFAULT_CODONS


@dataclass(slots=True)
class DNAParser:
    codon_registry: dict[str, CodonDefinition] = field(default_factory=lambda: dict(DEFAULT_CODONS))
    codon_size: int = 3

    def normalize_sequence(self, sequence: str) -> str:
        return "".join(base for base in sequence.upper() if base in {"A", "T", "C", "G"})

    def split_codons(self, sequence: str) -> list[str]:
        normalized = self.normalize_sequence(sequence)
        usable_length = len(normalized) - (len(normalized) % self.codon_size)
        return [
            normalized[index : index + self.codon_size]
            for index in range(0, usable_length, self.codon_size)
        ]

    def decode(self, sequence: str) -> list[CodonDefinition]:
        codons = self.split_codons(sequence)
        return [self.codon_registry[codon] for codon in codons if codon in self.codon_registry]

    def unknown_codons(self, sequence: str) -> list[str]:
        return [codon for codon in self.split_codons(sequence) if codon not in self.codon_registry]
