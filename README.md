# AI-DNA

AI-DNA is a runnable prototype of the "evolvable AI" concept described in the repository's original design note. It turns the document into a small Python system with four layers:

1. `A/T/C/G` primitive operators.
2. Codon-to-skill mappings.
3. Gene execution pipelines.
4. A genome with regulation, epigenetic tuning, mutation, crossover, and selection.

The project is deliberately lightweight. The core runtime uses only the Python standard library. A Hugging Face adapter is included as an optional way to test whether AI-DNA style routing and tool use can improve a small language model on grounded tasks.

## What is implemented

- Primitive operators for attention, transformation, control, and generation.
- A codon registry with six default skill units: semantic understanding, logic reasoning, planning, text generation, arithmetic calculation, and memory storage.
- A DNA parser that decodes nucleotide sequences into executable codons.
- Genes and genomes with regulation rules and scenario-specific epigenetic marks.
- Evolution helpers for mutation, crossover, natural selection, and short search loops.
- A benchmark harness that compares a baseline model against AI-DNA enhanced execution.
- A CLI for demos, benchmarking, and evolution experiments.

## Install

Core runtime:

```bash
pip install -e .
```

Optional Hugging Face model support:

```bash
pip install -e .[hf]
```

## Quick Start

Run the built-in demo:

```bash
python -m ai_dna demo
```

Run a single prompt:

```bash
python -m ai_dna run --task-type math --prompt "24 marbles are shared equally among 6 kids. How many marbles does each kid get?"
```

Run the baseline vs AI-DNA benchmark with the built-in rule-based baseline:

```bash
python -m ai_dna benchmark
```

Run the benchmark with a real Hugging Face model:

```bash
python -m ai_dna benchmark --model google/flan-t5-small
```

Run a short evolutionary search:

```bash
python -m ai_dna evolve --generations 8 --population 12
```

## Suggested model for the first experiment

Use a small instruction-following model first so the effect of AI-DNA routing is easy to observe on CPU:

- `google/flan-t5-small` for a very light experiment.
- `HuggingFaceTB/SmolLM2-135M-Instruct` if you want a tiny causal model.

The benchmark is intentionally grounded around exact math and context-bound QA. In these settings AI-DNA can improve a small base model without retraining by routing math questions through the `ACC` codon and context questions through the `TCG/TGG/ATC` path.

## Repo Layout

```text
ai_dna/
  __init__.py
  __main__.py
  benchmarks.py
  cli.py
  codons.py
  evolution.py
  genome.py
  models.py
  operators.py
  parser.py
  runtime.py
  types.py
examples/
tests/
```

## Notes

- The architecture is a prototype, not a replacement for model training.
- The clearest near-term gain comes from modular routing, tool invocation, and evolutionary search over sequencing and regulation.
- The benchmark is small on purpose so you can iterate on codons and genes quickly.
