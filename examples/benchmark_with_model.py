from ai_dna import AIDNAAgent, build_default_genome, build_model_adapter, run_benchmark


def main() -> None:
    model = build_model_adapter("google/flan-t5-small")
    agent = AIDNAAgent(genome=build_default_genome(), model=model)
    report = run_benchmark(agent)
    print(f"Baseline accuracy: {report.baseline_accuracy:.2%}")
    print(f"AI-DNA accuracy: {report.ai_dna_accuracy:.2%}")


if __name__ == "__main__":
    main()
