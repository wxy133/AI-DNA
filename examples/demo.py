from ai_dna import AIDNAAgent, build_default_genome
from ai_dna.models import RuleBasedModel


def main() -> None:
    agent = AIDNAAgent(genome=build_default_genome(), model=RuleBasedModel())
    prompts = [
        ("math", "18 pencils and 9 more are combined. How many pencils is that?", ()),
        (
            "context_qa",
            "Which city hosts the summit?",
            ("The summit opening is in Lisbon.", "The workshop takes place in Porto."),
        ),
    ]
    for task_type, prompt, support_context in prompts:
        result = agent.run(prompt, task_type=task_type, support_context=support_context)
        print(prompt)
        print(result.output)
        print(result.active_genes)
        print()


if __name__ == "__main__":
    main()
