"""Full entry point — runs IdeaAnalyzerTask with MockLLM."""

from reflectai.tasks.idea_analyzer import IdeaAnalyzerTask


def main():
    task = IdeaAnalyzerTask()  # uses MockLLM by default
    result = task.execute({
        "idea": "AI-powered meal planning app",
        "context": "Health-conscious millennials in Latin America",
    })

    print("ReflectAI v0.1.0 — Startup Idea Analyzer")
    print("=" * 50)
    print("\nAnalysis Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
