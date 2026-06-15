"""ReflectAI — Startup Idea Analyzer (AI-First Workshop)."""

from reflectai.config import get_settings
from reflectai.tasks.idea_analyzer import IdeaAnalyzerTask


def build_llm():
    settings = get_settings()
    if not settings.openrouter_api_key:
        return None  # falls back to MockLLM inside IdeaAnalyzerTask

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def main():
    llm = build_llm()
    using_real_llm = llm is not None

    task = IdeaAnalyzerTask(llm=llm)
    result = task.execute({
        "idea": "AI-powered meal planning app",
        "context": "Health-conscious millennials in LATAM",
    })

    print("=" * 50)
    print(f"ReflectAI v0.1.0 — {'Real LLM' if using_real_llm else 'MockLLM'}")
    print("=" * 50)
    print(f"Viability Score : {result.get('viability_score', 'N/A')}")
    print(f"\nStrengths:")
    for s in result.get("strengths", []):
        print(f"  - {s}")
    print(f"\nWeaknesses:")
    for w in result.get("weaknesses", []):
        print(f"  - {w}")
    print(f"\nSummary:\n  {result.get('summary', 'N/A')}")
    print(f"\nImprovements Made:")
    for i in result.get("improvements_made", []):
        print(f"  - {i}")
    print("=" * 50)


if __name__ == "__main__":
    main()
