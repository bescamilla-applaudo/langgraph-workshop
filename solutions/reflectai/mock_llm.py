"""Deterministic mock LLM for testing without API keys."""

from __future__ import annotations

from .schemas import AnalysisOutput, CritiqueOutput, RefinedOutput


class MockLLM:
    """Deterministic LLM that returns canned responses for testing."""

    def invoke(self, prompt: str) -> str:
        if "analyze" in prompt.lower() or "startup" in prompt.lower():
            return (
                "This startup idea shows moderate potential. "
                "Key strengths include market demand and scalability. "
                "Risks include competition and execution complexity."
            )
        if "critique" in prompt.lower():
            return (
                "The analysis covers the basics but lacks depth on market sizing "
                "and competitive landscape. Consider adding financial projections."
            )
        if "refine" in prompt.lower():
            return (
                "After incorporating feedback, the analysis now includes "
                "a more thorough competitive assessment and clearer metrics."
            )
        return "Mock LLM response for testing purposes."

    def with_structured_output(self, schema: type) -> MockStructuredLLM:
        return MockStructuredLLM(schema)


class MockStructuredLLM:
    """Returns properly structured Pydantic objects for testing."""

    def __init__(self, schema: type):
        self.schema = schema

    def invoke(self, prompt: str) -> AnalysisOutput | CritiqueOutput | RefinedOutput:
        if self.schema is AnalysisOutput:
            return AnalysisOutput(
                viability_score=0.72,
                strengths=[
                    "Strong market demand in health-tech",
                    "Scalable SaaS model",
                    "Clear target demographic",
                ],
                weaknesses=[
                    "Competitive market with established players",
                    "Requires significant content curation",
                    "User retention challenges",
                ],
                summary=(
                    "The idea shows solid potential with a clear target market. "
                    "Success depends on differentiation and content quality."
                ),
            )
        if self.schema is CritiqueOutput:
            return CritiqueOutput(
                has_issues=True,
                issues=[
                    "Market size not quantified",
                    "No mention of monetization strategy",
                    "Competitive analysis is surface-level",
                ],
                suggestions=[
                    "Add TAM/SAM/SOM estimates",
                    "Define pricing model and revenue projections",
                    "Deep-dive into top 3 competitors",
                ],
                quality_score=0.6,
            )
        if self.schema is RefinedOutput:
            return RefinedOutput(
                viability_score=0.78,
                strengths=[
                    "Strong market demand in health-tech",
                    "Scalable SaaS model with freemium potential",
                    "Clear target demographic with high willingness to pay",
                    "TAM estimated at $4.2B globally",
                ],
                weaknesses=[
                    "Competitive market — need clear differentiator",
                    "Content curation requires domain expertise",
                ],
                summary=(
                    "After deeper analysis, the idea demonstrates strong viability. "
                    "A freemium model targeting health-conscious millennials in LATAM "
                    "could capture a meaningful share of the $4.2B market."
                ),
                improvements_made=[
                    "Added market sizing (TAM/SAM/SOM)",
                    "Defined freemium monetization strategy",
                    "Expanded competitive analysis with differentiators",
                ],
            )
        raise ValueError(f"Unsupported schema: {self.schema}")
