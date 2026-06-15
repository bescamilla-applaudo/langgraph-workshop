from reflectai.schemas import AnalysisOutput, CritiqueOutput, RefinedOutput

_ANALYSIS_RESPONSE = AnalysisOutput(
    viability_score=0.72,
    strengths=[
        "Addresses a clear pain point in an underserved market segment",
        "Low marginal cost to scale once infrastructure is in place",
        "Recurring revenue model provides predictable cash flow",
        "Founder domain expertise reduces early execution risk",
    ],
    weaknesses=[
        "High customer acquisition cost in a crowded space",
        "Dependency on third-party APIs creates a single point of failure",
        "Long enterprise sales cycles will strain early runway",
        "Network effects take time to materialize — chicken-and-egg problem",
    ],
    summary=(
        "The idea shows solid fundamentals with a defensible niche and a scalable revenue model. "
        "Primary risks are go-to-market execution and third-party dependency. "
        "Worth pursuing with a focused MVP targeting a single customer segment first."
    ),
)

_CRITIQUE_RESPONSE = CritiqueOutput(
    has_issues=True,
    issues=[
        "Viability score does not account for current macroeconomic headwinds affecting SaaS multiples",
        "Weaknesses section omits regulatory risk, which is material for this vertical",
        "Strengths are generic — no mention of specific competitive moat or IP",
        "Summary recommends MVP without defining success criteria",
    ],
    suggestions=[
        "Adjust viability score downward by 0.05–0.10 to reflect tighter funding environment",
        "Add a weakness covering data privacy regulations (GDPR, CCPA) and compliance cost",
        "Specify what makes the technical approach defensible — patents, proprietary data, or switching costs",
        "Define a concrete MVP success metric (e.g. 10 paying customers within 6 months)",
    ],
    quality_score=0.58,
)

_REFINED_RESPONSE = RefinedOutput(
    viability_score=0.67,
    strengths=[
        "Addresses a clear pain point in an underserved market segment",
        "Low marginal cost to scale once infrastructure is in place",
        "Recurring revenue model provides predictable cash flow",
        "Founder domain expertise reduces early execution risk",
        "Proprietary training data creates a compounding moat competitors cannot easily replicate",
    ],
    weaknesses=[
        "High customer acquisition cost in a crowded space",
        "Dependency on third-party APIs creates a single point of failure",
        "Long enterprise sales cycles will strain early runway",
        "Network effects take time to materialize — chicken-and-egg problem",
        "GDPR and CCPA compliance adds 15–20% to initial engineering budget",
    ],
    summary=(
        "Revised analysis incorporates regulatory risk and tightens the viability score to reflect "
        "current market conditions. The idea remains worth pursuing, but success depends on reaching "
        "10 paying customers within 6 months of launch while keeping burn below $50k/month."
    ),
    improvements_made=[
        "Reduced viability score from 0.72 to 0.67 to account for macroeconomic headwinds",
        "Added proprietary data moat as a concrete competitive strength",
        "Added GDPR/CCPA compliance cost as an explicit weakness",
        "Anchored the summary recommendation with measurable success criteria",
    ],
)

_PLAIN_RESPONSE = (
    "This is a mock LLM response. The idea has potential but requires further validation "
    "with real customer discovery before committing significant resources."
)


class MockStructuredLLM:
    def __init__(self, schema):
        self._schema = schema

    def invoke(self, prompt: str):
        if self._schema is AnalysisOutput:
            return _ANALYSIS_RESPONSE
        if self._schema is CritiqueOutput:
            return _CRITIQUE_RESPONSE
        if self._schema is RefinedOutput:
            return _REFINED_RESPONSE
        raise ValueError(f"No mock data registered for schema: {self._schema}")


class MockLLM:
    def invoke(self, prompt: str) -> str:
        return _PLAIN_RESPONSE

    def with_structured_output(self, schema) -> MockStructuredLLM:
        return MockStructuredLLM(schema)
