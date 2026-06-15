from pydantic import BaseModel, Field


class AnalysisOutput(BaseModel):
    viability_score: float = Field(
        ge=0.0, le=1.0, description="Overall viability of the idea, from 0 (not viable) to 1 (highly viable)"
    )
    strengths: list[str] = Field(description="Key strengths or advantages of the idea")
    weaknesses: list[str] = Field(description="Key weaknesses or risks of the idea")
    summary: str = Field(description="Concise summary of the analysis")


class CritiqueOutput(BaseModel):
    has_issues: bool = Field(description="Whether the analysis has significant issues that warrant a revision")
    issues: list[str] = Field(description="Specific problems found in the analysis")
    suggestions: list[str] = Field(description="Actionable suggestions to address each issue")
    quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality of the analysis being critiqued, from 0 (poor) to 1 (excellent)"
    )


class RefinedOutput(BaseModel):
    viability_score: float = Field(
        ge=0.0, le=1.0, description="Revised viability score after incorporating critique feedback"
    )
    strengths: list[str] = Field(description="Key strengths or advantages of the idea")
    weaknesses: list[str] = Field(description="Key weaknesses or risks of the idea")
    summary: str = Field(description="Concise summary of the refined analysis")
    improvements_made: list[str] = Field(description="Specific changes made in response to the critique")
