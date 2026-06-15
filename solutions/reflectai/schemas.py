"""Pydantic output schemas for structured LLM responses."""

from pydantic import BaseModel, Field


class AnalysisOutput(BaseModel):
    """Initial analysis of a startup idea."""

    viability_score: float = Field(ge=0.0, le=1.0, description="Overall viability score")
    strengths: list[str] = Field(description="Key strengths of the idea")
    weaknesses: list[str] = Field(description="Key weaknesses or risks")
    summary: str = Field(description="Brief analysis summary")


class CritiqueOutput(BaseModel):
    """Critique of an analysis — identifies issues and suggestions."""

    has_issues: bool = Field(description="Whether the analysis has significant issues")
    issues: list[str] = Field(description="Specific issues found in the analysis")
    suggestions: list[str] = Field(description="Suggestions for improvement")
    quality_score: float = Field(ge=0.0, le=1.0, description="Quality score of the analysis")


class RefinedOutput(BaseModel):
    """Refined analysis after incorporating critique feedback."""

    viability_score: float = Field(ge=0.0, le=1.0, description="Updated viability score")
    strengths: list[str] = Field(description="Updated strengths")
    weaknesses: list[str] = Field(description="Updated weaknesses")
    summary: str = Field(description="Refined summary incorporating feedback")
    improvements_made: list[str] = Field(description="What was improved from the original")
