"""Reflection Pattern nodes: Generator → Critic → Refiner."""

from ..schemas import AnalysisOutput, CritiqueOutput, RefinedOutput
from ..state import ReflectionState


def create_generator_node(llm):
    """Create a generator node that produces an initial analysis."""

    def generator(state: ReflectionState) -> dict:
        structured = llm.with_structured_output(AnalysisOutput)
        result = structured.invoke(
            f"Analyze startup idea: {state['idea']}. Context: {state.get('context', '')}"
        )
        return {
            "initial_output": result.model_dump(),
            "attempts": state.get("attempts", 0) + 1,
            "messages": [f"Generated analysis (attempt {state.get('attempts', 0) + 1})"],
        }

    return generator


def create_critic_node(llm):
    """Create a critic node that evaluates the initial analysis."""

    def critic(state: ReflectionState) -> dict:
        structured = llm.with_structured_output(CritiqueOutput)
        result = structured.invoke(
            f"Critique this analysis: {state['initial_output']}. Original idea: {state['idea']}"
        )
        return {
            "critique": result.model_dump(),
            "messages": [f"Critique complete: {'issues found' if result.has_issues else 'looks good'}"],
        }

    return critic


def create_refiner_node(llm):
    """Create a refiner node that improves the analysis based on critique."""

    def refiner(state: ReflectionState) -> dict:
        structured = llm.with_structured_output(RefinedOutput)
        result = structured.invoke(
            f"Refine analysis based on critique. "
            f"Original: {state['initial_output']}. Critique: {state['critique']}"
        )
        return {
            "final_output": result.model_dump(),
            "messages": ["Refinement complete"],
        }

    return refiner
