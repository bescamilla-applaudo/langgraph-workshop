from reflectai.schemas import AnalysisOutput, CritiqueOutput, RefinedOutput
from reflectai.state import ReflectionState


def create_generator_node(llm):
    structured_llm = llm.with_structured_output(AnalysisOutput)

    def generator_node(state: ReflectionState) -> dict:
        prompt = (
            f"Analyze the following startup idea and provide a structured assessment.\n\n"
            f"Idea: {state['idea']}\n"
            f"Context: {state['context']}"
        )
        result: AnalysisOutput = structured_llm.invoke(prompt)
        return {
            "initial_output": result.model_dump(),
            "attempts": state["attempts"] + 1,
            "messages": [
                f"[generator] attempt={state['attempts'] + 1} "
                f"viability={result.viability_score:.2f} "
                f"strengths={len(result.strengths)} weaknesses={len(result.weaknesses)}"
            ],
        }

    return generator_node


def create_critic_node(llm):
    structured_llm = llm.with_structured_output(CritiqueOutput)

    def critic_node(state: ReflectionState) -> dict:
        prompt = (
            f"Critique the following analysis of a startup idea.\n\n"
            f"Idea: {state['idea']}\n"
            f"Analysis: {state['initial_output']}"
        )
        result: CritiqueOutput = structured_llm.invoke(prompt)
        return {
            "critique": result.model_dump(),
            "messages": [
                f"[critic] has_issues={result.has_issues} "
                f"quality={result.quality_score:.2f} "
                f"issues={len(result.issues)}"
            ],
        }

    return critic_node


def create_refiner_node(llm):
    structured_llm = llm.with_structured_output(RefinedOutput)

    def refiner_node(state: ReflectionState) -> dict:
        prompt = (
            f"Refine the analysis based on the critique below.\n\n"
            f"Idea: {state['idea']}\n"
            f"Original analysis: {state['initial_output']}\n"
            f"Critique: {state['critique']}"
        )
        result: RefinedOutput = structured_llm.invoke(prompt)
        return {
            "final_output": result.model_dump(),
            "messages": [
                f"[refiner] viability={result.viability_score:.2f} "
                f"improvements={len(result.improvements_made)}"
            ],
        }

    return refiner_node
