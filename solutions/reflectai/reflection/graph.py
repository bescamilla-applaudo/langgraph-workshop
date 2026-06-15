"""Reflection graph factory: Generator → Critic → (Refiner | Passthrough)."""

from langgraph.graph import END, START, StateGraph

from ..state import ReflectionState
from .nodes import create_critic_node, create_generator_node, create_refiner_node


def should_refine(state: ReflectionState) -> str:
    """Decide whether to refine or pass through based on critique."""
    if state.get("critique") and state["critique"].get("has_issues"):
        if state.get("attempts", 0) < 3:
            return "refiner"
    return "passthrough"


def passthrough(state: ReflectionState) -> dict:
    """Copy initial output as final when no refinement is needed."""
    return {"final_output": state["initial_output"], "messages": ["No refinement needed"]}


def create_reflection_graph(llm=None):
    """Build and compile the reflection graph.

    Args:
        llm: LLM instance to use. Defaults to MockLLM if None.
    """
    if llm is None:
        from ..mock_llm import MockLLM

        llm = MockLLM()

    generator = create_generator_node(llm)
    critic = create_critic_node(llm)
    refiner = create_refiner_node(llm)

    builder = StateGraph(ReflectionState)
    builder.add_node("generator", generator)
    builder.add_node("critic", critic)
    builder.add_node("refiner", refiner)
    builder.add_node("passthrough", passthrough)

    builder.add_edge(START, "generator")
    builder.add_edge("generator", "critic")
    builder.add_conditional_edges("critic", should_refine, ["refiner", "passthrough"])
    builder.add_edge("refiner", END)
    builder.add_edge("passthrough", END)

    return builder.compile()
