from langgraph.graph import END, START, StateGraph

from reflectai.mock_llm import MockLLM
from reflectai.reflection.nodes import (
    create_critic_node,
    create_generator_node,
    create_refiner_node,
)
from reflectai.state import ReflectionState

MAX_ATTEMPTS = 3


def should_refine(state: ReflectionState) -> str:
    critique = state.get("critique") or {}
    has_issues = critique.get("has_issues", False)
    under_limit = state.get("attempts", 0) < MAX_ATTEMPTS

    if has_issues and under_limit:
        return "refine"
    return "passthrough"


def passthrough(state: ReflectionState) -> dict:
    return {
        "final_output": state["initial_output"],
        "messages": ["[passthrough] no refinement needed — promoting initial output to final"],
    }


def create_reflection_graph(llm=None):
    if llm is None:
        llm = MockLLM()

    graph = StateGraph(ReflectionState)

    graph.add_node("generator", create_generator_node(llm))
    graph.add_node("critic", create_critic_node(llm))
    graph.add_node("refiner", create_refiner_node(llm))
    graph.add_node("passthrough", passthrough)

    graph.add_edge(START, "generator")
    graph.add_edge("generator", "critic")
    graph.add_conditional_edges(
        "critic",
        should_refine,
        {"refine": "refiner", "passthrough": "passthrough"},
    )
    graph.add_edge("refiner", END)
    graph.add_edge("passthrough", END)

    return graph.compile()
