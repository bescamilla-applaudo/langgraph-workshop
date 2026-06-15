"""Basic graph builder (Phase 1 — no LLM)."""

from langgraph.graph import END, START, StateGraph

from .nodes import analyze_text, clean_text, format_output
from .state import BasicState


def create_basic_graph():
    """Create and compile the basic text-processing graph."""
    builder = StateGraph(BasicState)
    builder.add_node("clean", clean_text)
    builder.add_node("analyze", analyze_text)
    builder.add_node("format", format_output)

    builder.add_edge(START, "clean")
    builder.add_edge("clean", "analyze")
    builder.add_edge("analyze", "format")
    builder.add_edge("format", END)

    return builder.compile()
