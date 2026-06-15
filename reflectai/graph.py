from langgraph.graph import END, START, StateGraph

from reflectai.nodes import analyze_text, clean_text, format_output
from reflectai.state import BasicState


def create_basic_graph():
    graph = StateGraph(BasicState)

    graph.add_node("clean", clean_text)
    graph.add_node("analyze", analyze_text)
    graph.add_node("format", format_output)

    graph.add_edge(START, "clean")
    graph.add_edge("clean", "analyze")
    graph.add_edge("analyze", "format")
    graph.add_edge("format", END)

    return graph.compile()
