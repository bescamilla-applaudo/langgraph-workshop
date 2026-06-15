"""Tests for the basic text-processing graph (Phase 1)."""

from reflectai.graph import create_basic_graph
from reflectai.nodes import analyze_text, clean_text


def test_clean_text(sample_basic_state):
    """clean_text removes special characters and normalizes whitespace."""
    result = clean_text(sample_basic_state)
    assert result["processed"] == "Hello World Testing 123"


def test_analyze_text_english():
    """analyze_text counts words and detects English."""
    state = {"processed": "Hello World Testing 123", "input": "", "word_count": 0, "language": ""}
    result = analyze_text(state)
    assert result["word_count"] == 4
    assert result["language"] == "en"


def test_analyze_text_spanish():
    """analyze_text detects Spanish when enough markers are present."""
    state = {"processed": "el mundo de la tecnología es increíble", "input": "", "word_count": 0, "language": ""}
    result = analyze_text(state)
    assert result["language"] == "es"
    assert result["word_count"] == 7


def test_full_basic_graph():
    """The complete basic graph processes input end-to-end."""
    graph = create_basic_graph()
    result = graph.invoke({
        "input": "  Hello, World!  Testing... 123  ",
        "processed": "",
        "word_count": 0,
        "language": "",
    })

    assert result["word_count"] == 4
    assert result["language"] == "en"
    assert "Words: 4" in result["processed"]
    assert "Language: en" in result["processed"]


def test_graph_structure():
    """The basic graph has the expected nodes."""
    graph = create_basic_graph()
    node_names = set(graph.get_graph().nodes.keys())
    # LangGraph adds __start__ and __end__ nodes
    assert "clean" in node_names
    assert "analyze" in node_names
    assert "format" in node_names
