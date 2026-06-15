import pytest

from reflectai.graph import create_basic_graph
from reflectai.nodes import analyze_text, clean_text, format_output
from reflectai.state import BasicState


def test_clean_text(sample_basic_state):
    result = clean_text(sample_basic_state)
    assert "processed" in result
    assert "!" not in result["processed"]
    assert "," not in result["processed"]
    assert "." not in result["processed"]
    assert "  " not in result["processed"]


def test_clean_text_normalizes_whitespace():
    state = BasicState(input="too   many    spaces", processed="", word_count=0, language="")
    result = clean_text(state)
    assert result["processed"] == "too many spaces"


def test_analyze_text_english():
    state = BasicState(
        input="",
        processed="the quick brown fox jumps over the lazy dog and the cat",
        word_count=0,
        language="",
    )
    result = analyze_text(state)
    assert result["language"] == "en"
    assert result["word_count"] == 12


def test_analyze_text_spanish():
    state = BasicState(
        input="",
        processed="el perro corre por el parque con la pelota",
        word_count=0,
        language="",
    )
    result = analyze_text(state)
    assert result["language"] == "es"
    assert result["word_count"] == 9


def test_clean_text_returns_only_processed_key(sample_basic_state):
    result = clean_text(sample_basic_state)
    assert set(result.keys()) == {"processed"}


def test_analyze_text_returns_only_its_keys():
    state = BasicState(input="", processed="the cat sat", word_count=0, language="")
    result = analyze_text(state)
    assert set(result.keys()) == {"word_count", "language"}


def test_full_basic_graph():
    graph = create_basic_graph()
    result = graph.invoke({
        "input": "Hello, world! This is a test sentence.",
        "processed": "",
        "word_count": 0,
        "language": "",
    })
    assert result["word_count"] > 0
    assert result["language"] in {"en", "es"}
    assert str(result["word_count"]) in result["processed"]


def test_graph_structure():
    graph = create_basic_graph()
    node_names = set(graph.get_graph().nodes.keys())
    assert "clean" in node_names
    assert "analyze" in node_names
    assert "format" in node_names
