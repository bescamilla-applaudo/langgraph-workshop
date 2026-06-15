"""Tests for the task system."""

from reflectai.mock_llm import MockLLM
from reflectai.tasks.base import BaseTask
from reflectai.tasks.idea_analyzer import IdeaAnalyzerTask


def test_idea_analyzer_execute():
    """IdeaAnalyzerTask.execute returns a result dict with expected keys."""
    task = IdeaAnalyzerTask()
    result = task.execute({"idea": "AI-powered meal planning app"})

    assert isinstance(result, dict)
    assert "viability_score" in result
    assert "strengths" in result
    assert "weaknesses" in result
    assert "summary" in result


def test_idea_analyzer_with_context():
    """Context is passed through to the graph state."""
    task = IdeaAnalyzerTask()
    result = task.execute({
        "idea": "Smart garden watering system",
        "context": "Urban apartments in Mexico City",
    })

    assert isinstance(result, dict)
    assert result["viability_score"] > 0


def test_base_task_graph_cached():
    """The graph property is lazy-initialized and cached."""
    task = IdeaAnalyzerTask()
    assert task._graph is None

    graph1 = task.graph
    assert task._graph is not None

    graph2 = task.graph
    assert graph1 is graph2


def test_extract_output_fallback():
    """extract_output falls back to initial_output when final_output is missing."""
    task = IdeaAnalyzerTask()

    # final_output present → use it
    result_with_final = task.extract_output({
        "final_output": {"score": 0.8},
        "initial_output": {"score": 0.5},
    })
    assert result_with_final == {"score": 0.8}

    # final_output missing → fall back to initial_output
    result_without_final = task.extract_output({
        "final_output": None,
        "initial_output": {"score": 0.5},
    })
    assert result_without_final == {"score": 0.5}

    # both missing → empty dict
    result_empty = task.extract_output({
        "final_output": None,
        "initial_output": None,
    })
    assert result_empty == {}


def test_idea_analyzer_uses_mock_llm_by_default():
    """IdeaAnalyzerTask defaults to MockLLM when no LLM is provided."""
    task = IdeaAnalyzerTask()
    assert task.llm is None  # MockLLM is injected inside create_reflection_graph

    # Should still execute successfully without any API key
    result = task.execute({"idea": "Test idea"})
    assert isinstance(result, dict)
    assert len(result) > 0


def test_idea_analyzer_with_explicit_mock(mock_llm):
    """IdeaAnalyzerTask works with an explicitly provided MockLLM."""
    task = IdeaAnalyzerTask(llm=mock_llm)
    result = task.execute({"idea": "Blockchain voting platform"})

    assert isinstance(result, dict)
    assert "viability_score" in result
