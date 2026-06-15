import pytest

from reflectai.tasks.base import BaseTask
from reflectai.tasks.idea_analyzer import IdeaAnalyzerTask


# --- helpers ---

class ConcreteTask(BaseTask):
    """Minimal BaseTask subclass for testing base-class behavior in isolation."""

    def create_graph(self):
        return _FakeGraph()

    def build_initial_state(self, request: dict) -> dict:
        return request


class _FakeGraph:
    def __init__(self):
        self.invoke_count = 0

    def invoke(self, state: dict) -> dict:
        self.invoke_count += 1
        return state


# --- IdeaAnalyzerTask tests ---

def test_idea_analyzer_execute(mock_llm):
    task = IdeaAnalyzerTask(llm=mock_llm)
    result = task.execute({"idea": "On-demand dog walking app", "context": "Consumer mobile, urban markets"})

    assert isinstance(result, dict)
    assert "viability_score" in result
    assert "strengths" in result
    assert "weaknesses" in result
    assert "summary" in result


def test_idea_analyzer_with_context(mock_llm):
    task = IdeaAnalyzerTask(llm=mock_llm)
    state = task.build_initial_state({
        "idea": "AI legal assistant",
        "context": "B2B SaaS for small law firms",
    })

    assert state["idea"] == "AI legal assistant"
    assert state["context"] == "B2B SaaS for small law firms"


def test_idea_analyzer_context_defaults_to_empty(mock_llm):
    task = IdeaAnalyzerTask(llm=mock_llm)
    state = task.build_initial_state({"idea": "No context provided"})
    assert state["context"] == ""


# --- BaseTask graph caching tests ---

def test_base_task_graph_cached():
    task = ConcreteTask()

    first = task.graph
    second = task.graph

    assert first is second


def test_base_task_graph_lazy():
    task = ConcreteTask()
    assert task._graph is None  # not built yet
    _ = task.graph
    assert task._graph is not None  # built on first access


def test_base_task_graph_calls_create_graph_once():
    call_count = 0
    original_create = ConcreteTask.create_graph

    class CountingTask(ConcreteTask):
        def create_graph(self):
            nonlocal call_count
            call_count += 1
            return super().create_graph()

    task = CountingTask()
    _ = task.graph
    _ = task.graph
    _ = task.graph

    assert call_count == 1


# --- extract_output fallback tests ---

def test_extract_output_returns_final_output():
    task = ConcreteTask()
    result = task.extract_output({"final_output": {"score": 0.9}, "initial_output": {"score": 0.5}})
    assert result == {"score": 0.9}


def test_extract_output_fallback_to_initial_output():
    task = ConcreteTask()
    result = task.extract_output({"final_output": None, "initial_output": {"score": 0.5}})
    assert result == {"score": 0.5}


def test_extract_output_fallback_to_empty_dict():
    task = ConcreteTask()
    result = task.extract_output({"final_output": None, "initial_output": None})
    assert result == {}
