"""Tests for the Reflection Pattern graph."""

from reflectai.mock_llm import MockLLM
from reflectai.reflection.graph import create_reflection_graph
from reflectai.reflection.nodes import (
    create_critic_node,
    create_generator_node,
    create_refiner_node,
)
from reflectai.schemas import AnalysisOutput, CritiqueOutput, RefinedOutput


def test_generator_node(mock_llm, sample_reflection_state):
    """Generator node creates an initial AnalysisOutput."""
    generator = create_generator_node(mock_llm)
    result = generator(sample_reflection_state)

    assert result["initial_output"] is not None
    assert "viability_score" in result["initial_output"]
    assert "strengths" in result["initial_output"]
    assert result["attempts"] == 1
    assert len(result["messages"]) == 1

    # Validate structure matches AnalysisOutput
    output = AnalysisOutput(**result["initial_output"])
    assert 0.0 <= output.viability_score <= 1.0
    assert len(output.strengths) > 0


def test_critic_node(mock_llm, sample_reflection_state):
    """Critic node produces a critique of the analysis."""
    # Set up state as if generator already ran
    sample_reflection_state["initial_output"] = {
        "viability_score": 0.72,
        "strengths": ["Good market"],
        "weaknesses": ["High competition"],
        "summary": "Promising idea",
    }

    critic = create_critic_node(mock_llm)
    result = critic(sample_reflection_state)

    assert result["critique"] is not None
    assert "has_issues" in result["critique"]
    assert "issues" in result["critique"]

    # Validate structure
    critique = CritiqueOutput(**result["critique"])
    assert isinstance(critique.has_issues, bool)
    assert len(result["messages"]) == 1


def test_refiner_node(mock_llm, sample_reflection_state):
    """Refiner node produces a refined output with improvements."""
    sample_reflection_state["initial_output"] = {
        "viability_score": 0.72,
        "strengths": ["Good market"],
        "weaknesses": ["High competition"],
        "summary": "Promising idea",
    }
    sample_reflection_state["critique"] = {
        "has_issues": True,
        "issues": ["Lacks market sizing"],
        "suggestions": ["Add TAM/SAM/SOM"],
        "quality_score": 0.6,
    }

    refiner = create_refiner_node(mock_llm)
    result = refiner(sample_reflection_state)

    assert result["final_output"] is not None
    assert "improvements_made" in result["final_output"]

    refined = RefinedOutput(**result["final_output"])
    assert len(refined.improvements_made) > 0
    assert refined.viability_score >= 0.0


def test_full_reflection_graph():
    """The complete reflection graph runs Generator → Critic → Refiner."""
    graph = create_reflection_graph()  # uses MockLLM
    result = graph.invoke({
        "idea": "AI-powered meal planning app",
        "context": "Health-conscious millennials",
        "initial_output": None,
        "critique": None,
        "final_output": None,
        "attempts": 0,
        "error": None,
        "messages": [],
    })

    assert result["initial_output"] is not None
    assert result["critique"] is not None
    assert result["final_output"] is not None
    assert result["attempts"] >= 1


def test_reflection_graph_produces_final_output():
    """The reflection graph always produces a final_output."""
    graph = create_reflection_graph()
    result = graph.invoke({
        "idea": "Drone delivery for rural pharmacies",
        "context": "",
        "initial_output": None,
        "critique": None,
        "final_output": None,
        "attempts": 0,
        "error": None,
        "messages": [],
    })

    assert result["final_output"] is not None
    assert "viability_score" in result["final_output"]
    assert "summary" in result["final_output"]


def test_reflection_messages_accumulate():
    """Messages list grows as the graph progresses through nodes."""
    graph = create_reflection_graph()
    result = graph.invoke({
        "idea": "Smart garden watering system",
        "context": "Urban apartments",
        "initial_output": None,
        "critique": None,
        "final_output": None,
        "attempts": 0,
        "error": None,
        "messages": [],
    })

    # Should have at least: generator message, critic message, refiner/passthrough message
    assert len(result["messages"]) >= 3
    assert any("Generated" in m for m in result["messages"])
    assert any("Critique" in m for m in result["messages"])
