import pytest

from reflectai.reflection.graph import create_reflection_graph
from reflectai.reflection.nodes import (
    create_critic_node,
    create_generator_node,
    create_refiner_node,
)
from reflectai.schemas import AnalysisOutput, CritiqueOutput, RefinedOutput


def test_generator_node(mock_llm, sample_reflection_state):
    generator = create_generator_node(mock_llm)
    result = generator(sample_reflection_state)

    assert "initial_output" in result
    assert "attempts" in result
    assert "messages" in result

    output = AnalysisOutput(**result["initial_output"])
    assert 0.0 <= output.viability_score <= 1.0
    assert len(output.strengths) > 0
    assert len(output.weaknesses) > 0
    assert output.summary

    assert result["attempts"] == 1
    assert len(result["messages"]) == 1
    assert "[generator]" in result["messages"][0]


def test_critic_node(mock_llm, sample_reflection_state):
    sample_reflection_state["initial_output"] = AnalysisOutput(
        viability_score=0.72,
        strengths=["Strong market demand"],
        weaknesses=["High competition"],
        summary="Decent idea with execution risk.",
    ).model_dump()

    critic = create_critic_node(mock_llm)
    result = critic(sample_reflection_state)

    assert "critique" in result
    assert "messages" in result

    critique = CritiqueOutput(**result["critique"])
    assert isinstance(critique.has_issues, bool)
    assert 0.0 <= critique.quality_score <= 1.0
    assert isinstance(critique.issues, list)
    assert isinstance(critique.suggestions, list)

    assert len(result["messages"]) == 1
    assert "[critic]" in result["messages"][0]


def test_refiner_node(mock_llm, sample_reflection_state):
    sample_reflection_state["initial_output"] = AnalysisOutput(
        viability_score=0.72,
        strengths=["Strong market demand"],
        weaknesses=["High competition"],
        summary="Decent idea with execution risk.",
    ).model_dump()
    sample_reflection_state["critique"] = CritiqueOutput(
        has_issues=True,
        issues=["Missing regulatory analysis"],
        suggestions=["Add GDPR compliance cost estimate"],
        quality_score=0.58,
    ).model_dump()

    refiner = create_refiner_node(mock_llm)
    result = refiner(sample_reflection_state)

    assert "final_output" in result
    assert "messages" in result

    refined = RefinedOutput(**result["final_output"])
    assert 0.0 <= refined.viability_score <= 1.0
    assert len(refined.improvements_made) > 0

    assert len(result["messages"]) == 1
    assert "[refiner]" in result["messages"][0]


def test_full_reflection_graph(mock_llm, sample_reflection_state):
    graph = create_reflection_graph(llm=mock_llm)
    final_state = graph.invoke(sample_reflection_state)

    assert final_state["initial_output"] is not None
    assert final_state["critique"] is not None
    assert final_state["final_output"] is not None
    assert final_state["attempts"] >= 1


def test_reflection_graph_produces_final_output(mock_llm, sample_reflection_state):
    graph = create_reflection_graph(llm=mock_llm)
    final_state = graph.invoke(sample_reflection_state)

    assert final_state["final_output"] is not None
    assert "viability_score" in final_state["final_output"]
    assert "summary" in final_state["final_output"]


def test_reflection_messages_accumulate(mock_llm, sample_reflection_state):
    graph = create_reflection_graph(llm=mock_llm)
    final_state = graph.invoke(sample_reflection_state)

    messages = final_state["messages"]
    assert len(messages) >= 3

    sources = {msg.split("]")[0].lstrip("[") for msg in messages}
    assert "generator" in sources
    assert "critic" in sources
    # refiner or passthrough ran — at least one must be present
    assert sources & {"refiner", "passthrough"}
