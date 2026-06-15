"""Shared fixtures for ReflectAI tests."""

import pytest

from reflectai.mock_llm import MockLLM
from reflectai.state import BasicState, ReflectionState


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def sample_basic_state():
    return BasicState(
        input="  Hello, World!  Testing... 123  ",
        processed="",
        word_count=0,
        language="",
    )


@pytest.fixture
def sample_reflection_state():
    return ReflectionState(
        idea="AI-powered meal planning app",
        context="Health-conscious millennials",
        initial_output=None,
        critique=None,
        final_output=None,
        attempts=0,
        error=None,
        messages=[],
    )
