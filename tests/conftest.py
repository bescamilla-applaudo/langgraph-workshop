import pytest

from reflectai.mock_llm import MockLLM
from reflectai.state import BasicState, ReflectionState


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def sample_basic_state() -> BasicState:
    return BasicState(
        input="Hello, world! This is a test sentence with some punctuation...",
        processed="",
        word_count=0,
        language="",
    )


@pytest.fixture
def sample_reflection_state() -> ReflectionState:
    return ReflectionState(
        idea="An AI-powered code review tool that detects security vulnerabilities",
        context="B2B SaaS targeting mid-size engineering teams of 20–200 developers",
        initial_output=None,
        critique=None,
        final_output=None,
        attempts=0,
        error=None,
        messages=[],
    )
