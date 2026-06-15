"""State definitions for ReflectAI graphs."""

from typing import Annotated, TypedDict

import operator


class BasicState(TypedDict):
    """State for the basic text-processing graph (Phase 1)."""

    input: str
    processed: str
    word_count: int
    language: str


class ReflectionState(TypedDict):
    """State for the Reflection Pattern graph (Generator → Critic → Refiner)."""

    idea: str
    context: str
    initial_output: dict | None
    critique: dict | None
    final_output: dict | None
    attempts: int
    error: str | None
    messages: Annotated[list[str], operator.add]
