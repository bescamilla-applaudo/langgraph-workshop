import operator
from typing import Annotated, TypedDict


class BasicState(TypedDict):
    input: str
    processed: str
    word_count: int
    language: str


class ReflectionState(TypedDict):
    idea: str
    context: str
    initial_output: dict | None
    critique: dict | None
    final_output: dict | None
    attempts: int
    error: str | None
    messages: Annotated[list[str], operator.add]
