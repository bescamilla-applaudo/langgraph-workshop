"""Concrete task: Startup Idea Analyzer using the Reflection Pattern."""

from .base import BaseTask


class IdeaAnalyzerTask(BaseTask):
    """Analyzes a startup idea using Generator → Critic → Refiner."""

    def create_graph(self):
        from ..reflection.graph import create_reflection_graph

        return create_reflection_graph(llm=self.llm)

    def build_initial_state(self, request: dict) -> dict:
        return {
            "idea": request["idea"],
            "context": request.get("context", ""),
            "initial_output": None,
            "critique": None,
            "final_output": None,
            "attempts": 0,
            "error": None,
            "messages": [],
        }
