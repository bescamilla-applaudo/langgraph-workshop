from reflectai.reflection.graph import create_reflection_graph
from reflectai.tasks.base import BaseTask


class IdeaAnalyzerTask(BaseTask):
    def create_graph(self):
        return create_reflection_graph(llm=self._llm)

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
