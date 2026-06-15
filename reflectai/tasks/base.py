from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(self, llm=None):
        self._llm = llm
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.create_graph()
        return self._graph

    @abstractmethod
    def create_graph(self):
        """Build and return the compiled LangGraph for this task."""

    @abstractmethod
    def build_initial_state(self, request: dict) -> dict:
        """Convert a raw request dict into a valid initial state for the graph."""

    def extract_output(self, result: dict) -> dict:
        return result.get("final_output") or result.get("initial_output") or {}

    def execute(self, request: dict) -> dict:
        initial_state = self.build_initial_state(request)
        result = self.graph.invoke(initial_state)
        return self.extract_output(result)
