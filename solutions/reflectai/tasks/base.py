"""Abstract base task for graph-backed operations."""

from abc import ABC, abstractmethod


class BaseTask(ABC):
    """Base class for tasks that wrap a LangGraph graph."""

    def __init__(self, llm=None):
        self.llm = llm
        self._graph = None

    @property
    def graph(self):
        """Lazy-initialize and cache the compiled graph."""
        if self._graph is None:
            self._graph = self.create_graph()
        return self._graph

    @abstractmethod
    def create_graph(self):
        """Build and return the compiled graph."""
        ...

    def execute(self, request: dict) -> dict:
        """Execute the task with the given request."""
        initial_state = self.build_initial_state(request)
        result = self.graph.invoke(initial_state)
        return self.extract_output(result)

    @abstractmethod
    def build_initial_state(self, request: dict) -> dict:
        """Convert a request dict into the graph's initial state."""
        ...

    def extract_output(self, result: dict) -> dict:
        """Extract the meaningful output from the graph result."""
        return result.get("final_output") or result.get("initial_output") or {}
