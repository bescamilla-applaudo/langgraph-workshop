# ReflectAI — Workshop Progress

## Setup
- [x] Python 3.11+ installed
- [x] Virtual environment created (`.venv`)
- [x] Dependencies installed (`pip install -e ".[dev]"`)
- [x] `python main.py` prints "ReflectAI v0.1.0"

## Phase 1 — The Basic Graph (No LLM)
- [x] `reflectai/state.py` — BasicState and ReflectionState with TypedDict
- [x] `reflectai/nodes.py` — clean_text, analyze_text, format_output
- [x] `reflectai/graph.py` — create_basic_graph() with 3 nodes
- [x] The graph processes text and returns correct word_count + language
- [x] I understand what a reducer is (Annotated[list, operator.add])

## Phase 2 — Schemas and Mock LLM
- [ ] `reflectai/schemas.py` — AnalysisOutput, CritiqueOutput, RefinedOutput
- [ ] `reflectai/mock_llm.py` — MockLLM + MockStructuredLLM
- [ ] MockStructuredLLM returns valid Pydantic instances
- [ ] I understand with_structured_output and why it's key for reliable AI

## Phase 3 — The Reflection Pattern
- [ ] `reflectai/reflection/nodes.py` — Factory functions for Generator, Critic, Refiner
- [ ] `reflectai/reflection/graph.py` — create_reflection_graph() with conditional edges
- [ ] The graph runs Generator → Critic → (Refiner | Passthrough) → END
- [ ] Messages accumulate correctly (reducer works)
- [ ] I understand why Generator → Critic → Refiner improves output quality

## Phase 4 — Task System
- [ ] `reflectai/tasks/base.py` — Abstract BaseTask with lazy graph caching
- [ ] `reflectai/tasks/idea_analyzer.py` — Concrete IdeaAnalyzerTask
- [ ] task.execute() works and returns dict with viability_score, strengths, etc.
- [ ] I understand the encapsulation pattern: Task wraps Graph wraps Nodes

## Phase 5 — Configuration
- [ ] `reflectai/config.py` — Settings with pydantic-settings
- [ ] get_settings() works with and without .env file
- [ ] I understand env_prefix vs alias for OPENROUTER_API_KEY

## Phase 6 — Testing
- [ ] `tests/conftest.py` — Fixtures with MockLLM and sample states
- [ ] `tests/test_basic_graph.py` — 5 tests for the basic graph
- [ ] `tests/test_reflection.py` — 6 tests for the Reflection Pattern
- [ ] `tests/test_tasks.py` — 6 tests for the task system
- [ ] 17 tests pass with `pytest tests/ -v`

## Phase 7 — Entry Point & Demo
- [ ] `main.py` runs IdeaAnalyzerTask and displays result
- [ ] Everything works without API key (MockLLM)
- [ ] (Bonus) Tested with real LLM (OPENROUTER_API_KEY in .env)
