# ReflectAI — Startup Idea Analyzer

> **AI-First Workshop:** You don't write code from scratch. You ask Copilot to generate it, run it, understand it, and tweak it.

## What You'll Build

**ReflectAI** — an AI service that analyzes startup ideas using the **Reflection Pattern** from LangGraph: Generator → Critic → Refiner. The model generates an analysis, self-critiques, and produces an improved version.

```
reflectai/
├── state.py               ← TypedDict states (what "travels" between nodes)
├── schemas.py             ← Pydantic output schemas (what the LLM returns)
├── nodes.py               ← Node functions (Phase 1, no LLM)
├── graph.py               ← Basic graph builder
├── mock_llm.py            ← Mock LLM for testing without API key
├── config.py              ← Settings with pydantic-settings
├── reflection/
│   ├── nodes.py           ← Generator, Critic, Refiner (with LLM)
│   └── graph.py           ← create_reflection_graph() factory
└── tasks/
    ├── base.py            ← Abstract BaseTask
    └── idea_analyzer.py   ← Concrete IdeaAnalyzerTask
```

---

## Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. (Optional) Configure API key for real LLM
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY

# 4. Verify installation
python main.py
# → "ReflectAI v0.1.0"
```

---

## Phase 1 — The Basic Graph (No LLM)

**Goal:** Understand StateGraph, nodes, edges, and how state flows.

### Prompt 1.1 — State definition

```
Create the file reflectai/state.py with two TypedDict:

1. BasicState: for a text processing pipeline
   - input (str), processed (str), word_count (int), language (str)

2. ReflectionState: for the Reflection Pattern
   - idea (str), context (str)
   - initial_output, critique, final_output (dict | None)
   - attempts (int), error (str | None)
   - messages: Annotated[list[str], operator.add] (reducer that accumulates)

Explain what a reducer is and why messages uses operator.add.
```

### Prompt 1.2 — Node functions

```
Create reflectai/nodes.py with 3 functions for a text pipeline:

1. clean_text(state: BasicState) → dict: remove special characters, normalize whitespace
2. analyze_text(state: BasicState) → dict: count words, detect language (simple heuristic)
3. format_output(state: BasicState) → dict: build formatted result string

Each function receives the full state but returns ONLY the keys it changes (partial update).
```

### Prompt 1.3 — Graph builder

```
Create reflectai/graph.py with a function create_basic_graph() that:

1. Creates a StateGraph(BasicState)
2. Adds 3 nodes: clean → analyze → format
3. Connects: START → clean → analyze → format → END
4. Returns the compiled graph

Then explain the difference between add_edge (fixed connection) and add_conditional_edges (branching).
```

### Phase 1 Validation

```bash
cd solutions && PYTHONPATH=. python -c "
from reflectai.graph import create_basic_graph
g = create_basic_graph()
r = g.invoke({'input': '  Hello, World!  Testing... 123  ', 'processed': '', 'word_count': 0, 'language': ''})
print(r)
assert r['word_count'] == 4
assert r['language'] == 'en'
print('✅ Phase 1 complete')
"
```

---

## Phase 2 — Schemas and Mock LLM

**Goal:** Define the data structures the LLM must return (structured output) and create a mock for testing.

### Prompt 2.1 — Pydantic schemas

```
Create reflectai/schemas.py with 3 Pydantic BaseModel:

1. AnalysisOutput: viability_score (0-1), strengths (list[str]), weaknesses (list[str]), summary (str)
2. CritiqueOutput: has_issues (bool), issues (list[str]), suggestions (list[str]), quality_score (0-1)
3. RefinedOutput: same as AnalysisOutput + improvements_made (list[str])

Use Field with ge/le to validate ranges and description to document each field.
```

### Prompt 2.2 — Mock LLM

```
Create reflectai/mock_llm.py with two classes:

1. MockLLM: has invoke(prompt) that returns strings, and with_structured_output(schema) that returns a MockStructuredLLM
2. MockStructuredLLM: has invoke(prompt) that returns Pydantic schema instances with deterministic startup analysis data

The mock should return realistic data: scores, strengths, weaknesses, etc.
This allows testing everything without an API key.
```

### Phase 2 Validation

```bash
cd solutions && PYTHONPATH=. python -c "
from reflectai.mock_llm import MockLLM
from reflectai.schemas import AnalysisOutput
llm = MockLLM()
structured = llm.with_structured_output(AnalysisOutput)
result = structured.invoke('test')
print(f'Score: {result.viability_score}')
print(f'Strengths: {result.strengths}')
assert 0 <= result.viability_score <= 1
print('✅ Phase 2 complete')
"
```

---

## Phase 3 — The Reflection Pattern

> **Switch model → Claude Sonnet 4.6 or GPT-5.4**
> The Reflection Pattern involves graphs with cycles, conditional routing, and LLM dependency injection.
> Gemini 3 Flash tends to confuse node structure or generate routing that never terminates.

**Goal:** Implement Generator → Critic → Refiner — the heart of the architecture.

### Prompt 3.1 — Reflection nodes

```
Create reflectai/reflection/nodes.py with 3 factory functions:

1. create_generator_node(llm): returns function that analyzes the idea with structured output (AnalysisOutput)
2. create_critic_node(llm): returns function that critiques the analysis (CritiqueOutput)
3. create_refiner_node(llm): returns function that refines based on the critique (RefinedOutput)

Use dependency injection: the LLM is passed as an argument, not instantiated inside.
Each node updates the ReflectionState with its output + a message in messages[].
```

### Prompt 3.2 — Reflection graph factory

```
Create reflectai/reflection/graph.py with:

1. should_refine(state): routing function that decides whether to refine (when there are issues and attempts < 3)
2. passthrough(state): copies initial_output to final_output when refinement isn't needed
3. create_reflection_graph(llm=None): factory that builds the complete graph

The graph: START → generator → critic → (refiner | passthrough) → END
If llm is None, use MockLLM by default.
```

### Phase 3 Validation

```bash
cd solutions && PYTHONPATH=. python -c "
from reflectai.reflection.graph import create_reflection_graph
g = create_reflection_graph()
r = g.invoke({
    'idea': 'AI-powered meal planning app',
    'context': 'Health-conscious millennials',
    'initial_output': None, 'critique': None, 'final_output': None,
    'attempts': 0, 'error': None, 'messages': []
})
print(f'Messages: {r[\"messages\"]}')
print(f'Final score: {r[\"final_output\"][\"viability_score\"]}')
assert r['final_output'] is not None
assert len(r['messages']) >= 3
print('✅ Phase 3 complete')
"
```

---

## Phase 4 — Task System

**Goal:** Encapsulate graphs into reusable tasks with a common interface.

### Prompt 4.1 — Base task

```
Create reflectai/tasks/base.py with an abstract class BaseTask:

- __init__(self, llm=None): stores the LLM
- graph (property): lazy-init with cache (creates the graph only the first time)
- create_graph() (abstract): each task defines its own graph
- execute(request: dict) → dict: invokes the graph and extracts the output
- build_initial_state(request) (abstract): converts request to initial state
- extract_output(result): returns final_output, fallback to initial_output, or {}
```

### Prompt 4.2 — Idea analyzer task

```
Create reflectai/tasks/idea_analyzer.py with IdeaAnalyzerTask(BaseTask):

- create_graph(): uses create_reflection_graph(llm=self.llm)
- build_initial_state(request): maps {idea, context} to the full ReflectionState

The task is used like this:
  task = IdeaAnalyzerTask()
  result = task.execute({"idea": "...", "context": "..."})
```

### Phase 4 Validation

```bash
cd solutions && PYTHONPATH=. python -c "
from reflectai.tasks.idea_analyzer import IdeaAnalyzerTask
task = IdeaAnalyzerTask()
result = task.execute({'idea': 'Smart garden watering', 'context': 'Urban Mexico'})
for k, v in result.items():
    print(f'  {k}: {v}')
assert 'viability_score' in result
assert 'improvements_made' in result
print('✅ Phase 4 complete')
"
```

---

## Phase 5 — Configuration

**Goal:** Centralize configuration with pydantic-settings.

### Prompt 5.1 — Settings

```
Create reflectai/config.py with:

- Settings(BaseSettings): openrouter_api_key, llm_model, llm_temperature, debug
- model_config with env_file=".env"
- openrouter_api_key uses Field(alias="OPENROUTER_API_KEY") because it has no prefix
- The rest use env_prefix="REFLECTAI_"
- get_settings() with @lru_cache for singleton

Explain how pydantic-settings resolves env_prefix vs alias.
```

### Phase 5 Validation

```bash
cd solutions && PYTHONPATH=. python -c "
from reflectai.config import get_settings
s = get_settings()
print(f'Model: {s.llm_model}')
print(f'Temperature: {s.llm_temperature}')
print(f'Debug: {s.debug}')
print('✅ Phase 5 complete')
"
```

---

## Phase 6 — Testing

> **You can switch back to Gemini 3 Flash**
> Tests use MockLLM and don't require reasoning about the full graph. Flash generates fixtures and pytest assertions well.

**Goal:** Write tests that validate the entire architecture WITHOUT an API key.

### Prompt 6.1 — Fixtures

```
Create tests/conftest.py with pytest fixtures:

- mock_llm: returns MockLLM()
- sample_basic_state: BasicState with test input
- sample_reflection_state: ReflectionState with a startup idea
```

### Prompt 6.2 — Basic graph tests

```
Create tests/test_basic_graph.py with tests for:

- test_clean_text: verifies special characters are cleaned
- test_analyze_text_english: detects English and counts words
- test_analyze_text_spanish: detects Spanish
- test_full_basic_graph: invokes the complete graph end-to-end
- test_graph_structure: verifies nodes exist
```

### Prompt 6.3 — Reflection Pattern tests

```
Create tests/test_reflection.py with tests for:

- test_generator_node: generates valid AnalysisOutput
- test_critic_node: produces CritiqueOutput
- test_refiner_node: produces RefinedOutput with improvements
- test_full_reflection_graph: runs the complete flow
- test_reflection_graph_produces_final_output: always has final_output
- test_reflection_messages_accumulate: messages grow with each node
```

### Prompt 6.4 — Task system tests

```
Create tests/test_tasks.py with tests for:

- test_idea_analyzer_execute: execute returns dict with expected keys
- test_idea_analyzer_with_context: context is passed to state
- test_base_task_graph_cached: graph is lazy-initialized and cached
- test_extract_output_fallback: fallback to initial_output if no final_output
```

### Phase 6 Validation

```bash
cd solutions && PYTHONPATH=. pytest tests/ -v
# → 17 passed
```

---

## Phase 7 — Entry Point & Demo

**Goal:** Connect everything in an executable main.py.

### Prompt 7.1 — Main script

```
Create a main.py that:

1. Imports IdeaAnalyzerTask
2. Runs it with an example idea
3. Prints the formatted result (viability_score, strengths, weaknesses, summary, improvements)

Must work without an API key (uses MockLLM by default).
```

### Phase 7 Validation

```bash
# Starter
python main.py
# → "ReflectAI v0.1.0"

# Complete solution
cd solutions && PYTHONPATH=. python main.py
# → Analysis Result with score, strengths, weaknesses, summary, improvements

# Full tests
cd solutions && PYTHONPATH=. pytest tests/ -v
# → 17 passed
```

---

## Quick Commands

| Action | Command |
|---|---|
| Activate environment | `source .venv/bin/activate` |
| Run starter | `python main.py` |
| Run solution | `cd solutions && PYTHONPATH=. python main.py` |
| Run tests | `cd solutions && PYTHONPATH=. pytest tests/ -v` |
| Specific test | `cd solutions && PYTHONPATH=. pytest tests/test_reflection.py -v` |
| View graph | `cd solutions && PYTHONPATH=. python -c "from reflectai.graph import create_basic_graph; create_basic_graph().get_graph().print_ascii()"` |

---

## Key LangGraph Concepts

| Concept | What It Does | Example |
|---|---|---|
| `StateGraph` | Typed state graph | Defines nodes and connections |
| `TypedDict` | Defines state shape | Dictionary with fixed types |
| `add_node(fn)` | Registers a node | Function that transforms state |
| `add_edge(A, B)` | Fixed connection A → B | Sequential flow |
| `add_conditional_edges` | Condition-based branching | Decision based on state |
| `compile()` | Builds the executable graph | Validates and prepares the graph |
| `invoke()` | Runs and waits for result | Executes the full pipeline |
| Reducer (`Annotated[list, add]`) | Accumulates values instead of replacing | Message history |
| Reflection Pattern | Generator → Critic → Refiner | Self-improvement loop |
| `with_structured_output` | Forces the LLM to return Pydantic | Typed and validated output |

---

## Tips for Copilot (Gemini 3 Flash)

1. **Be specific with types:** "Use TypedDict, not dataclass" — Gemini respects explicit constraints.
2. **Ask for dependency injection:** "The LLM is passed as an argument, not instantiated inside the node."
3. **Ask for explanations after the code:** "Generate X, then explain why you used Y."
4. **If an import fails:** Verify you're in `solutions/` with `PYTHONPATH=.`
5. **If a test fails:** Ask Copilot "this test fails with [error]. Diagnose the problem."
