# LangGraph: From Zero to AI Engineer
> **Context:** A plan designed to understand, read, and extend a production AI service that uses LangGraph with the **Reflection Pattern**, subgraphs, typed state, streaming, and multiple chained workflows.
>
> **Your advantage:** If you understand the concept of "data flowing through steps", you already have the foundation. A LangGraph graph is state flowing through nodes with transition logic.
>
> **Source:** Plan built with Context7 from official LangGraph documentation + direct analysis of real production projects.
>
> **The key insight:** A production AI project doesn't use generic graphs — it uses a specific pattern called **Reflection** where each task has three nodes: `Generator → Critic → Refiner`. Understanding this is understanding 90% of any AI service that uses LangGraph.

---

## Mental Map: Key LangGraph Concepts

```
LangGraph Concept              What It Does
─────────────────────────────────────────────────────────────────
StateGraph                → Defines a typed state graph
TypedDict State           → Dictionary with fixed shape (the data that travels)
add_node(fn)              → Registers a function as a graph node
add_edge(A, B)            → Fixed connection: after A, execute B
add_conditional_edges()   → Branching — chooses next node based on state
compile()                 → Validates and builds the executable graph
invoke()                  → Executes the graph and waits for the result
stream()                  → Executes and emits partial results (streaming)
checkpointer              → State persistence between executions
interrupt()               → Pauses the graph to wait for human input
subgraph                  → Nested graph with its own internal state
```

---

## Phase 1 — The Minimal Graph (The Conceptual Foundation)
**Estimated duration:** 1-2 days
**Goal:** Understand what a directed state graph is and why it's better than a linear chain for AI.

### 1.1 Why graphs and not chains?

```
CHAIN (limited):
  A → B → C → D → End
  
  Problem: Can't go back, can't branch, can't iterate.

GRAPH (LangGraph):
  START → A → B → condition? → C → End
                       ↓
                       D → A  (improvement loop)
  
  Advantage: Can iterate, branch, and resume from any point.
```

### 1.2 Your first graph — no LLM
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Define the state — what "travels" between nodes
class State(TypedDict):
    input: str
    processed: str
    score: float
    attempts: int

# 2. Define the nodes — functions that receive and return PARTIAL state
def process_node(state: State) -> dict:
    """Processes the input and returns only the keys it changes."""
    return {
        "processed": state["input"].upper(),
        "attempts": state.get("attempts", 0) + 1
    }

def score_node(state: State) -> dict:
    """Evaluates the quality of the result."""
    score = len(state["processed"]) / 100
    return {"score": min(score, 1.0)}

def should_retry(state: State) -> str:
    """Routing function — decides which node comes next."""
    if state["score"] < 0.5 and state["attempts"] < 3:
        return "process_node"   # go back and reprocess
    return END                  # finish

# 3. Build the graph
builder = StateGraph(State)
builder.add_node("process_node", process_node)
builder.add_node("score_node", score_node)

# 4. Define the connections
builder.add_edge(START, "process_node")
builder.add_edge("process_node", "score_node")
builder.add_conditional_edges(    # conditional branching
    "score_node",
    should_retry,
    ["process_node", END]         # possible destinations
)

# 5. Compile
graph = builder.compile()

# 6. Execute
result = graph.invoke({"input": "hello world", "processed": "", "score": 0.0, "attempts": 0})
print(result)
# {"input": "hello world", "processed": "HELLO WORLD", "score": 0.11, "attempts": 1}
```

### 1.3 Reducers — how state accumulates
```python
# Without reducer: the new value REPLACES the previous one
class State(TypedDict):
    current_step: str   # "generate" → "critique" → "refine"

# With reducer: the new value ACCUMULATES
from typing import Annotated
import operator

class State(TypedDict):
    # operator.add makes each update APPEND to the list (not replace)
    messages: Annotated[list, operator.add]
    # Result: ["msg1"] + ["msg2"] = ["msg1", "msg2"]
```

### Practice 1 — Text pipeline without LLM
Build a graph with 3 nodes: `clean_text → analyze → format_output`.
- `clean_text`: removes special characters, normalizes whitespace
- `analyze`: counts words, detects language (simple heuristic)
- `format_output`: builds the final response dict

**Success criteria:** The graph returns a correct result with `graph.invoke()` and you can visualize its structure with `graph.get_graph().print_ascii()`.

---

## Phase 2 — LLMs in the Graph
**Estimated duration:** 2-3 days
**Goal:** Integrate a language model as a node and understand how a production service calls an LLM via OpenRouter.

### 2.1 Connecting an LLM to a node
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState

# MessagesState is a TypedDict with messages: Annotated[list, add_messages]
# It's the most common state for conversational apps

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0.7,
    api_key="YOUR_OPENROUTER_API_KEY",
    base_url="https://openrouter.ai/api/v1",
)

# Node that calls the LLM
def llm_node(state: MessagesState) -> dict:
    response = llm.invoke([
        SystemMessage("You are an expert startup assistant."),
        *state["messages"]   # message history
    ])
    return {"messages": [response]}  # ACCUMULATES thanks to the reducer

# The simplest possible flow
graph = (
    StateGraph(MessagesState)
    .add_node("llm", llm_node)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile()
)

result = graph.invoke({"messages": [HumanMessage("What is a value proposition?")]})
print(result["messages"][-1].content)
```

### 2.2 Structured Output — how to get JSON from the LLM
```python
from pydantic import BaseModel, Field

# Expected response schema from the LLM
class ValuePropositionOutput(BaseModel):
    primary_value: str = Field(description="The main value proposition in one sentence")
    differentiators: list[str] = Field(description="3 key differentiators")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Model confidence 0-1")

# Force the LLM to return this exact schema
structured_llm = llm.with_structured_output(ValuePropositionOutput)

def generate_node(state: State) -> dict:
    prompt = f"Generate a value proposition for: {state['idea']}"
    output: ValuePropositionOutput = structured_llm.invoke(prompt)
    return {
        "primary_value": output.primary_value,
        "differentiators": output.differentiators,
        "confidence_score": output.confidence_score
    }
```

### 2.3 Prompt Templates — how to build reusable prompts
```python
from langchain_core.prompts import ChatPromptTemplate

# Reusable template with variables
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Respond in {language}."),
    ("human", "Analyze this business idea: {idea}\n\nContext: {context}")
])

# Compose the prompt with concrete values
prompt = template.format_messages(
    domain="tech startups",
    language="English",
    idea="Healthy food delivery app",
    context="Latin American market, millennial users"
)

response = llm.invoke(prompt)
```

### Practice 2 — Business Idea Generator
Graph with 2 nodes:
1. `generate`: calls the LLM with structured output to generate `IdeaOutput(title, description, target_market)`
2. `format`: formats the result into readable text

---

## Phase 3 — The Reflection Pattern (The Heart of the Architecture)
**Estimated duration:** 3-4 days
**Goal:** Master the self-correcting generation pattern used in production AI services.

### 3.1 What is the Reflection Pattern?

```
WITHOUT Reflection (single pass):
  Prompt → LLM → Result
  Quality: depends on the model's first attempt

WITH Reflection (production):
  Prompt → Generator → initial_output
              ↓
           Critic → critique  (what's wrong? what's missing?)
              ↓
           Refiner → final_output  (improved version)
  Quality: the model self-corrects systematically
```

### 3.2 Complete pattern implementation
```python
from typing import TypedDict, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

# --- SCHEMAS (what each node returns) ---
class GeneratorOutput(BaseModel):
    value_proposition: str
    confidence_score: float

class CriticOutput(BaseModel):
    has_errors: bool
    issues: list[str]
    suggestions: list[str]

class RefinerOutput(BaseModel):
    value_proposition: str
    improvements_made: list[str]

# --- STATE ---
class ReflectionState(TypedDict):
    # Input
    idea: str
    context: dict
    # Outputs from each node (filled progressively)
    initial_output: Optional[dict]
    critique: Optional[dict]
    final_output: Optional[dict]
    # Metadata
    error: Optional[str]

# --- NODES ---
generator_llm = llm.with_structured_output(GeneratorOutput)
critic_llm = llm.with_structured_output(CriticOutput)
refiner_llm = llm.with_structured_output(RefinerOutput)

def generator_node(state: ReflectionState) -> dict:
    """First pass: generates the initial content."""
    output = generator_llm.invoke(
        f"Generate a value proposition for: {state['idea']}"
    )
    return {"initial_output": output.model_dump()}

def critic_node(state: ReflectionState) -> dict:
    """Evaluates the generator's output and finds issues."""
    output = critic_llm.invoke(
        f"""Evaluate this value proposition:
        {state['initial_output']['value_proposition']}
        
        Original idea: {state['idea']}
        Does it have errors? What's missing?"""
    )
    return {"critique": output.model_dump()}

def should_skip_refiner(state: ReflectionState) -> str:
    """If the critic found no errors, skip the refiner."""
    if not state["critique"]["has_errors"]:
        return "skip"   # initial_output is already good
    return "refine"

def refiner_node(state: ReflectionState) -> dict:
    """Improves the output based on the critique."""
    output = refiner_llm.invoke(
        f"""Improve this value proposition based on the critique:
        
        Original: {state['initial_output']['value_proposition']}
        Issues found: {state['critique']['issues']}
        Suggestions: {state['critique']['suggestions']}"""
    )
    return {"final_output": output.model_dump()}

def passthrough_node(state: ReflectionState) -> dict:
    """If there was no critique, final_output = initial_output."""
    return {"final_output": state["initial_output"]}

# --- GRAPH ---
builder = StateGraph(ReflectionState)
builder.add_node("generator", generator_node)
builder.add_node("critic", critic_node)
builder.add_node("refiner", refiner_node)
builder.add_node("passthrough", passthrough_node)

builder.add_edge(START, "generator")
builder.add_edge("generator", "critic")
builder.add_conditional_edges(
    "critic",
    should_skip_refiner,
    {"refine": "refiner", "skip": "passthrough"}
)
builder.add_edge("refiner", END)
builder.add_edge("passthrough", END)

graph = builder.compile()
```

### 3.3 How it's abstracted in production
In production, the Reflection Pattern is abstracted into a reusable module. Each task reuses it:

```python
# Production pattern (simplified):
from src.shared.patterns.reflection.graph import create_reflection_graph
from src.shared.patterns.reflection.nodes import create_reflection_nodes

# Nodes are created with factories (Builder pattern)
generator_node, critic_node, refiner_node = create_reflection_nodes(
    config=config,
    generator_schema=GeneratorOutput,
    critic_schema=CriticOutput,
    refiner_schema=RefinerOutput,
    generator_prompt_builder=build_generate_generator_prompt,
    critic_prompt_builder=build_generate_critic_prompt,
    refiner_prompt_builder=build_generate_refiner_prompt,
)

# The graph is created with the factory
graph = create_reflection_graph(
    state_schema=GenerateState,
    generator_node=generator_node,
    critic_node=critic_node,
    refiner_node=refiner_node,
    enable_conditional_skip=True,
    confidence_threshold=0.8,
)
```

**The production architecture is:**
```
BaseTask (abstract class)
    ↓ inherits
ValuePropositionTask → creates its graph with create_reflection_graph()
ObjectivesTask       → creates its graph with create_reflection_graph()
TargetAudienceTask   → creates its graph with create_reflection_graph()
TechnicalArchTask    → creates its graph with create_reflection_graph()
```

### Practice 3 — Your own Reflection Pattern
Implement the Reflection Pattern to generate **product taglines**:
- `generator`: generates 3 tagline options
- `critic`: evaluates which is most memorable and why (has_errors: true if none are good)
- `refiner`: improves the options based on the critique

---

## Phase 4 — Persistence and Memory (Checkpointer)
**Estimated duration:** 2 days
**Goal:** Understand how the graph "remembers" previous conversations.

### 4.1 InMemorySaver — for development
```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

# The checkpointer saves state snapshots at each node
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# thread_id = conversation identifier (like session_id)
config = {"configurable": {"thread_id": "user-123-session-1"}}

# First invocation
result1 = graph.invoke({"messages": [HumanMessage("My idea is X")]}, config)

# Second invocation — the graph REMEMBERS the first one
result2 = graph.invoke({"messages": [HumanMessage("How can I improve it?")]}, config)
# The graph has access to the full history of "user-123-session-1"
```

### 4.2 Viewing state history
```python
# View all checkpoints for a thread
history = list(graph.get_state_history(config))

for checkpoint in history:
    print(f"Step: {checkpoint.next}")
    print(f"State: {checkpoint.values}")
    print("---")

# View current state
current_state = graph.get_state(config)
print(current_state.values)
```

### 4.3 Human-in-the-loop with interrupt()
```python
from langgraph.types import interrupt, Command

def approval_node(state: State) -> dict:
    # The graph PAUSES here and waits for the user
    user_decision = interrupt({
        "question": "Do you approve this result?",
        "current_output": state["final_output"]
    })
    
    if user_decision == "approve":
        return {"approved": True}
    else:
        return {"approved": False, "feedback": user_decision}

# In the FastAPI endpoint:
@router.post("/approve/{thread_id}")
async def approve(thread_id: str, decision: str):
    config = {"configurable": {"thread_id": thread_id}}
    # Resume the graph with the user's decision
    result = graph.invoke(Command(resume=decision), config)
    return result
```

---

## Phase 5 — Streaming (The Real-Time Experience)
**Estimated duration:** 2 days
**Goal:** Understand how clients receive streaming responses from the AI service.

### 5.1 stream() vs invoke()
```python
# invoke() — waits for the ENTIRE graph to finish
result = graph.invoke({"input": "..."})  # waits 10s, then returns

# stream() — emits events as they happen
for event in graph.stream({"input": "..."}, stream_mode="updates"):
    print(event)
    # {"generator": {"initial_output": {...}}}   ← immediate after generator
    # {"critic": {"critique": {...}}}             ← immediate after critic
    # {"refiner": {"final_output": {...}}}        ← immediate after refiner
```

### 5.2 Streaming from FastAPI (how it's done in production)
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

@router.post("/generate-stream")
async def generate_stream(request: GenerateRequest):
    async def event_generator():
        async for event in graph.astream(
            {"idea": request.idea},
            stream_mode="updates"
        ):
            # Each event is a dict with the node name and its output
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"  # Server-Sent Events
    )
```

### 5.3 Consuming the stream from a client
```python
# Python client — how to consume the AI service stream
import httpx

async def stream_generation(idea: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", "/api/generate-stream",
            json={"idea": idea}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    import json
                    event = json.loads(line[6:])
                    # Process each node event
                    print(event)
```

---

## Phase 6 — Subgraphs and Multi-Task Architecture
**Estimated duration:** 3 days
**Goal:** Understand how multiple independent workflows are organized.

### 6.1 Subgraphs — graphs within graphs
```python
# Validation subgraph
class ValidationState(TypedDict):
    input: str
    is_valid: bool
    errors: list[str]

validation_graph = (
    StateGraph(ValidationState)
    .add_node("validate", validate_node)
    .add_edge(START, "validate")
    .add_edge("validate", END)
    .compile()
)

# Main graph that uses the subgraph as a node
class MainState(TypedDict):
    idea: str
    is_valid: bool
    errors: list[str]
    final_output: str

main_graph = (
    StateGraph(MainState)
    .add_node("validate", validation_graph)  # the subgraph IS a node
    .add_node("generate", generate_node)
    .add_edge(START, "validate")
    .add_conditional_edges("validate", route_after_validation, ["generate", END])
    .add_edge("generate", END)
    .compile()
)
```

### 6.2 The task architecture in production
```
src/tasks/
├── base.py                          ← BaseTask: common interface for all tasks
├── registry.py                      ← Registry of all available tasks
│
├── ideation_and_business_foundation/
│   ├── value_proposition/
│   │   ├── generate/
│   │   │   ├── state.py    ← GenerateState (TypedDict)
│   │   │   ├── graph.py    ← create_reflection_graph()
│   │   │   ├── schemas.py  ← Pydantic output schemas
│   │   │   └── service.py  ← ValuePropositionService(BaseTask)
│   │   ├── confidence/     ← Separate task for calculating confidence
│   │   └── improvement/    ← Separate task for improvement
│   │
│   ├── objectives_and_kpis/
│   ├── target_audience_context/
│   └── entry_and_executive_summary/
│
└── technical_architecture/
    └── (same structure — state, graph, schemas, service)
```

**The rule:** Each user "action" (generate proposition, improve objectives, calculate confidence) is an independent LangGraph graph with its own state, schemas, and reflection logic.

### Practice 6 — Startup content system
Implement 2 tasks following the production pattern:
1. `IdeaValidationTask` — validates if an idea is viable (reflection: generates analysis → critiques → refines)
2. `TaglineGeneratorTask` — generates taglines (Practice 3 encapsulated as a BaseTask)

Both must inherit from `BaseTask` and expose an `execute(request)` method.

---

## Phase 7 — Integration with FastAPI
**Estimated duration:** 2-3 days
**Goal:** Expose a LangGraph graph as a FastAPI endpoint with streaming.

### 7.1 The structure of a production service
```python
# service.py — encapsulates the graph and exposes execute()
from app.tasks.base import BaseTask
from app.tasks.value_proposition.generate.graph import get_graph
from app.tasks.value_proposition.generate.schemas import GenerateRequest, GenerateResponse
from app.tasks.value_proposition.generate.state import GenerateState

class ValuePropositionGenerateService(BaseTask[GenerateState, GenerateRequest, GenerateResponse]):
    
    def create_graph(self):
        return get_graph(field_name="Primary Value")
    
    async def execute(self, request: GenerateRequest) -> GenerateResponse:
        await self.validate_input(request)
        
        initial_state = GenerateState(
            generate={"name": request.field_name, "description": request.description},
            context_user=request.context,
            initial_output={},
            critique={},
            final_output={},
            error=None,
            processing_time=0.0
        )
        
        result = await self.graph.ainvoke(initial_state)
        
        if result.get("error"):
            raise ValueError(result["error"])
            
        return GenerateResponse(**result["final_output"])
```

### 7.2 The FastAPI router
```python
# router.py
from fastapi import APIRouter, Depends
from app.tasks.value_proposition.generate.service import ValuePropositionGenerateService
from app.tasks.value_proposition.generate.schemas import GenerateRequest, GenerateResponse

router = APIRouter(prefix="/value-proposition", tags=["value-proposition"])

# Service singleton (the graph is compiled once with @lru_cache)
_service = ValuePropositionGenerateService()

@router.post("/generate", response_model=GenerateResponse)
async def generate_value_proposition(request: GenerateRequest):
    return await _service.execute(request)
```

---

## Verification Checklist

### Did I complete Phase 1?
- [ ] I can draw on paper the difference between a graph with `add_edge` and one with `add_conditional_edges`
- [ ] I understand why a node returns a PARTIAL dict (only the keys it changes)
- [ ] I know what a reducer is and can use `Annotated[list, operator.add]`

### Did I complete Phase 2?
- [ ] I can connect ChatOpenAI (via OpenRouter) to a node
- [ ] I can use `.with_structured_output(PydanticModel)` to get reliable JSON
- [ ] I understand the difference between `HumanMessage`, `SystemMessage`, and `AIMessage`

### Did I complete Phase 3?
- [ ] I can implement the Reflection Pattern (Generator → Critic → Refiner) from scratch
- [ ] I understand why the Reflection Pattern is abstracted as a reusable module
- [ ] I can read any task's graph.py and understand its flow

### Did I complete Phase 4?
- [ ] I can create a graph with `InMemorySaver` and maintain multi-turn conversations
- [ ] I understand what `thread_id` is and how it maps to a user session
- [ ] I know how to implement `interrupt()` to pause the graph and wait for a human decision

### Did I complete Phase 5?
- [ ] I can expose a graph with `StreamingResponse` from FastAPI
- [ ] I understand the difference between `stream_mode="updates"` and `stream_mode="values"`
- [ ] I can consume the stream from the frontend with `ReadableStream`

### Did I complete Phases 6-7?
- [ ] I can read the complete `src/tasks/` structure and navigate without getting lost
- [ ] I can add a new task following the existing pattern
- [ ] I can code-review a PR for an AI service with real technical judgment

---

## The Final Project: ReflectAI Service
**Estimated duration:** 5-7 days

Build an AI service with the production architecture:

```
langgraph-workshop/final-project/
├── src/
│   ├── shared/
│   │   └── patterns/
│   │       └── reflection/
│   │           ├── nodes.py    # create_reflection_nodes()
│   │           └── graph.py    # create_reflection_graph()
│   └── tasks/
│       ├── base.py
│       └── startup_analyzer/
│           ├── state.py
│           ├── graph.py
│           ├── schemas.py
│           └── service.py
└── main.py   # FastAPI app with the /analyze endpoint
```
