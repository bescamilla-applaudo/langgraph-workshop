# LangGraph Workshop — AI-First AI Engineering

Hands-on workshop to learn **LangGraph** from scratch using AI as your copilot. You'll build **ReflectAI**, an AI service that analyzes startup ideas using the **Reflection Pattern**.

## What You'll Learn

- StateGraph: typed state graphs with TypedDict
- Nodes, edges, and state flow
- Pydantic schemas for structured LLM output
- Reflection Pattern: Generator → Critic → Refiner
- Dependency injection for LLMs (MockLLM for testing)
- Task system with abstract BaseTask
- Full testing without API keys

## Prerequisites

- Python 3.11+
- VS Code with GitHub Copilot
- Basic Python and terminal knowledge

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Verify
python main.py
# → "ReflectAI v0.1.0"
```

## Structure

| Folder | Purpose |
|---|---|
| `reflectai/` | Your work — build the AI service phase by phase |
| `tests/` | Your work — tests with pytest (no API key needed) |
| `solutions/` | Complete reference (do not modify) |

## How to Use

1. Open [INSTRUCTIONS.md](INSTRUCTIONS.md) — it has ready-made prompts for Copilot Chat
2. Follow the phases in order (1→7)
3. Each phase has validation with real commands
4. If you get stuck, check `solutions/`

## Quick Commands

```bash
source .venv/bin/activate                          # Activate environment
python main.py                                     # Run starter
cd solutions && PYTHONPATH=. python main.py         # Run complete solution
cd solutions && PYTHONPATH=. pytest tests/ -v       # Run tests (17 tests)
```

## Theory Reference

See [LANGGRAPH.md](LANGGRAPH.md) for in-depth concepts.
