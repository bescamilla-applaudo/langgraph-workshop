"""Node functions for the basic text-processing graph (Phase 1)."""

import re

from .state import BasicState


def clean_text(state: BasicState) -> dict:
    """Remove special characters and normalize whitespace."""
    text = state["input"]
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"processed": text}


def analyze_text(state: BasicState) -> dict:
    """Count words and detect language using a simple heuristic."""
    text = state["processed"]
    words = text.split()
    word_count = len(words)

    spanish_markers = {"el", "la", "los", "las", "de", "en", "es", "un", "una", "por", "con", "para"}
    text_words = {w.lower() for w in words}
    spanish_overlap = text_words & spanish_markers

    language = "es" if len(spanish_overlap) >= 2 else "en"

    return {"word_count": word_count, "language": language}


def format_output(state: BasicState) -> dict:
    """Build the formatted result string."""
    result = (
        f"Processed: {state['processed']} | "
        f"Words: {state['word_count']} | "
        f"Language: {state['language']}"
    )
    return {"processed": result}
