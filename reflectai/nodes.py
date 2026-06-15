import re

from reflectai.state import BasicState

_SPANISH_MARKERS = {"el", "la", "los", "las", "de", "en", "es", "un", "una", "por", "con", "para"}


def clean_text(state: BasicState) -> dict:
    text = state["input"]
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"processed": text}


def analyze_text(state: BasicState) -> dict:
    words = state["processed"].split()
    word_set = {w.lower() for w in words}
    language = "es" if len(word_set & _SPANISH_MARKERS) >= 2 else "en"
    return {"word_count": len(words), "language": language}


def format_output(state: BasicState) -> dict:
    result = (
        f"Processed: {state['processed']} | "
        f"Words: {state['word_count']} | "
        f"Language: {state['language']}"
    )
    return {"processed": result}
