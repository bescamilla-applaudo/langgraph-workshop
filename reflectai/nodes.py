import re

from reflectai.state import BasicState

# Common words used as a heuristic to detect language
_SPANISH_MARKERS = {"el", "la", "los", "las", "de", "en", "que", "es", "un", "una", "y", "con"}
_FRENCH_MARKERS  = {"le", "la", "les", "de", "en", "que", "est", "un", "une", "et", "avec"}
_ENGLISH_MARKERS = {"the", "is", "are", "in", "of", "and", "a", "an", "to", "with", "that"}


def clean_text(state: BasicState) -> dict:
    text = state["input"]
    text = re.sub(r"[^a-zA-ZáéíóúàèìòùâêîôûäëïöüñçÑÇ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"processed": text}


def analyze_text(state: BasicState) -> dict:
    words = state["processed"].lower().split()
    word_set = set(words)

    scores = {
        "english": len(word_set & _ENGLISH_MARKERS),
        "spanish": len(word_set & _SPANISH_MARKERS),
        "french":  len(word_set & _FRENCH_MARKERS),
    }
    language = max(scores, key=scores.get)
    if all(v == 0 for v in scores.values()):
        language = "unknown"

    return {"word_count": len(words), "language": language}


def format_output(state: BasicState) -> dict:
    result = (
        f"[{state['language'].upper()} | {state['word_count']} words]\n"
        f"{state['processed']}"
    )
    return {"processed": result}
