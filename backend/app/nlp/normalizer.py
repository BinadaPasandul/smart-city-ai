"""Small normalization helpers used only for local text analysis."""

import re
import unicodedata


_PUNCTUATION_OR_SYMBOL = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize matching text while leaving the caller's original text untouched."""
    compatible = unicodedata.normalize("NFKC", text)
    without_punctuation = _PUNCTUATION_OR_SYMBOL.sub(" ", compatible)
    return _WHITESPACE.sub(" ", without_punctuation).strip().casefold()
