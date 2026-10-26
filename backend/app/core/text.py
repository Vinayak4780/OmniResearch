"""Text normalization and tokenization shared by BM25, lexical reranking and citation matching."""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the this to was were will with
    which what how why when where who whom do does did can could should would may might than then there these
    those such into over under about between their them they we our you your not no""".split()
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # de-hyphenate line breaks
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())
    return [t for t in toks if not (drop_stopwords and t in STOPWORDS)]


def match_key(text: str) -> str:
    """Aggressive normalization for excerpt matching (case, whitespace, quotes, dashes, ligatures)."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("­", "")
    t = re.sub(r"[‘’‚‛`´]", "'", t)
    t = re.sub(r"[“”„‟]", '"', t)
    t = re.sub(r"[‐-―−]", "-", t)
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip()


_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?")


def extract_numbers(text: str) -> list[str]:
    """Numbers as they appear, normalized (thousand separators removed, trailing zeros trimmed)."""
    out = []
    for m in _NUMBER.findall(text):
        n = m.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        n = n.lstrip("+")
        if n and n not in ("-",):
            out.append(n)
    return out
