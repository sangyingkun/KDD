from __future__ import annotations

import re


NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
CAMEL_BOUNDARY_PATTERN = re.compile(r"([a-z0-9])([A-Z])")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "list",
    "of",
    "on",
    "or",
    "please",
    "return",
    "show",
    "the",
    "their",
    "to",
    "what",
    "which",
    "with",
}


def normalize_identifier(value: str) -> str:
    lowered = CAMEL_BOUNDARY_PATTERN.sub(r"\1_\2", value).lower()
    normalized = NON_ALNUM_PATTERN.sub("_", lowered).strip("_")
    return normalized


def identifier_aliases(value: str) -> tuple[str, ...]:
    normalized = normalize_identifier(value)
    tokens = [_normalize_token(token) for token in normalized.split("_") if token]
    aliases = {value, normalized, " ".join(tokens), "".join(tokens)}
    return tuple(sorted(alias for alias in aliases if alias))


def question_terms(text: str) -> tuple[str, ...]:
    normalized = normalize_identifier(text)
    tokens = [
        _normalize_token(token)
        for token in normalized.split("_")
        if token and token not in STOP_WORDS
    ]
    return tuple(tokens)


def question_ngrams(text: str, *, max_n: int = 4) -> tuple[str, ...]:
    tokens = list(question_terms(text))
    ngrams: list[str] = []
    for size in range(1, max_n + 1):
        for index in range(0, len(tokens) - size + 1):
            ngrams.append(" ".join(tokens[index : index + size]))
    return tuple(dict.fromkeys(ngrams))


def token_overlap_score(left: str, right: str) -> int:
    left_tokens = set(_normalized_tokens(left))
    right_tokens = set(_normalized_tokens(right))
    if not left_tokens or not right_tokens:
        return 0
    return len(left_tokens & right_tokens)


def _normalized_tokens(value: str) -> tuple[str, ...]:
    normalized = normalize_identifier(value)
    return tuple(_normalize_token(token) for token in TOKEN_PATTERN.findall(normalized) if token)


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token
