from __future__ import annotations

import json
import re
from pathlib import Path

from data_agent_competition.adapters.doc_adapter import load_document_text


def load_knowledge_facts(path: Path) -> tuple[str, ...]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _flatten_json_facts(payload)
    if suffix in {".md", ".txt", ".docx", ".pdf"}:
        text = load_document_text(path).text
        return _text_to_facts(text)
    return ()


def _text_to_facts(text: str) -> tuple[str, ...]:
    raw_lines = [line.strip(" -\t") for line in text.splitlines()]
    facts = [line.strip() for line in raw_lines if line.strip()]
    return tuple(facts)


def _flatten_json_facts(payload: object, *, prefix: str = "") -> tuple[str, ...]:
    facts: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}{key}".strip()
            facts.extend(_flatten_json_facts(value, prefix=f"{next_prefix}: "))
    elif isinstance(payload, list):
        for item in payload:
            facts.extend(_flatten_json_facts(item, prefix=prefix))
    else:
        scalar = re.sub(r"\s+", " ", str(payload)).strip()
        if scalar:
            facts.append(f"{prefix}{scalar}".strip())
    return tuple(facts)
