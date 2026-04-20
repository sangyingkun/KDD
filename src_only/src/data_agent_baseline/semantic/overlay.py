from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from data_agent_baseline.semantic.catalog import EntitySpec, SemanticCatalog


def load_overlay_file(path: Path) -> dict[str, Any]:
    """
    Load a declarative semantic overlay file.

    Overlays are optional: if the file is missing, return an empty payload.
    """

    if not path.exists():
        return {}

    if not path.is_file():
        raise ValueError(f"Overlay path is not a file: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read overlay file: {path}") from exc

    try:
        payload = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in overlay file: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Overlay payload in {path} must be a mapping, got {type(payload).__name__}")

    if payload.get("version", 1) != 1:
        raise ValueError(f"Unsupported overlay version in {path}")

    return payload


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return []


def _merge_unique_strings(base: list[str], extra: Iterable[str]) -> list[str]:
    merged = list(base)
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged


def apply_overlay(catalog: SemanticCatalog, payload: dict[str, Any]) -> SemanticCatalog:
    entity_updates = payload.get("entities", {})
    if not isinstance(entity_updates, dict):
        raise ValueError("Overlay payload.entities must be a mapping of entity names to updates.")

    entities: list[EntitySpec] = []
    for entity in catalog.entities:
        update = entity_updates.get(entity.name)
        if not isinstance(update, dict) or not update:
            entities.append(entity)
            continue

        aliases = _merge_unique_strings(entity.aliases, _coerce_string_list(update.get("aliases")))
        primary_keys = _merge_unique_strings(
            entity.primary_keys, _coerce_string_list(update.get("primary_keys"))
        )

        description = update.get("description")
        if description is None:
            description = entity.description
        confidence = update.get("confidence")
        if confidence is None:
            confidence = entity.confidence

        entities.append(
            replace(
                entity,
                aliases=aliases,
                description=str(description),
                primary_keys=primary_keys,
                confidence=str(confidence),
                provenance="overlay",
            )
        )

    return replace(catalog, entities=entities)
