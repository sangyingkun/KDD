from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_agent_competition.semantic.types import AssetKind, ContextAsset


DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
CSV_SUFFIXES = {".csv", ".tsv"}
JSON_SUFFIXES = {".json"}
DOC_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}
KNOWLEDGE_STEMS = {"knowledge"}


@dataclass(frozen=True, slots=True)
class TaskAssetInventory:
    assets: tuple[ContextAsset, ...]

    def by_kind(self, kind: AssetKind) -> tuple[ContextAsset, ...]:
        return tuple(asset for asset in self.assets if asset.kind == kind)


def build_asset_inventory(context_dir: Path) -> TaskAssetInventory:
    assets: list[ContextAsset] = []
    for path in sorted(item for item in context_dir.rglob("*") if item.is_file()):
        relative_parts = path.relative_to(context_dir).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        relative_path = path.relative_to(context_dir).as_posix()
        assets.append(
            ContextAsset(
                relative_path=relative_path,
                absolute_path=path,
                kind=_classify_asset(path),
                size_bytes=path.stat().st_size,
            )
        )
    return TaskAssetInventory(assets=tuple(assets))


def _classify_asset(path: Path) -> AssetKind:
    suffix = path.suffix.lower()
    stem = path.stem.lower()
    if stem in KNOWLEDGE_STEMS or path.name.lower().startswith("knowledge."):
        return AssetKind.KNOWLEDGE
    if suffix in DB_SUFFIXES:
        return AssetKind.DB
    if suffix in CSV_SUFFIXES:
        return AssetKind.CSV
    if suffix in JSON_SUFFIXES:
        return AssetKind.JSON
    if suffix in DOC_SUFFIXES:
        return AssetKind.DOC
    return AssetKind.OTHER
