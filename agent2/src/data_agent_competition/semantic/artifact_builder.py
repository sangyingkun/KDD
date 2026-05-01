from __future__ import annotations

from pathlib import Path

from data_agent_competition.adapters.csv_adapter import inspect_delimited_file
from data_agent_competition.adapters.db_adapter import inspect_database
from data_agent_competition.adapters.doc_adapter import chunk_document, load_document_text
from data_agent_competition.adapters.json_adapter import inspect_json_file
from data_agent_competition.adapters.knowledge_adapter import load_knowledge_facts
from data_agent_competition.artifacts.schema import (
    ArtifactAsset,
    DocumentChunk,
    GraphEdge,
    GraphNode,
    JoinCandidate,
    KnowledgeFact,
    RetrievalDocument,
    SemanticArtifact,
    SourceDescriptor,
    SourceField,
)
from data_agent_competition.semantic.indexing_extractor import extract_business_relations
from data_agent_competition.semantic.indexing_merger import merge_business_relations_into_graph
from data_agent_competition.semantic.graph_builder import build_task_semantic_graph
from data_agent_competition.semantic.llm_support import SemanticRuntime
from data_agent_competition.semantic.normalization import identifier_aliases, normalize_identifier
from data_agent_competition.semantic.types import AssetKind, ContextAsset, TaskBundle

ARTIFACT_VERSION_MARKER = "semantic_artifact_version=graph_rag_llm_indexing_v1"


def build_semantic_artifact(
    bundle: TaskBundle,
    runtime: SemanticRuntime | None = None,
    *,
    enable_llm_indexing: bool = False,
) -> SemanticArtifact:
    assets = [_artifact_asset(asset) for asset in bundle.assets]
    knowledge_facts = _build_knowledge_facts(bundle)
    sources = _build_sources(bundle)
    _inject_knowledge_aliases(knowledge_facts, sources)
    join_candidates = _build_join_candidates(sources)
    doc_chunks = _build_doc_chunks(bundle)
    graph_nodes, graph_edges, retrieval_documents = _build_graph_artifacts(
        sources=sources,
        knowledge_facts=knowledge_facts,
        join_candidates=join_candidates,
        doc_chunks=doc_chunks,
    )
    extraction = (
        extract_business_relations(
            knowledge_facts=knowledge_facts,
            sources=sources,
            runtime=runtime,
        )
        if enable_llm_indexing
        else extract_business_relations(
            knowledge_facts=[],
            sources=[],
            runtime=None,
        )
    )
    graph_nodes, graph_edges = merge_business_relations_into_graph(
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        relations=extraction.relations,
    )
    retrieval_documents = _rebuild_retrieval_documents(graph_nodes, graph_edges)

    return SemanticArtifact(
        task_id=bundle.task_id,
        assets=assets,
        knowledge_facts=knowledge_facts,
        sources=sources,
        join_candidates=join_candidates,
        doc_chunks=doc_chunks,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        retrieval_documents=retrieval_documents,
        notes=[
            ARTIFACT_VERSION_MARKER,
            "Built at runtime by the agent2 semantic layer.",
            "This semantic snapshot is an internal kernel representation, not a public run artifact.",
            "Graph-native retrieval documents derived from the task semantic graph.",
            *extraction.notes,
        ],
    )


def _artifact_asset(asset: ContextAsset) -> ArtifactAsset:
    return ArtifactAsset(
        asset_id=asset.relative_path,
        relative_path=asset.relative_path,
        kind=asset.kind.value,
        format=asset.absolute_path.suffix.lower().lstrip("."),
        size_bytes=asset.size_bytes,
        description="",
    )


def _build_knowledge_facts(bundle: TaskBundle) -> list[KnowledgeFact]:
    knowledge_facts: list[KnowledgeFact] = []
    for asset in bundle.assets_by_kind(AssetKind.KNOWLEDGE):
        current_section = "introduction"
        for index, fact in enumerate(load_knowledge_facts(asset.absolute_path), start=1):
            current_section = _knowledge_section(fact, current_section)
            knowledge_facts.append(
                KnowledgeFact(
                    fact_id=f"{asset.relative_path}::fact_{index}",
                    statement=fact,
                    evidence=[fact],
                    source_path=asset.relative_path,
                    tags=[current_section],
                )
            )
    return knowledge_facts


def _build_sources(bundle: TaskBundle) -> list[SourceDescriptor]:
    sources: list[SourceDescriptor] = []

    for asset in bundle.assets:
        if asset.kind == AssetKind.DB:
            sources.extend(_sources_from_database(asset))
        elif asset.kind == AssetKind.CSV:
            sources.append(_source_from_csv(asset))
        elif asset.kind == AssetKind.JSON:
            source = _source_from_json(asset)
            if source is not None:
                sources.append(source)
    return sources


def _sources_from_database(asset: ContextAsset) -> list[SourceDescriptor]:
    sources: list[SourceDescriptor] = []
    for table in inspect_database(asset.absolute_path):
        fields = [
            SourceField(
                field_name=column.name,
                dtype=column.declared_type,
                aliases=list(identifier_aliases(column.name)),
                nullable=column.nullable,
                semantic_tags=["primary_key"] if column.is_primary_key else [],
                sample_values=[_stringify_value(row.get(column.name)) for row in table.sample_rows[:3]],
            )
            for column in table.columns
        ]
        sources.append(
            SourceDescriptor(
                source_id=f"{asset.relative_path}::{table.name}",
                source_kind=AssetKind.DB.value,
                asset_path=asset.relative_path,
                object_name=table.name,
                description="sqlite table",
                grain_hint=[column.name for column in table.columns if column.is_primary_key],
                fields=fields,
            )
        )
    return sources


def _source_from_csv(asset: ContextAsset) -> SourceDescriptor:
    preview = inspect_delimited_file(asset.absolute_path)
    fields = [
        SourceField(
            field_name=column,
            dtype=preview.dtypes.get(column, ""),
            aliases=list(identifier_aliases(column)),
            sample_values=[row.get(column, "") for row in preview.sample_rows[:3]],
        )
        for column in preview.columns
    ]
    return SourceDescriptor(
        source_id=asset.relative_path,
        source_kind=AssetKind.CSV.value,
        asset_path=asset.relative_path,
        object_name=Path(asset.relative_path).stem,
        description="delimited file",
        grain_hint=[],
        fields=fields,
    )


def _source_from_json(asset: ContextAsset) -> SourceDescriptor | None:
    preview = inspect_json_file(asset.absolute_path)
    if not preview.field_names:
        return None
    fields = [
        SourceField(
            field_name=field_name,
            dtype="json",
            aliases=list(identifier_aliases(field_name)),
            sample_values=[_stringify_value(record.get(field_name)) for record in preview.sample_records[:3]],
        )
        for field_name in preview.field_names
    ]
    return SourceDescriptor(
        source_id=asset.relative_path,
        source_kind=AssetKind.JSON.value,
        asset_path=asset.relative_path,
        object_name=Path(asset.relative_path).stem,
        description=f"json root={preview.root_type}",
        grain_hint=[],
        fields=fields,
    )


def _build_join_candidates(sources: list[SourceDescriptor]) -> list[JoinCandidate]:
    candidates: list[JoinCandidate] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()
    for index, left_source in enumerate(sources):
        for right_source in sources[index + 1 :]:
            for left_field in left_source.fields:
                left_norm = normalize_identifier(left_field.field_name)
                for right_field in right_source.fields:
                    right_norm = normalize_identifier(right_field.field_name)
                    if not _is_join_match(left_norm, right_norm):
                        continue
                    key = (
                        left_source.source_id,
                        left_field.field_name,
                        right_source.source_id,
                        right_field.field_name,
                    )
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    confidence = 0.9 if left_norm == right_norm else 0.5
                    candidates.append(
                        JoinCandidate(
                            left_source_id=left_source.source_id,
                            left_field=left_field.field_name,
                            right_source_id=right_source.source_id,
                            right_field=right_field.field_name,
                            rationale="normalized field-name overlap",
                            confidence=confidence,
                        )
                    )
                    continue
            for left_field in left_source.fields:
                for right_field in right_source.fields:
                    inferred = _infer_reference_join(left_source, left_field.field_name, right_source, right_field.field_name)
                    if inferred is None:
                        continue
                    key = (
                        inferred.left_source_id,
                        inferred.left_field,
                        inferred.right_source_id,
                        inferred.right_field,
                    )
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    candidates.append(inferred)
    return candidates


def _inject_knowledge_aliases(
    knowledge_facts: list[KnowledgeFact],
    sources: list[SourceDescriptor],
) -> None:
    alias_pairs = _knowledge_alias_pairs(knowledge_facts)
    if not alias_pairs:
        return
    for source in sources:
        for field in source.fields:
            field_norm = normalize_identifier(field.field_name)
            extra_aliases = alias_pairs.get(field_norm, set())
            if not extra_aliases:
                continue
            field.aliases = list(dict.fromkeys([*field.aliases, *sorted(extra_aliases)]))


def _knowledge_alias_pairs(knowledge_facts: list[KnowledgeFact]) -> dict[str, set[str]]:
    alias_pairs: dict[str, set[str]] = {}
    for fact in knowledge_facts:
        statement = fact.statement
        lowered = statement.lower()
        if "same attribute" not in lowered and "use '" not in lowered:
            continue
        cleaned = statement.replace("**", "")
        if " vs. " not in cleaned:
            continue
        left, remainder = cleaned.split(" vs. ", maxsplit=1)
        if ":" not in remainder:
            continue
        right, _ = remainder.split(":", maxsplit=1)
        left_norm = normalize_identifier(left)
        right_norm = normalize_identifier(right)
        if not left_norm or not right_norm:
            continue
        alias_pairs.setdefault(left_norm, set()).add(right)
        alias_pairs.setdefault(right_norm, set()).add(left)
    return alias_pairs


def _build_doc_chunks(bundle: TaskBundle) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    doc_assets = bundle.assets_by_kind(AssetKind.DOC) + bundle.assets_by_kind(AssetKind.KNOWLEDGE)
    for asset in doc_assets:
        if asset.absolute_path.suffix.lower() not in {".md", ".txt", ".docx", ".pdf"}:
            continue
        try:
            document = load_document_text(asset.absolute_path)
        except Exception:
            continue
        chunk_sections = _chunk_sections(document.text)
        for index, chunk in enumerate(chunk_document(document.text), start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{asset.relative_path}::chunk_{index}",
                    asset_path=asset.relative_path,
                    text=chunk,
                    section=chunk_sections[index - 1] if index - 1 < len(chunk_sections) else "",
                    keywords=_chunk_keywords(chunk),
                )
            )
    return chunks


def _build_graph_artifacts(
    sources: list[SourceDescriptor],
    knowledge_facts: list[KnowledgeFact],
    join_candidates: list[JoinCandidate],
    doc_chunks: list[DocumentChunk],
) -> tuple[list[GraphNode], list[GraphEdge], list[RetrievalDocument]]:
    return build_task_semantic_graph(
        sources=sources,
        knowledge_facts=knowledge_facts,
        join_candidates=join_candidates,
        doc_chunks=doc_chunks,
    )


def _rebuild_retrieval_documents(
    graph_nodes: list[GraphNode],
    graph_edges: list[GraphEdge],
) -> list[RetrievalDocument]:
    from data_agent_competition.semantic.graph_builder import build_graph_retrieval_documents

    return build_graph_retrieval_documents(nodes=graph_nodes, edges=graph_edges)


def _stringify_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _is_join_match(left_norm: str, right_norm: str) -> bool:
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm and "id" in left_norm.split("_"):
        return True
    if not (left_norm.endswith("_id") and right_norm.endswith("_id")):
        return False
    left_core = tuple(token for token in left_norm.split("_") if token != "id")
    right_core = tuple(token for token in right_norm.split("_") if token != "id")
    return bool(left_core) and left_core == right_core


def _infer_reference_join(
    left_source: SourceDescriptor,
    left_field_name: str,
    right_source: SourceDescriptor,
    right_field_name: str,
) -> JoinCandidate | None:
    left_norm = normalize_identifier(left_field_name)
    right_norm = normalize_identifier(right_field_name)
    right_object = normalize_identifier(right_source.object_name)
    left_object = normalize_identifier(left_source.object_name)
    if left_norm == f"link_to_{right_object}" and right_norm == f"{right_object}_id":
        return JoinCandidate(
            left_source_id=left_source.source_id,
            left_field=left_field_name,
            right_source_id=right_source.source_id,
            right_field=right_field_name,
            rationale="reference link inferred from source object name",
            confidence=0.8,
        )
    if right_norm == f"link_to_{left_object}" and left_norm == f"{left_object}_id":
        return JoinCandidate(
            left_source_id=right_source.source_id,
            left_field=right_field_name,
            right_source_id=left_source.source_id,
            right_field=left_field_name,
            rationale="reference link inferred from source object name",
            confidence=0.8,
        )
    return None


def _knowledge_section(statement: str, current_section: str) -> str:
    normalized = normalize_identifier(statement).replace("_", " ")
    if normalized.startswith("1 introduction"):
        return "introduction"
    if normalized.startswith("2 core entities"):
        return "core_entities"
    if normalized.startswith("3 metric definitions"):
        return "metrics"
    if normalized.startswith("4 constraints"):
        return "constraints"
    if normalized.startswith("5 exemplar use cases"):
        return "use_cases"
    if normalized.startswith("6 ambiguity resolution"):
        return "ambiguity"
    return current_section


def _chunk_sections(text: str) -> list[str]:
    sections: list[str] = []
    current_section = "document"
    for chunk in chunk_document(text):
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if lines:
            current_section = _knowledge_section(lines[0], current_section)
        sections.append(current_section)
    return sections


def _chunk_keywords(chunk: str) -> list[str]:
    tokens = [token for token in normalize_identifier(chunk).split("_") if len(token) > 2]
    keywords = list(dict.fromkeys(tokens))
    return keywords[:12]
