from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, TypedDict


class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int


@dataclass(frozen=True, slots=True)
class QueryParam:
    """
    Adapted from LightRAG's public query contract.

    Source reference:
    https://github.com/HKUDS/LightRAG/blob/main/lightrag/base.py
    """

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int = 12
    chunk_top_k: int = 8
    max_entity_tokens: int = 4000
    max_relation_tokens: int = 4000
    max_total_tokens: int = 12000
    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    history_turns: int = 3
    model_func: Any = None
    user_prompt: str | None = None
    enable_rerank: bool = True
    include_references: bool = False


@dataclass
class StorageNameSpace(ABC):
    namespace: str
    workspace: str
    global_config: dict[str, Any]

    async def initialize(self) -> None:
        return None

    async def finalize(self) -> None:
        return None

    @abstractmethod
    async def index_done_callback(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def drop(self) -> dict[str, str]:
        raise NotImplementedError


@dataclass
class BaseVectorStorage(StorageNameSpace, ABC):
    embedding_func: Any
    cosine_better_than_threshold: float = 0.2
    meta_fields: set[str] = field(default_factory=set)

    @abstractmethod
    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass
class BaseKVStorage(StorageNameSpace, ABC):
    embedding_func: Any = None

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def filter_keys(self, keys: set[str]) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def is_empty(self) -> bool:
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    embedding_func: Any = None

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        raise NotImplementedError

    @abstractmethod
    async def get_all_nodes(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_all_edges(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class QueryResult:
    content: str | None = None
    response_iterator: AsyncIterator[str] | None = None
    raw_data: dict[str, Any] | None = None
    is_streaming: bool = False

    @property
    def reference_list(self) -> list[dict[str, str]]:
        if not self.raw_data:
            return []
        return list(self.raw_data.get("data", {}).get("references", []))

    @property
    def metadata(self) -> dict[str, Any]:
        if not self.raw_data:
            return {}
        return dict(self.raw_data.get("metadata", {}))


@dataclass(frozen=True, slots=True)
class QueryContextResult:
    context: str
    raw_data: dict[str, Any]

    @property
    def reference_list(self) -> list[dict[str, str]]:
        return list(self.raw_data.get("data", {}).get("references", []))
