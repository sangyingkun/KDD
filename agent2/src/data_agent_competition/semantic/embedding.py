from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import logging
import math
from pathlib import Path

from data_agent_competition.artifacts.schema import RetrievalDocument
from data_agent_competition.runtime.config import AgentConfig
from data_agent_competition.semantic.normalization import normalize_identifier

logger = logging.getLogger("agent2.semantic.embedding")

_HASH_DIGEST_SIZE = 8


@dataclass(frozen=True, slots=True)
class DenseScore:
    doc_id: str
    score: float


class EmbeddingProvider:
    def score_documents(
        self,
        *,
        query: str,
        documents: tuple[RetrievalDocument, ...],
        limit: int,
    ) -> tuple[DenseScore, ...]:
        raise NotImplementedError


class TokenSpaceEmbeddingProvider(EmbeddingProvider):
    def score_documents(
        self,
        *,
        query: str,
        documents: tuple[RetrievalDocument, ...],
        limit: int,
    ) -> tuple[DenseScore, ...]:
        query_vector = _token_vector(query)
        scores: list[DenseScore] = []
        for document in documents:
            doc_vector = _token_vector(document.text)
            score = _cosine(query_vector, doc_vector)
            if score <= 0:
                continue
            scores.append(DenseScore(doc_id=document.doc_id, score=score))
        scores.sort(key=lambda item: (-item.score, item.doc_id))
        return tuple(scores[:limit])


@dataclass(frozen=True, slots=True)
class HashEmbeddingProvider(EmbeddingProvider):
    dimension: int = 1024

    def __post_init__(self) -> None:
        if self.dimension < _HASH_DIGEST_SIZE:
            raise ValueError(
                f"HashEmbeddingProvider dimension must be >= {_HASH_DIGEST_SIZE}, got {self.dimension}."
            )

    def score_documents(
        self,
        *,
        query: str,
        documents: tuple[RetrievalDocument, ...],
        limit: int,
    ) -> tuple[DenseScore, ...]:
        query_vector = _hashed_vector(query, self.dimension)
        scores: list[DenseScore] = []
        for document in documents:
            doc_vector = _hashed_vector(document.text, self.dimension)
            score = _dense_cosine(query_vector, doc_vector)
            if score <= 0:
                continue
            scores.append(DenseScore(doc_id=document.doc_id, score=score))
        scores.sort(key=lambda item: (-item.score, item.doc_id))
        return tuple(scores[:limit])


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str,
        batch_size: int,
        local_files_only: bool,
        query_prefix: str,
        document_prefix: str,
        fallback_provider: EmbeddingProvider,
    ) -> None:
        self._model_name_or_path = model_name_or_path.strip()
        self._device = device.strip() or "cpu"
        self._batch_size = max(int(batch_size), 1)
        self._local_files_only = bool(local_files_only)
        self._query_prefix = query_prefix.strip()
        self._document_prefix = document_prefix.strip()
        self._fallback_provider = fallback_provider
        self._cache: dict[str, tuple[float, ...]] = {}
        self._model: object | None = None
        self._disabled_reason: str | None = None
        if not self._model_name_or_path:
            self._disabled_reason = "missing_embedding_model_name"

    @property
    def enabled(self) -> bool:
        return self._disabled_reason is None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def score_documents(
        self,
        *,
        query: str,
        documents: tuple[RetrievalDocument, ...],
        limit: int,
    ) -> tuple[DenseScore, ...]:
        if not self.enabled:
            return self._fallback_provider.score_documents(query=query, documents=documents, limit=limit)
        try:
            query_vector = self._embedding_for_text(query, prefix=self._query_prefix)
            document_vectors = self._embeddings_for_documents(documents)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local embedding failed, falling back to deterministic embeddings: %s", exc)
            return self._fallback_provider.score_documents(query=query, documents=documents, limit=limit)
        scores: list[DenseScore] = []
        for document in documents:
            vector = document_vectors.get(document.doc_id)
            if vector is None:
                continue
            score = _dense_cosine(query_vector, vector)
            if score <= 0:
                continue
            scores.append(DenseScore(doc_id=document.doc_id, score=score))
        scores.sort(key=lambda item: (-item.score, item.doc_id))
        return tuple(scores[:limit])

    def probe(self) -> None:
        vector = self._embedding_for_text("embedding provider health check", prefix=self._query_prefix)
        if not vector:
            raise RuntimeError("Local embedding provider returned an empty probe vector.")

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("sentence-transformers is not available for local embeddings.") from exc

        model_reference = self._model_name_or_path
        if Path(model_reference).exists():
            model_reference = str(Path(model_reference))
        self._model = SentenceTransformer(
            model_reference,
            device=self._device,
            trust_remote_code=True,
            local_files_only=self._local_files_only,
        )
        return self._model

    def _embeddings_for_documents(
        self,
        documents: tuple[RetrievalDocument, ...],
    ) -> dict[str, tuple[float, ...]]:
        vectors: dict[str, tuple[float, ...]] = {}
        pending: list[tuple[str, str]] = []
        for document in documents:
            cache_key = _cache_key(f"{self._document_prefix}::{document.text}")
            cached = self._cache.get(cache_key)
            if cached is not None:
                vectors[document.doc_id] = cached
                continue
            pending.append((document.doc_id, document.text))
        if not pending:
            return vectors

        encoded = self._encode([text for _, text in pending], prefix=self._document_prefix)
        for (doc_id, text), vector in zip(pending, encoded, strict=False):
            normalized = tuple(float(value) for value in vector)
            self._cache[_cache_key(f"{self._document_prefix}::{text}")] = normalized
            vectors[doc_id] = normalized
        return vectors

    def _embedding_for_text(self, text: str, *, prefix: str) -> tuple[float, ...]:
        cache_key = _cache_key(f"{prefix}::{text}")
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        encoded = self._encode([text], prefix=prefix)
        if len(encoded) != 1:
            raise RuntimeError("Local embedding provider returned an unexpected query batch size.")
        vector = tuple(float(value) for value in encoded[0])
        self._cache[cache_key] = vector
        return vector

    def _encode(self, texts: list[str], *, prefix: str) -> list[tuple[float, ...]]:
        if not texts:
            return []
        model = self._load_model()
        inputs = _normalize_text_input(texts, prefix=prefix)
        matrix = model.encode(
            inputs,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if getattr(matrix, "ndim", 1) == 1:
            matrix = [matrix]
        return [tuple(float(value) for value in row) for row in matrix]


def build_embedding_provider(config: AgentConfig) -> EmbeddingProvider:
    lexical_fallback = TokenSpaceEmbeddingProvider()
    deterministic_fallback = HashEmbeddingProvider(dimension=max(config.embedding_dimension, 128))
    if not config.embedding_enabled:
        return lexical_fallback

    model_reference = _resolve_local_model_reference(config)
    try:
        provider = SentenceTransformerEmbeddingProvider(
            model_name_or_path=model_reference,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
            local_files_only=config.embedding_local_files_only,
            query_prefix=config.embedding_query_prefix,
            document_prefix=config.embedding_document_prefix,
            fallback_provider=deterministic_fallback,
        )
        provider.probe()
        logger.info("Using local sentence-transformers embeddings from %s", model_reference)
        return provider
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Falling back to deterministic embeddings after local provider init failure for %s: %s",
            model_reference,
            exc,
        )
        return deterministic_fallback


def _resolve_local_model_reference(config: AgentConfig) -> str:
    if config.embedding_model_path.strip():
        return config.embedding_model_path.strip()
    model_name = config.embedding_model_name.strip()
    if not model_name:
        return model_name
    normalized_cache_name = model_name.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{normalized_cache_name}"
    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda item: item.name,
            reverse=True,
        )
        for snapshot in snapshots:
            if (snapshot / "config.json").exists() or (snapshot / "modules.json").exists():
                return str(snapshot)
        if snapshots:
            return str(snapshots[0])
    return model_name


def _normalize_text_input(texts: list[str], *, prefix: str = "") -> list[str]:
    normalized_prefix = prefix.strip()
    normalized: list[str] = []
    for text in texts:
        body = " ".join(str(text).split())
        if normalized_prefix:
            normalized.append(f"{normalized_prefix} {body}".strip())
        else:
            normalized.append(body)
    return normalized


def _token_vector(text: str) -> Counter[str]:
    normalized = normalize_identifier(text)
    tokens = [token for token in normalized.split("_") if token]
    return Counter(tokens)


def _hashed_vector(text: str, dimension: int) -> tuple[float, ...]:
    slots = [0.0] * dimension
    tokens = [token for token in str(text).lower().split() if token]
    if not tokens:
        return tuple(slots)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=_HASH_DIGEST_SIZE).digest()
        for chunk_index in range(0, len(digest), 2):
            slot = int.from_bytes(digest[chunk_index : chunk_index + 2], "little") % dimension
            sign = 1.0 if digest[chunk_index] % 2 == 0 else -1.0
            slots[slot] += sign
    norm = math.sqrt(sum(value * value for value in slots))
    if norm == 0:
        return tuple(slots)
    return tuple(value / norm for value in slots)


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _dense_cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left_item * right_item for left_item, right_item in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
