from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from openai import OpenAI

from data_agent_baseline.config import RetrievalConfig

logger = logging.getLogger("dabench.semantic.embedding")

_HASH_DIGEST_SIZE = 8


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Embedding matrix must be 2-dimensional.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms


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


def _resolve_model_reference(config: RetrievalConfig) -> str:
    if config.embedding_model_path.strip():
        return config.embedding_model_path.strip()
    model_name = config.embedding_model_name.strip()
    if not model_name:
        return model_name

    normalized_cache_name = model_name.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{normalized_cache_name}"
    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        ranked_snapshots = sorted(snapshots, key=lambda item: item.name, reverse=True)
        for snapshot in ranked_snapshots:
            if (snapshot / "config.json").exists() or (snapshot / "modules.json").exists():
                return str(snapshot)
        if ranked_snapshots:
            return str(ranked_snapshots[0])
    return model_name


@dataclass(frozen=True, slots=True)
class HashEmbeddingProvider:
    dimension: int = 1024

    def __post_init__(self) -> None:
        if self.dimension < _HASH_DIGEST_SIZE:
            raise ValueError(
                f"HashEmbeddingProvider dimension must be >= {_HASH_DIGEST_SIZE}, got {self.dimension}."
            )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = [token for token in str(text).lower().split() if token]
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=_HASH_DIGEST_SIZE).digest()
                for chunk_index in range(0, len(digest), 2):
                    slot = int.from_bytes(digest[chunk_index : chunk_index + 2], "little") % self.dimension
                    sign = 1.0 if digest[chunk_index] % 2 == 0 else -1.0
                    matrix[row_index, slot] += sign
        return _normalize_rows(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


@dataclass(frozen=True, slots=True)
class OpenAIEmbeddingProvider:
    model: str
    api_base: str
    api_key: str
    query_prefix: str = ""
    document_prefix: str = ""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if not self.api_key:
            raise RuntimeError("Missing embedding API key.")
        client = OpenAI(api_key=self.api_key, base_url=self.api_base.rstrip("/") or None)
        response = client.embeddings.create(
            model=self.model,
            input=_normalize_text_input(texts, prefix=self.document_prefix),
        )
        matrix = np.array([item.embedding for item in response.data], dtype=np.float32)
        return _normalize_rows(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError("Missing embedding API key.")
        client = OpenAI(api_key=self.api_key, base_url=self.api_base.rstrip("/") or None)
        response = client.embeddings.create(
            model=self.model,
            input=_normalize_text_input([text], prefix=self.query_prefix),
        )
        matrix = np.array([item.embedding for item in response.data], dtype=np.float32)
        return _normalize_rows(matrix)[0]


@dataclass(slots=True)
class SentenceTransformerEmbeddingProvider:
    model_name_or_path: str
    device: str = "cpu"
    batch_size: int = 16
    local_files_only: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    _model: object | None = None

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("sentence-transformers is not available for local embeddings.") from exc

        model_name_or_path = self.model_name_or_path
        if Path(model_name_or_path).exists():
            model_name_or_path = str(Path(model_name_or_path))

        self._model = SentenceTransformer(
            model_name_or_path,
            device=self.device,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        return self._model

    def _encode(self, texts: list[str], *, prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load_model()
        inputs = _normalize_text_input(texts, prefix=prefix)
        matrix = model.encode(
            inputs,
            batch_size=max(1, self.batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        encoded = np.asarray(matrix, dtype=np.float32)
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        return _normalize_rows(encoded)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prefix=self.document_prefix)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], prefix=self.query_prefix)[0]


def create_embedding_provider(config: RetrievalConfig) -> EmbeddingProvider:
    provider_name = config.embedding_provider.strip().lower()
    if not config.enable_dense_linking:
        logger.info("Dense linking disabled; using deterministic hash embeddings.")
        return HashEmbeddingProvider(dimension=config.embedding_dimension)

    if provider_name == "api":
        if not config.embedding_api_key:
            logger.warning("API embedding provider requested without an API key; using hash embeddings.")
            return HashEmbeddingProvider(dimension=config.embedding_dimension)
        if not config.embedding_api_base:
            logger.warning("API embedding provider requested without an API base; using hash embeddings.")
            return HashEmbeddingProvider(dimension=config.embedding_dimension)
        try:
            return OpenAIEmbeddingProvider(
                model=config.embedding_model_name,
                api_base=config.embedding_api_base,
                api_key=config.embedding_api_key,
                query_prefix=config.embedding_query_prefix,
                document_prefix=config.embedding_document_prefix,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to hash embeddings after API provider init failure: %s", exc)
            return HashEmbeddingProvider(dimension=config.embedding_dimension)

    if provider_name == "local":
        model_reference = _resolve_model_reference(config)
        try:
            provider = SentenceTransformerEmbeddingProvider(
                model_name_or_path=model_reference,
                device=config.embedding_device,
                batch_size=config.embedding_batch_size,
                local_files_only=config.embedding_local_files_only,
                query_prefix=config.embedding_query_prefix,
                document_prefix=config.embedding_document_prefix,
            )
            # Force a tiny encode up front so failures happen at startup, not mid-run.
            probe = provider.embed_query("embedding provider health check")
            if probe.ndim != 1 or probe.size == 0:
                raise RuntimeError("Local embedding provider returned an empty probe vector.")
            logger.info("Using local sentence-transformers embeddings from %s", model_reference)
            return provider
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Falling back to hash embeddings after local provider init failure for %s: %s",
                model_reference,
                exc,
            )
            return HashEmbeddingProvider(dimension=config.embedding_dimension)

    logger.warning(
        "Unknown embedding provider '%s'; using deterministic hash embeddings.",
        config.embedding_provider,
    )
    return HashEmbeddingProvider(dimension=config.embedding_dimension)
