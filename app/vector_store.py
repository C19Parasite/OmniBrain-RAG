"""Embedding and vector storage for OmniBrain-RAG.

The module is independent of FastAPI and document parsing. Feed it normalised
chunks from ``DocumentProcessor.process_file()`` and it will embed, persist,
retrieve, list, and remove them.

Required packages:
    pip install qdrant-client sentence-transformers numpy
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)


class EmbeddingModel(Protocol):
    """Minimum interface needed from an embedding model."""

    def encode(self, sentences: str | Sequence[str], **kwargs: Any) -> Any: ...


class VectorStoreError(RuntimeError):
    """Raised when a vector-store operation cannot be completed safely."""


class VectorStoreManager:
    """Store document chunks in a local Qdrant collection.

    ``storage_path`` is persistent by default; pass ``None`` for an in-memory
    store in tests. The Sentence Transformer model is loaded lazily, and a
    custom ``embedding_model`` can be injected to keep tests offline.
    """

    def __init__(
        self,
        collection_name: str = "omnibrain_documents",
        *,
        storage_path: str | Path | None = "data/vector_db",
        model_name: str = "all-MiniLM-L6-v2",
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        self.collection_name = collection_name
        self.model_name = model_name
        self._embedding_model = embedding_model
        self._vector_size: int | None = self._model_dimension(embedding_model)

        if storage_path is None:
            self.client = QdrantClient(":memory:")
        else:
            storage_dir = Path(storage_path)
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_dir))

    @staticmethod
    def _model_dimension(model: EmbeddingModel | None) -> int | None:
        if model is None:
            return None
        get_dimension = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(get_dimension):
            dimension = get_dimension()
            if dimension:
                return int(dimension)
        return None

    def _get_embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise VectorStoreError(
                    "sentence-transformers is required to create embeddings. "
                    "Install the vector-store dependencies."
                ) from exc
            self._embedding_model = SentenceTransformer(self.model_name)
            self._vector_size = self._model_dimension(self._embedding_model)
        return self._embedding_model

    def _embed(self, texts: str | Sequence[str]) -> np.ndarray:
        model = self._get_embedding_model()
        values = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise VectorStoreError("Embedding model returned invalid vectors")

        dimension = int(vectors.shape[1])
        if self._vector_size is None:
            self._vector_size = dimension
        elif self._vector_size != dimension:
            raise VectorStoreError(
                f"Embedding dimension changed from {self._vector_size} to {dimension}"
            )
        self._ensure_collection()
        return vectors

    def _ensure_collection(self) -> None:
        if self._vector_size is None:
            raise VectorStoreError("Cannot create a collection before embedding a value")
        names = {item.name for item in self.client.get_collections().collections}
        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    @staticmethod
    def _normalise_chunk(chunk: dict[str, Any], index: int) -> dict[str, Any]:
        text = str(chunk.get("text", "")).strip()
        if not text:
            raise ValueError(f"Chunk {index} has no text")

        source = str(chunk.get("source", "unknown"))
        chunk_id = str(chunk.get("chunk_id", index))
        payload = {
            "chunk_id": chunk_id,
            "source": source,
            "page": chunk.get("page"),
            "text": text,
        }
        payload.update({
            key: value
            for key, value in chunk.items()
            if key not in payload and value is not None
        })
        return payload

    @staticmethod
    def _point_id(source: str, chunk_id: str) -> str:
        """Stable IDs ensure re-ingesting a source updates its chunks."""
        return str(uuid5(NAMESPACE_URL, f"omnibrain://{source}/{chunk_id}"))

    def add_documents(self, chunks: Iterable[dict[str, Any]]) -> int:
        """Embed and upsert parser chunks, returning the number stored."""
        normalised = [self._normalise_chunk(chunk, index) for index, chunk in enumerate(chunks)]
        if not normalised:
            return 0

        vectors = self._embed([chunk["text"] for chunk in normalised])
        points = [
            PointStruct(
                id=self._point_id(chunk["source"], chunk["chunk_id"]),
                vector=vector.tolist(),
                payload=chunk,
            )
            for chunk, vector in zip(normalised, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(points)

    # Compatibility alias for the existing FastAPI ingestion route.
    add_chunks = add_documents

    def add_document(self, filename: str, text: str) -> int:
        """Store one already-extracted document as a single chunk."""
        return self.add_documents([{
            "chunk_id": "0",
            "source": filename,
            "page": 1,
            "text": text,
        }])

    def search(self, query: str, *, top_k: int = 4) -> list[dict[str, Any]]:
        """Return best-matching chunks in an application-friendly shape."""
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self._embed(query)[0].tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "id": str(point.id),
                "score": float(point.score),
                **(point.payload or {}),
            }
            for point in results
        ]

    def search_similar(self, query: str, top_k: int = 4) -> dict[str, Any]:
        """Legacy response format used by the current RAG pipeline."""
        matches = self.search(query, top_k=top_k)
        return {
            "documents": [[match["text"] for match in matches]],
            "metadatas": [[
                {
                    "source": match.get("source", "unknown"),
                    "page": match.get("page"),
                    "chunk_id": match.get("chunk_id"),
                }
                for match in matches
            ]],
            "scores": [match["score"] for match in matches],
        }

    def get_documents(self, *, limit: int = 100, source: str | None = None) -> list[dict[str, Any]]:
        """List stored chunks, optionally restricted to one source document."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if self.collection_name not in {item.name for item in self.client.get_collections().collections}:
            return []
        query_filter = None
        if source is not None:
            query_filter = Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=source))
            ])
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [{"id": str(point.id), **(point.payload or {})} for point in points]

    def delete_document(self, source: str) -> None:
        """Delete all chunks that originated from ``source``."""
        if not source:
            raise ValueError("source cannot be empty")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=source))
            ]),
            wait=True,
        )

    def delete_chunks(self, point_ids: Sequence[str]) -> None:
        """Delete explicit Qdrant point IDs returned by ``search``/``get_documents``."""
        if point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=list(point_ids)),
                wait=True,
            )

    def clear(self) -> None:
        """Remove the current collection; its configured storage remains available."""
        if self.collection_name in {item.name for item in self.client.get_collections().collections}:
            self.client.delete_collection(self.collection_name)

    def count(self) -> int:
        """Return the number of stored chunks without loading their payloads."""
        if self.collection_name not in {item.name for item in self.client.get_collections().collections}:
            return 0
        return self.client.count(collection_name=self.collection_name, exact=True).count
