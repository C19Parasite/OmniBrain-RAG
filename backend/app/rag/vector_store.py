import chromadb
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..config import settings
from .hybrid import BM25Index, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """
    Chroma-backed persistent vector store for OmniBrain.
    Stores and retrieves both 'text' and 'visual' chunks with rich source metadata.
    Integrates sparse BM25 indexing and Reciprocal Rank Fusion (RRF) for hybrid retrieval.
    """

    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = "multimodal_chunks"):
        self.persist_dir = Path(persist_dir or settings.VECTOR_STORE_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # In-memory store for chunk image payloads
        self._image_store: Dict[str, str] = {}
        
        # Sparse BM25 index for keyword & exact numeric matching
        self.bm25_index = BM25Index()
        self._sync_bm25_from_collection()

    @staticmethod
    def _validate_embedding(embedding: List[float], expected_dimension: Optional[int] = None) -> None:
        if embedding is None or len(embedding) == 0:
            raise ValueError("Embedding cannot be empty.")
        if expected_dimension is not None and len(embedding) != expected_dimension:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match expected {expected_dimension}."
            )
        if not all(math.isfinite(float(value)) for value in embedding):
            raise ValueError("Embedding contains non-finite values.")
        if not any(float(value) != 0.0 for value in embedding):
            raise ValueError("Embedding cannot be all zeros.")

    def ensure_embedding_space(self, embeddings: Any) -> bool:
        """Re-embed persisted chunks when their provider/model space changes."""
        expected_space = embeddings.embedding_space
        metadata = dict(self.collection.metadata or {})
        if metadata.get("embedding_space") == expected_space:
            return False

        count = self.collection.count()
        if count:
            data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            existing = data.get("embeddings")
            if existing is not None and len(existing):
                self._validate_embedding(existing[0], embeddings.dimension)
            pending = [
                (chunk_id, document, dict(metadata or {}))
                for chunk_id, document, metadata in zip(
                    list(data["ids"]),
                    list(data.get("documents") or []),
                    list(data.get("metadatas") or []),
                )
                if (metadata or {}).get("embedding_space") != expected_space
            ]
            logger.info("Re-embedding %s of %s chunks into %s.", len(pending), count, expected_space)
            # Commit each provider batch immediately. If a quota error occurs,
            # the next startup resumes only the unfinished chunks.
            for start in range(0, len(pending), 10):
                batch = pending[start:start + 10]
                vectors = embeddings.embed_documents([item[1] for item in batch])
                for vector in vectors:
                    self._validate_embedding(vector, embeddings.dimension)
                metadatas = []
                for _, _, metadata in batch:
                    metadata["embedding_space"] = expected_space
                    metadatas.append(metadata)
                self.collection.upsert(
                    ids=[item[0] for item in batch],
                    documents=[item[1] for item in batch],
                    metadatas=metadatas,
                    embeddings=vectors,
                )

            still_pending = self.collection.get(
                include=["metadatas"]
            ).get("metadatas") or []
            if any((metadata or {}).get("embedding_space") != expected_space for metadata in still_pending):
                raise RuntimeError("Embedding-space migration did not complete.")

        metadata["embedding_space"] = expected_space
        metadata["embedding_dimension"] = embeddings.dimension
        self.collection.modify(metadata=metadata)
        return count > 0

    def _sync_bm25_from_collection(self):
        """Populates BM25 index from existing persistent ChromaDB collection."""
        try:
            count = self.collection.count()
            if count > 0:
                data = self.collection.get(include=["documents", "metadatas"])
                if data and data.get("ids"):
                    chunks = []
                    for idx, cid in enumerate(data["ids"]):
                        doc = data["documents"][idx] if data.get("documents") else ""
                        meta = data["metadatas"][idx] if data.get("metadatas") else {}
                        chunks.append({
                            "id": cid,
                            "text": doc,
                            "source_document": meta.get("source_document", "unknown"),
                            "page_number": int(meta.get("page_number", 1)),
                            "chunk_type": meta.get("chunk_type", "text"),
                            "section_title": meta.get("section_title", "General"),
                            "doc_id": meta.get("doc_id", "doc_0"),
                            "image_base64": self._image_store.get(cid)
                        })
                    self.bm25_index.add_chunks(chunks)
        except Exception as e:
            print(f"[VectorStore] Notice: BM25 sync on init: {e}")

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Adds multimodal chunks (text or visual) to ChromaDB.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        for embedding in embeddings:
            self._validate_embedding(embedding)

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = []

        for c in chunks:
            chunk_type = str(c.get("chunk_type", "text"))
            chunk_id = c["id"]

            if c.get("image_base64"):
                self._image_store[chunk_id] = c["image_base64"]

            metadatas.append({
                "source_document": str(c.get("source_document", "unknown")),
                "page_number": int(c.get("page_number", 1)),
                "chunk_type": chunk_type,
                "section_title": str(c.get("section_title", "General")),
                "doc_id": str(c.get("doc_id", "doc_0"))
            })

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        # Update BM25 sparse index
        self.bm25_index.add_chunks(chunks)
        return len(chunks)

    def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        chunk_type_filter: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search against multimodal chunks.
        Supports filtering by chunk_type and specific document IDs.
        """
        self._validate_embedding(query_embedding)
        if min_similarity is None:
            min_similarity = settings.SIMILARITY_THRESHOLD

        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        
        # Build where filter
        where_filter = None
        conditions = []
        if chunk_type_filter:
            conditions.append({"chunk_type": chunk_type_filter})
        if doc_ids and len(doc_ids) > 0:
            if len(doc_ids) == 1:
                conditions.append({"doc_id": doc_ids[0]})
            else:
                conditions.append({"doc_id": {"$in": doc_ids}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            # Fallback if where filter didn't match any documents
            return []

        formatted_results = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for idx in range(len(ids)):
                dist = float(distances[idx])
                similarity = round(max(0.0, min(1.0, 1.0 - dist)), 4)
                if similarity < min_similarity:
                    continue
                meta = metas[idx]
                chunk_id = ids[idx]

                formatted_results.append({
                    "id": chunk_id,
                    "text": docs[idx],
                    "source_document": meta.get("source_document", "unknown"),
                    "page_number": int(meta.get("page_number", 1)),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "section_title": meta.get("section_title", "General"),
                    "similarity_score": similarity,
                    "image_base64": self._image_store.get(chunk_id),
                    "doc_id": meta.get("doc_id", "doc_0"),
                    "retrieval_method": "dense"
                })

        logger.info(
            "Dense retrieval: requested=%s returned=%s threshold=%.4f scores=%s",
            top_k, len(formatted_results), min_similarity,
            [(item["id"], item["similarity_score"]) for item in formatted_results],
        )

        return formatted_results

    def search_hybrid(
        self,
        query: str = "",
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        chunk_type_filter: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        mode: str = "dense"
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining Dense Semantic Cosine Search and Sparse BM25 Search
        via Reciprocal Rank Fusion (RRF).
        Supports modes: 'hybrid', 'dense', 'bm25' (or 'sparse').
        """
        if mode == "dense":
            if not query_embedding:
                return []
            return self.search_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                chunk_type_filter=chunk_type_filter,
                doc_ids=doc_ids
            )

        if mode in ("sparse", "bm25"):
            if not query:
                return []
            return self.bm25_index.search(
                query=query,
                top_k=top_k,
                doc_ids=doc_ids,
                chunk_type_filter=chunk_type_filter
            )

        # Mode: Hybrid (RRF)
        candidate_k = max(top_k * 2, 8)
        dense_results: List[Dict[str, Any]] = []
        if query_embedding and any(v != 0.0 for v in query_embedding):
            dense_results = self.search_chunks(
                query_embedding=query_embedding,
                top_k=candidate_k,
                chunk_type_filter=chunk_type_filter,
                doc_ids=doc_ids
            )

        sparse_results: List[Dict[str, Any]] = []
        if query and query.strip():
            sparse_results = self.bm25_index.search(
                query=query,
                top_k=candidate_k,
                doc_ids=doc_ids,
                chunk_type_filter=chunk_type_filter
            )

        if not dense_results and not sparse_results:
            return []
        if not dense_results:
            # BM25 may re-rank semantic candidates, but cannot manufacture
            # literal-overlap results after dense retrieval rejects them.
            return []
        if not sparse_results:
            return dense_results[:top_k]

        dense_ids = {item["id"] for item in dense_results}
        sparse_results = [item for item in sparse_results if item.get("id") in dense_ids]
        if not sparse_results:
            return dense_results[:top_k]

        return reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k
        )

    def search_text(
        self,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        query: str = "",
        mode: str = "dense"
    ) -> List[Dict[str, Any]]:
        """High-level text search supporting hybrid, dense, or sparse modes."""
        return self.search_hybrid(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            doc_ids=doc_ids,
            mode=mode
        )

    def add_text_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        return self.add_chunks(chunks, embeddings)

    def count(self) -> int:
        return self.collection.count()

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Deletes all chunks matching doc_id from the vector store and BM25 index."""
        try:
            results = self.collection.get(where={"doc_id": doc_id})
            if results and results.get("ids"):
                ids_to_del = results["ids"]
                self.collection.delete(ids=ids_to_del)
                for cid in ids_to_del:
                    self._image_store.pop(cid, None)
                self.bm25_index.delete_by_doc_id(doc_id)
                return len(ids_to_del)
        except Exception:
            pass
        return 0

    def clear(self):
        self._image_store.clear()
        self.bm25_index.clear()
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
