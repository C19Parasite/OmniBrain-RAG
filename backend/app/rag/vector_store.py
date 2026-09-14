import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..config import settings
from .hybrid import BM25Index, reciprocal_rank_fusion

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
        doc_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search against multimodal chunks.
        Supports filtering by chunk_type and specific document IDs.
        """
        if not query_embedding or not any(v != 0.0 for v in query_embedding):
            return []

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

        return formatted_results

    def search_hybrid(
        self,
        query: str = "",
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        chunk_type_filter: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        mode: str = "hybrid"
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
            return sparse_results[:top_k]
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
        mode: str = "hybrid"
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
