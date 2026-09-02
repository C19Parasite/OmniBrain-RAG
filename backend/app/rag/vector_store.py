import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..config import settings

class ChromaVectorStore:
    """
    Chroma-backed persistent vector store for OmniBrain.
    Stores and retrieves both 'text' and 'visual' chunks with rich source metadata.
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
        return len(chunks)

    def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        chunk_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search against multimodal chunks.
        Returns top-k matching chunks tagged as 'text' or 'visual' with source metadata.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        where_filter = {"chunk_type": chunk_type_filter} if chunk_type_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

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
                    "image_base64": self._image_store.get(chunk_id)
                })

        return formatted_results

    # Alias for backward compatibility
    def search_text(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search_chunks(query_embedding, top_k=top_k)

    def add_text_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        return self.add_chunks(chunks, embeddings)

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self._image_store.clear()
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
