from typing import List, Dict, Any, Optional
from ..config import settings
from ..rag.embeddings import TextEmbeddings
from ..rag.vector_store import ChromaVectorStore

class SearchAgent:
    """
    Semantic Vector Search Agent for Text-based Financial RAG.
    Embeds incoming queries and retrieves top-k chunks with rich source metadata.
    Supports filtering by specific document IDs.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embeddings: Optional[TextEmbeddings] = None
    ):
        self.embeddings = embeddings or TextEmbeddings()
        self.vector_store = vector_store or ChromaVectorStore()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector search for a natural language query.
        Returns top-k matching text chunks with source document, page, and similarity scores.
        """
        if not query or not query.strip():
            return []

        k = top_k if (top_k is not None and top_k > 0) else settings.TOP_K

        # 1. Embed query
        query_vector = self.embeddings.embed_text(query)

        # 2. Query ChromaDB vector store with optional doc_ids filter
        results = self.vector_store.search_text(
            query_embedding=query_vector,
            top_k=k,
            doc_ids=doc_ids
        )

        return results
