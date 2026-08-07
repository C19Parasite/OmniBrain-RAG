from app.vector_store import VectorStoreManager

class RAGPipeline:
    """
    RAG Orchestrator connecting retrieved ChromaDB contexts to generation logic.
    """
    def __init__(self, vector_store: VectorStoreManager = None):
        self.vector_store = vector_store or VectorStoreManager()

    def generate_response(self, query: str, top_k: int = 3) -> dict:
        """
        Retrieves matching contexts and structures a grounded RAG response prompt.
        """
        search_results = self.vector_store.search_similar(query, top_k=top_k)
        
        retrieved_docs = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]

        context_blocks = []
        for doc, meta in zip(retrieved_docs, metadatas):
            page_info = f"[Source: {meta.get('source', 'N/A')}, Page {meta.get('page', 'N/A')}]"
            context_blocks.append(f"{page_info}\n{doc}")

        combined_context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        # Augmented prompt template ready for an LLM integration (OpenAI/Ollama/HuggingFace)
        system_prompt = (
            f"You are OmniBrain AI. Answer the query based strictly on the retrieved context below:\n\n"
            f"--- CONTEXT ---\n{combined_context}\n---------------\n\n"
            f"User Query: {query}"
        )

        return {
            "query": query,
            "context": combined_context,
            "sources": metadatas,
            "prompt": system_prompt
        }

if __name__ == "__main__":
    print("RAG Pipeline module created successfully.")
