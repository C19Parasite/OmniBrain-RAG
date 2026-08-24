from app.vector_store import VectorStoreManager
from app.llm import LLMManager


class RAGPipeline:
    """
    RAG orchestrator connecting ChromaDB retrieval with LLM generation.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager = None,
        llm: LLMManager = None
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.llm = llm or LLMManager()

    def generate_response(self, query: str, top_k: int = 3) -> dict:
        """
        Retrieve relevant documents and generate a grounded answer.
        """

        if not query or not query.strip():
            return {
                "query": query,
                "question": query,
                "answer": "Please provide a question.",
                "context": "",
                "sources": []
            }

        search_results = self.vector_store.search_similar(
            query,
            top_k=top_k
        )

        retrieved_docs = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]

        context_blocks = []
        sources = []

        for doc, meta in zip(retrieved_docs, metadatas):
            source = meta.get("source", "N/A")
            page = meta.get("page", "N/A")

            context_blocks.append(
                f"[Source: {source}, Page {page}]\n{doc}"
            )

            sources.append({
                "source": source,
                "page": page
            })

        combined_context = (
            "\n\n".join(context_blocks)
            if context_blocks
            else "No relevant context found."
        )

        system_prompt = (
            "You are OmniBrain AI. Answer the user's question using "
            "only the retrieved context. If the context does not contain "
            "enough information, clearly say that the information is "
            "not available in the provided documents."
        )

        prompt = (
            f"Retrieved Context:\n\n"
            f"{combined_context}\n\n"
            f"User Question: {query}"
        )

        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )

        return {
            "query": query,
            "question": query,
            "answer": answer,
            "context": combined_context,
            "sources": sources
        }


if __name__ == "__main__":
    print("RAG Pipeline module created successfully.")