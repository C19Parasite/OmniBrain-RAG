"""
Enhanced RAG Pipeline with LangGraph Agentic Orchestration.

Combines the existing RAGPipeline with intelligent agent routing:
- semantic_search → Uses vector DB (your existing RAG)
- sql_query → Uses SQL agent for structured data
- vision_analysis → Uses vision agent for charts/tables
- hybrid → Uses multiple agents and synthesizes results
"""

from app.vector_store import VectorStoreManager
from app.llm import LLMManager
from app.agents.graph import build_agentic_graph
from app.schemas.state import AgentState
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    """
    Enhanced RAG orchestrator with agentic routing.
    
    Supports:
    1. Original RAG (semantic search) - Your existing code
    2. Agentic orchestration (intelligent routing) - New
    
    Usage:
        pipeline = RAGPipeline()
        
        # Use intelligent routing (default)
        result = pipeline.generate_response("What was the stock price?")
        
        # Or use original RAG (for backward compatibility)
        result = pipeline.generate_response_rag("What does CEO say?")
    """

    def __init__(
        self,
        vector_store: VectorStoreManager = None,
        llm: LLMManager = None,
        use_agentic: bool = True,
        provider: str = "groq"
    ):
        """
        Initialize Enhanced RAG Pipeline
        
        Args:
            vector_store: Your existing VectorStoreManager
            llm: Your existing LLMManager
            use_agentic: Use intelligent agentic routing (default: True)
            provider: LLM provider ("groq" or "openai")
        """
        self.vector_store = vector_store or VectorStoreManager()
        self.llm = llm or LLMManager()
        self.use_agentic = use_agentic
        self.provider = provider
        
        # Initialize agentic orchestrator if enabled
        if self.use_agentic:
            try:
                self.agentic_graph = build_agentic_graph(provider=provider)
                self.agentic_enabled = True
            except Exception as e:
                print(f"⚠️  Warning: Agentic mode disabled - {str(e)}")
                self.agentic_enabled = False
                self.use_agentic = False
        else:
            self.agentic_enabled = False

    # ===== ORIGINAL RAG METHOD (Backward Compatible) =====

    def generate_response_rag(self, query: str, top_k: int = 3) -> dict:
        """
        Original RAG pipeline using vector DB semantic search.
        
        This is the original method - use when you want pure semantic search.
        
        Args:
            query: User's question
            top_k: Number of top results to retrieve
            
        Returns:
            Dictionary with answer, context, and sources
        """
        
        if not query or not query.strip():
            return {
                "query": query,
                "question": query,
                "answer": "Please provide a question.",
                "context": "",
                "sources": [],
                "method": "rag"
            }

        # Search vector database
        search_results = self.vector_store.search_similar(
            query,
            top_k=top_k
        )

        retrieved_docs = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]

        # Build context blocks
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

        # Generate answer
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
            "sources": sources,
            "method": "rag",
            "routing_decision": "semantic_search"
        }

    # ===== ENHANCED AGENTIC METHOD (New) =====

    def generate_response(self, query: str, top_k: int = 3, pdf_path: Optional[str] = None) -> dict:
        """
        Enhanced RAG with intelligent agentic routing.
        
        Routes to:
        - semantic_search → Vector DB search (your existing RAG)
        - sql_query → SQL database queries
        - vision_analysis → Chart/table analysis
        - hybrid → Multiple agents working together
        
        Args:
            query: User's question
            top_k: Number of results (for vector search)
            pdf_path: Path to PDF for vision analysis (optional)
            
        Returns:
            Dictionary with answer, routing info, and sources
        """
        
        # If agentic mode is disabled, fall back to original RAG
        if not self.use_agentic or not self.agentic_enabled:
            return self.generate_response_rag(query, top_k)
        
        if not query or not query.strip():
            return {
                "query": query,
                "question": query,
                "answer": "Please provide a question.",
                "context": "",
                "sources": [],
                "routing_decision": "none",
                "method": "agentic"
            }

        try:
            # Create initial state for agentic orchestration
            initial_state = AgentState(
                user_query=query,
                query_analysis=None,
                search_results=[],
                sql_results=[],
                vision_results=[],
                intermediate_steps=[],
                final_response="",
                errors=[],
                routing_decision=""
            )
            
            # Add PDF path if provided
            if pdf_path:
                initial_state["pdf_path"] = pdf_path
            
            # Execute agentic orchestration
            result = self.agentic_graph.invoke(initial_state)
            
            # Extract results based on routing decision
            routing_decision = result["routing_decision"]
            final_response = result["final_response"]
            
            # Build sources from results
            sources = []
            
            # Add search results as sources
            for search_result in result.get("search_results", []):
                sources.append({
                    "type": "semantic_search",
                    "source": search_result.get("source", "N/A"),
                    "page": search_result.get("page", "N/A"),
                    "score": search_result.get("score", 0)
                })
            
            # Add SQL results as sources
            for sql_result in result.get("sql_results", []):
                sources.append({
                    "type": "sql_query",
                    "data": sql_result
                })
            
            # Add vision results as sources
            for vision_result in result.get("vision_results", []):
                sources.append({
                    "type": "vision_analysis",
                    "page": vision_result.get("page", "N/A"),
                    "extracted_data": vision_result.get("extracted_data", "")
                })
            
            # Build context from search results
            context_blocks = []
            for search_result in result.get("search_results", []):
                content = search_result.get("content", "")
                source = search_result.get("source", "N/A")
                page = search_result.get("page", "N/A")
                context_blocks.append(f"[{source}:Page {page}]\n{content}")
            
            combined_context = "\n\n".join(context_blocks) if context_blocks else ""
            
            return {
                "query": query,
                "question": query,
                "answer": final_response,
                "context": combined_context,
                "sources": sources,
                "routing_decision": routing_decision,
                "method": "agentic",
                "intermediate_steps": [step[0] for step in result.get("intermediate_steps", [])],
                "confidence": result["query_analysis"].confidence if result.get("query_analysis") else 0,
                "errors": result.get("errors", [])
            }
        
        except Exception as e:
            # If agentic routing fails, fall back to original RAG
            print(f"⚠️  Agentic routing failed: {str(e)}. Falling back to RAG.")
            return self.generate_response_rag(query, top_k)

    # ===== UTILITY METHODS =====

    def toggle_agentic(self, enabled: bool):
        """Enable/disable agentic mode"""
        self.use_agentic = enabled and self.agentic_enabled

    def get_routing_info(self, query: str) -> dict:
        """
        Get routing decision for a query without generating response.
        
        Useful for debugging or understanding query classification.
        """
        if not self.agentic_enabled:
            return {"error": "Agentic mode not enabled"}
        
        try:
            from app.agents.supervisor import QueryRouter
            router = QueryRouter(provider=self.provider)
            analysis = router.analyze_query(query)
            
            return {
                "query": query,
                "query_type": analysis.query_type,
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning
            }
        except Exception as e:
            return {"error": str(e)}

    def search_documents(self, query: str, top_k: int = 5) -> dict:
        """
        Direct vector database search (for backward compatibility).
        """
        search_results = self.vector_store.search_similar(query, top_k=top_k)
        
        results = []
        for doc, meta in zip(
            search_results.get("documents", [[]])[0],
            search_results.get("metadatas", [[]])[0]
        ):
            results.append({
                "content": doc,
                "source": meta.get("source", "N/A"),
                "page": meta.get("page", "N/A")
            })
        
        return {"query": query, "results": results}


if __name__ == "__main__":
    print("Enhanced RAG Pipeline module created successfully.")
    
   
